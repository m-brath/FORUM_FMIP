#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alternative sensor-response convolution: apply the SRF to BRIGHTNESS TEMPERATURE.

This is the alternative to convolve_sensor_response.py -- that script is left
untouched and this script only ever writes new files (prefix
Reference_BTconvolved_*, vs. Reference_convolved_* of the standard pipeline).

While convolve_sensor_response.py follows the physical instrument response
(SRF convolution of the high-resolution RADIANCE, then Planck inversion of
the convolved radiance), this script reverses the order:

    1. Planck-invert every high-resolution radiance point to a
       monochromatic brightness temperature,
    2. apply the sensor response function (SRF) as a spectral weighted
       average of BRIGHTNESS TEMPERATURE, evaluated at the sensor channel
       centres.

As in convolve_sensor_response.py, the output is on the sensor channel grid
(IASI: 8461 channels, 645-2760 cm^-1, 0.25 cm^-1 spacing), not on the
simulation frequency grid: the SRF-weighted averages are computed directly at
the channel centres (frtmip_aux_functions.build_srf_channel_operator).

The two are NOT equivalent: Planck inversion is nonlinear, so averaging in
BT space underestimates the spectrally-integrated emission compared to the
radiance-space convolution (the classic Planck-mean vs. brightness-temperature
mean difference).  This product is therefore a diagnostic alternative that
shows the sensitivity of the convolved T_B to the convolution order, not a
physically correct instrument simulation.

Environment (mamba/conda):    pyarts
                              (same as fmip_radiance_batch.py)

Input data: Spectral radiance chunks from fmip_radiance_batch.py /
fmip_radiance_slurm_sub.py in the run sub folder selected by the run_name
constant, results/RADIANCE/<run_name>/:
    Reference_radiance_TOA_Nf*.nc  (one file per chunk)
    If the chunk files are no longer present (e.g. they were removed after
    merge_radiance_chunks.py ran with delete_chunk_files = True), the merged
    files Reference_radiance_TOA_merged_*.nc are used instead.
    data/sensor_response.nc  (SRF wavenumber offsets and response)
    data/satdata.nc          (IASI channel centres, "wvn" -- the output grid)

Output:
    results/RADIANCE/<run_name>/Reference_BTconvolved_*_chunk*.nc
    (one per input radiance chunk, written to the same run sub folder as
    its input)
    Each output file contains:
    - brightness_temperature: SRF-convolved brightness temperature (frequency, idx)
    - spectral_BT_monochromatic (optional): un-convolved monochromatic BT,
      kept on the simulation grid (its own frequency_sim axis)
    - Metadata: SRF name, convolution order, physical constants used
    The frequency / wavenumber coordinates of brightness_temperature are the
    IASI channel centres (8461 points) -- the sensor's spectral resolution,
    not the simulation grid of the input file.
    The BT variables are stored as float32 (half the float64 file size;
    float32 gives ~1e-7 K relative precision, far below any relevant BT
    tolerance).

Example
-------
Set the run_name constant to the run sub folder to process, then run the
script; it discovers all radiance chunks in
results/RADIANCE/<run_name>/ and processes them sequentially.

    $ python convolve_sensor_response_bt.py

Notes
-----
* The input radiance grid is already extended to cover the SRF bandwidth
  (see fmip_radiance_batch.py), so no additional padding is needed.
  Applying the SRF needs the full spectrum around every channel, so the input
  files must cover the whole satellite range (merged files or a non-chunked
  run) -- frequency-chunk files of a SLURM run have to be merged first.
* The Planck inversion (frtmip_aux_functions.radiance2brightness_temp) is
  done FIRST, on the full-resolution spectrum; the SRF filter
  (frtmip_aux_functions.build_srf_channel_operator /
  convolve_with_srf_to_channels) is then a unit-sum weighted average of BT
  evaluated at the channel centres -- it does not care about the physical
  units of the array it filters, only that frequency is axis 0.  The
  monochromatic BT of step 1 therefore lives on the simulation grid while the
  convolved BT of step 2 lives on the channel grid.
* Output file names use Reference_BTconvolved_* so they neither overwrite
  nor collide with the Reference_convolved_* files of the standard pipeline;
  in particular, compare_convolved_with_obs.py (pattern Reference_convolved_*.nc)
  does not pick these files up.
* Processing is fast: the Planck inversion runs on the full-resolution input,
  building the SRF channel operator takes ~1 s, and applying it to a chunk
  takes a few seconds.
"""

import time
import pathlib
import glob
import re

import numpy as np
import xarray as xr
import scipy.constants as scc

import frtmip_aux_functions as fax


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

# Location of this script
script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"

# run sub folder of results/RADIANCE/ written by one launch
# (run_name = <setup_name>_Nf<N_wvn>_Nza<N_za>)
run_name = "fmip_radiance_batch_slurm_Nf108951_Nza20"
results_folder = script_dir.parent / "results" / "RADIANCE" / run_name

# Input data files
sensor_response_file = data_dir / "sensor_response.nc"
# output grid: sensor (IASI) channel centres, taken from the "wvn" coordinate
# of the satellite data file (same grid define_frequency_grid uses to build
# the simulation range)
satdata_file = data_dir / "satdata.nc"

# Whether to also convolve the full 5D field (frequency, idx, lev, zenith,
# azimuth) if present in the input file
# (normally only TOA slice is used, but enabling this may help with debugging)
include_full_field = False

# Output options
export_results = True
export_monochromatic_bt = False  # save the un-convolved monochromatic BT
# (full-resolution BT field, same size as the radiance input -- keep False
# unless needed for diagnostics)
export_convolved_bt = True       # save the SRF-convolved brightness temperature


# =============================================================================
# %% helpers
# =============================================================================

def discover_radiance_chunks(results_folder, run_name):
    """
    Discover all radiance output files of one run, pinned to run_name.

    Same file-discovery logic as convolve_sensor_response.py (kept local so
    the two scripts stay independent): the search covers the run sub folder
    selected by the run_name constant (results/RADIANCE/<run_name>/); the
    recursive glob also picks up any files in nested sub folders.  Only files
    whose names belong to run_name are processed:

    * Unmerged chunk/non-chunked files (pattern Reference_radiance_TOA_Nf*.nc)
      must contain the setup-name part of run_name plus its _Nf<..>_/ _Nza<..>_
      tags.  run_name is <setup_name>_Nf<N_wvn>_Nza<N_za>, but the file names
      put the Nf/Nza tags first (Reference_radiance_TOA_Nf*_Nza*_<setup_name>_
      ...), so the parts are extracted from run_name and matched separately.
    * If no unmerged files are found -- e.g. a SLURM run whose chunk files
      were deleted after merge_radiance_chunks.py ran with
      delete_chunk_files = True -- falls back to the merged files, whose
      names are Reference_radiance_TOA_merged_<run_name>[_chunk###].nc,
      matched by the full run_name.  Merged files hold the same variables
      and coordinates as the chunk files.

    Preferring unmerged over merged files avoids processing the same
    frequencies twice when both kinds of files are present.

    Parameters
    ----------
    results_folder : pathlib.Path
        Path to the run sub folder results/RADIANCE/<run_name>/.
    run_name : str
        Run identifier (<setup_name>_Nf<N_wvn>_Nza<N_za>) used to pin the
        file name patterns to this run.

    Returns
    -------
    list of pathlib.Path
        Sorted list of NetCDF files matching the chunk pattern
        Reference_radiance_TOA_Nf*.nc for this run, or the merged pattern
        Reference_radiance_TOA_merged_<run_name>*.nc as fallback.
    """
    # parse run_name = <setup_name>_Nf<N_wvn>_Nza<N_za>
    # (fall back to matching only the full run_name if the layout differs)
    match = re.match(r"(.+)_Nf(\d+)_Nza(\d+)$", run_name)
    if match:
        setup_name, nf_tag, nza_tag = (
            match.group(1), f"_Nf{match.group(2)}_", f"_Nza{match.group(3)}_"
        )
    else:
        setup_name, nf_tag, nza_tag = run_name, None, None

    pattern = str(results_folder / "**" / "Reference_radiance_TOA_Nf*.nc")
    files = [
        pathlib.Path(f)
        for f in glob.glob(pattern, recursive=True)
        if setup_name in pathlib.Path(f).name
        and (nf_tag is None or nf_tag in pathlib.Path(f).name)
        and (nza_tag is None or nza_tag in pathlib.Path(f).name)
    ]
    if not files:
        merged_pattern = str(
            results_folder / "**"
            / f"Reference_radiance_TOA_merged_{run_name}*.nc"
        )
        files = [pathlib.Path(f) for f in glob.glob(merged_pattern, recursive=True)]
    return sorted(files)


def validate_and_normalize_srf(srf_wvn, srf_response):
    """
    Validate SRF and normalize to unit area integral if needed.

    Parameters
    ----------
    srf_wvn : ndarray
        SRF wavenumber offsets [cm^-1].
    srf_response : ndarray
        SRF response values.

    Returns
    -------
    srf_wvn : ndarray
        (Validated) SRF wavenumber offsets.
    srf_response : ndarray
        Normalized SRF response (integral ≈ 1.0 in wavenumber space).
    norm_factor : float
        Normalization factor applied.
    """
    # Sort by wavenumber for integration
    sorted_idx = np.argsort(srf_wvn)
    wvn_sorted = srf_wvn[sorted_idx]
    resp_sorted = srf_response[sorted_idx]

    # Integrate using trapezoidal rule (wavenumber space)
    integral = np.trapezoid(resp_sorted, wvn_sorted)

    if abs(integral) < 1e-10:
        raise ValueError(f"SRF integral is too small ({integral}), cannot normalize.")

    norm_factor = integral
    srf_response_normalized = srf_response / norm_factor

    print(f"  SRF area normalization factor: {norm_factor:.6f}")
    if abs(norm_factor - 1.0) > 0.01:
        print(f"  WARNING: SRF was not unit area normalized; "
              f"applied normalization factor {norm_factor:.6f}")

    return srf_wvn, srf_response_normalized, norm_factor


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()

    # -------------------------------------------------------------------------
    # Load sensor response function
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Brightness-temperature convolution (SRF applied to BT, not radiance)")
    print("=" * 70)

    print("\nLoading sensor response...")
    sensor_response = xr.open_dataset(sensor_response_file)
    srf_wvn = sensor_response["wvn"].values
    srf_response = sensor_response["response"].values
    sensor_response.close()

    print(f"  SRF shape: {srf_response.shape}")
    print(f"  SRF wavenumber range: {srf_wvn.min():.6f} to {srf_wvn.max():.6f} cm^-1")
    print(f"  SRF response range: {srf_response.min():.6e} to {srf_response.max():.6e}")

    # Validate and normalize SRF
    srf_wvn, srf_response, srf_norm_factor = validate_and_normalize_srf(srf_wvn, srf_response)

    # -------------------------------------------------------------------------
    # Load the sensor (IASI) channel grid -- the spectral grid of the output
    # -------------------------------------------------------------------------
    print("\nLoading sensor channel grid...")
    if not satdata_file.is_file():
        raise FileNotFoundError(
            f"{satdata_file} not found.  It provides the sensor (IASI) channel "
            f"centres that the convolved products are written on; run "
            f"prepare_data.py first."
        )
    satdata = xr.open_dataset(satdata_file)
    channel_wvn_all = np.sort(satdata["wvn"].values)
    satdata.close()
    print(f"  {channel_wvn_all.size} channels, {channel_wvn_all.min():.2f} to "
          f"{channel_wvn_all.max():.2f} cm^-1, spacing "
          f"{np.median(np.diff(channel_wvn_all)):.4f} cm^-1")

    # -------------------------------------------------------------------------
    # Discover radiance chunks
    # -------------------------------------------------------------------------
    print("\nDiscovering radiance chunks...")
    if not results_folder.is_dir():
        results_root = results_folder.parent
        available = sorted(
            p.name for p in results_root.iterdir() if p.is_dir()
        ) if results_root.is_dir() else []
        raise FileNotFoundError(
            f"Run sub folder {results_folder} does not exist. "
            f"Set run_name to one of the available runs: "
            f"{', '.join(available) if available else '(none)'}"
        )
    radiance_files = discover_radiance_chunks(results_folder, run_name)

    if not radiance_files:
        raise FileNotFoundError(
            f"No radiance output files (Reference_radiance_TOA_Nf*.nc or "
            f"Reference_radiance_TOA_merged_*.nc) found in {results_folder}. "
            f"Run fmip_radiance_batch.py / fmip_radiance_slurm_main.py first."
        )

    print(f"  Found {len(radiance_files)} radiance chunk(s):")
    for rf in radiance_files:
        print(f"    - {rf.relative_to(results_folder)}")

    # -------------------------------------------------------------------------
    # Process each radiance chunk
    # -------------------------------------------------------------------------
    all_stats = []

    # SRF channel operator: built from the first input file's frequency grid
    # and reused for all files of the run (all files carry the same grid)
    channel_operator = None
    channel_wvn = None
    channel_frequency = None
    f_grid_ref = None
    t_operator = 0.0

    for ck, radiance_file in enumerate(radiance_files):
        print("\n" + "=" * 70)
        print(f"Chunk {ck + 1}/{len(radiance_files)}: {radiance_file.name}")
        print("=" * 70)

        t_chunk = time.time()

        # Load radiance data
        print("  Loading radiance data...")
        radiance_ds = xr.open_dataset(radiance_file)
        spectral_radiance_TOA = radiance_ds["spectral_radiance_TOA"].values
        f_grid = radiance_ds.coords["frequency"].values
        wvn_grid = radiance_ds.coords["wavenumber"].values
        idx_coords = radiance_ds.coords["idx"].values

        # optional: full altitude-resolved cube, if exported by the batch run
        spectral_radiance_field = None
        if include_full_field and "spectral_radiance_field" in radiance_ds:
            spectral_radiance_field = radiance_ds["spectral_radiance_field"].values

        # Get metadata from the dataset
        zenith_angle = radiance_ds["spectral_radiance_TOA"].attrs.get(
            "zenith_angle", 180.0
        )

        n_wvn, n_idx = spectral_radiance_TOA.shape
        print(f"  Radiance shape: {spectral_radiance_TOA.shape}")
        print(f"  Frequency range: {f_grid.min():.3e} to {f_grid.max():.3e} Hz")
        print(f"  Wavenumber range: {wvn_grid.min():.2f} to {wvn_grid.max():.2f} cm^-1")
        print(f"  Number of columns: {n_idx}")

        # =====================================================================
        # Step 0: Build the SRF channel operator (once per run)
        # =====================================================================
        if channel_operator is None:
            print("  Building SRF channel operator "
                  "(SRF-weighted quadrature at the channel centres)...")
            t_op = time.time()
            channel_operator, channel_wvn = fax.build_srf_channel_operator(
                f_grid, srf_wvn, srf_response, channel_wvn_all
            )
            t_operator = time.time() - t_op
            channel_frequency = channel_wvn * 100.0 * scc.c  # Hz
            f_grid_ref = f_grid
            print(f"    Operator built in {t_operator:.2f} s; output grid: "
                  f"{channel_wvn.size} channels, {channel_wvn.min():.2f} to "
                  f"{channel_wvn.max():.2f} cm^-1")
        elif not np.array_equal(f_grid, f_grid_ref):
            raise ValueError(
                f"{radiance_file.name}: frequency grid differs from the first "
                f"input file of this run -- the SRF channel operator built "
                f"from it does not apply. Process files of one grid together."
            )

        # =====================================================================
        # Step 1: Planck inversion FIRST -- monochromatic brightness temperature
        # =====================================================================
        print("  Computing monochromatic brightness temperature via Planck "
              "inversion (before convolution)...")
        t_bt = time.time()
        spectral_BT_monochromatic = fax.radiance2brightness_temp(
            spectral_radiance_TOA, wvn_grid
        )
        if spectral_radiance_field is not None:
            # frequency is axis 0 of the full cube as well
            spectral_BT_field_monochromatic = fax.radiance2brightness_temp(
                spectral_radiance_field, wvn_grid
            )
        t_bt_elapsed = time.time() - t_bt
        print(f"    Planck inversion completed in {t_bt_elapsed:.2f} s")

        # Sanity check: monochromatic BT should be in a reasonable range
        bt_valid = spectral_BT_monochromatic[
            (spectral_BT_monochromatic > 50) & (spectral_BT_monochromatic < 400)
        ]
        if bt_valid.size:
            print(f"  Monochromatic BT (valid range): {bt_valid.min():.1f} to "
                  f"{bt_valid.max():.1f} K")
        print(f"  Monochromatic BT range (all): {spectral_BT_monochromatic.min():.1f} to "
              f"{spectral_BT_monochromatic.max():.1f} K")

        # =====================================================================
        # Step 2: Apply the SRF to the BRIGHTNESS TEMPERATURE and reduce to
        #         the sensor channel grid
        # =====================================================================
        print("  Applying SRF filter to brightness temperature at the "
              "channel centres...")
        t_conv = time.time()
        convolved_BT = fax.convolve_with_srf_to_channels(
            spectral_BT_monochromatic, channel_operator
        )
        if spectral_radiance_field is not None:
            # frequency is axis 0 of the full cube as well
            convolved_BT_field = fax.convolve_with_srf_to_channels(
                spectral_BT_field_monochromatic, channel_operator
            )
        t_conv_elapsed = time.time() - t_conv
        print(f"    Convolution completed in {t_conv_elapsed:.2f} s "
              f"({n_wvn} simulation frequencies -> {channel_wvn.size} channels)")

        # Sanity check: BT convolution (unit-sum weighted average) should not
        # move BT outside the range of the monochromatic BT
        print(f"  Input BT range: {spectral_BT_monochromatic.min():.1f} to "
              f"{spectral_BT_monochromatic.max():.1f} K")
        print(f"  Convolved BT range: {convolved_BT.min():.1f} to "
              f"{convolved_BT.max():.1f} K")

        # BT-mean shift diagnostic: the SRF-averaged channel BT vs. the
        # monochromatic BT at the same channel centres quantifies how much the
        # SRF smoothing changes the mean (both averaged over columns first, so
        # the two means refer to the same channel grid)
        mono_at_channels = np.interp(channel_wvn, wvn_grid,
                                     np.nanmean(spectral_BT_monochromatic, axis=1))
        bt_shift = np.nanmean(convolved_BT) - np.nanmean(mono_at_channels)
        print(f"  Mean BT shift from SRF convolution: {bt_shift:+.3f} K")

        # =====================================================================
        # Step 3: Create output xarray.Dataset
        # =====================================================================
        print("  Creating output dataset...")

        output_vars = {}
        if export_monochromatic_bt:
            # full-resolution diagnostic: stays on the simulation grid, so it
            # gets its own axis (the file's frequency axis is the channel grid)
            output_vars["spectral_BT_monochromatic"] = (
                ["frequency_sim", "idx"],
                spectral_BT_monochromatic.astype(np.float32),
                {
                    "units": "K",
                    "description": "Monochromatic brightness temperature before "
                                   "SRF convolution (Planck inversion of the "
                                   "full-resolution radiance), on the "
                                   "simulation grid frequency_sim",
                    "zenith_angle": float(zenith_angle),
                },
            )
        if export_convolved_bt:
            output_vars["brightness_temperature"] = (
                ["frequency", "idx"],
                convolved_BT.astype(np.float32),
                {
                    "units": "K",
                    "description": "Brightness temperature after SRF convolution "
                                   "in BT space, sensor channel grid (NOT the "
                                   "physical instrument response -- see "
                                   "planck_inversion_method attr)",
                    "zenith_angle": float(zenith_angle),
                    "srf_normalization_factor": float(srf_norm_factor),
                },
            )

        output_ds = xr.Dataset(
            output_vars,
            coords={
                "idx": ("idx", idx_coords),
                # sensor (IASI) channel centres -- the spectral grid of the
                # convolved product, replacing the simulation grid
                "frequency": ("frequency", channel_frequency, {"units": "Hz"}),
                "wavenumber": ("frequency", channel_wvn, {"units": "cm^-1"}),
            },
        )
        if export_monochromatic_bt:
            # simulation grid of the input file, kept for the full-resolution
            # monochromatic diagnostic
            output_ds.coords["frequency_sim"] = ("frequency_sim", f_grid,
                                                {"units": "Hz"})
            output_ds.coords["wavenumber_sim"] = ("frequency_sim", wvn_grid,
                                                 {"units": "cm^-1"})

        if spectral_radiance_field is not None:
            # full cube, same layout as the input
            # (frequency, idx, lev, zenith, azimuth), frequency axis now the
            # sensor channel grid
            output_ds["BT_field_convolved"] = (
                ["frequency", "idx", "lev", "zenith", "azimuth"],
                convolved_BT_field.astype(np.float32),
                {
                    "units": "K",
                    "description": "Full BT cube (Planck-inverted before SRF "
                                   "convolution), convolved in BT space, sensor "
                                   "channel grid",
                    "note": "level axis surface-first (index 0 = SFC, -1 = TOA)",
                },
            )

        # Copy attributes from input dataset if available
        output_ds.attrs.update(radiance_ds.attrs)
        output_ds.attrs["convolution_method"] = (
            "srf_weighted_average_in_BT_space_evaluated_at_channel_centres"
        )
        output_ds.attrs["srf_file"] = str(sensor_response_file.name)
        output_ds.attrs["channel_grid_file"] = str(satdata_file.name)
        output_ds.attrs["spectral_grid"] = (
            f"sensor channel grid: {channel_wvn.size} channels, "
            f"{channel_wvn.min():.2f}-{channel_wvn.max():.2f} cm^-1 "
            f"(input simulation grid: {n_wvn} points, "
            f"{wvn_grid.min():.2f}-{wvn_grid.max():.2f} cm^-1)"
        )
        output_ds.attrs["planck_inversion_method"] = (
            "analytic_planck_inversion_BEFORE_convolution"
        )
        output_ds.attrs["note"] = (
            "Alternative to Reference_convolved_* products: the SRF is applied "
            "to the monochromatic brightness temperature instead of the "
            "radiance.  Planck inversion is nonlinear, so this differs from "
            "the physical radiance-space convolution and is a diagnostic only."
        )

        # =====================================================================
        # Step 4: Export results
        # =====================================================================
        if export_results:
            if not output_vars:
                print("  Skipping export: both export_monochromatic_bt and "
                      "export_convolved_bt are False")
            else:
                # mirror the input file name, radiance_TOA -> BTconvolved
                # (keeps Nf/Nza/setup/chunk id from fmip_radiance_batch.py,
                # but uses a distinct prefix so the standard pipeline's
                # Reference_convolved_* files are never touched), and write
                # it to the same run sub folder as the input
                out_name = radiance_file.stem.replace("radiance_TOA", "BTconvolved")
                out_file = radiance_file.parent / f"{out_name}.nc"

                output_ds.to_netcdf(str(out_file))
                print(f"  Exported {out_file}")

        radiance_ds.close()

        # =====================================================================
        # Collect statistics
        # =====================================================================
        chunk_stats = {
            "chunk": ck,
            "file": radiance_file.name,
            "n_columns": n_idx,
            "n_sim_frequencies": n_wvn,
            "n_channels": channel_wvn.size,
            "time_planck": t_bt_elapsed,
            "time_convolution": t_conv_elapsed,
            "bt_shift_mean": bt_shift,
            "bt_min": convolved_BT.min(),
            "bt_max": convolved_BT.max(),
            "bt_mean": np.nanmean(convolved_BT),
        }
        all_stats.append(chunk_stats)

        print(f"  Chunk completed in {time.time() - t_chunk:.1f} s")

    # =========================================================================
    # Post-processing summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)

    print(f"Total processing time: {time.time() - t_start:.1f} s "
          f"(SRF channel operator: {t_operator:.1f} s)")
    print(f"Chunks processed: {len(all_stats)}")
    print(f"Output spectral grid: {channel_wvn.size} sensor channels, "
          f"{channel_wvn.min():.2f}-{channel_wvn.max():.2f} cm^-1")

    print("\nPer-chunk statistics:")
    print(f"{'Chunk':<6} {'Columns':<10} {'Channels':<10} {'Conv Time':<12} "
          f"{'BT Range':<25} {'Mean shift':<11}")
    print("-" * 84)
    for stats in all_stats:
        bt_range = f"{stats['bt_min']:.1f}–{stats['bt_max']:.1f} K"
        print(
            f"{stats['chunk']:<6} {stats['n_columns']:<10} "
            f"{stats['n_channels']:<10} {stats['time_convolution']:<12.2f} "
            f"{bt_range:<25} {stats['bt_shift_mean']:>+10.3f} "
        )

    # Brightness temperature range check
    all_bt_min = min(s["bt_min"] for s in all_stats)
    all_bt_max = max(s["bt_max"] for s in all_stats)
    print(f"\nConvolved BT range (all chunks): {all_bt_min:.1f} to {all_bt_max:.1f} K")
    if all_bt_min < 100 or all_bt_max > 400:
        print(f"  NOTE: BT outside [100K, 400K]. Low values at strong absorption "
              f"band centers (very small radiance, emission from the coldest "
              f"levels) can be physical; check the input radiance otherwise.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
