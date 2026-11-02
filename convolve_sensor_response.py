#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply sensor response function (SRF) to spectral radiance and compute brightness temperature.

This script reads the output from fmip_radiance_batch.py (high-resolution spectral
radiance at TOA), applies the sensor response function (SRF) to simulate
instrument filtering, and converts the result to brightness temperature via
Planck inversion.

The SRF defines the spectral response characteristics of a sensor (e.g., IASI).
Applying it smooths the high-resolution radiance spectrum to the instrument's
effective resolution AND reduces the spectral grid to the instrument's channel
grid: the output carries the IASI channel centres (8461 channels,
645-2760 cm^-1, 0.25 cm^-1 spacing) instead of the simulation frequency grid.
For every channel centre the SRF-weighted average of the high-resolution
spectrum is evaluated directly (unit-sum quadrature over the SRF support, see
frtmip_aux_functions.build_srf_channel_operator) -- the smoothed spectrum is
never resampled, so no resolution is lost or invented by interpolation.
Planck inversion then converts the channel radiance to brightness temperature
(the equivalent blackbody temperature).

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
    results/RADIANCE/<run_name>/Reference_convolved_*_chunk*.nc
    (one per input radiance chunk, written to the same run sub folder as
    its input)
    Each output file contains:
    - spectral_radiance_convolved: convolved radiance (frequency, idx)
    - brightness_temperature: brightness temperature (frequency, idx)
    - Metadata: SRF name, convolution method, physical constants used
    Both variables live on the IASI channel grid: the "frequency" /
    "wavenumber" coordinates are the channel centres, not the simulation grid
    (8461 points instead of e.g. 108951 -- the files are correspondingly
    smaller, ~135 MB instead of ~1.7 GB per variable for the full SLURM run).

Example
-------
Set the run_name constant to the run sub folder to process, then run the
script; it discovers all radiance chunks in
results/RADIANCE/<run_name>/ and processes them sequentially.

    $ python convolve_sensor_response.py

Notes
-----
* The input radiance grid is already extended to cover the SRF bandwidth
  (see fmip_radiance_batch.py), so no additional padding is needed.
* Applying the SRF needs the full high-resolution spectrum around every
  channel, so the input files must cover the complete satellite range
  (merged files or a non-chunked run).  A frequency-chunk file of a SLURM run
  only holds a short slice of the spectrum and cannot be convolved -- merge
  the run with merge_radiance_chunks.py first.
* The SRF filter is applied independently per column (and per level / viewing
  direction of the optional full field) to preserve spatial structure.
  Because the operator is the same for all columns it is built once per run
  and reused for every input file.
* Planck inversion is the exact analytic inversion of the frequency-domain
  Planck function (radiance per Hz, matching the ARTS/DISORT output units;
  see frtmip_aux_functions.radiance2brightness_temp).
* Output files carry the IASI channel grid (channel centres from
  data/satdata.nc), the same column indices (idx) and the same metadata as
  the input; compare_convolved_with_obs.py therefore matches the observation
  grid exactly instead of collocating by nearest simulation channel.
* Processing is fast: building the SRF channel operator takes ~1 s, applying
  it to a chunk takes a few seconds.
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
export_convolved_radiance = False  # save the convolved radiance
# (the channel-grid products are small -- ~135 MB per variable and 2000-column
# idx chunk, ~270 MB with both exports -- so enabling this no longer strains
# the disk; keep False to stay comparable with the BT-only comparison runs)
export_brightness_temperature = True  # save the brightness temperature


# =============================================================================
# %% helpers
# =============================================================================

def discover_radiance_chunks(results_folder, run_name):
    """
    Discover all radiance output files of one run, pinned to run_name.

    The search covers the run sub folder selected by the run_name constant
    (results/RADIANCE/<run_name>/); the recursive glob also picks up any
    files in nested sub folders.  Only files whose names belong to
    run_name are processed:

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
    print("Sensor response convolution and brightness temperature computation")
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
        # Step 1: Apply the SRF and reduce to the sensor channel grid
        # =====================================================================
        print("  Applying SRF convolution at the channel centres...")
        t_conv = time.time()
        spectral_radiance_convolved = fax.convolve_with_srf_to_channels(
            spectral_radiance_TOA, channel_operator
        )
        if spectral_radiance_field is not None:
            # frequency is axis 0 of the full cube as well
            spectral_radiance_field_convolved = fax.convolve_with_srf_to_channels(
                spectral_radiance_field, channel_operator
            )
        t_conv_elapsed = time.time() - t_conv
        print(f"    Convolution completed in {t_conv_elapsed:.2f} s "
              f"({n_wvn} simulation frequencies -> {channel_wvn.size} channels)")

        # Sanity check: convolution should not dramatically change the range
        print(f"  Input radiance range: {spectral_radiance_TOA.min():.3e} to "
              f"{spectral_radiance_TOA.max():.3e}")
        print(f"  Convolved radiance range: {spectral_radiance_convolved.min():.3e} to "
              f"{spectral_radiance_convolved.max():.3e}")

        # Filtering check: the channel values are unit-sum weighted averages of
        # the input spectrum, so the band-average of a locally smooth spectrum
        # must be preserved.  Compare the mean channel radiance with the mean
        # of the input spectrum sampled at the channel centres.
        input_at_channels = np.interp(channel_wvn, wvn_grid,
                                      spectral_radiance_TOA.mean(axis=1))
        if input_at_channels.mean() > 0:
            mean_ratio = spectral_radiance_convolved.mean() / input_at_channels.mean()
            print(f"  Band-mean radiance ratio (channel values / input at "
                  f"channel centres): {mean_ratio:.4f}")
        else:
            mean_ratio = np.nan

        # =====================================================================
        # Step 2: Convert to brightness temperature
        # =====================================================================
        print("  Computing brightness temperature via Planck inversion...")
        t_bt = time.time()
        brightness_temperature = fax.radiance2brightness_temp(
            spectral_radiance_convolved, channel_wvn
        )
        t_bt_elapsed = time.time() - t_bt
        print(f"    Planck inversion completed in {t_bt_elapsed:.2f} s")

        # Sanity check: brightness temperature should be in a reasonable range
        bt_valid = brightness_temperature[(brightness_temperature > 50) & (brightness_temperature < 400)]
        if bt_valid.size:
            print(f"  Brightness temperature (valid range): {bt_valid.min():.1f} to "
                  f"{bt_valid.max():.1f} K")
        print(f"  Brightness temperature range (all): {brightness_temperature.min():.1f} to "
              f"{brightness_temperature.max():.1f} K")

        # =====================================================================
        # Step 3: Create output xarray.Dataset
        # =====================================================================
        print("  Creating output dataset...")

        output_vars = {}
        if export_convolved_radiance:
            output_vars["spectral_radiance_convolved"] = (
                ["frequency", "idx"],
                spectral_radiance_convolved,
                {
                    "units": "W m^-2 Hz^-1 sr^-1",
                    "description": "Spectral radiance after SRF convolution, "
                                   "sensor channel grid",
                    "zenith_angle": float(zenith_angle),
                    "srf_normalization_factor": float(srf_norm_factor),
                },
            )
        if export_brightness_temperature:
            output_vars["brightness_temperature"] = (
                ["frequency", "idx"],
                brightness_temperature,
                {
                    "units": "K",
                    "description": "Brightness temperature from Planck inversion "
                                   "of the channel radiance, sensor channel grid",
                    "zenith_angle": float(zenith_angle),
                },
            )

        output_ds = xr.Dataset(
            output_vars,
            coords={
                "idx": ("idx", idx_coords),
                # sensor (IASI) channel centres -- the instrument's spectral
                # grid, replacing the simulation grid of the input file
                "frequency": ("frequency", channel_frequency, {"units": "Hz"}),
                "wavenumber": ("frequency", channel_wvn, {"units": "cm^-1"}),
            },
        )

        if spectral_radiance_field is not None:
            # full cube, same layout as the input
            # (frequency, idx, lev, zenith, azimuth), frequency axis now the
            # sensor channel grid
            output_ds["spectral_radiance_field_convolved"] = (
                ["frequency", "idx", "lev", "zenith", "azimuth"],
                spectral_radiance_field_convolved,
                {
                    "units": "W m^-2 Hz^-1 sr^-1",
                    "description": "Full radiance cube after SRF convolution, "
                                   "sensor channel grid",
                    "note": "level axis surface-first (index 0 = SFC, -1 = TOA)",
                },
            )

        # Copy attributes from input dataset if available
        output_ds.attrs.update(radiance_ds.attrs)
        output_ds.attrs["convolution_method"] = (
            "srf_weighted_average_evaluated_at_channel_centres"
        )
        output_ds.attrs["srf_file"] = str(sensor_response_file.name)
        output_ds.attrs["channel_grid_file"] = str(satdata_file.name)
        output_ds.attrs["spectral_grid"] = (
            f"sensor channel grid: {channel_wvn.size} channels, "
            f"{channel_wvn.min():.2f}-{channel_wvn.max():.2f} cm^-1 "
            f"(input simulation grid: {n_wvn} points, "
            f"{wvn_grid.min():.2f}-{wvn_grid.max():.2f} cm^-1)"
        )
        output_ds.attrs["planck_inversion_method"] = "analytic_planck_inversion"

        # =====================================================================
        # Step 4: Export results
        # =====================================================================
        if export_results:
            if not output_vars:
                print("  Skipping export: both export_convolved_radiance and "
                      "export_brightness_temperature are False")
            else:
                # mirror the input file name, radiance_TOA -> convolved
                # (keeps Nf/Nza/setup/chunk id from fmip_radiance_batch.py),
                # and write it to the same run sub folder as the input
                out_name = radiance_file.stem.replace("radiance_TOA", "convolved")
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
            "time_convolution": t_conv_elapsed,
            "time_planck": t_bt_elapsed,
            "mean_ratio": mean_ratio,
            "bt_min": brightness_temperature.min(),
            "bt_max": brightness_temperature.max(),
            "bt_mean": np.nanmean(brightness_temperature),
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
          f"{'BT Range':<25}")
    print("-" * 72)
    for stats in all_stats:
        bt_range = f"{stats['bt_min']:.1f}–{stats['bt_max']:.1f} K"
        print(
            f"{stats['chunk']:<6} {stats['n_columns']:<10} "
            f"{stats['n_channels']:<10} {stats['time_convolution']:<12.2f} "
            f"{bt_range:<25}"
        )

    # Band-mean filtering check
    mean_ratios = [s["mean_ratio"] for s in all_stats if not np.isnan(s["mean_ratio"])]
    if mean_ratios:
        avg_mean_ratio = np.mean(mean_ratios)
        print(f"\nAverage band-mean radiance ratio: {avg_mean_ratio:.4f}")
        if not 0.9 <= avg_mean_ratio <= 1.1:
            print("  WARNING: band-mean radiance ratio far from 1; check the "
                  "SRF operator and the input radiance.")

    # Brightness temperature range check
    all_bt_min = min(s["bt_min"] for s in all_stats)
    all_bt_max = max(s["bt_max"] for s in all_stats)
    print(f"\nBrightness temperature range (all chunks): {all_bt_min:.1f} to {all_bt_max:.1f} K")
    if all_bt_min < 100 or all_bt_max > 400:
        print(f"  NOTE: BT outside [100K, 400K]. Low values at strong absorption "
              f"band centers (very small radiance, emission from the coldest "
              f"levels) can be physical; check the input radiance otherwise.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
