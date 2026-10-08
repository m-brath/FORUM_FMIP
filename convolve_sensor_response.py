#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply sensor response function (SRF) to spectral radiance and compute brightness temperature.

This script reads the output from fmip_radiance_batch.py (high-resolution spectral
radiance at TOA), applies the sensor response function (SRF) via spectral convolution
to simulate instrument filtering, and optionally converts the result to brightness
temperature via Planck inversion.

The SRF defines the spectral response characteristics of a sensor (e.g., IASI).
Convolution with the SRF smooths the high-resolution radiance spectrum to the
instrument's effective resolution, while Planck inversion converts the resulting
convolved radiance to brightness temperature (the equivalent blackbody temperature).

Environment (mamba/conda):    pyarts
                              (same as fmip_radiance_batch.py)

Input data: Spectral radiance chunks from fmip_radiance_batch.py:
    results/RADIANCE/Reference_radiance_TOA_Nf*_*.nc  (one file per chunk)
    data/sensor_response.nc  (SRF wavenumber offsets and response)

Output:
    results/RADIANCE/Reference_convolved_*_chunk*.nc (one per input radiance chunk)
    Each output file contains:
    - spectral_radiance_convolved: convolved radiance (frequency, idx)
    - brightness_temperature: brightness temperature (frequency, idx)
    - Metadata: SRF name, convolution method, physical constants used

Example
-------
Simply run the script; it will automatically discover all radiance chunks
in results/RADIANCE/ and process them sequentially.

    $ python convolve_sensor_response.py

Notes
-----
* The input radiance grid is already extended to cover the SRF bandwidth
  (see fmip_radiance_batch.py), so no additional padding is needed.
* Convolution is performed independently per column to preserve spatial structure.
  The simulation grid is uniform in wavenumber, so the SRF sampled at the grid
  offsets is one stationary kernel and the convolution is a single vectorized
  filter pass (see frtmip_aux_functions.convolve_with_srf).
* Planck inversion is the exact analytic inversion of the frequency-domain
  Planck function (radiance per Hz, matching the ARTS/DISORT output units;
  see frtmip_aux_functions.radiance2brightness_temp).
* Output files mirror the input radiance structure: same frequency grid,
  same column indices (idx), same metadata where applicable.
* Processing is fast: for ~2180 frequencies x 500 columns, expect well under
  a second per chunk.
"""

import time
import pathlib
import glob

import numpy as np
import xarray as xr

import frtmip_aux_functions as fax


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

# Location of this script
script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
results_folder = script_dir.parent / "results" / "RADIANCE"

# Input data files
sensor_response_file = data_dir / "sensor_response.nc"

# Whether to also convolve the full 5D field (frequency, idx, lev, zenith,
# azimuth) if present in the input file
# (normally only TOA slice is used, but enabling this may help with debugging)
include_full_field = False

# Output options
export_results = True
export_convolved_radiance = True  # save the convolved radiance
export_brightness_temperature = True  # save the brightness temperature


# =============================================================================
# %% helpers
# =============================================================================

def discover_radiance_chunks(results_folder):
    """
    Discover all radiance output files matching the fmip_radiance_batch pattern.

    Parameters
    ----------
    results_folder : pathlib.Path
        Path to results/RADIANCE/ folder.

    Returns
    -------
    list of pathlib.Path
        Sorted list of NetCDF files matching the pattern
        Reference_radiance_TOA_Nf*.nc (chunks and non-chunked).
    """
    pattern = str(results_folder / "Reference_radiance_TOA_Nf*.nc")
    files = sorted(glob.glob(pattern))
    return [pathlib.Path(f) for f in files]


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
    # Discover radiance chunks
    # -------------------------------------------------------------------------
    print("\nDiscovering radiance chunks...")
    radiance_files = discover_radiance_chunks(results_folder)

    if not radiance_files:
        raise FileNotFoundError(
            f"No radiance output files found in {results_folder}. "
            f"Run fmip_radiance_batch.py first."
        )

    print(f"  Found {len(radiance_files)} radiance chunk(s):")
    for rf in radiance_files:
        print(f"    - {rf.name}")

    # -------------------------------------------------------------------------
    # Process each radiance chunk
    # -------------------------------------------------------------------------
    all_stats = []

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
        # Step 1: Apply SRF convolution
        # =====================================================================
        print("  Applying SRF convolution...")
        t_conv = time.time()
        spectral_radiance_convolved = fax.convolve_with_srf(
            spectral_radiance_TOA, f_grid, srf_wvn, srf_response
        )
        if spectral_radiance_field is not None:
            # frequency is axis 0 of the full cube as well
            spectral_radiance_field_convolved = fax.convolve_with_srf(
                spectral_radiance_field, f_grid, srf_wvn, srf_response
            )
        t_conv_elapsed = time.time() - t_conv
        print(f"    Convolution completed in {t_conv_elapsed:.2f} s")

        # Sanity check: convolution should not dramatically change the range
        print(f"  Input radiance range: {spectral_radiance_TOA.min():.3e} to "
              f"{spectral_radiance_TOA.max():.3e}")
        print(f"  Convolved radiance range: {spectral_radiance_convolved.min():.3e} to "
              f"{spectral_radiance_convolved.max():.3e}")

        # Estimate energy conservation
        energy_in = np.sum(spectral_radiance_TOA)
        energy_out = np.sum(spectral_radiance_convolved)
        if energy_in > 0:
            energy_ratio = energy_out / energy_in
            print(f"  Energy conservation ratio: {energy_ratio:.4f}")
        else:
            energy_ratio = np.nan

        # =====================================================================
        # Step 2: Convert to brightness temperature
        # =====================================================================
        print("  Computing brightness temperature via Planck inversion...")
        t_bt = time.time()
        brightness_temperature = fax.radiance2brightness_temp(
            spectral_radiance_convolved, wvn_grid
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
                    "description": "Spectral radiance after SRF convolution",
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
                    "description": "Brightness temperature from Planck inversion",
                    "zenith_angle": float(zenith_angle),
                },
            )

        output_ds = xr.Dataset(
            output_vars,
            coords={
                "idx": ("idx", idx_coords),
                "frequency": ("frequency", f_grid, {"units": "Hz"}),
                "wavenumber": ("frequency", wvn_grid, {"units": "cm^-1"}),
            },
        )

        if spectral_radiance_field is not None:
            # full cube, same layout as the input
            # (frequency, idx, lev, zenith, azimuth)
            output_ds["spectral_radiance_field_convolved"] = (
                ["frequency", "idx", "lev", "zenith", "azimuth"],
                spectral_radiance_field_convolved,
                {
                    "units": "W m^-2 Hz^-1 sr^-1",
                    "description": "Full radiance cube after SRF convolution",
                    "note": "level axis surface-first (index 0 = SFC, -1 = TOA)",
                },
            )

        # Copy attributes from input dataset if available
        output_ds.attrs.update(radiance_ds.attrs)
        output_ds.attrs["convolution_method"] = "spectral_srf_convolution"
        output_ds.attrs["srf_file"] = str(sensor_response_file.name)
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
                # (keeps Nf/Nza/setup/chunk id from fmip_radiance_batch.py)
                out_name = radiance_file.stem.replace("radiance_TOA", "convolved")
                out_file = results_folder / f"{out_name}.nc"

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
            "n_frequencies": n_wvn,
            "time_convolution": t_conv_elapsed,
            "time_planck": t_bt_elapsed,
            "energy_ratio": energy_ratio,
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

    print(f"Total processing time: {time.time() - t_start:.1f} s")
    print(f"Chunks processed: {len(all_stats)}")

    print("\nPer-chunk statistics:")
    print(f"{'Chunk':<6} {'Columns':<10} {'Conv Time':<12} {'BT Range':<25}")
    print("-" * 60)
    for stats in all_stats:
        bt_range = f"{stats['bt_min']:.1f}–{stats['bt_max']:.1f} K"
        print(
            f"{stats['chunk']:<6} {stats['n_columns']:<10} "
            f"{stats['time_convolution']:<12.2f} {bt_range:<25}"
        )

    # Energy conservation check
    energy_ratios = [s["energy_ratio"] for s in all_stats if not np.isnan(s["energy_ratio"])]
    if energy_ratios:
        avg_energy_ratio = np.mean(energy_ratios)
        print(f"\nAverage energy conservation ratio: {avg_energy_ratio:.4f}")
        if avg_energy_ratio < 0.98:
            print("  WARNING: Energy ratio < 0.98; check convolution implementation.")

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
