#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the BT-space SRF-convolved products (Reference_BTconvolved_*) with IASI
satellite observations -- the comparison companion to
convolve_sensor_response_bt.py.

This wrapper reuses the streaming comparison machinery of
compare_convolved_with_obs.py (which is imported, NOT modified) and only
differs in two points:

    1. it discovers the Reference_BTconvolved_* chunk files of the run
       (convolve_sensor_response_bt.py output, BT-only: the comparison runs
       in brightness-temperature mode, radiance quantities are skipped),
    2. it labels all outputs with the tag <run_name>_BTconvolved, so the
       statistics file and figures of the standard radiance-space pipeline
       (comparison_stats_<run_name>.nc, fig_*_<run_name>.png/pdf) are never
       overwritten.

This makes the two convolution orders directly comparable: same observation
grid, same collocation, same statistics/histogram definitions -- only the
convolution order differs.

Environment (mamba/conda):    pyarts
                              (same as compare_convolved_with_obs.py)

Input data:
    data/satdata.nc                                   (observed T_b)
    results/RADIANCE/<run_name>/Reference_BTconvolved_*.nc
    (searched recursively within the run sub folder, file names pinned to
    run_name)

Output:
    results/COMPARISON/comparison_stats_<run_name>_BTconvolved.nc
    plot/comparison/fig_mean_bt_<run_name>_BTconvolved.png / .pdf
    plot/comparison/fig_bt_stats_<run_name>_BTconvolved.png / .pdf
    plot/comparison/fig_bt_hist_<run_name>_BTconvolved.png / .pdf
    (same figure set as compare_convolved_with_obs.py, including the
    mean-spectra figure with the +-1 std bands, only tagged differently)

Example
-------
Set the run_name constant to the run sub folder to compare, then run the
script; it discovers the run's BTconvolved chunks automatically.

    $ python compare_btconvolved_with_obs.py

Notes
-----
* All statistics definitions follow compare_convolved_with_obs.py (its
  constants -- collocation tolerance, valid T_b range, histogram bins --
  apply unchanged, so results are directly comparable to the standard
  Reference_convolved_* comparison).
"""

import time
import pathlib
import glob
import re

import numpy as np
import xarray as xr

import compare_convolved_with_obs as cco


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

# run sub folder of results/RADIANCE/ whose BTconvolved products to compare
# against the observations (same constant name as in
# convolve_sensor_response_bt.py / compare_convolved_with_obs.py)
run_name = "fmip_radiance_batch_slurm_Nf108951_Nza20"

# tag used for all output file names; keeps the BT-space product separate
# from the standard radiance-space comparison outputs
tag = f"{run_name}_BTconvolved"


# =============================================================================
# %% helpers
# =============================================================================

def discover_btconvolved_chunks(results_folder, run_name):
    """
    Sorted list of Reference_BTconvolved_* chunk files of one run.

    Same name-pinning logic as compare_convolved_with_obs.discover_convolved_chunks,
    but matching the Reference_BTconvolved_* prefix written by
    convolve_sensor_response_bt.py (radiance_TOA -> BTconvolved).
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

    pattern = str(results_folder / "**" / "Reference_BTconvolved_*.nc")
    files = [
        pathlib.Path(f)
        for f in glob.glob(pattern, recursive=True)
        if setup_name in pathlib.Path(f).name
        and (nf_tag is None or nf_tag in pathlib.Path(f).name)
        and (nza_tag is None or nza_tag in pathlib.Path(f).name)
    ]
    return sorted(files)


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("Simulation vs. observation comparison (BT-space SRF-convolved products)")
    print("=" * 70)

    cco.stats_folder.mkdir(parents=True, exist_ok=True)
    cco.plot_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load observations (identical to compare_convolved_with_obs.main)
    # -------------------------------------------------------------------------
    print("\nLoading observation data...")
    satdata = xr.open_dataset(cco.satdata_file)
    # NB: the spectral axis of T_b is mislabelled 'lev' in satdata.nc
    obs_Tb = satdata["T_b"].rename({"lev": "wvn_obs"})
    wvn_obs = satdata.coords["wvn"].values
    n_fill = int(((obs_Tb < cco.bt_valid_range[0])
                  | (obs_Tb > cco.bt_valid_range[1])).sum())
    print(f"  Observation shape: {obs_Tb.shape} (idx, wvn)")
    print(f"  Wavenumber range: {wvn_obs.min():.2f} to {wvn_obs.max():.2f} cm^-1")
    print(f"  Fill values masked (outside {cco.bt_valid_range}): {n_fill}")
    obs_Tb_values = obs_Tb.values.astype(np.float32)  # (idx, wvn)

    # -------------------------------------------------------------------------
    # Discover the BTconvolved chunks of the selected run
    # -------------------------------------------------------------------------
    results_folder = cco.results_root / run_name
    if not results_folder.is_dir():
        available = sorted(
            p.name for p in cco.results_root.iterdir() if p.is_dir()
        ) if cco.results_root.is_dir() else []
        raise FileNotFoundError(
            f"Run sub folder {results_folder} does not exist. "
            f"Set run_name to one of the available runs: "
            f"{', '.join(available) if available else '(none)'}"
        )
    chunk_files = discover_btconvolved_chunks(results_folder, run_name)
    if not chunk_files:
        raise FileNotFoundError(
            f"No BTconvolved output files (Reference_BTconvolved_* belonging "
            f"to '{run_name}') found in {results_folder}. "
            f"Run convolve_sensor_response_bt.py first."
        )

    print(f"  Found {len(chunk_files)} BTconvolved chunk(s):")
    for cf in chunk_files:
        print(f"    - {cf.relative_to(results_folder)}")

    # -------------------------------------------------------------------------
    # Streaming comparison with the shared machinery
    # (BT-only files -> brightness-temperature comparison; the tag keeps all
    # outputs separate from the Reference_convolved_* comparison)
    # -------------------------------------------------------------------------
    stats = cco.run_comparison(chunk_files, obs_Tb_values, wvn_obs, tag)

    if cco.export_stats:
        cco.export_stats(stats, wvn_obs,
                         cco.stats_folder / f"comparison_stats_{tag}.nc")

    if cco.make_plots:
        cco.make_setup_plots(stats, wvn_obs)
        print(f"  Figures written to {cco.plot_folder}")

    satdata.close()

    print(f"\nTotal processing time: {time.time() - t_start:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
