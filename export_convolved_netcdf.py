#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the SRF-convolved products of one run (radiance-space convolution)
as ONE NetCDF file with complete run metadata.

convolve_sensor_response.py writes its products one file per input radiance
column chunk (Reference_convolved_*_chunk###.nc, IASI channel grid).  This
script concatenates those chunk files along the column axis (idx) into a
single file

    results/RADIANCE/<run_name>/Reference_convolved_<run_name>_ALL.nc

and attaches self-describing metadata for the whole processing chain, so the
file can be shared or archived without the surrounding run folder:

    * spectral grids: simulation frequency grid (number of frequencies,
      wavenumber range -- from the spectral_grid attribute written by
      convolve_sensor_response.py) and sensor (IASI) channel grid of the
      product (number of channels, centre range, spacing)
    * simulation setup: number of columns, ACTUAL number of atmospheric
      levels used by the radiative transfer (input-level count plus the
      profile refinement applied by the run, recomputed from
      data/atmdata.nc and the run's JSON configs in configs/<run_name>/),
      angular quadrature (N_za / N_aa), viewing zenith angle, absorber
      species from the input data
    * software versions: ARTS/pyarts (the pyarts version pins the ARTS
      library version), Python, numpy, xarray
    * processing provenance: convolution method, SRF / channel-grid input
      files, Planck inversion method, list of source chunk files, timestamp

Environment (mamba/conda):    pyarts
                              (same as convolve_sensor_response.py)

Input data:
    results/RADIANCE/<run_name>/Reference_convolved_*.nc
    (the run's convolved chunk files, searched recursively, file names
    pinned to run_name; Reference_BTconvolved_* of the BT-space variant
    script are NOT matched)
    data/atmdata.nc            (input level count and species list)
    configs/<run_name>/*.json  (run settings of the SLURM launcher,
                                optional -- refinement / angular grid)

Output:
    results/RADIANCE/<run_name>/Reference_convolved_<run_name>_ALL.nc

Example
-------
Set the run_name constant to the run sub folder to process, then run the
script; it discovers all convolved chunks of that run.

    $ python export_convolved_netcdf.py

Notes
-----
* Only products of the RADIANCE-space convolution
  (convolve_sensor_response.py) are exported; the BT-space diagnostic of
  convolve_sensor_response_bt.py is a different product and is skipped.
* Variables are carried through unchanged (brightness_temperature, plus
  spectral_radiance_convolved if the convolve run exported it).  The data
  are loaded and concatenated in memory (the full-SLURM BT product is
  ~1.3 GB float64), which needs roughly twice the file size in RAM.
* The number of atmospheric levels reported is the one the simulation
  actually used: with profile refinement (as in the SLURM run) the input
  file's lev count is refined to
      n_lev = (n_lev_in - 1) * (points_added_per_step + 1) ** steps + 1
  (formula of fmip_radiance_batch.refine_all_profiles).  If the run's JSON
  configs are no longer available, the input level count is reported and
  n_atm_levels_refined is noted as unknown.
"""

import time
import pathlib
import glob
import re
import sys
import json
import platform

import numpy as np
import xarray as xr


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

# Location of this script
script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
configs_root = script_dir.parent / "configs"

# run sub folder of results/RADIANCE/ whose convolved products are exported
# (run_name = <setup_name>_Nf<N_wvn>_Nza<N_za>, same constant as in
# convolve_sensor_response.py)
run_name = "fmip_radiance_batch_slurm_Nf108951_Nza20"
results_folder = script_dir.parent / "results" / "RADIANCE" / run_name

# input files consulted for the metadata (same files the run itself used)
atmdata_file = data_dir / "atmdata.nc"

# single output file (written into the run sub folder)
out_file = results_folder / f"Reference_convolved_{run_name}_ALL.nc"


# =============================================================================
# %% helpers
# =============================================================================

def discover_convolved_chunks(results_folder, run_name):
    """
    Discover the convolved output files of one run (radiance-space SRF
    convolution), pinned to run_name.

    Mirrors the discovery of compare_convolved_with_obs.py: pattern
    Reference_convolved_*.nc, but only files belonging to run_name -- either
    the full run_name in the file name (merged-file lineage) or the setup-
    name part plus the _Nf<..>_ / _Nza<..>_ tags (direct chunk lineage).
    Reference_BTconvolved_* files (BT-space convolution, different product)
    do not match the pattern.

    Parameters
    ----------
    results_folder : pathlib.Path
        Path to the run sub folder results/RADIANCE/<run_name>/.
    run_name : str
        Run identifier (<setup_name>_Nf<N_wvn>_Nza<N_za>).

    Returns
    -------
    list of pathlib.Path
        Sorted list of convolved NetCDF files belonging to run_name.
    """
    match = re.match(r"(.+)_Nf(\d+)_Nza(\d+)$", run_name)
    if match:
        setup_name, nf_tag, nza_tag = (
            match.group(1), f"_Nf{match.group(2)}_", f"_Nza{match.group(3)}_"
        )
    else:
        setup_name, nf_tag, nza_tag = run_name, None, None

    pattern = str(results_folder / "**" / "Reference_convolved_*.nc")
    files = []
    for f in glob.glob(pattern, recursive=True):
        name = pathlib.Path(f).name
        if name.endswith("_ALL.nc"):
            continue    # never re-absorb a previous single-file export
        if setup_name in name and (
            run_name in name
            or (nf_tag in name and nza_tag in name)
        ):
            files.append(pathlib.Path(f))
    return sorted(files)


def refined_level_count(n_lev_in, refinement_steps, points_added_per_step):
    """
    Level count after profile refinement (same formula as
    fmip_radiance_batch.refine_all_profiles).
    """
    return (n_lev_in - 1) * (points_added_per_step + 1) ** refinement_steps + 1


def read_run_config(configs_folder):
    """
    Read the run settings from one of the SLURM launcher's JSON configs
    (all configs of a run carry the same setup, only f_grid/start_column
    differ).  Returns None if the folder is gone (e.g. configs cleaned up).
    """
    configs = sorted(glob.glob(str(configs_folder / "*.json")))
    if not configs:
        return None
    with open(configs[0]) as fh:
        return json.load(fh)


def software_versions():
    """
    Collect the software versions of the current environment.  The pyarts
    package bundles the ARTS library, and its version pins the ARTS version
    (pyarts 2.6.20 == ARTS 2.6.20).
    """
    versions = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "xarray_version": xr.__version__,
    }
    try:
        import pyarts
        versions["pyarts_version"] = pyarts.__version__
        versions["arts_version"] = pyarts.__version__
        versions["arts_version_note"] = (
            "pyarts bundles the ARTS library; the pyarts version pins the "
            "ARTS version"
        )
    except Exception:
        versions["pyarts_version"] = "pyarts not importable in this env"
    return versions


def parse_sim_grid(spectral_grid_attr):
    """
    Extract (n_sim_points, wvn_min, wvn_max) of the simulation grid from the
    spectral_grid attribute written by convolve_sensor_response.py, e.g.
    "... (input simulation grid: 108951 points, 613.00-2792.00 cm^-1)".
    Returns None on an unexpected layout.
    """
    m = re.search(
        r"input simulation grid: (\d+) points, ([\d.]+)-([\d.]+) cm",
        spectral_grid_attr,
    )
    if not m:
        return None
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("Single-file export of the SRF-convolved products (radiance space)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Discover convolved chunk files
    # -------------------------------------------------------------------------
    if not results_folder.is_dir():
        raise FileNotFoundError(
            f"Run sub folder {results_folder} does not exist. Set run_name "
            f"to an existing run under results/RADIANCE/."
        )
    files = discover_convolved_chunks(results_folder, run_name)
    if not files:
        raise FileNotFoundError(
            f"No convolved files (Reference_convolved_*.nc belonging to "
            f"{run_name}) found in {results_folder}. Run "
            f"convolve_sensor_response.py first."
        )
    print(f"\nFound {len(files)} convolved chunk file(s):")
    for f in files:
        print(f"  - {f.name}")

    # -------------------------------------------------------------------------
    # Concatenate along the column axis (idx carries the global column
    # index, so the chunk files concatenate without overlap)
    # -------------------------------------------------------------------------
    print("\nLoading and concatenating...")
    datasets = []
    for f in files:
        ds = xr.open_dataset(f)
        if "brightness_temperature" not in ds and \
                "spectral_radiance_convolved" not in ds:
            raise ValueError(
                f"{f.name}: neither brightness_temperature nor "
                f"spectral_radiance_convolved found -- not a product of "
                f"convolve_sensor_response.py?"
            )
        datasets.append(ds)
    ds_all = xr.concat(datasets, dim="idx")
    for ds in datasets:
        ds.close()

    wvn = ds_all.coords["wavenumber"].values
    freq = ds_all.coords["frequency"].values
    idx = ds_all.coords["idx"].values
    n_channels = wvn.size
    n_columns = idx.size
    if not np.array_equal(idx, np.arange(n_columns)):
        print(f"  NOTE: idx is not a plain 0..{n_columns - 1} range "
              f"({idx.min()}..{idx.max()}) -- global column indices kept.")
    dw = np.diff(wvn)
    print(f"  Product grid: {n_channels} channels, {wvn.min():.2f}-"
          f"{wvn.max():.2f} cm^-1 (spacing min {dw.min():.4f}, max "
          f"{dw.max():.4f}), {n_columns} columns")
    for var in ds_all.data_vars:
        print(f"  Variable: {var} {ds_all[var].shape} "
              f"{ds_all[var].dtype}")

    # -------------------------------------------------------------------------
    # Metadata: run setup from configs/ and atmdata.nc, grids from the file
    # -------------------------------------------------------------------------
    print("\nCollecting run metadata...")
    attrs = {}

    # -- run identification ---------------------------------------------------
    attrs["run_name"] = run_name
    m = re.match(r"(.+)_Nf(\d+)_Nza(\d+)$", run_name)
    if m:
        attrs["setup_name"] = m.group(1)
    attrs["product"] = (
        "SRF-convolved TOA brightness temperature / radiance at the IASI "
        "sensor channel grid (radiance-space convolution, "
        "convolve_sensor_response.py)"
    )

    # -- spectral grids ---------------------------------------------------------
    sim_grid = None
    for k in ("convolution_method", "srf_file", "channel_grid_file",
              "spectral_grid", "planck_inversion_method"):
        if k in ds_all.attrs:
            attrs[k] = ds_all.attrs[k]
    if "spectral_grid" in ds_all.attrs:
        sim_grid = parse_sim_grid(ds_all.attrs["spectral_grid"])
    if sim_grid:
        attrs["n_simulation_frequencies"] = sim_grid[0]
        attrs["simulation_wavenumber_range_cm-1"] = (
            f"{sim_grid[1]:.2f}-{sim_grid[2]:.2f}"
        )
    attrs["n_sensor_channels"] = n_channels
    attrs["sensor_wavenumber_range_cm-1"] = (
        f"{wvn.min():.2f}-{wvn.max():.2f}"
    )
    attrs["sensor_channel_spacing_cm-1"] = (
        f"{dw.min():.4f}-{dw.max():.4f}"
    )

    # -- simulation setup -------------------------------------------------------
    n_lev_in = None
    species = None
    if atmdata_file.is_file():
        atm = xr.open_dataset(atmdata_file)
        n_lev_in = int(atm.sizes["lev"])
        n_cols_in = int(atm.sizes["idx"])
        species = [spc for spc in atm.data_vars if spc not in ("p", "t", "alt")]
        atm.close()
        attrs["n_input_atm_levels"] = n_lev_in
        attrs["n_input_columns"] = n_cols_in
        attrs["absorber_species_n"] = len(species)
        attrs["absorber_species"] = ", ".join(species)
    else:
        print(f"  WARNING: {atmdata_file} not found -- level/species "
              f"metadata incomplete")

    cfg = read_run_config(configs_root / run_name)
    if cfg is not None and n_lev_in is not None:
        if cfg.get("refine"):
            n_lev_used = refined_level_count(
                n_lev_in, int(cfg["refinement_steps"]),
                int(cfg["points_added_per_step"]),
            )
            attrs["atm_level_refinement"] = (
                f"log10(p) refinement: {cfg['refinement_steps']} step(s), "
                f"{cfg['points_added_per_step']} point(s) added per interval "
                f"(fmip_radiance_batch.refine_all_profiles)"
            )
        else:
            n_lev_used = n_lev_in
        attrs["n_atm_levels"] = n_lev_used
        attrs["n_zenith_angles"] = int(cfg["N_za"])
        attrs["n_azimuth_angles"] = int(cfg["N_aa"])
    else:
        print("  WARNING: run configs (configs/{run_name}) or atmdata.nc "
              "not available -- cannot report the actual level count / "
              "angular grid")
        if n_lev_in is not None:
            attrs["n_atm_levels"] = n_lev_in
            attrs["n_atm_levels_note"] = (
                "input level count; run configs missing, refinement status "
                "unknown"
            )

    for var in ds_all.data_vars:
        za = ds_all[var].attrs.get("zenith_angle")
        if za is not None:
            attrs[f"viewing_zenith_angle_deg_{var}"] = float(za)

    # -- software versions --------------------------------------------------------
    attrs.update(software_versions())

    # -- provenance -----------------------------------------------------------------
    attrs["n_source_chunk_files"] = len(files)
    attrs["source_chunk_files"] = ", ".join(f.name for f in files)
    attrs["exported_by"] = "export_convolved_netcdf.py"
    attrs["export_date"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    ds_all.attrs = attrs
    ds_all.coords["idx"].attrs = {
        "description": "global column index into data/atmdata.nc",
    }

    # -------------------------------------------------------------------------
    # Write single file
    # -------------------------------------------------------------------------
    print(f"\nWriting {out_file.name} ...")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds_all.to_netcdf(str(out_file))
    size_mb = out_file.stat().st_size / 1e6
    print(f"  Exported {out_file}  ({size_mb:.0f} MB)")

    print(f"\nTotal time: {time.time() - t_start:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
