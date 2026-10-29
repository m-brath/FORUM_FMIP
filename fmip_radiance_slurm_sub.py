#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMIP batch spectral-radiance benchmark -- worker for ONE frequency chunk.

Counterpart of fmip_radiance_slurm_main.py (pattern: fast_scheme
flux_simulator_slurm_main.py / _slurm_sub.py). One process per frequency
chunk, typically one SLURM job on levante:

    python fmip_radiance_slurm_sub.py <config.json>

The config (written by the launcher) carries the sub frequency grid, the
column subset / column chunking, the angular grid, the refinement and LUT
flags, and the per-chunk FluxSimulator setup name.

Behaviour
---------
* Same simulation setup as fmip_radiance_batch.py (longwave emission = 1,
  gas_scattering = False, no sun, reflectivity = 1 - emissivity interpolated
  onto the SIMULATION wavenumber grid -- here the sub grid of this chunk),
  and the same TOA extraction (zenith angle 180 degrees, last level of the
  surface-first level axis).
* The batch LUT is built for THIS chunk only: setup name
  "<setup_base>_<identifier>" -> own cache dir <cwd>/cache/<setup>/, LUT
  built on the chunk's sub frequency grid. Like fmip_radiance_batch.py it
  is calculated from ALL atmospheres of the input file, so its state
  ranges (T/p/VMR nodes) are identical to the LUT of a single full-grid
  run and every frequency chunk sees the same node grid (recalc_lut in the
  config forces a rebuild). The LUT cache is not shared between jobs, but
  the per-chunk LUTs are mutually consistent.
* The column subset is simulated in column chunks exactly as in
  fmip_radiance_batch.py (one fresh FluxSimulator instance per chunk, LUT
  read from the cache with recalc=False).
* Output: results/RADIANCE/<run_name>/Reference_radiance_TOA_Nf<N_chunk>_
  Nza<N_za>_<setup>[_chunk###].nc, with the run sub folder taken from the
  config ("run_name", falling back to the setup base name for old configs).
  convolve_sensor_response.py discovers these files recursively;
  merge_radiance_chunks.py re-assembles the full frequency grid.

The script chdirs to its own folder so the FluxSimulator LUT cache always
lives at <scripts>/cache/<setup> on both machines.
"""

import os
import sys
import json
import time
import pathlib

import numpy as np
import xarray as xr
from pyarts import arts

import FluxSimulator as fsm

# helpers shared with the non-chunked benchmark (import is side-effect free:
# fmip_radiance_batch.main() is __main__-guarded)
from fmip_radiance_batch import (
    define_abs_species,
    make_gridded_field,
    make_reflectivities,
    refine_all_profiles,
)


# =============================================================================
# %% paths
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
results_folder = script_dir.parent / "results" / "RADIANCE"

atmdata_file = data_dir / "atmdata.nc"
sfcdata_file = data_dir / "sfcdata.nc"
surface_emissivity_file = data_dir / "surface_emissivity.nc"


# =============================================================================
# %% main
# =============================================================================

def simulate_frequency_chunk(config):
    """Run the radiance batch for one frequency chunk (see module docstring)."""
    t_start = time.time()

    # FluxSimulator LUT cache is <cwd>/cache/<setup>/ -> pin the cwd
    os.chdir(script_dir)

    identifier = config["identifier"]
    flxsim_setup = config["flxsim_setup"]

    # run sub folder in results/RADIANCE/, shared by every frequency chunk of
    # one launch (configs written by fmip_radiance_slurm_main.py carry it);
    # for old configs without the key, fall back to the setup base name,
    # i.e. flxsim_setup with the per-chunk identifier stripped
    run_name = config.get("run_name") or flxsim_setup.removesuffix(
        f"_{identifier}")
    run_results_folder = results_folder / run_name

    f_grid = np.asarray(config["f_grid"], dtype=np.float64)
    wvn_grid = arts.convert.freq2kaycm(f_grid)

    start_column = int(config["start_column"])
    chunk_size = int(config["chunk_size"])
    N_za = int(config["N_za"])
    N_aa = int(config["N_aa"])
    refine = bool(config["refine"])
    refinement_steps = int(config["refinement_steps"])
    points_added_per_step = int(config["points_added_per_step"])
    recalc_lut = bool(config["recalc_lut"])
    export_full_field = bool(config["export_full_field"])

    # -------------------------------------------------------------------------
    # open the input data (same files as fmip_radiance_batch.py)
    # -------------------------------------------------------------------------
    atmdata = xr.open_dataset(atmdata_file)
    sfcdata = xr.open_dataset(sfcdata_file)
    emissivity = xr.open_dataset(surface_emissivity_file)

    n_total = int(atmdata.sizes["idx"])
    n_lev = int(atmdata.sizes["lev"])

    if start_column >= n_total:
        raise ValueError(
            f"start_column={start_column} exceeds the number of columns "
            f"in the input data ({n_total})"
        )
    end_column = min(start_column + int(config["n_columns"]), n_total)
    col_ids = list(range(start_column, end_column))   # original idx values
    n_to_run = len(col_ids)
    n_chunks = int(np.ceil(n_to_run / chunk_size))

    # atmospheric species in the input data (volume mixing ratios)
    species_list_of_data = [
        spc for spc in atmdata.data_vars if spc not in ("p", "t", "alt")
    ]

    # load all profiles from the file (files are TOA-first). Like in
    # fmip_radiance_batch.py the batch LUT is built from EVERY column of
    # the input file, not just the columns this job simulates: the LUT
    # state-grid (T/p nodes) depends only on the set of atmospheres, so
    # this keeps the per-chunk LUTs mutually consistent AND identical to
    # the LUT of a single full-grid fmip_radiance_batch.py run. The
    # simulation itself still only uses the columns of this job.
    p_data = atmdata["p"].values            # (n_total, n_lev)
    t_data = atmdata["t"].values
    z_data = atmdata["alt"].values
    spc_data = {
        spc: atmdata[spc].values
        for spc in species_list_of_data
    }

    if refine:
        n_orig = n_lev
        p_data, t_data, z_data, spc_data, n_lev = refine_all_profiles(
            p_data, t_data, z_data, spc_data,
            refinement_steps, points_added_per_step
        )
        print(f"  profile refinement: "
              f"{n_orig} -> {n_lev} levels "
              f"(steps={refinement_steps}, "
              f"points_added_per_step={points_added_per_step})")

    # sfcdata fields are 1-D coordinate arrays -> positional indexing
    T_sfc_file = sfcdata["temperature"].values
    lat_file = sfcdata["lat"].values
    lon_file = sfcdata["lon"].values

    wvn_em = emissivity["wvn"].values
    eps_subset = emissivity["emissivity"].isel(
        idx=slice(start_column, end_column)).values

    # -------------------------------------------------------------------------
    # run summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("FMIP batch spectral-radiance (TOA) -- frequency chunk worker")
    print(f"  identifier  : {identifier}")
    print(f"  spectrum    : {wvn_grid.min():.1f} - {wvn_grid.max():.1f} cm^-1, "
          f"N_wvn = {len(f_grid)}")
    print(f"  columns     : {col_ids[0]} .. {col_ids[-1]} "
          f"({n_to_run} columns, {n_chunks} column chunk(s) of <= "
          f"{chunk_size})")
    print(f"  levels      : {n_lev}"
          + (f" (refined, {refinement_steps}/{points_added_per_step})"
             if refine else ""))
    print(f"  angular     : N_za = {N_za}, N_aa = {N_aa}")
    print(f"  emission    : longwave/thermal, no sun, gas_scattering = False")
    print(f"  species     : {len(species_list_of_data)} in the data")
    print(f"  setup name  : {flxsim_setup}")
    print(f"  LUT cache   : {os.path.join(os.getcwd(), 'cache', flxsim_setup)}")
    print(f"  recalc LUT  : {recalc_lut}")
    print(f"  results     : {run_results_folder}")
    print("=" * 70)

    run_results_folder.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # %% atmospheres for ALL columns of the input file (surface-first)
    # ==========================================================================
    # Same as in fmip_radiance_batch.py: the full set of atmospheres is built
    # up front so the batch LUT is computed from the full input (consistent
    # state ranges / node grids for every frequency chunk and for the
    # non-chunked benchmark).
    all_atms = []
    T_surfs_all = []
    geo_positions_all = []
    for j in range(n_total):
        all_atms.append(make_gridded_field(
            p_data[j][::-1],
            t_data[j][::-1],
            z_data[j][::-1],
            {spc: spc_data[spc][j][::-1]
             for spc in species_list_of_data},
            species_list_of_data,
        ))
        T_surfs_all.append(float(T_sfc_file[j]))
        geo_positions_all.append([float(lat_file[j]), float(lon_file[j])])

    # ==========================================================================
    # %% LUT for this frequency chunk (own setup/cache, sub f_grid)
    # ==========================================================================
    t_lut = time.time()
    Flxsim_lut = fsm.FluxSimulator(flxsim_setup)
    Flxsim_lut.set_frequency_grid(f_grid)

    # data species added to the FluxSimulator defaults
    # (H2O + MT-CKD 4.3 continuums, O2/N2 CIA, CO2-CKDMT252, O3, O3-XFIT),
    # same as fmip_radiance_batch.py
    abs_species = define_abs_species(Flxsim_lut, species_list_of_data)
    Flxsim_lut.add_species(abs_species, verbose=True)

    Flxsim_lut.emission = 1
    Flxsim_lut.gas_scattering = False

    Flxsim_lut.get_lookuptableBatch(all_atms, recalc=recalc_lut)
    print(f"  LUT ready in {time.time() - t_lut:.1f} s\n")

    # ==========================================================================
    # %% simulation, column chunk by column chunk
    # ==========================================================================
    for ck in range(n_chunks):
        c0 = ck * chunk_size
        c1 = min(c0 + chunk_size, n_to_run)

        print("=" * 70)
        print(f"column chunk {ck + 1}/{n_chunks}: "
              f"columns {col_ids[c0]} .. {col_ids[c1 - 1]} ({c1 - c0} columns)")
        print("=" * 70)

        # chunk views of the up-front built full-file arrays (shifted by
        # start_column, since the lists cover ALL columns of the input file)
        atms = all_atms[c0 + start_column:c1 + start_column]
        T_surfs = T_surfs_all[c0 + start_column:c1 + start_column]
        geo_positions = geo_positions_all[c0 + start_column:c1 + start_column]

        # surface altitudes from the (surface-first) compact atmospheres
        surface_altitudes = [atm[1, 0, 0, 0] for atm in atms]

        # per-column surface reflectivity vectors (1 - eps) on the SUB grid
        reflectivities = make_reflectivities(
            eps_subset[c0:c1], wvn_em, wvn_grid)

        # no sun for longwave
        sun_positions = [[] for _ in atms]

        # ---------------------------------------------------------------------
        # FluxSimulator setup (fresh instance per column chunk)
        # ---------------------------------------------------------------------
        t_chunk = time.time()
        Flxsim = fsm.FluxSimulator(flxsim_setup)
        Flxsim.set_frequency_grid(f_grid)

        abs_species = define_abs_species(Flxsim, species_list_of_data)
        Flxsim.add_species(abs_species, verbose=(ck == 0))

        Flxsim.emission = 1
        Flxsim.gas_scattering = False

        # recalc=False reads the LUT this job built up front for its own
        # sub frequency grid (setup name is chunk-specific, so no other job
        # can interfere)
        results = Flxsim.radiance_simulator_batch(
            atms,
            T_surfs,
            surface_altitudes,
            reflectivities,
            geo_positions,
            sun_positions,
            N_za=N_za,
            N_aa=N_aa,
            end_index=-1,
            recalc=False,
        )
        print(f"  chunk simulation done in {time.time() - t_chunk:.1f} s\n")

        # ---------------------------------------------------------------------
        # gather the chunk results (TOA spectral radiance field)
        # ---------------------------------------------------------------------
        n_chunk = c1 - c0

        f_out = np.array(results["f_grid"])
        za_grid = np.array(results["zenith_angle"])
        aa_grid = np.array(results["azimuth_angle"])
        N_wvn_out, N_za_out, N_aa_out = (
            len(f_out), len(za_grid), len(aa_grid)
        )

        # radiance field is (idx, frequency, lev, zenith, azimuth), with the
        # level axis surface-first (index 0 = SFC, last index = TOA)
        field = np.zeros((n_chunk, N_wvn_out, n_lev, N_za_out, N_aa_out))
        pressure = np.zeros((n_chunk, n_lev))
        altitude = np.zeros((n_chunk, n_lev))
        for i in range(n_chunk):
            rad = np.array(
                results["array_of_spectral_radiance_clearsky"][i]
            )                                          # (N_wvn, n_lev, N_za, N_aa)
            if rad.shape[1] != n_lev:
                raise AssertionError(
                    f"unexpected radiance level axis {rad.shape}, "
                    f"expected {n_lev} levels"
                )
            field[i] = rad
            # stored surface-first, consistent with the radiance field
            pressure[i] = results["array_of_pressure"][i]
            altitude[i] = results["array_of_altitude"][i]

        # TOA slice: highest model level (last index, since surface-first).
        # Keep only the radiance at zenith angle 180 degrees.
        za_index = int(np.argmin(np.abs(za_grid - 180.0)))
        if not np.isclose(za_grid[za_index], 180.0):
            raise AssertionError(
                f"zenith angle grid does not contain 180 degrees: {za_grid}"
            )
        toa_rad = field[:, :, -1, za_index, 0]       # (idx, N_wvn)

        # quick sanity check for the first profile of the chunk
        r0 = toa_rad[0]
        print(f"  TOA spectral radiance (profile {col_ids[c0]}): "
              f"shape {r0.shape}, min {r0.min():.3e}, max {r0.max():.3e}")
        print(f"  selected zenith angle: {za_grid[za_index]} degrees\n")

        # ---------------------------------------------------------------------
        # export the chunk results
        # ---------------------------------------------------------------------
        ds = xr.Dataset(
            {
                # transpose to the conventional (frequency, idx) layout
                "spectral_radiance_TOA":
                    (["frequency", "idx"],
                     toa_rad.transpose(1, 0),
                     {"units": "W m^-2 Hz^-1 sr^-1 "
                               "(ARTS/DISORT spectral radiance)",
                      "zenith_angle": float(za_grid[za_index])}),
                "altitude_min": (["idx"], altitude.min(axis=1),
                                  {"units": "m"}),
                "altitude_max": (["idx"], altitude.max(axis=1),
                                  {"units": "m"}),
            },
            coords={
                "idx": np.array(col_ids[c0:c1], dtype=np.int64),
                "frequency": (["frequency"], f_out, {"units": "Hz"}),
                "wavenumber": (["frequency"],
                                arts.convert.freq2kaycm(f_out),
                                {"units": "cm^-1"}),
            },
        )

        if export_full_field:
            # full altitude-resolved cube, (frequency, idx, lev, zenith, azim),
            # level axis kept surface-first (index 0 = SFC, -1 = TOA)
            ds["spectral_radiance_field"] = (
                ["frequency", "idx", "lev", "zenith", "azimuth"],
                field.transpose(1, 0, 2, 3, 4),
                {"units": "W m^-2 Hz^-1 sr^-1",
                 "note": "level axis surface-first (index 0 = SFC, -1 = TOA)"},
            )

        fname = (
            f"Reference_radiance_TOA_Nf{N_wvn_out}"
            f"_Nza{N_za}_{flxsim_setup}"
        )
        if n_chunks > 1:
            fname += f"_chunk{ck:03d}"
        out_file = run_results_folder / f"{fname}.nc"
        ds.to_netcdf(str(out_file))
        print(f"  exported {out_file}\n")

    print(f"frequency chunk {identifier}: "
          f"total wall time {time.time() - t_start:.1f} s")


# =============================================================================
# %% entry point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python fmip_radiance_slurm_sub.py <config.json>"
        )

    config_file = sys.argv[1]

    with open(config_file) as fin:
        config = json.load(fin)

    print("======================================================================")
    print("\n...start calculating")
    print(config_file)

    simulate_frequency_chunk(config)

    print("\n...done calculating")
    print("______________________________________________________________________\n")
