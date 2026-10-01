#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMIP batch spectral-radiance (top-of-atmosphere) benchmark.

Longwave spectral-radiance batch RTE script, modeled after
fmip_lw_benchmark_batch.py, but based on the FluxSimulator batch-radiance
example (examples/flux_simulator_batch_radiance_example.py), i.e. it uses the
**FluxSimulator.radiance_simulator_batch** member function instead of
``flux_simulator_batch``.

Environment (mamba/conda):    pyarts
                              (/Users/u242031/miniforge3/envs/pyarts,
                               pyarts 2.6.20 + FluxSimulator -- NOT pyarts3)

Input data: the same files as used by fmip_lw_benchmark_batch.py
(created by prepare_data.py), in <repo>/data:
    atmdata.nc             columns (idx) x levels (lev), TOA-first
    sfcdata.nc             surface temperature, elevation, lat/lon per column
    surface_emissivity.nc  per-column spectral emissivity on a fixed wvn grid

Result of interest
------------------
The top-of-atmosphere (TOA) spectral radiance at zenith angle 180 degrees,
plus the minimum and maximum altitude of every column:
    spectral_radiance_TOA  (frequency, idx)
    altitude_min, altitude_max  (idx)

Notes
-----
* Only a subset of columns is processed. All run parameters are the constants
  at the top of this file.
* The angular grid is shared by every profile of the batch and set once:
  N_za = 20 zenith angles, N_aa = 1 azimuth angle (see the radiance example).
  DISORT internally uses nstreams = int(N_za // 2) * 2 = 20 streams.
* Longwave / thermal: emission = 1, gas_scattering = False, no sun
  (sun_positions empty for every column, so no sun is registered).
* Surface reflectivity is passed per column as a vector of N_wvn values
  (1 - emissivity), interpolated from the emissivity-file wavenumber grid
  onto the simulation wavenumber grid. DISORT derives the surface emissivity
  as 1 - albedo, i.e. the true spectral emissivity.
* Radiances returned by radiance_simulator_batch are stored per profile with
  shape (N_wvn, n_lev, N_za, N_aa), and the LEVEL axis is in the same
  (surface-first) order as the compact atmospheres fed in (index 0 = SFC,
  last index = TOA). Hence the TOA field is the slice
      radiance[i][:, -1, :, :]
* One fresh FluxSimulator instance per chunk: repeated
  radiance_simulator_batch calls on one instance are not supported by this
  ARTS workspace (e.g. NumericCreate("DummyVariable") in the batch method
  fails on already existing variables).
* LUT: the batch LUT is calculated up front, before the per-chunk
  simulations, from ALL atmospheres of the input file (independent of the
  column subset / chunks) via get_lookuptableBatch, so its temperature /
  pressure / VMR state ranges are wide enough for any selection. Later chunk
  instances only read the stored LUT (set recalc_lut=True in the edit section
  to force a rebuild). Cache location: <cwd>/cache/<setup_name>/ (default;
  same for the presimulation and every chunk instance, because the whole
  batch shares setup_name = fmip_radiance_batch). Delete the cache folder
  manually to force a rebuild if the spectrum or species setup is changed.
* Radiance units: the stored radiance is the raw ARTS/DISORT spectral
  radiance (W m^-2 Hz^-1 sr^-1). Confirm against a first run and adjust the
  NetCDF attribute if your convention differs.
"""

import os
import time
import pathlib

import numpy as np
import xarray as xr
from pyarts import arts

import FluxSimulator as fsm


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

# Location of this script and of the input data
script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"

atmdata_file = data_dir / "atmdata.nc"
sfcdata_file = data_dir / "sfcdata.nc"
surface_emissivity_file = data_dir / "surface_emissivity.nc"

# results folder
results_folder = script_dir.parent / "results" / "RADIANCE"

# -----------------------------------------------------------------------------
# longwave (wavenumber) grid, in cm^-1 (matches the surface_emissivity grid)
# (same spectrum as fmip_lw_benchmark_batch.py)
# -----------------------------------------------------------------------------
wvn_min = 645.0      # cm^-1
wvn_max = 2760.0     # cm^-1
N_wvn = 100          # number of wavenumbers (changeable later)
wvn_grid = np.linspace(wvn_min, wvn_max, N_wvn)
f_grid = arts.convert.kaycm2freq(wvn_grid)   # Hz

# -----------------------------------------------------------------------------
# angular grid, shared by every profile of the batch
# -----------------------------------------------------------------------------
N_za = 20            # number of zenith angles
N_aa = 1             # number of azimuth angles

# -----------------------------------------------------------------------------
# column subset / batching
# -----------------------------------------------------------------------------
start_column = 0     # first (file order) column to process
n_columns = 20       # number of columns to process
chunk_size = 20      # number of columns per radiance_simulator_batch call

# output options
export_results = True
export_full_field = False   # also store the full (frequency, idx, lev, zenith,
                            # azimuth) cube, not just the TOA slice

# If True, FORCE recalculation of the batch LUT before the simulations
# (i.e. overwrite any stored LUT in the cache folder even if it exists).
# If False, the stored LUT is used when one exists; a LUT is only calculated
# from scratch if the cache does not contain one yet. Because the LUT is
# calculated up front from ALL input atmospheres (see below), a stored LUT
# is always wide enough. Delete <cwd>/cache/<setup_name>/ manually if the
# spectrum or species setup has changed.
recalc_lut = False

# FluxSimulator setup name.
# A NEW setup name -> its own LUT cache dir <cwd>/cache/<setup_name>/ is
# used; nothing is shared with the fmip_batch (flux) cache.
setup_name = "fmip_radiance_batch"


# =============================================================================
# %% helpers
# =============================================================================

def define_abs_species(Flxsim, species_list_of_data):
    """
    Analog of rte_aux_functions.define_abs_species.

    For every species present in the atmospheric data, return the absorption
    species to pass to FluxSimulator.add_species: the plain name if it is a
    line species, the name + "-XFIT" if it is a cross-section species.
    Species not found in the catalog are skipped with a warning.
    """
    abs_species = []
    for spc in species_list_of_data:
        in_lines = any(spc_arts == spc
                       for spc_arts in Flxsim.line_species_available)
        in_xsec = any(spc_arts == spc
                      for spc_arts in Flxsim.xsec_species_available)
        if in_lines:
            abs_species.append(spc)
        if in_xsec:
            abs_species.append(spc + "-XFIT")
        if not (in_lines or in_xsec):
            print(f"WARNING: species {spc} not found in catalog lines/ or "
                  f"xsec/, skipping it\n")
    return abs_species


def make_gridded_field(p_prof, t_prof, z_prof, spc_profiles, species_list):
    """
    Build the compact ARTS atmosphere (GriddedField4) for one column.

    All profiles must be in surface-first order (pressure decreasing, i.e.
    reversed w.r.t. the input files, which are TOA-first).
    """
    atm_field = arts.GriddedField4()

    atm_field.set_grid(
        0, ["T", "z"] + [f"abs_species-{spc}" for spc in species_list]
    )
    atm_field.set_grid(1, p_prof.astype(np.float64))

    data = np.zeros(
        (len(atm_field.grids[0]), len(atm_field.grids[1]), 1, 1)
    )
    # first two dimensions are temperature and altitude
    data[0, :, 0, 0] = t_prof
    data[1, :, 0, 0] = z_prof
    # the remaining dimensions are the gas volume mixing ratios
    for i, spc in enumerate(species_list):
        data[i + 2, :, 0, 0] = spc_profiles[spc]

    # clip tiny negative mixing ratios (mixing-ratio preparation artifacts)
    data[data < 0] = 0

    atm_field.data = data
    return atm_field


def make_reflectivities(eps_profiles, wvn_em):
    """
    Surface reflectivity vectors for a set of columns.

    Per column the spectral emissivity is interpolated from the emissivity
    file wavenumber grid onto the simulation wavenumber grid, and the
    reflectivity is 1 - emissivity (one value per frequency of f_grid).

    Returns a list of (N_wvn,) vectors, one per column.
    """
    refls = []
    for j in range(eps_profiles.shape[0]):
        eps = np.interp(wvn_grid, wvn_em, eps_profiles[j])
        refls.append(np.clip(1.0 - eps, 0.0, 1.0))
    return refls


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()

    # -------------------------------------------------------------------------
    # open the input data
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
    end_column = min(start_column + n_columns, n_total)
    col_ids = list(range(start_column, end_column))     # original idx values
    n_to_run = len(col_ids)
    n_chunks = int(np.ceil(n_to_run / chunk_size))

    # atmospheric species in the input data (volume mixing ratios)
    species_list_of_data = [
        spc for spc in atmdata.data_vars if spc not in ("p", "t", "alt")
    ]

    # load all profiles from the file (files are TOA-first -> flip to
    # surface-first). The LUT is calculated from EVERY column of the file
    # (independent of the subset run below), so all of them are needed.
    p_data = atmdata["p"].values            # (n_total, n_lev)
    t_data = atmdata["t"].values
    z_data = atmdata["alt"].values
    spc_data = {
        spc: atmdata[spc].values
        for spc in species_list_of_data
    }

    # sfcdata fields are 1-D coordinate arrays (no shared 'idx' dim),
    # hence positional indexing
    T_sfc_file = sfcdata["temperature"].values
    lat_file = sfcdata["lat"].values
    lon_file = sfcdata["lon"].values

    wvn_em = emissivity["wvn"].values
    # emissivities are only needed for the simulated subset
    eps_subset = emissivity["emissivity"].isel(
        idx=slice(start_column, end_column)).values

    # -------------------------------------------------------------------------
    # run summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("FMIP batch spectral-radiance (TOA) benchmark")
    print(f"  spectrum    : {wvn_min:.0f} - {wvn_max:.0f} cm^-1, "
          f"N_wvn = {N_wvn}")
    print(f"  columns     : {col_ids[0]} .. {col_ids[-1]} "
          f"({n_to_run} columns, {n_chunks} chunk(s) of <= {chunk_size})")
    print(f"  levels      : {n_lev}")
    print(f"  angular     : N_za = {N_za}, N_aa = {N_aa}")
    print(f"  emission    : longwave/thermal, no sun, gas_scattering = False")
    print(f"  species     : {len(species_list_of_data)} in the data")
    print(f"  setup name  : {setup_name}")
    print(f"  LUT cache   : {os.path.join(os.getcwd(), 'cache', setup_name)}")
    print(f"  recalc LUT  : {recalc_lut}")
    print("=" * 70)

    if export_results:
        results_folder.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # %% atmospheres for ALL columns of the input file (surface-first)
    # ==========================================================================
    # All profiles of the file are built once up front so the batch LUT can
    # be calculated from the full set of atmospheres (independent of the
    # start_column / n_columns subset run below).
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
    # %% LUT setup up front, from ALL atmospheres of the input file
    # ==========================================================================
    # Calculate (or read) the batch LUT before any simulation runs. Passing
    # EVERY atmosphere in the input file (not only the column subset that is
    # simulated below) guarantees the LUT state ranges (temperature, pressure,
    # mixing ratios) cover any possible chunk/selection. recalc_lut (edit
    # section) forces a rebuild of any stored LUT.
    t_lut = time.time()
    Flxsim_lut = fsm.FluxSimulator(setup_name)
    Flxsim_lut.set_frequency_grid(f_grid)

    # data species added to the FluxSimulator defaults
    # (H2O + MT-CKD 4.3 continuums, O2/N2 CIA, CO2-CKDMT252, O3, O3-XFIT),
    # mirroring the LW part of rte_benchmarks_batch.py
    abs_species = define_abs_species(Flxsim_lut, species_list_of_data)
    Flxsim_lut.add_species(abs_species, verbose=True)

    Flxsim_lut.emission = 1
    Flxsim_lut.gas_scattering = False

    Flxsim_lut.get_lookuptableBatch(all_atms, recalc=recalc_lut)
    print(f"  LUT ready in {time.time() - t_lut:.1f} s\n")

    # ==========================================================================
    # %% simulation, chunk by chunk
    # ==========================================================================
    for ck in range(n_chunks):
        c0 = ck * chunk_size
        c1 = min(c0 + chunk_size, n_to_run)

        print("=" * 70)
        print(f"chunk {ck + 1}/{n_chunks}: "
              f"columns {col_ids[c0]} .. {col_ids[c1 - 1]} ({c1 - c0} columns)")
        print("=" * 70)

        # chunk views of the up-front built full-file arrays (shifted by
        # start_column, since the lists cover ALL columns of the input file)
        atms = all_atms[c0 + start_column:c1 + start_column]
        T_surfs = T_surfs_all[c0 + start_column:c1 + start_column]
        geo_positions = geo_positions_all[c0 + start_column:c1 + start_column]

        # surface altitudes from the (surface-first) compact atmospheres
        surface_altitudes = [atm[1, 0, 0, 0] for atm in atms]

        # per-column surface reflectivity vectors (1 - eps), length == f_grid
        reflectivities = make_reflectivities(
            eps_subset[c0:c1], wvn_em)

        # no sun for longwave
        sun_positions = [[] for _ in atms]

        # ---------------------------------------------------------------------
        # FluxSimulator setup (fresh instance per chunk, see module docstring)
        # ---------------------------------------------------------------------
        t_chunk = time.time()
        Flxsim = fsm.FluxSimulator(setup_name)
        Flxsim.set_frequency_grid(f_grid)

        # data species added to the FluxSimulator defaults
        # (H2O + MT-CKD 4.3 continuums, O2/N2 CIA, CO2-CKDMT252, O3, O3-XFIT),
        # mirroring the LW part of rte_benchmarks_batch.py
        abs_species = define_abs_species(Flxsim, species_list_of_data)
        Flxsim.add_species(abs_species, verbose=(ck == 0))

        Flxsim.emission = 1
        Flxsim.gas_scattering = False

        # the simulation (radiance batch derives nstreams from N_za internally).
        # recalc=False is passed through to the internal get_lookuptableBatch
        # call, which then just reads the LUT that was set up up front from
        # ALL atmospheres; the recalc fallback only triggers if the stored
        # LUT is missing or unfit.
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

        # authoritative output grids (as returned by the simulator)
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

        if export_results:
            fname = (
                f"Reference_radiance_TOA_Nf{N_wvn}"
                f"_Nza{N_za}_Naa{N_aa}"
            )
            if n_chunks > 1:
                fname += f"_chunk{ck:03d}"
            out_file = results_folder / f"{fname}.nc"
            ds.to_netcdf(str(out_file))
            print(f"  exported {out_file}\n")

    print(f"total wall time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
