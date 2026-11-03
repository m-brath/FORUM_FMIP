#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal test for the CO2 nu3 (4.3 micron) band-edge bias (analysis note
results/COMPARISON/analysis_co2_nu3_wing_2026-11-03.md).

Idea: rerun a small radiance benchmark over 2330-2560 cm^-1 with every CO2
line (all isotopologues) whose centre lies in 2398-2520 cm^-1 scaled to 10x
its catalog intensity, and check whether the smooth +4 K sim-obs deficit in
that range shrinks while the -9 K lobe at 2382-2396 does NOT collapse
(a column-like "missing absorption" fix cannot explain the lobe pair).

Two variants, run as separate processes because the ARTS catalog path is a
global library setting:

    CO2_SCALE_VARIANT=baseline  normal arts-cat-data-2.6.20 line catalogs
    CO2_SCALE_VARIANT=scaled    patched catalog (catalog_co2x10/, built by
                                hand: symlink farm of arts-cat-data-2.6.20
                                with lines/CO2-*.xml intensities x10 for
                                line centres in 2398-2520 cm^-1)

Both variants share every other setup (frequency grid 2330-2560 cm^-1 at
0.02 cm^-1, same column subset, nadir-only DISORT, LUT built from the same
subset atmospheres). Each variant writes its SRF-convolved channel-grid
brightness temperature to results/RADIANCE/co2_scale_test/. After the
second variant ran, the comparison against the observations and against the
full-run Reference_convolved products is done automatically, and
plot/comparison/fig_co2_nu3_scaling_test.png is written.

Run from the scripts/ directory:

    CO2_SCALE_VARIANT=baseline  .../python test_co2_intensity_scaling.py
    CO2_SCALE_VARIANT=scaled    .../python test_co2_intensity_scaling.py
"""

# =============================================================================
# %% imports / constants  (edit here)
# =============================================================================

import os
import time
import pathlib

import numpy as np
import xarray as xr
from pyarts import arts

import FluxSimulator as fsm

import frtmip_aux_functions as fax
import fmip_radiance_batch as frb   # reused helpers, no side effects

script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
results_dir = script_dir.parent / "results" / "RADIANCE" / "co2_scale_test"
plot_dir = script_dir.parent / "plot" / "comparison"

# run variant: env CO2_SCALE_VARIANT = "baseline" (default) or "scaled"
variant = os.environ.get("CO2_SCALE_VARIANT", "baseline")
assert variant in ("baseline", "scaled")

# patched catalog used by the "scaled" variant
patched_catalog = script_dir.parent / "catalog_co2x10"
scale_range = (2398.0, 2520.0)     # cm^-1, documented in the file attrs
scale_factor = 10.0

# mini simulation grid (covers IASI channels 2362-2528 with full SRF support)
wvn_min, wvn_max, wvn_step = 2330.0, 2560.0, 0.05
N_wvn = int(round((wvn_max - wvn_min) / wvn_step)) + 1

# column subset (identical for both variants) and DISORT angular grid
start_column = 0
n_columns = 200
chunk_size = 50
N_za = 20               # DISORT agenda requires za_grid >= 20 (as in the
N_aa = 1                # main runs); additional angles are ~free there

setup_name = f"co2scale_{variant}"


def point_at_patched_catalog():
    """
    Replace the arts-cat-data entry of the ARTS global datapath by the
    patched catalog. Must be called after EVERY FluxSimulator construction,
    because FluxSimulator.__init__ re-resolves the catalog via
    pyarts.cat.download.retrieve().
    """
    dp = [str(p) for p in arts.globals.parameters.datapath]
    dp = [str(patched_catalog) if "arts-cat-data" in p else p for p in dp]
    arts.globals.parameters.datapath = dp


# =============================================================================
# %% simulation of one variant
# =============================================================================

def run_variant():
    t0 = time.time()

    atmdata = xr.open_dataset(frb.atmdata_file)
    sfcdata = xr.open_dataset(frb.sfcdata_file)
    emissivity = xr.open_dataset(frb.surface_emissivity_file)

    wvn_grid = np.linspace(wvn_min, wvn_max, N_wvn)
    f_grid = arts.convert.kaycm2freq(wvn_grid)

    n_total = int(atmdata.sizes["idx"])
    end_column = min(start_column + n_columns, n_total)
    col_ids = np.arange(start_column, end_column)
    species_list = [spc for spc in atmdata.data_vars
                    if spc not in ("p", "t", "alt")]

    atms, T_surfs, geo_positions = [], [], []
    for j in col_ids:
        atms.append(frb.make_gridded_field(
            atmdata["p"].values[j][::-1],
            atmdata["t"].values[j][::-1],
            atmdata["alt"].values[j][::-1],
            {spc: atmdata[spc].values[j][::-1] for spc in species_list},
            species_list,
        ))
        T_surfs.append(float(sfcdata["temperature"].values[j]))
        geo_positions.append([float(sfcdata["lat"].values[j]),
                              float(sfcdata["lon"].values[j])])

    wvn_em = emissivity["wvn"].values
    eps = emissivity["emissivity"].isel(
        idx=slice(start_column, end_column)).values

    print("=" * 70)
    print(f"CO2 scaling test, variant = {variant}")
    print(f"  spectrum  : {wvn_min}-{wvn_max} cm^-1 @ {wvn_step} (Nf={N_wvn})")
    print(f"  columns   : {start_column}..{end_column - 1}")
    print(f"  setup     : {setup_name} (LUT cache cache/{setup_name})")
    print("=" * 70)

    # --- LUT up front from the SAME atmospheres as the simulation ---------
    Flxsim = fsm.FluxSimulator(setup_name)
    if variant == "scaled":
        point_at_patched_catalog()
    Flxsim.set_frequency_grid(f_grid)
    abs_species = frb.define_abs_species(Flxsim, species_list)
    if variant == "scaled":
        point_at_patched_catalog()
    Flxsim.add_species(abs_species)
    Flxsim.emission = 1
    Flxsim.gas_scattering = False
    t_lut = time.time()
    # recalc=False: reuse the LUT built by the first attempt of this setup
    # (the LUT depends on spectrum/species/atmospheres only, which are
    # unchanged here; it is recomputed automatically if absent)
    Flxsim.get_lookuptableBatch(atms, recalc=False)
    print(f"  LUT ready in {time.time() - t_lut:.1f} s")

    # --- radiance, one fresh instance per chunk ----------------------------
    srf = xr.open_dataset(frb.sensor_response_file)
    srf_wvn = srf["wvn"].values
    srf_response = srf["response"].values \
        if "response" in srf else srf["srf"].values
    channel_wvn_all = xr.open_dataset(frb.satdata_file)["wvn"].values

    operator = None
    channel_wvn = None
    bt_all = []

    for c0 in range(0, len(col_ids), chunk_size):
        c1 = min(c0 + chunk_size, len(col_ids))
        Flxsim = fsm.FluxSimulator(setup_name)
        if variant == "scaled":
            point_at_patched_catalog()
        Flxsim.set_frequency_grid(f_grid)
        frb.define_abs_species(Flxsim, species_list)
        if variant == "scaled":
            point_at_patched_catalog()
        Flxsim.add_species(abs_species)
        Flxsim.emission = 1
        Flxsim.gas_scattering = False

        results = Flxsim.radiance_simulator_batch(
            atms[c0:c1],
            T_surfs[c0:c1],
            [atm[1, 0, 0, 0] for atm in atms[c0:c1]],
            frb.make_reflectivities(eps[c0:c1], wvn_em, wvn_grid),
            geo_positions[c0:c1],
            [[] for _ in atms[c0:c1]],
            N_za=N_za,
            N_aa=N_aa,
            end_index=-1,
            recalc=False,
        )

        za_grid = np.array(results["zenith_angle"])
        zi = int(np.argmin(np.abs(za_grid - 180.0)))
        if not np.isclose(za_grid[zi], 180.0):
            print(f"  WARNING: zenith grid {za_grid} has no 180 deg, "
                  f"using {za_grid[zi]}")
        toa = np.array([
            np.array(r)[:, -1, zi, 0] for r in
            results["array_of_spectral_radiance_clearsky"]
        ]).T                          # (N_wvn, n_col), TOA level = last

        if operator is None:
            operator, channel_wvn = fax.build_srf_channel_operator(
                np.array(results["f_grid"]), srf_wvn, srf_response,
                channel_wvn_all)
            print(f"  channels kept: {channel_wvn.min():.1f}-"
                  f"{channel_wvn.max():.1f} cm^-1 "
                  f"({channel_wvn.size} channels)")

        bt = fax.radiance2brightness_temp(
            fax.convolve_with_srf_to_channels(toa, operator), channel_wvn)
        bt_all.append(bt)
        print(f"  chunk {c0 // chunk_size + 1}: columns {col_ids[c0]}.."
              f"{col_ids[c1 - 1]} done ({time.time() - t0:.0f} s)")

    bt_all = np.concatenate(bt_all, axis=1)        # (N_channels, n_columns)

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"Reference_BT_{variant}_Nf{N_wvn}.nc"
    ds = xr.Dataset(
        {"brightness_temperature": (["frequency", "idx"], bt_all)},
        coords={"idx": col_ids,
                "frequency": channel_wvn * 100.0 * 2.99792458e10,
                "wavenumber": channel_wvn},
        attrs={"variant": variant,
               "setup_name": setup_name,
               "co2_intensity_scale_range_cm": f"{scale_range[0]}-{scale_range[1]}",
               "co2_intensity_scale_factor": scale_factor,
               "catalog": str(patched_catalog) if variant == "scaled"
                         else "arts-cat-data-2.6.20",
               "note": "mini run for the CO2 nu3 band-edge causal test; "
                       "LUT from the subset atmospheres only"},
    )
    ds.to_netcdf(str(out))
    print(f"  exported {out}")


# =============================================================================
# %% comparison (runs once both variants are present)
# =============================================================================

def compare_and_plot():
    import matplotlib.pyplot as plt
    import compare_convolved_with_obs as ccow

    files = {v: results_dir / f"Reference_BT_{v}_Nf{N_wvn}.nc"
             for v in ("baseline", "scaled")}
    if not all(f.exists() for f in files.values()):
        return

    bt = {v: xr.open_dataset(f)["brightness_temperature"].values
          for v, f in files.items()}          # (N_ch, n_col)
    ds = xr.open_dataset(files["baseline"])
    wch = ds["wavenumber"].values
    cols = ds["idx"].values

    sat = xr.open_dataset(frb.satdata_file)
    # channels are a subset of the sat grid: nearest-channel collocation
    wobs = sat["wvn"].values
    ci = np.abs(wch[:, None] - wobs[None, :]).argmin(axis=1)
    obs = sat["T_b"].values[cols, :][:, ci].T         # (N_ch, n_col)

    # full-run reference for the same columns (chunk000 holds col 0..1999)
    import glob
    full = None
    for f in sorted(glob.glob(str(
            script_dir.parent / "results/RADIANCE/"
            "fmip_radiance_batch_slurm_Nf108951_Nza20/"
            "Reference_convolved_*.nc"))):
        d = xr.open_dataset(f)
        idx = d["idx"].values
        if idx.min() <= cols[0] and idx.max() >= cols[-1]:
            full = d["brightness_temperature"].values[
                np.ix_(ci, idx.searchsorted(cols))]      # (N_ch, n_col)
            break

    def prof(x):
        return np.nanmean(x, axis=1)

    m = (wch >= 2340) & (wch <= 2530)
    print("\n  wvn     full_run   mini_base   mini_x10      obs   "
          "d_base  d_x10")
    for lo in range(2360, 2520, 10):
        s = m & (wch >= lo) & (wch < lo + 10)
        pb, ps, po = prof(bt["baseline"][s]), prof(bt["scaled"][s]), prof(obs[s])
        pf = prof(full[s]) if full is not None else np.full_like(pb, np.nan)
        print(f"  {lo:5.0f} {pf.mean():9.2f} {pb.mean():10.2f} "
              f"{ps.mean():9.2f} {po.mean():8.2f} "
              f"{pb.mean() - po.mean():+7.2f} {ps.mean() - po.mean():+7.2f}")

    ccow.apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2), sharex=True)
    ax = axes[0]
    if full is not None:
        ax.plot(wch[m], prof(full[m]), color="0.6", lw=1.0,
                label="full run (Nf108951)")
    ax.plot(wch[m], prof(bt["baseline"][m]), color=ccow.palette["sim"],
            lw=ccow.line_width, label="mini baseline")
    ax.plot(wch[m], prof(bt["scaled"][m]), color=ccow.palette["diff"],
            lw=ccow.line_width,
            label="mini, CO2 lines 2398-2520 x10")
    ax.plot(wch[m], prof(obs[m]), color=ccow.palette["obs"], ls="--",
            lw=ccow.line_width, label="IASI obs")
    ax.set_ylabel("Mean $T_B$ / K")
    ax.legend(loc="upper left")

    ax = axes[1]
    if full is not None:
        ax.plot(wch[m], prof(full[m]) - prof(obs[m]), color="0.6", lw=1.0,
                label="full run - obs")
    ax.plot(wch[m], prof(bt["baseline"][m]) - prof(obs[m]),
            color=ccow.palette["sim"], lw=ccow.line_width,
            label="mini baseline - obs")
    ax.plot(wch[m], prof(bt["scaled"][m]) - prof(obs[m]),
            color=ccow.palette["diff"], lw=ccow.line_width,
            label="mini x10 - obs")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvspan(*scale_range, color="0.85", zorder=0)
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel("$T_B$ difference / K")
    ax.set_xlim(2340, 2530)
    ax.legend(loc="upper left")
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(r"CO$_2$ $\nu_3$ band-edge causal test: CO2 line "
                 r"intensities 2398--2520 cm$^{-1}$ scaled $\times$10",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_file = plot_dir / "fig_co2_nu3_scaling_test.png"
    fig.savefig(fig_file, dpi=200)
    print(f"  wrote {fig_file}")


# =============================================================================
# %% main
# =============================================================================

if __name__ == "__main__":
    run_variant()
    compare_and_plot()
