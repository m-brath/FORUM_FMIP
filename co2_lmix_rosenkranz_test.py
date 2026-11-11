#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2 line-mixing (ECS / Rosenkranz adaptation) test for the 2398-2520 cm^-1
band-wing deficit (see results/COMPARISON analysis notes; the 2349 cm^-1
CO2 nu3 marker in compare_convolved_with_obs.py).

Idea: rerun the production-style TOA-radiance pipeline for the single FMIP
column atm_index with ARTS' ECS-based CO2 line mixing, in three arms --

    none : untouched arts-cat-data-2.6.20 line catalogs (no line mixing)
    ra0  : ECS line mixing fitted through the EIGENVALUES of the ECS
           relaxation matrix (workspace-method default,
           rosenkranz_adaptation = 0)
    ra1  : ECS line mixing fitted to the DIRECT first/second-order
           Rosenkranz Y/G/DV expressions (rosenkranz_adaptation = 1)

and quantify whether / how much the line mixing closes the model-vs-IASI
T_B deficit in the CO2 nu3 short branch (2398-2520 cm^-1).

Mechanism (verified against the ARTS 2.6 source, src/m_linemixing.cc,
src/linemixing.cc):

* ws.abs_lines_per_speciesAdaptOnTheFlyLineMixing(abs_lines_per_species,
  ecs_data, t_grid, pressure, order, robust, rosenkranz_adaptation)
  converts bands whose population flag is ByRovibLinearDipoleLineMixing
  (linear-molecule rovibrational ECS -- the CO2 path; ByMakarovFullRelmat
  is the O2 machinery and needs local quantum number N, which the HITRAN
  CO2 bands do not carry) into plain LTE/SFS Voigt bands with per-line
  Rosenkranz Y (order >= 1) and G/DV (order == 2) POLY parameters.  The
  ECS relaxation matrix itself is only defined for CO2-626
  (LinearRovibErrorCorrectedSudden::erot_selection), so only the CO2-626
  bands whose lines overlap the simulated window (padded by the 750 GHz
  FluxSimulator line cutoff plus adapt_wvn_pad) are adapted; all other
  catalog files are carried through unchanged.
* Failed band adaptations fall back to the no-lmix state silently
  (robust != 0) with an OUT3 warning "Bad eigenvalue adaptation"; the
  script checks per band whether Y parameters were actually set.
* ECS data are compiled into ARTS: ecs_dataAddRodrigues1997() provides
  CO2-626/628/636 x N2/O2 collider sets, and ecs_dataAddMeanAir() builds
  the AIR/Bath entry the shipped bands' AIR broadener needs (without it
  the ECS lookup falls back to a missing Bath entry).
* FluxSimulator hard-codes abs_lines_per_speciesReadSpeciesSplitCatalog
  (basename "lines/") in get_lookuptableBatch, so the adapted catalogs
  are injected through a SHADOW CATALOG TREE: results/LMIX_TEST/
  catdata_<variant>/ is a symlink farm of arts-cat-data-2.6.20 whose
  CO2-*.xml line files are the rewritten (adapted) copies.  The ARTS
  global datapath is pointed at the shadow tree after every
  FluxSimulator construction (retrieve() inside FluxSimulator.__init__
  resets it; same pattern as test_co2_intensity_scaling.py).
* FluxSimulator applies Cutoff ByLine 750 GHz and then sets lbl_checked
  = 1, bypassing the ARTS check that rejects ByLine cutoff combined with
  Rosenkranz Y/DV.  The main arms keep cutoff = True for comparability
  with production; set run_nocutoff_arm = True for a cutoff = False
  ra1 cross-check arm that brackets the artifact.

Run from the scripts/ directory:

    /Users/u242031/miniforge3/envs/pyarts/bin/python co2_lmix_rosenkranz_test.py

Outputs:

    results/LMIX_TEST/catdata_<variant>/            shadow catalog trees
    results/LMIX_TEST/<setup>_Nf<Nf>_Nza20/         TOA radiance NetCDF
    plot/lmix_test/fig_lmix_spectra.png             spectra + BT + obs
    plot/lmix_test/fig_lmix_diffs.png               arm differences
"""

# =============================================================================
# %% imports / constants  (edit here)
# =============================================================================

import os
import time
import pathlib

import numpy as np
import xarray as xr
from pyarts import arts, cat
from pyarts.workspace import Workspace

import FluxSimulator as fsm

import frtmip_aux_functions as fax
import fmip_radiance_batch as frb   # reused helpers, no side effects

script_dir = pathlib.Path(__file__).parent.resolve()
results_base = script_dir.parent / "results" / "LMIX_TEST"
plot_dir = script_dir.parent / "plot" / "lmix_test"

# -----------------------------------------------------------------------------
# physics / run setup
# -----------------------------------------------------------------------------
atm_index = 12345                # FMIP column under test (atmdata.nc idx)

wvn_min, wvn_max = 2350.0, 2550.0   # cm^-1, window around the deficit
N_wvn = 4001                     # 0.05 cm^-1 spacing (grid hits 2398 and
                                 # 2520 exactly); 2001 = 0.1 cm^-1 if too slow
N_za, N_aa = 20, 1               # mirror production angular grid

deficit_range_cm = (2398.0, 2520.0)  # CO2 nu3 short branch (the deficit)

# FluxSimulator setup suffix -> rosenkranz_adaptation (None = no lmix arm)
lmix_variants = {"none": None, "ra0": 0, "ra1": 1}
run_nocutoff_arm = False         # extra "ra1_nocutoff" arm (cutoff=False).
                                 # NOTE: prohibitively slow at N_wvn = 4001 --
                                 # without the 750 GHz ByLine cutoff ARTS evaluates
                                 # the Lorentz wings of every line at every
                                 # spectral point (LUT build > 50 min vs ~400 s
                                 # with cutoff; killed at Nf4001, not viable at
                                 # final resolution)
cutoff = True                    # line cutoff in the main arms (= production)

adapt_order = 2                  # 2 = Y + G + DV; 1 would be Y only
adapt_pressure = 50e3            # Pa, reference pressure of the adaptation
adapt_tgrid = np.linspace(160.0, 280.0, 13)  # K, adaptation temperature grid
adapt_wvn_pad = 40.0             # cm^-1 band-selection pad around the window
                                 # (> 750 GHz cutoff = 25 cm^-1)
ecs_air_vmr = [0.78, 0.21]       # N2 / O2 VMRs for the AIR (Bath) ECS entry

rebuild_catalogs = False         # force rebuild of the shadow catalog trees
recalc_lut = False               # force FluxSimulator LUT rebuild

setup_base = "co2_lmix"          # per-variant setup -> own cache/<setup> LUT


def setup_name(variant):
    """FluxSimulator setup name of one arm (Nf in the name so that a
    changed frequency grid cannot silently reuse an old LUT cache)."""
    return f"{setup_base}_{variant}_Nf{N_wvn}"


def run_folder(variant):
    return results_base / f"{setup_base}_{variant}_Nf{N_wvn}_Nza{N_za}"


def shadow_cat_dir(variant):
    return results_base / f"catdata_{variant}"


# =============================================================================
# %% shadow catalog construction (ECS adaptation)
# =============================================================================

def point_at_catalog(shadow_dir):
    """
    Replace the arts-cat-data entry of the ARTS global datapath by a shadow
    catalog tree.  Must be called after EVERY FluxSimulator construction,
    because FluxSimulator.__init__ re-resolves the catalog via
    pyarts.cat.download.retrieve().
    """
    dp = [str(p) for p in arts.globals.parameters.datapath]
    dp = [str(shadow_dir) if "arts-cat-data" in p else p for p in dp]
    arts.globals.parameters.datapath = dp


def build_shadow_catalog(variant, ra):
    """
    Build the shadow catalog tree for one lmix arm.

    Reads the CO2 line catalog from the default arts-cat-data, flags every
    CO2-626 band whose lines overlap [wvn_min, wvn_max] + adapt_wvn_pad as
    ByRovibLinearDipoleLineMixing, adapts them with the ECS machinery
    (Rodrigues1997 N2/O2 colliders + MeanAir Bath entry), and writes the
    full CO2 catalog (adapted + untouched bands) into
    <shadow>/lines/CO2-*.xml.  Everything else in the catalog directory
    is symlinked from the original tree.

    Returns (n_selected, n_adapted, n_fallback).
    """
    src = next(pathlib.Path(str(p))
               for p in arts.globals.parameters.datapath
               if "arts-cat-data" in str(p))
    shadow = shadow_cat_dir(variant)
    lines_dir = shadow / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)

    # symlink everything except the CO2 line files (those get rewritten)
    for entry in src.iterdir():
        if entry.name == "lines":
            continue
        link = shadow / entry.name
        if not link.is_symlink() or link.readlink() != entry:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(entry)
    for entry in sorted((src / "lines").iterdir()):
        if entry.name.startswith("CO2-"):
            continue                     # written by ARTS below
        link = lines_dir / entry.name
        if not link.is_symlink() or link.readlink() != entry:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(entry)

    # --- read CO2 catalogs, select window-overlapping 626 bands ------------
    ws = Workspace(verbosity=0)
    ws.abs_species = [[arts.SpeciesTag("CO2")]]
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    lines = ws.abs_lines_per_species.value

    f_lo = arts.convert.kaycm2freq(wvn_min - adapt_wvn_pad)
    f_hi = arts.convert.kaycm2freq(wvn_max + adapt_wvn_pad)
    selected_pos = []
    for i, band in enumerate(lines[0]):
        if str(band.quantumidentity).split()[0] != "CO2-626":
            continue
        f0 = np.array([ln.F0 for ln in band.lines])
        if f0.max() >= f_lo and f0.min() <= f_hi:
            band.population = "ByRovibLinearDipoleLineMixing"
            selected_pos.append(i)
    print(f"  {variant}: {len(selected_pos)} CO2-626 bands overlap "
          f"[{wvn_min - adapt_wvn_pad:.0f}, {wvn_max + adapt_wvn_pad:.0f}] "
          f"cm^-1 -> selected for ECS adaptation")

    # --- ECS adaptation (verbosity 3 so OUT3 fallback warnings reach stdout)
    ws_ad = Workspace(verbosity=3)
    ws_ad.abs_species = [[arts.SpeciesTag("CO2")]]
    ws_ad.abs_lines_per_species = lines
    ws_ad.Wigner6Init()
    ws_ad.ecs_dataInit()
    ws_ad.ecs_dataAddRodrigues1997()
    ws_ad.ecs_dataAddMeanAir(vmrs=ecs_air_vmr,
                             specs=[arts.SpeciesTag("N2"),
                                    arts.SpeciesTag("O2")])
    t0 = time.time()
    ws_ad.abs_lines_per_speciesAdaptOnTheFlyLineMixing(
        t_grid=adapt_tgrid, pressure=adapt_pressure,
        order=adapt_order, robust=1, rosenkranz_adaptation=ra)
    print(f"  {variant}: ECS adaptation (order={adapt_order}, "
          f"P0={adapt_pressure/1e3:.0f} kPa, rosenkranz_adaptation={ra}) "
          f"done in {time.time() - t0:.0f} s")

    # --- G4 check: every selected band must carry Y (and DV) parameters ----
    # (read the adapted bands back out of the workspace; the python-side
    # copies were not touched by the method)
    adapted = ws_ad.abs_lines_per_species.value
    n_ok = n_bad = 0
    for i in selected_pos:
        band = adapted[0][i]
        lss = band.lines[0].lineshape       # per-broadener shape models
        order_ok = (str(lss[0].Y.type) != "None"
                    and (adapt_order < 2 or str(lss[0].DV.type) != "None"))
        if order_ok:
            n_ok += 1
        else:
            n_bad += 1
            print(f"    WARNING: band {band.quantumidentity} fell back to "
                  f"no line mixing (Bad eigenvalue adaptation)")
    print(f"  {variant}: bands adapted {n_ok}, fallback (no Y/DV) {n_bad}")

    # --- write the full CO2 catalog into the shadow tree --------------------
    ws_ad.abs_lines_per_speciesWriteSpeciesSplitCatalog(
        output_file_format="ascii", basename=str(lines_dir) + "/")
    written = sorted(p.name for p in lines_dir.glob("CO2-*.xml"))
    print(f"  {variant}: wrote {written} into {lines_dir}")
    return len(selected_pos), n_ok, n_bad


# =============================================================================
# %% single-column simulation (mirrors fmip_radiance_batch.py)
# =============================================================================

def run_arm(variant, ra, f_grid, wvn_grid, atm, T_surf, geo, reflectivity,
            species_list, use_cutoff):
    """
    One arm: LUT up front + radiance_simulator_batch for the single column.

    Fresh FluxSimulator instances (one for the LUT, one for the radiance
    call), distinct setup_name per arm -> own cache/<setup> LUT directory.
    The shadow catalog (if any) is injected by pointing the ARTS global
    datapath at it right after each FluxSimulator construction.
    """
    setup = setup_name(variant)
    shadow = None if ra is None else shadow_cat_dir(variant)

    def new_simulator():
        flx = fsm.FluxSimulator(setup)
        if shadow is not None:
            point_at_catalog(shadow)
        flx.set_frequency_grid(f_grid)
        abs_species = frb.define_abs_species(flx, species_list)
        if shadow is not None:
            point_at_catalog(shadow)
        flx.add_species(abs_species)
        flx.emission = 1
        flx.gas_scattering = False
        return flx

    print("=" * 70)
    print(f"arm {variant}: setup {setup}, "
          f"catalog {'shadow ' + str(shadow) if shadow else 'default'}, "
          f"cutoff={use_cutoff}")
    print("=" * 70)

    # --- LUT up front from the single tested atmosphere --------------------
    t0 = time.time()
    flx = new_simulator()
    flx.get_lookuptableBatch([atm], recalc=recalc_lut, cutoff=use_cutoff)
    print(f"  LUT ready in {time.time() - t0:.1f} s")

    # --- radiance (fresh instance; reads the stored LUT) -------------------
    t0 = time.time()
    flx = new_simulator()
    results = flx.radiance_simulator_batch(
        [atm],
        [T_surf],
        [float(atm[1, 0, 0, 0])],          # surface altitude
        [reflectivity],
        [geo],
        [[]],                             # no sun (longwave)
        N_za=N_za,
        N_aa=N_aa,
        end_index=-1,
        recalc=False,
        cutoff=use_cutoff,
    )
    print(f"  radiance done in {time.time() - t0:.1f} s")

    f_out = np.array(results["f_grid"])
    za_grid = np.array(results["zenith_angle"])
    za_index = int(np.argmin(np.abs(za_grid - 180.0)))
    if not np.isclose(za_grid[za_index], 180.0):
        raise AssertionError(
            f"zenith angle grid does not contain 180 degrees: {za_grid}")

    # (frequency, idx) TOA slice, surface-first level axis -> TOA = last
    toa_rad = np.array(results["array_of_spectral_radiance_clearsky"][0]
                       )[:, -1, za_index, 0]
    altitude = np.array(results["array_of_altitude"][0])
    print(f"  TOA radiance at za={za_grid[za_index]:.1f} deg: "
          f"min {toa_rad.min():.3e}, max {toa_rad.max():.3e}")

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    folder = run_folder(variant)
    folder.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "spectral_radiance_TOA":
                (["frequency", "idx"], toa_rad[:, None],
                 {"units": "W m^-2 Hz^-1 sr^-1 (ARTS/DISORT spectral radiance)",
                  "zenith_angle": float(za_grid[za_index])}),
            "altitude_min": (["idx"], [altitude.min()], {"units": "m"}),
            "altitude_max": (["idx"], [altitude.max()], {"units": "m"}),
        },
        coords={
            "idx": np.array([atm_index], dtype=np.int64),
            "frequency": (["frequency"], f_out, {"units": "Hz"}),
            "wavenumber": (["frequency"], arts.convert.freq2kaycm(f_out),
                           {"units": "cm^-1"}),
        },
    )
    ds.attrs = {
        "title": "CO2 line-mixing (ECS Rosenkranz adaptation) test, "
                 "single-column TOA radiance",
        "variant": variant,
        "rosenkranz_adaptation": -1 if ra is None else int(ra),
        "lmix_ecs_source": "ecs_dataAddRodrigues1997 (CO2 x N2/O2) + "
                           "ecs_dataAddMeanAir (AIR/Bath, "
                           f"vmr N2={ecs_air_vmr[0]}, O2={ecs_air_vmr[1]})",
        "lmix_adapt_order": int(adapt_order),
        "lmix_adapt_pressure_Pa": float(adapt_pressure),
        "lmix_adapt_t_grid_K": " ".join(f"{t:.1f}" for t in adapt_tgrid),
        "lmix_adapt_wvn_pad_cm": float(adapt_wvn_pad),
        "lmix_bands_adapted": "CO2-626 only (ECS erot defined for 626; "
                              "ByRovibLinearDipoleLineMixing)",
        "catalog": str(shadow_cat_dir(variant)) if shadow is not None
                   else "arts-cat-data (unmodified)",
        "fluxsimulator_cutoff": "True" if use_cutoff else "False",
        "lbl_checked": "bypassed by FluxSimulator (cutoff arm); "
                       "ByLine cutoff + Rosenkranz Y/DV not officially "
                       "supported -- see ra1_nocutoff arm",
        "setup_name": setup,
        "atm_index": int(atm_index),
        "N_za": int(N_za), "N_aa": int(N_aa),
        "simulated_window_cm": f"{wvn_grid.min():.2f}-{wvn_grid.max():.2f}",
    }
    out_file = folder / (f"Reference_radiance_TOA_{setup}"
                         f"_Nza{N_za}_co2_lmix_{variant}.nc")
    ds.to_netcdf(str(out_file))
    print(f"  exported {out_file}\n")
    return ds


# =============================================================================
# %% gates and plots
# =============================================================================

def arm_bt(ds):
    """Planck brightness temperature of an arm's monochromatic TOA radiance."""
    wvn = ds["wavenumber"].values
    rad = ds["spectral_radiance_TOA"].values[:, 0]
    return fax.radiance2brightness_temp(rad, wvn)


def gate_g2_none_vs_production(ds_none):
    """
    G2: compare the none arm against the production run
    (fmip_radiance_batch_slurm_Nf108951_Nza20) at atm_index on the
    exactly-coincident wavenumber points (production grid is 0.02 cm^-1,
    this run 0.05 -- every other point lies on a production grid point).
    Reports max/mean abs difference in radiance -- the pipeline mirror
    should be faithful up to LUT interpolation (production LUT covers all
    19706 columns, this run only the tested column).
    """
    import glob
    prod_files = sorted(glob.glob(
        str(script_dir.parent / "results/RADIANCE/"
            "fmip_radiance_batch_slurm_Nf108951_Nza20/"
            "Reference_radiance_TOA_merged_*.nc")))
    if not prod_files:
        print("G2: no production merged file found, gate SKIPPED")
        return
    for f in prod_files:
        prod = xr.open_dataset(f)
        idx = prod["idx"].values
        if idx.min() <= atm_index <= idx.max():
            break
    else:
        print(f"G2: production files do not cover idx {atm_index}, skipped")
        return

    j = int(np.where(idx == atm_index)[0][0])
    prod_wvn = prod["wavenumber"].values
    prod_rad = prod["spectral_radiance_TOA"].values[:, j]

    wvn = ds_none["wavenumber"].values
    rad = ds_none["spectral_radiance_TOA"].values[:, 0]
    # Nearest-point comparison is meaningless on the CO2 line cores when
    # the two grids do not coincide (a 0.01 cm^-1 offset alone moves the
    # production spectrum by ~37% of peak here), so the headline number
    # uses only the exactly-coincident grid points; the nearest-point and
    # production self-interpolation numbers are printed as context.
    k = np.abs(wvn[:, None] - prod_wvn[None, :]).argmin(axis=1)
    d_nearest = np.abs(rad - prod_rad[k])
    exact = np.abs(wvn - prod_wvn[k]) < 1e-9
    d_exact = d_nearest[exact]
    scale = np.abs(prod_rad[k]).max()
    prod_interp = np.interp(wvn, prod_wvn, prod_rad)
    d_grid = np.abs(prod_rad[k] - prod_interp)
    print(f"G2: none vs production (idx {atm_index}, {len(wvn)} common "
          f"points, {int(exact.sum())} exactly coincident): "
          f"max |diff| on coincident points = {d_exact.max():.4e} "
          f"({100 * d_exact.max() / scale:.2f}% of peak), "
          f"mean |diff| = {d_exact.mean():.4e}")
    print(f"    context: nearest-point max |diff| = {d_nearest.max():.4e} "
          f"(grid offsets up to {np.abs(wvn - prod_wvn[k]).max():.3f} cm^-1); "
          f"production self-interpolation at the same offsets moves "
          f"{d_grid.max():.4e} -- grid-offset artifact, not physics")


def make_plots(arms):
    """
    fig_lmix_spectra.png : monochromatic TOA radiance + Planck BT of all
                           arms, IASI obs T_B of the column as context
    fig_lmix_diffs.png   : BT differences ra0-none / ra1-none / ra1-ra0,
                           2398-2520 cm^-1 shaded
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import compare_convolved_with_obs as ccow

    ccow.apply_plot_style()

    sat = xr.open_dataset(frb.satdata_file)
    obs_wvn = sat["wvn"].values
    obs_bt = sat["T_b"].values[atm_index, :]
    m_obs = (obs_wvn >= wvn_min) & (obs_wvn <= wvn_max)

    color = {"none": ccow.palette["sim"],
             "ra0": "#1a7f37",
             "ra1": "#8250df",
             "ra1_nocutoff": "#b08800"}
    label = {"none": "no lmix",
             "ra0": "lmix ECS eigenvalues (ra0)",
             "ra1": "lmix Rosenkranz direct (ra1)",
             "ra1_nocutoff": "ra1, cutoff off"}

    variants = list(arms)

    # ---- spectra ----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2), sharex=True)
    ax = axes[0]
    for v in variants:
        w = arms[v]["wavenumber"].values
        r = arms[v]["spectral_radiance_TOA"].values[:, 0]
        ax.plot(w, r, color=color[v], lw=ccow.line_width, label=label[v])
    ax.set_ylabel(r"TOA spectral radiance / W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$")
    ax.legend(loc="upper left")

    ax = axes[1]
    for v in variants:
        w = arms[v]["wavenumber"].values
        ax.plot(w, arm_bt(arms[v]), color=color[v], lw=ccow.line_width,
                label=label[v])
    ax.plot(obs_wvn[m_obs], obs_bt[m_obs], ".", ms=2,
            color=ccow.palette["obs"], label="IASI obs $T_B$ (context)")
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(r"Monochromatic $T_B$ / K")
    ax.set_xlim(wvn_min, wvn_max)
    ax.axvspan(*deficit_range_cm, color=ccow.palette["shade"], zorder=0)
    ccow.annotate_spectral_features(ax, shade_window=False)
    ax.legend(loc="upper left")
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(r"CO$_2$ ECS line mixing, FMIP column idx "
                 f"{atm_index}: monochromatic TOA radiance", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_file = plot_dir / "fig_lmix_spectra.png"
    fig.savefig(fig_file, dpi=200)
    plt.close(fig)
    print(f"  wrote {fig_file}")

    # ---- differences ------------------------------------------------------
    diffs = []
    if "ra0" in arms and "none" in arms:
        diffs.append(("ra0 - none", arm_bt(arms["ra0"]) - arm_bt(arms["none"]),
                      color["ra0"]))
    if "ra1" in arms and "none" in arms:
        diffs.append(("ra1 - none", arm_bt(arms["ra1"]) - arm_bt(arms["none"]),
                      color["ra1"]))
    if "ra1" in arms and "ra0" in arms:
        diffs.append(("ra1 - ra0", arm_bt(arms["ra1"]) - arm_bt(arms["ra0"]),
                      color["ra1"], "--"))
    if "ra1_nocutoff" in arms and "ra1" in arms:
        diffs.append(("ra1_nocutoff - ra1",
                      arm_bt(arms["ra1_nocutoff"]) - arm_bt(arms["ra1"]),
                      color["ra1_nocutoff"], ":"))

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    w = arms[variants[0]]["wavenumber"].values
    for d in diffs:
        ax.plot(w, d[1], color=d[2], lw=ccow.line_width,
                ls=(d[3] if len(d) > 3 else "-"), label=d[0])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvspan(*deficit_range_cm, color=ccow.palette["shade"], zorder=0)
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(r"$T_B$ difference / K")
    ax.set_xlim(wvn_min, wvn_max)
    ccow.annotate_spectral_features(ax, shade_window=False)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.suptitle(r"$T_B$ (monochromatic) line-mixing increments, "
                 f"FMIP column idx {atm_index}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_file = plot_dir / "fig_lmix_diffs.png"
    fig.savefig(fig_file, dpi=200)
    plt.close(fig)
    print(f"  wrote {fig_file}")


def print_gate_g3(arms):
    """G3: line-mixing BT/radiance increments inside the deficit range."""
    if "none" not in arms:
        return
    w = arms["none"]["wavenumber"].values
    m = (w >= deficit_range_cm[0]) & (w <= deficit_range_cm[1])
    pairs = [("ra0", "none"), ("ra1", "none"), ("ra1_nocutoff", "none"),
             ("ra1", "ra0"), ("ra1_nocutoff", "ra1")]
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        dbt = arm_bt(arms[a]) - arm_bt(arms[b])
        dr = (arms[a]["spectral_radiance_TOA"].values[:, 0]
              - arms[b]["spectral_radiance_TOA"].values[:, 0])
        print(f"G3 {a} - {b} over {deficit_range_cm[0]:.0f}-"
              f"{deficit_range_cm[1]:.0f} cm^-1: "
              f"T_B mean {dbt[m].mean():+.3f} K "
              f"(min {dbt[m].min():+.3f}, max {dbt[m].max():+.3f}), "
              f"radiance mean {dr[m].mean():+.3e} "
              f"({dr[m].min():+.3e} .. {dr[m].max():+.3e}) "
              f"W m^-2 Hz^-1 sr^-1")


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()

    cat.download.retrieve(verbose=False)
    print("ARTS datapath:", [str(p) for p in arts.globals.parameters.datapath])

    # ------------------------------------------------------------------
    # input data for the single tested column
    # ------------------------------------------------------------------
    atmdata = xr.open_dataset(frb.atmdata_file)
    sfcdata = xr.open_dataset(frb.sfcdata_file)
    emissivity = xr.open_dataset(frb.surface_emissivity_file)

    n_total = int(atmdata.sizes["idx"])
    if atm_index >= n_total:
        raise ValueError(f"atm_index {atm_index} >= {n_total} columns")

    species_list = [spc for spc in atmdata.data_vars
                    if spc not in ("p", "t", "alt")]

    j = int(np.where(atmdata["idx"].values == atm_index)[0][0]) \
        if "idx" in atmdata.coords else atm_index
    # surface-first profiles (input files are TOA-first)
    atm = frb.make_gridded_field(
        atmdata["p"].values[j][::-1],
        atmdata["t"].values[j][::-1],
        atmdata["alt"].values[j][::-1],
        {spc: atmdata[spc].values[j][::-1] for spc in species_list},
        species_list,
    )
    T_surf = float(sfcdata["temperature"].values[j])
    geo = [float(sfcdata["lat"].values[j]), float(sfcdata["lon"].values[j])]

    wvn_grid = np.linspace(wvn_min, wvn_max, N_wvn)
    f_grid = arts.convert.kaycm2freq(wvn_grid)
    reflectivity = frb.make_reflectivities(
        emissivity["emissivity"].isel(idx=j).values[None, :],
        emissivity["wvn"].values, wvn_grid)[0]

    arms_def = dict(lmix_variants)
    if run_nocutoff_arm:
        arms_def["ra1_nocutoff"] = 1

    print("=" * 70)
    print("CO2 line-mixing (ECS Rosenkranz adaptation) single-column test")
    print(f"  column    : idx {atm_index} "
          f"(T_sfc {T_surf:.1f} K, {atmdata.sizes['lev']} levels)")
    print(f"  spectrum  : {wvn_min:.0f}-{wvn_max:.0f} cm^-1, "
          f"N_wvn = {N_wvn} (df = "
          f"{(wvn_max - wvn_min) / max(N_wvn - 1, 1):.3f} cm^-1)")
    print(f"  angular   : N_za = {N_za}, N_aa = {N_aa}, TOA za = 180 deg")
    print(f"  arms      : { {v: r for v, r in arms_def.items()} } "
          f"(cutoff={cutoff})")
    print(f"  adaptation: order={adapt_order}, P0={adapt_pressure/1e3:.0f} "
          f"kPa, T {adapt_tgrid[0]:.0f}-{adapt_tgrid[-1]:.0f} K "
          f"({adapt_tgrid.size} pts)")
    print(f"  results   : {results_base}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # shadow catalogs (lmix arms); "none" uses the unmodified catalog
    # ------------------------------------------------------------------
    for variant, ra in arms_def.items():
        if ra is None:
            continue
        marker = shadow_cat_dir(variant) / "lines" / "CO2-626.xml"
        if marker.exists() and not rebuild_catalogs:
            print(f"  {variant}: shadow catalog exists "
                  f"({shadow_cat_dir(variant)}), keeping it "
                  f"(rebuild_catalogs=True to force)")
            continue
        build_shadow_catalog(variant, ra)

    # ------------------------------------------------------------------
    # simulate all arms
    # ------------------------------------------------------------------
    arms = {}
    for variant, ra in arms_def.items():
        use_cutoff = cutoff and not (variant == "ra1_nocutoff")
        arms[variant] = run_arm(variant, ra, f_grid, wvn_grid, atm, T_surf,
                                geo, reflectivity, species_list, use_cutoff)

    # ------------------------------------------------------------------
    # gates / diagnostics / plots
    # ------------------------------------------------------------------
    print("-" * 70)
    gate_g2_none_vs_production(arms["none"])
    print_gate_g3(arms)
    print("-" * 70)
    make_plots(arms)

    print(f"total wall time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
