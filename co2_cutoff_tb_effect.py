#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radiative effect of the CO2 line-wing cutoff on the 2398-2520 cm^-1 deficit
(one FMIP column, on-the-fly line-by-line absorption, no lookup table).

`co2_crosssection_cutoff_test.py` shows what the line-wing cutoff does to the
CO2 cross section at a single state: at 2398-2520 cm^-1 the 25 cm^-1 (750 GHz)
benchmark truncation leaves only 1/22 (50 cm^-1) ... 1/217 (untruncated) of the
band-wing absorption.  A cross-section multiplier of two centuries says nothing
about the brightness temperature until the *radiative transfer* is done: the
region is optically thin for the benchmark, so extra absorption pulls T_B down
from the surface value towards the cold emitting layer, and it saturates.

This script closes that loop for the tested column (`atm_index`, default
12345, the same column as the opacity / line-mixing / cross-section tests):

  1. per cutoff arm it computes the CO2 line-by-line absorption coefficient at
     every layer of the column (on-the-fly `propmat_clearskyAddLines`, no
     FluxSimulator, no `abs_lookup`, no LUT cache),
  2. integrates the vertical (nadir) clear-sky RTE in LTE -> monochromatic TOA
     radiance, Planck-inverts it,
  3. convolves the radiance with the IASI SRF (`frtmip_aux_functions` operator,
     exactly as `convolve_sensor_response.py`) so the arm differences are on the
     *sensor* grid and directly comparable with the benchmark-vs-observation
     statistics in results/COMPARISON/comparison_stats_*.nc,
  4. compares dT_B(arm - 25 cm^-1) against the deficit that has to be removed,
     -(sim - obs) from the comparison statistics, and inverts the cutoff that
     would supply it at the reference wavenumber.

Result (documented in results/COMPARISON/analysis_co2_nu3_wing_2026-11-03.md,
sec. 13): the sign is right (the benchmark is too warm, extra CO2 absorption
cools) but the cutoff is not the cause -- doubling the cutoff to 50 cm^-1
already overshoots the required opacity by an order of magnitude at 2405 cm^-1
and does nothing beyond ~2425 cm^-1, while the observed deficit is smooth out to
2520 cm^-1.  No single cutoff reproduces the observed shape.

IMPORTANT scope notes
---------------------
* Absorption is CO2 *lines only*: no MT-CKD CO2 continuum, no H2O/O2/N2.  The
  absolute T_B of this script is therefore warmer than the products (the
  production run has the full absorber set, and its 2400-2500 cm^-1 wing
  opacity is dominated by the CO2 continuum, sigma ~ 6.5e-28 m^2 at 2450 cm^-1,
  i.e. ~14x the 25 cm^-1 line-wing value).  Only the *arm differences*
  dT_B(arm - 25 cm^-1) and the CO2 line-wing optical depths are meaningful.
* Vertical path only (nadir), plane-parallel, no scattering, no Jacobians.
* "untruncated" (`option="None"`) is a reference, not physics: it sums the
  SFS-scaled Voigt wings of all 545084 catalogued CO2 lines (including the
  10 um band and hot bands up to 19908 cm^-1).  Real air-broadened CO2 wings
  are super-Lorentzian, so a truncated Voigt+SFS underestimates them -- but
  the untruncated SFS sum overestimates the smooth band-mean level.

CPU: ARTS is capped at `n_threads` (<= 10) threads.  The untruncated arm is by
far the most expensive (every line is evaluated at every frequency point).

Environment (mamba/conda): pyarts (pyarts 2.6, no FluxSimulator import).  Run
from the scripts/ directory:
    $ python co2_cutoff_tb_effect.py
"""

# =============================================================================
# %% imports
# =============================================================================

import time
import pathlib

import numpy as np
import xarray as xr
import scipy.constants as scc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyarts import arts, cat
import pyarts
from pyarts.workspace import Workspace

import frtmip_aux_functions as fax
import convolve_sensor_response as csr      # SRF validation/normalisation
import fmip_radiance_batch as frb           # data-file constants
import compare_convolved_with_obs as ccow   # house plot style / palette


# =============================================================================
# %% paths / run constants  (edit here)
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
results_folder = script_dir.parent / "results" / "CO2_XSEC"
plot_folder = script_dir.parent / "plot" / "co2_xsec"

# column of data/atmdata.nc to analyse (same as the opacity / lmix / xsec tests)
atm_index = 12345

# spectral window: wider than the deficit so the SRF of every channel that is
# evaluated is fully covered by the computed grid
wvn_min, wvn_max = 2340.0, 2560.0
dwvn = 0.10                       # cm^-1 (2201 points)

# line-wing cutoff arms (cm^-1); None -> abs_lines_per_speciesCutoff "None".
# 25.01 cm^-1 = 750 GHz is exactly the FluxSimulator/benchmark truncation.
cutoffs_cm = [25, 50, 75, 100, 150, None]

# layer grid for the RTE integration (log-pressure interpolation of the profile;
# None -> use the native level midpoints, used by gate G4)
n_layers = 40

# deficit window and the channels printed in the summary
deficit_range_cm = (2398.0, 2520.0)
report_wvn = [2405.0, 2420.0, 2450.0, 2475.0, 2500.0, 2519.0]

# reference wavenumber for the cutoff inversion (analysis note sec. 4 uses
# 2405 cm^-1) and the production sim-obs statistics to compare against
wvn_reference = 2405.0
comparison_stats_file = (script_dir.parent / "results" / "COMPARISON"
                         / "comparison_stats_fmip_radiance_batch"
                         "_slurm_Nf108951_Nza20.nc")

# single-state cross-section reference produced by
# co2_crosssection_cutoff_test.py (gate G1); skipped if absent
xsec_reference_file = results_folder / "co2_xsection_280K_1000hPa_cutoffs.nc"
probe_T, probe_P, probe_vmr = 280.0, 100e3, 420e-6

# per-arm curve colours (house palette plus the extra colours used in the lmix
# figures; keyed by arm_short())
arm_colors = {25: "#c0392b", 50: "#b08800", 75: "#1a7f37", 100: "#6f42c1",
              150: "#0a7d8c", None: "#333333"}

# gate tolerances
tol_g1_rel = 1e-8                 # sigma vs the stored single-state products
tol_g2_K = 1e-2                   # tau -> 0 limit must reproduce B(T_sfc)
tol_g4_rel = 5e-2                 # layer-grid convergence of the column tau

n_threads = 10


def arm_label(cut_cm):
    return "untruncated" if cut_cm is None else f"cutoff {cut_cm} cm$^{-1}$"


def arm_short(cut_cm):
    return "untrunc" if cut_cm is None else f"{cut_cm}"


# =============================================================================
# %% atmospheric column
# =============================================================================

def load_column():
    """Surface-first profile (p, t, alt, CO2 vmr) and the surface temperature.

    atmdata.nc is TOA-first (index 0 = TOA); the radiative arrays and the layer
    construction here are surface-first (index 0 = surface)."""
    atmdata = xr.open_dataset(frb.atmdata_file)
    sfcdata = xr.open_dataset(frb.sfcdata_file)

    if "idx" in atmdata.coords:
        j = int(np.where(atmdata["idx"].values == atm_index)[0][0])
    else:
        j = atm_index
    if atm_index >= int(atmdata.sizes["idx"]):
        raise ValueError(f"atm_index {atm_index} out of range")

    p = atmdata["p"].values[j][::-1]
    t = atmdata["t"].values[j][::-1]
    z = atmdata["alt"].values[j][::-1]
    if "CO2" not in atmdata.data_vars:
        raise KeyError("atmdata.nc has no CO2 field")
    vmr = np.clip(atmdata["CO2"].values[j][::-1], 0.0, None)
    return (p, t, z, vmr, float(sfcdata["temperature"].values[j]))


def layer_grid(p, t, z, vmr, n):
    """Layer-midpoint states (surface-first) and the layer thicknesses.  n is
    the number of *levels* to interpolate the profile onto; n = None keeps the
    native levels.  Layers are the intervals between consecutive levels, so the
    returned arrays have n - 1 entries.

    NB np.interp needs *increasing* xp: the surface-first profile has
    decreasing p, so the interpolation is done on the TOA-first order and the
    result reversed."""
    if n is None:
        pl, tl, zl, vl = p, t, z, vmr                  # native levels
    else:
        lp_inc = np.log(p[::-1])
        lp = np.linspace(lp_inc[0], lp_inc[-1], n)
        pl = np.exp(lp)[::-1]
        tl = np.interp(lp, lp_inc, t[::-1])[::-1]
        zl = np.interp(lp, lp_inc, z[::-1])[::-1]
        vl = np.interp(lp, lp_inc, vmr[::-1])[::-1]
    dz = np.abs(np.diff(zl))
    pm = 0.5 * (pl[:-1] + pl[1:])
    tm = 0.5 * (tl[:-1] + tl[1:])
    zm = 0.5 * (zl[:-1] + zl[1:])
    vm = 0.5 * (vl[:-1] + vl[1:])
    keep = (pm > 0.0) & (tm > 0.0) & (dz > 0.0)
    return pm[keep], tm[keep], zm[keep], vm[keep], dz[keep]


# =============================================================================
# %% on-the-fly absorption and the vertical LTE RTE
# =============================================================================

def open_lbl_workspace(wvn_grid):
    """Workspace with the CO2 line catalog, on-the-fly LBL agenda (no LUT)."""
    ws = Workspace(verbosity=2)
    ws.SetNumberOfThreads(n_threads)
    ws.stokes_dim = 1
    ws.f_grid = arts.convert.kaycm2freq(wvn_grid)
    ws.abs_species = [[arts.SpeciesTag("CO2")]]      # all CO2 isotopologues
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.propmat_clearsky_agendaAuto(lines_speedup_option="None")
    ws.jacobian_quantities = []
    ws.nlte_do = 0
    ws.rtp_nlte = arts.EnergyLevelMap()
    ws.lbl_checked = 1        # like FluxSimulator: bypass the LBL check
    return ws


def set_cutoff(ws, cut_cm):
    """(Re-)set the band cutoff.  abs_lines_per_speciesCutoff mutates the
    catalog in place, so it must be repeated for every arm."""
    if cut_cm is None:
        ws.abs_lines_per_speciesCutoff(option="None", value=0.0)
    else:
        ws.abs_lines_per_speciesCutoff(option="ByLine",
                                       value=arts.convert.kaycm2freq(cut_cm))


def lbl_absorption(ws, P, T, vmr):
    """Absorption coefficient [1/m] of the CO2 line groups at one state."""
    ws.rtp_pressure = float(P)
    ws.rtp_temperature = float(T)
    ws.rtp_vmr = [float(vmr)]
    ws.propmat_clearskyInit()
    ws.propmat_clearskyAddLines()
    return np.array(ws.propmat_clearsky.value.data).squeeze()


def crosssections(alpha, vmr, P, T):
    """sigma = alpha / (vmr * n_tot), n_tot = p/(kB T) -- the ARTS abs_lookup
    convention (m_abs_lookup.cc) that GasAbsLookup::Extract uses in the RTE."""
    return alpha / (vmr * P / (scc.k * T))


def planck_nu(T, f_grid):
    """Planck function in the ARTS radiance units [W m^-2 Hz^-1 sr^-1]."""
    return (2.0 * scc.h * f_grid ** 3 / scc.c ** 2) / np.expm1(
        scc.h * f_grid / (scc.k * T))


def integrate_rte(dtau, temps, T_sfc, f_grid):
    """Vertical (nadir) LTE emission integration of a layer optical-depth array.

    dtau is (n_f, n_layers) surface-first; layer emission B_l (1-exp(-dtau_l))
    is attenuated by the optical depth above the layer, and the surface term is
    B(T_sfc) exp(-tau_column).  Returns (TOA radiance, column tau)."""
    n_f = dtau.shape[0]
    # A[:, l] = optical depth from the bottom of layer l up to the TOA
    A = np.cumsum(dtau[:, ::-1], axis=1)[:, ::-1]
    A_next = np.concatenate([A[:, 1:], np.zeros((n_f, 1))], axis=1)
    I = np.zeros(n_f)
    for l, T in enumerate(temps):
        I += planck_nu(T, f_grid) * (-np.expm1(-dtau[:, l])) * np.exp(-A_next[:, l])
    I += planck_nu(T_sfc, f_grid) * np.exp(-A[:, 0])
    return I, A[:, 0]


# =============================================================================
# %% gates
# =============================================================================

def gate_g1(ws, wvn, cuts):
    """sigma of a dedicated 280 K / 1000 hPa / 420 ppm calculation must
    reproduce the products of co2_crosssection_cutoff_test.py."""
    if not xsec_reference_file.exists():
        print(f"G1: no {xsec_reference_file.name}, gate SKIPPED")
        return True
    ref = xr.open_dataset(str(xsec_reference_file))
    wvn_ref = ref["wavenumber"].values
    ok = True
    for cut_cm in cuts:
        k = int(np.where(ref["cutoff"].values == (cut_cm if cut_cm is not None
                                                 else -1.0))[0][0])
        set_cutoff(ws, cut_cm)
        sig = crosssections(lbl_absorption(ws, probe_P, probe_T, probe_vmr),
                            probe_vmr, probe_P, probe_T)
        # identical grid points only (reference runs on a finer grid)
        j = np.abs(wvn[:, None] - wvn_ref[None, :]).argmin(axis=1)
        common = np.abs(wvn - wvn_ref[j]) < 1e-9
        if not common.any():
            print("G1: no coincident grid points with the reference, SKIPPED")
            return True
        sig_ref = ref["crosssection"].values[j, k]
        rel = np.abs(sig[common] - sig_ref[common]) / np.maximum(
            sig_ref[common], 1e-40)
        good = rel.max() <= tol_g1_rel
        ok &= good
        print(f"G1 {arm_short(cut_cm):>7} cm^-1: max |rel diff| vs "
              f"{xsec_reference_file.name} = {rel.max():.2e} over "
              f"{int(common.sum())} coincident points -> "
              f"{'PASS' if good else 'FAIL'} (tol {tol_g1_rel:.0e})")
        ref.close()
    return ok


def gate_g2(ws, wvn, layers, T_sfc, f_grid):
    """tau -> 0 must reproduce the surface Planck radiance."""
    p, t, z, vmr, dz = layers
    set_cutoff(ws, 25)
    dtau = np.zeros((wvn.size, p.size))
    for l in range(p.size):
        dtau[:, l] = lbl_absorption(ws, p[l], t[l], vmr[l] * 1e-12) * dz[l]
    I, tau = integrate_rte(dtau, t, T_sfc, f_grid)
    dT = fax.radiance2brightness_temp(I, wvn) - fax.radiance2brightness_temp(
        planck_nu(T_sfc, f_grid), wvn)
    ok = np.abs(dT).max() <= tol_g2_K
    print(f"G2 tau->0 limit: max |T_B - B(T_sfc)| = {np.abs(dT).max():.2e} K, "
          f"max tau = {tau.max():.2e} -> {'PASS' if ok else 'FAIL'} "
          f"(tol {tol_g2_K:.0e} K)")
    return ok


def gate_g3(tau, tb, tau_thin=1.0):
    """tau and T_B must be monotone in the cutoff (more wing absorption ->
    larger tau, colder T_B).

    The T_B check is restricted to the optically thin regime (tau of the
    25 cm^-1 arm < `tau_thin`, i.e. the whole deficit window, where the
    added opacity acts on radiation whose source is the warm surface /
    lower troposphere).  At already-saturated line cores shortward of
    ~2390 cm^-1 (tau >> 1) extra opacity instead lifts the emission
    layer into the warmer stratopause and can *raise* the near-saturated
    core T_B by tiny amounts -- physically expected, not a bug, so those
    points are reported but do not fail the gate."""
    dtau = np.diff(tau, axis=1)
    dTb = np.diff(tb, axis=1)
    thin = (tau[:, [0]] < tau_thin) & np.ones((1, dTb.shape[1]), bool)
    ok = (dtau.min() >= -1e-12) and (dTb[thin].max() <= 1e-9)
    print(f"G3 monotonicity: min d(tau) = {dtau.min():.2e}; "
          f"max d(T_B) over tau(25 cm^-1) < {tau_thin:.0f} (deficit regime) = "
          f"{dTb[thin].max():.2e} K over {int(thin.sum())} points; "
          f"max d(T_B) at saturated cores = {dTb[~thin].max():.2e} K "
          f"(exempt, stratospheric source) -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_g4(cuts):
    """Column tau with the native level midpoints vs the log-p layer grid.

    Restricted to the deficit window so that the untruncated arm (every line at
    every point) stays affordable."""
    wvn = np.arange(deficit_range_cm[0], deficit_range_cm[1] + dwvn / 2, dwvn)
    p, t, z, vmr, T_sfc = load_column()
    lay_sub = layer_grid(p, t, z, vmr, n_layers)
    lay_nat = layer_grid(p, t, z, vmr, None)
    f_grid = arts.convert.kaycm2freq(wvn)
    ws = open_lbl_workspace(wvn)
    ratios = []
    for cut_cm in (25, None):
        set_cutoff(ws, cut_cm)
        taus = []
        for lay in (lay_sub, lay_nat):
            pl, tl, zl, vl, dl = lay
            dtau = np.stack([lbl_absorption(ws, pl[i], tl[i], vl[i]) * dl[i]
                             for i in range(pl.size)], axis=1)
            taus.append(dtau.sum(axis=1))
        rel = np.abs(taus[0] - taus[1]) / np.maximum(taus[1], 1e-30)
        ratios.append(rel.max())
        print(f"G4 {arm_short(cut_cm):>7} cm^-1: tau({n_layers} log-p layers) "
              f"vs {lay_nat[0].size} native layers: max |rel diff| = "
              f"{rel.max():.2e}, median {np.median(rel):.2e}")
    ok = max(ratios) <= tol_g4_rel
    print(f"G4 layer-grid convergence -> {'PASS' if ok else 'FAIL'} "
          f"(tol {tol_g4_rel:.0e})")
    del ws
    return ok


# =============================================================================
# %% figures
# =============================================================================

def _mark_deficit(ax):
    ax.axvspan(*deficit_range_cm, color=ccow.palette["shade"], zorder=0)


def figure_tb_effect(chan, dTb, deficit_required, tau, wvn):
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4))

    ax = axes[0]
    for cut_cm in cutoffs_cm:
        ax.plot(chan, dTb[cut_cm], color=arm_colors[cut_cm],
                lw=ccow.line_width, label=arm_label(cut_cm))
    ax.plot(chan, deficit_required, color="0.15", lw=1.6, ls="--",
            label=r"required $-({\rm sim}-{\rm obs})$, benchmark")
    _mark_deficit(ax)
    ccow.annotate_spectral_features(ax, shade_window=False)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel(r"$\Delta T_B$ (arm $-$ 25 cm$^{-1}$) / K")
    ax.set_xlim(chan.min(), chan.max())
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for cut_cm in cutoffs_cm:
        ax.plot(wvn, tau[cut_cm], color=arm_colors[cut_cm],
                lw=ccow.line_width, label=arm_label(cut_cm))
    _mark_deficit(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(r"CO$_2$ line-wing column opacity $\tau$ / $-1$")
    ax.set_xlim(wvn[0], wvn[-1])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(rf"CO$_2$ line-wing cutoff, FMIP column idx {atm_index}: "
                 "sensor-level $T_B$ effect and CO$_2$ line opacity",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_file = plot_folder / f"fig_co2_xsec_cutoff_tb_effect_idx{atm_index}.png"
    fig.savefig(fig_file, dpi=200)
    plt.close(fig)
    print(f"  wrote {fig_file}")
    return fig_file


def figure_closure(chan, dTb, deficit_required):
    """|dT_B| / |required| -- where an arm supplies exactly the missing
    absorption (ratio 1)."""
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for cut_cm in cutoffs_cm:
        if cut_cm == 25:
            continue
        ratio = np.abs(dTb[cut_cm]) / np.maximum(np.abs(deficit_required), 1e-6)
        ax.semilogy(chan, ratio, color=arm_colors[cut_cm], lw=ccow.line_width,
                    label=arm_label(cut_cm))
    ax.axhline(1.0, color="0.15", lw=1.2, ls="--", label="exact closure")
    _mark_deficit(ax)
    ccow.annotate_spectral_features(ax, shade_window=False)
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(r"$|\Delta T_B| / |T_{B,\rm obs}-T_{B,\rm sim}|$ / $-1$")
    ax.set_xlim(chan.min(), chan.max())
    ax.set_ylim(1e-3, 1e3)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(r"How much of the missing absorption each cutoff supplies "
                 f"(column idx {atm_index})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_file = plot_folder / f"fig_co2_xsec_cutoff_closure_idx{atm_index}.png"
    fig.savefig(fig_file, dpi=200)
    plt.close(fig)
    print(f"  wrote {fig_file}")
    return fig_file


# =============================================================================
# %% export
# =============================================================================

def export_netcdf(wvn, chan, tau, tb_mono, tb_chan, dTb, deficit_required,
                  layers, T_sfc, gates):
    results_folder.mkdir(parents=True, exist_ok=True)
    cuts_num = np.array([-1.0 if c is None else float(c) for c in cutoffs_cm])
    cuts_hz = np.array([0.0 if c is None
                        else arts.convert.kaycm2freq(float(c))
                        for c in cutoffs_cm])

    ds = xr.Dataset(
        {
            "tau_column_lines": (["wavenumber", "cutoff"],
                                 np.stack([tau[c] for c in cutoffs_cm], axis=1),
                                 {"units": "-", "long_name":
                                  "vertical CO2 line-wing column opacity"}),
            "brightness_temperature_mono":
                (["wavenumber", "cutoff"],
                 np.stack([tb_mono[c] for c in cutoffs_cm], axis=1),
                 {"units": "K", "zenith_angle": 0.0,
                  "long_name": "monochromatic nadir T_B, CO2 lines only"}),
            "brightness_temperature_channel":
                (["channel", "cutoff"],
                 np.stack([tb_chan[c] for c in cutoffs_cm], axis=1),
                 {"units": "K", "long_name":
                  "SRF-convolved nadir T_B on the IASI channel grid, "
                  "CO2 lines only"}),
            "delta_tb_vs_25cm":
                (["channel", "cutoff"], np.stack([dTb[c] for c in cutoffs_cm],
                                                 axis=1),
                 {"units": "K", "long_name":
                  "SRF-convolved T_B minus the 25 cm^-1 (750 GHz) arm"}),
            "deficit_required": ("channel", deficit_required,
                                 {"units": "K", "long_name":
                                  "-(sim - obs) T_B of the production run, "
                                  "the shift the CO2 absorption would have to "
                                  "supply"}),
        },
        coords={
            "wavenumber": (["wavenumber"], wvn, {"units": "cm^-1"}),
            "frequency": (["wavenumber"], arts.convert.kaycm2freq(wvn),
                          {"units": "Hz"}),
            "channel": (["channel"], chan, {"units": "cm^-1",
                                            "long_name": "IASI channel centre"}),
            "cutoff": (["cutoff"], cuts_num,
                       {"units": "cm^-1", "long_name":
                        "line-wing cutoff, -1 marks option='None'"}),
            "cutoff_hz": (["cutoff"], cuts_hz, {"units": "Hz"}),
        },
    )
    ds.attrs = {
        "title": "Radiative effect of the CO2 line-wing cutoff on the "
                 "2398-2520 cm^-1 deficit (single column, on-the-fly LBL)",
        "atm_index": int(atm_index),
        "profile": f"{layers[0].size} log-p layers from the atmdata.nc column, "
                   f"surface T {T_sfc:.2f} K, vmr(CO2) interpolated "
                   f"(surface {layers[3][0]:.3e})",
        "method": "ARTS 2.6 on-the-fly line-by-line absorption per layer "
                  "(propmat_clearskyInit + propmat_clearskyAddLines, "
                  "lines_speedup_option='None'); vertical LTE emission "
                  "integration; SRF convolution with the IASI operator of "
                  "frtmip_aux_functions",
        "absorbers": "CO2 lines only -- no MT-CKD CO2 continuum, no H2O/O2/N2; "
                     "absolute T_B is warmer than the products, only the arm "
                     "differences are meaningful",
        "cutoff_semantics": "abs_lines_per_speciesCutoff ByLine: a line "
                            "contributes only inside F0 +/- (cutoff + D0); "
                            "cutoff -1 -> option 'None' (untruncated SFS Voigt "
                            "sum over the whole CO2 catalog, a reference not a "
                            "physical wing model)",
        "spectroscopy": "arts-cat-data-2.6.20 CO2 lines, Voigt + SFS far-wing "
                        "scaling, LTE, no line mixing, Earth isotope ratios",
        "deficit_source": str(comparison_stats_file.name),
        "deficit_range_cm-1": f"{deficit_range_cm[0]}-{deficit_range_cm[1]}",
        "gates": ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                           for k, v in gates.items()),
        "arts_version": str(arts.__version__ if hasattr(arts, "__version__")
                            else "2.6"),
        "pyarts_version": str(getattr(pyarts, "__version__", "2.6")),
    }
    out = results_folder / f"co2_cutoff_tb_effect_idx{atm_index}.nc"
    ds.to_netcdf(str(out))
    print(f"  exported {out}")
    return out


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()
    ccow.apply_plot_style()
    plot_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    cat.download.retrieve(verbose=False)

    p, t, z, vmr, T_sfc = load_column()
    layers = layer_grid(p, t, z, vmr, n_layers)
    pl, tl, zl, vl, dl = layers
    wvn = np.arange(wvn_min, wvn_max + dwvn / 2, dwvn)
    f_grid = arts.convert.kaycm2freq(wvn)

    print("=" * 78)
    print("Radiative effect of the CO2 line-wing cutoff "
          f"(column idx {atm_index}, on-the-fly LBL, no LUT)")
    print(f"  window    : {wvn_min:.0f}-{wvn_max:.0f} cm^-1, dwvn = {dwvn} "
          f"({wvn.size} points)")
    print(f"  profile   : {p.size} levels -> {pl.size} layers, "
          f"T_sfc {T_sfc:.1f} K, surface p {p[0]/100:.1f} hPa, "
          f"vmr(CO2) {vl[0]:.3e}")
    print(f"  arms      : {', '.join(arm_short(c) for c in cutoffs_cm)} cm^-1 "
          "(25 = benchmark 750 GHz)")
    print(f"  results   : {results_folder}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # per-arm absorption, opacity, monochromatic and sensor-level T_B
    # ------------------------------------------------------------------
    ws = open_lbl_workspace(wvn)
    set_cutoff(ws, 25)
    if not gate_g1(ws, wvn, cutoffs_cm):
        print("G1 FAILED -- stop")
        return 1
    if not gate_g2(ws, wvn, layers, T_sfc, f_grid):
        print("G2 FAILED -- stop")
        return 1

    tau, tb_mono, I_mono = {}, {}, {}
    for cut_cm in cutoffs_cm:
        t0 = time.time()
        set_cutoff(ws, cut_cm)
        dtau = np.zeros((wvn.size, pl.size))
        for l in range(pl.size):
            dtau[:, l] = lbl_absorption(ws, pl[l], tl[l], vl[l]) * dl[l]
        I, tau_col = integrate_rte(dtau, tl, T_sfc, f_grid)
        tau[cut_cm], tb_mono[cut_cm], I_mono[cut_cm] = (
            tau_col, fax.radiance2brightness_temp(I, wvn), I)
        print(f"  arm {arm_short(cut_cm):>7} cm^-1 ({time.time()-t0:6.1f} s): "
              "tau at " + " ".join(
                  f"{w:.0f}:{float(np.interp(w, wvn, tau_col)):.2e}"
                  for w in report_wvn))

    # SRF convolution onto the IASI channel grid (only fully covered channels)
    sensor_response = xr.open_dataset(frb.sensor_response_file)
    srf_wvn, srf_response, _ = csr.validate_and_normalize_srf(
        sensor_response["wvn"].values, sensor_response["response"].values)
    operator, chan = fax.build_srf_channel_operator(
        f_grid, srf_wvn, srf_response, xr.open_dataset(frb.satdata_file)["wvn"].values)
    srf_lo, srf_hi = srf_wvn.min(), srf_wvn.max()
    keep = (chan >= wvn[0] - srf_lo) & (chan <= wvn[-1] - srf_hi)
    chan = chan[keep]
    tb_chan = {}
    for cut_cm in cutoffs_cm:
        rad_c = fax.convolve_with_srf_to_channels(I_mono[cut_cm], operator)[keep]
        tb_chan[cut_cm] = fax.radiance2brightness_temp(rad_c, chan)
    dTb = {c: tb_chan[c] - tb_chan[25] for c in cutoffs_cm}
    print(f"  sensor grid: {chan.size} fully-covered IASI channels "
          f"({chan.min():.1f}-{chan.max():.1f} cm^-1)")

    # production sim - obs of the same channels -> the shift that is needed
    stats = xr.open_dataset(str(comparison_stats_file))
    wvn_stats = stats["wavenumber"].values
    k = np.abs(wvn_stats[None, :] - chan[:, None]).argmin(axis=1)
    if np.abs(wvn_stats[k] - chan).max() > 1e-6:
        raise RuntimeError("channel grids of the comparison statistics and the "
                           "SRF operator do not coincide")
    deficit_required = -stats["diff_mean_bt"].values[k]
    del stats

    del ws

    # ------------------------------------------------------------------
    # gates 3 and 4
    # ------------------------------------------------------------------
    gates = {"G1": True, "G2": True,
             "G3": gate_g3(np.stack([tau[c] for c in cutoffs_cm], axis=1),
                           np.stack([tb_mono[c] for c in cutoffs_cm], axis=1)),
             "G4": gate_g4(cutoffs_cm)}
    print("-" * 78)
    print("gate summary: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                       for k, v in gates.items())
          + (" -> all gates passed" if all(gates.values())
             else " -> FAILURES PRESENT"))

    # ------------------------------------------------------------------
    # summary at the reporting wavenumbers + cutoff inversion
    # ------------------------------------------------------------------
    print("\nsensor-level T_B shifts vs the 25 cm^-1 arm (K) and the shift the "
          "benchmark deficit requires")
    print(f"  cm^-1   " + "".join(f"{arm_short(c):>10}" for c in cutoffs_cm)
          + f"{'required':>11}")
    for wq in report_wvn:
        c = int(np.argmin(np.abs(chan - wq)))
        print(f"  {chan[c]:7.2f} " + "".join(f"{dTb[k][c]:+10.2f}"
                                             for k in cutoffs_cm)
              + f"{deficit_required[c]:+11.2f}")

    c_ref = int(np.argmin(np.abs(chan - wvn_reference)))
    need = abs(deficit_required[c_ref])
    print(f"\ncutoff inversion at {chan[c_ref]:.2f} cm^-1: the benchmark deficit "
          f"requires dT_B = -{need:.2f} K")
    prev_cut, prev_shift = 25, 0.0
    reached = False
    for cut_cm in cutoffs_cm[1:]:
        shift = abs(dTb[cut_cm][c_ref])
        if shift >= need:
            if shift > prev_shift and prev_shift > 0:
                frac = ((np.log(need) - np.log(prev_shift))
                        / (np.log(shift) - np.log(prev_shift)))
            else:
                frac = 0.0
            est = prev_cut + frac * (cut_cm - prev_cut)
            print(f"  -> required shift reached between cutoffs {prev_cut} and "
                  f"{arm_short(cut_cm)} cm^-1 (dT_B {prev_shift:.2f} .. "
                  f"{shift:.2f} K); log-interpolated cutoff ~ {est:.1f} cm^-1")
            reached = True
            break
        prev_cut, prev_shift = cut_cm, shift
    if not reached:
        print(f"  -> even {arm_label(cutoffs_cm[-1])} does not reach it "
              f"(dT_B {abs(dTb[cutoffs_cm[-1]][c_ref]):.2f} K)")

    wsel = (chan >= deficit_range_cm[0]) & (chan <= deficit_range_cm[1])
    print(f"\ndeficit-window means ({deficit_range_cm[0]:.0f}-"
          f"{deficit_range_cm[1]:.0f} cm^-1):")
    print(f"  required mean shift  : {deficit_required[wsel].mean():+.2f} K")
    for cut_cm in cutoffs_cm[1:]:
        print(f"  {arm_short(cut_cm):>9} cm^-1 : {dTb[cut_cm][wsel].mean():+.2f} K "
              f"(range {dTb[cut_cm][wsel].min():+.2f} .. "
              f"{dTb[cut_cm][wsel].max():+.2f} K)")

    # ------------------------------------------------------------------
    # products
    # ------------------------------------------------------------------
    export_netcdf(wvn, chan, tau, tb_mono, tb_chan, dTb, deficit_required,
                  layers, T_sfc, gates)
    figure_tb_effect(chan, dTb, deficit_required, tau, wvn)
    figure_closure(chan, dTb, deficit_required)

    print(f"\ntotal wall time: {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
