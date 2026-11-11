#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2 absorption cross sections at ONE thermodynamic state as a function of the
line-wing cutoff (on-the-fly line-by-line, no FluxSimulator, no lookup table).

State: T = 280 K, total pressure 1000 hPa, CO2 dilute in air (vmr 420e-6),
spectral region 2350-2550 cm^-1 (a bit wider than the 2398-2520 cm^-1 deficit
window so that the CO2 nu3 band edge at ~2397 cm^-1 is visible).

This is the clean single-state, line-resolved counterpart of

  * plot_nu3_wing_opacity_and_crosssections.py  -- column opacity and tau = 1
    cross sections of *all* species, computed through ARTS `abs_lookupCalc`;
  * probe_co2_wing_cutoff_sensitivity.py        -- cutoff sensitivity of the
    *column* opacity (still `abs_lookup`, coarse grid, whole profile).

Here nothing but a plain line-by-line calculation is used: the cross sections
come from `propmat_clearskyInit` + `propmat_clearskyAddLines` with
`lines_speedup_option = "None"` (no sparse grids, no speedup, no LUT).  The
quantity is the per-molecule cross section of the CO2 *species group*,

    sigma(f) = alpha(f) / ( vmr_CO2 * n_tot ),   n_tot = P / (kB T),

which is exactly the convention ARTS stores in `abs_lookup` and which
GasAbsLookup::Extract multiplies back inside the RTE -- so the numbers here are
directly comparable with the LUT/opacity scripts.  Note that the group cross
section *includes* the terrestrial isotopologue ratios of the catalogue
(CO2-626 0.9842, -627, -628, -636, ...): they are part of a "molecule of
natural CO2", and gate G2 makes this explicit.

Physics / interpretation
------------------------
The radiance benchmark (FluxSimulator default) truncates every line at
750 GHz = 25.01 cm^-1 (`abs_lines_per_speciesCutoff` ByLine, i.e. a line
contributes only inside F0 +/- (cutoff + D0)).  The CO2 nu3 band of the cold
lines ends at ~2397 cm^-1, so with a 25 cm^-1 cutoff the region above ~2422
cm^-1 is fed only by hot lines and by the wings of lines that happen to lie
close by -- everything else is exactly zero.  This script quantifies how much
absorption the cutoff throws away across the deficit window at a fixed state,
which is the pure *truncation* part of the model deficit there.

The line shape is still the ARTS default of the shipped catalogue: Voigt (VP)
with simple frequency scaling of the far wings (SFS) and LTE populations at
T0 = 296 K, without line mixing.  Real air-broadened CO2 wings are known to be
*super*-Lorentzian, so even the untruncated arm of this script underestimates
the true far-wing absorption -- the cutoff test is a lower bound on the
missing opacity, not a complete error budget.  The CO2 MT-CKD continuum
(CO2-CKDMT252) is deliberately EXCLUDED from the cross sections (it is a
continuum model without any cutoff); its size is only measured once for
context, because at 2450 cm^-1 / 1000 hPa / 420 ppm it is of the same order as
the *line-wing* cross section obtained with the production cutoff.

Gates / self-checks (all printed as PASS/FAIL)
---------------------------------------------
G1  two-path consistency: at the same state, a tiny two-species `abs_lookup`
    (H2O + CO2 groups, 9 abs_t_pert books spanning +/-30 K, fine p step, ByLine
    at the 25 cm^-1 cutoff) versus the direct on-the-fly alpha.  This is the
    only place where a lookup table is touched at all.
G2  Doppler-limit normalisation: one strong isolated nu3 line at P = 1 Pa
    compared with the analytic I0 * sqrt(ln2/pi) / gamma_D (CO2 mass,
    gamma_D = F0 * sqrt(2 kB T ln2 / (m c^2))) -- catches unit/scale errors in
    the sigma extraction.
G3  pointwise monotonicity of sigma in the cutoff, plus the catalogue
    inventory around the window (which bands/lines can contribute at all).
G4  spectral resolution: one arm recomputed at dwvn/2.

Environment (mamba/conda): pyarts (pyarts 2.6 + FluxSimulator, but this script
does not import FluxSimulator).  Run from the scripts/ directory:
    $ python co2_crosssection_cutoff_test.py
The ARTS data path must contain lines/ (default ~/.cache/arts/
arts-cat-data-2.6.20).
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

import compare_convolved_with_obs as ccow      # house plot style / palette


# =============================================================================
# %% paths / run constants  (edit here)
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
results_folder = script_dir.parent / "results" / "CO2_XSEC"
plot_folder = script_dir.parent / "plot" / "co2_xsec"

# -- the single thermodynamic state -------------------------------------------
T_state = 280.0                 # K
P_state = 100e3                 # Pa  (= 1000 hPa total pressure)
vmr_co2 = 420e-6                # CO2 dilute in air -> ARTS AIR *bath* = 1 - vmr

# -- spectrum ------------------------------------------------------------------
wvn_min, wvn_max = 2350.0, 2550.0   # cm^-1 (wider than the deficit window:
                                    # the nu3 band edge ~2397 cm^-1 stays visible)
dwvn = 0.02                     # cm^-1 production spacing; Lorentz HWHG at
                                # 1000 hPa is ~0.069 cm^-1 -> cores resolved
deficit_range_cm = (2398.0, 2520.0)   # the FORUM/FMIP deficit window

# -- line-wing cutoff arms (cm^-1); None -> option="None" (untruncated) --------
# 25.01 cm^-1 = 750 GHz is exactly the FluxSimulator/benchmark cutoff; the
# 25.0 cm^-1 arm is that arm to within 0.04 %.
cutoffs_cm = [25.0, 50.0, 75.0, 100.0, 150.0, None]
reference_cutoff_cm = 25.0      # arm all ratios are quoted against
untrunc_marker_cm = -1.0        # value stored in the NetCDF for the None arm

# wavenumbers of the printed table columns (cm^-1)
report_wvn = [2398.0, 2405.0, 2420.0, 2450.0, 2475.0, 2500.0, 2519.0]

# -- ARTS workspace -------------------------------------------------------------
n_threads = 10                  # hard cap so the machine stays usable
ws_verbosity = 1

# -- gate settings ---------------------------------------------------------------
g1_p_step10 = 0.05              # log10(p/Pa) spacing of the tiny LUT p grid
g1_p_top = 200e2                # Pa, top of the tiny 3-level test profile
g1_t_pert = np.linspace(-30.0, 30.0, 9)   # K, abs_t_pert books (must be >= 2
#                                 for the T interpolation; a single book makes
#                                 propmat_clearskyAddFromLookup bail out)
g1_tol_rel = 0.02               # gate tolerance (planning run: ratio == 1.0)

g2_pressure = 1.0               # Pa -- Doppler-limit test pressure
g2_pressure_pure = 1e-4         # Pa, extra pure-Doppler arm for context
g2_grid_halfwidth_GD = 8.0      # tiny f_grid: +/- 8 Doppler 1/e widths
g2_grid_points = 33             # odd -> the line centre is a grid point
g2_min_line_gap_cm = 5e-4       # isolation criterion for the test line
g2_tol_rel = 0.02

g4_refine = 2                   # G4 recomputes the reference arm at dwvn/g4_refine
g4_tol_rel = 0.05

# -- plotting -------------------------------------------------------------------
xsec_floor = 1e-34              # log-axis floor (m^2 molecule^-1)
cutoff_colors = {25.0: ccow.palette["sim"],
                 50.0: "#b08800",
                 75.0: "#1a7f37",
                 100.0: "#8250df",
                 150.0: "#0b7285",
                 None: "0.25"}


# =============================================================================
# %% ARTS on-the-fly LBL machinery
# =============================================================================

def open_lbl_workspace(wvn_grid):
    """ARTS workspace with the CO2 line catalogue and the state of the run.

    API gotchas (verified against pyarts 2.6.20):
      * there is NO `propmat_clearskyCalc` and no `ws.Execute(...)` in pyarts
        2.6 -- `propmat_clearskyInit()` + `propmat_clearskyAddLines()` are the
        agenda steps that read rtp_* and write propmat_clearsky; the agenda
        itself is only needed to be *checked*, which agendaAuto does;
      * `lines_speedup_option = "None"` switches off the sparse-grid LBL
        speedups, so every line is evaluated at every grid point;
      * `rtp_nlte` must be the ARTS type `EnergyLevelMap()`, not a python {};
      * `lbl_checked` must come from `lbl_checkedCalc()`, not from setting it
        to 1 by hand;
      * `propmat_clearsky.value.data` has shape (1, 1, n_f, 1).
    """
    ws = Workspace(verbosity=ws_verbosity)
    ws.SetNumberOfThreads(n_threads)
    ws.stokes_dim = 1
    ws.f_grid = arts.convert.kaycm2freq(np.asarray(wvn_grid, dtype=float))

    # one species GROUP -> every CO2 isotopologue band of the catalogue; the
    # lines carry SELF + AIR POLY broadeners and ARTS resolves AIR as the bath
    # = 1 - sum(matched broadener VMRs), so no N2/O2 groups are needed
    ws.abs_species = [[arts.SpeciesTag("CO2")]]
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    ws.propmat_clearsky_agendaAuto(lines_speedup_option="None")
    ws.jacobian_quantities = []
    ws.nlte_do = 0
    ws.rtp_pressure = P_state
    ws.rtp_temperature = T_state
    ws.rtp_vmr = [vmr_co2]
    ws.rtp_nlte = arts.EnergyLevelMap()
    ws.lbl_checkedCalc()
    return ws


def set_cutoff(ws, cut_cm):
    """Re-set the line-wing cutoff of every band (the call mutates the
    catalogue in place; the cutoff is a band attribute `cutoff`/`cutofffreq`).

    ByLine: a line contributes only inside F0 +/- (cutofffreq + D0).  With
    option "None" the Lorentz wings are evaluated over the whole grid.
    """
    if cut_cm is None:
        ws.abs_lines_per_speciesCutoff(option="None", value=0.0)
    else:
        ws.abs_lines_per_speciesCutoff(option="ByLine",
                                       value=arts.convert.kaycm2freq(float(cut_cm)))


def lbl_absorption(ws, cut_cm, wvn_grid=None):
    """On-the-fly LBL absorption coefficient alpha (1/m) on `wvn_grid`."""
    if wvn_grid is not None:
        ws.f_grid = arts.convert.kaycm2freq(np.asarray(wvn_grid, dtype=float))
        # f_grid must be strictly increasing whenever a cutoff is in place
        ws.lbl_checkedCalc()
    set_cutoff(ws, cut_cm)
    ws.propmat_clearskyInit()
    ws.propmat_clearskyAddLines()
    return np.array(ws.propmat_clearsky.value.data).squeeze()


def crosssections(alpha):
    """alpha (1/m) -> sigma (m^2 molecule^-1) of the CO2 species group."""
    n_tot = P_state / (scc.k * T_state)
    return np.asarray(alpha, dtype=float) / (vmr_co2 * n_tot)


def arm_label(cut_cm):
    """Short arm name for tables and tick labels."""
    return "untrunc" if cut_cm is None else f"{cut_cm:.0f}"


def arm_legend(cut_cm):
    """Descriptive arm name for figure legends."""
    if cut_cm is None:
        return "untruncated (no cutoff)"
    if cut_cm == reference_cutoff_cm:
        return f"cutoff {cut_cm:.0f} cm$^{{-1}}$ (benchmark, 750 GHz)"
    return f"cutoff {cut_cm:.0f} cm$^{{-1}}$"


# =============================================================================
# %% catalogue inventory (line-resolved bookkeeping around the window)
# =============================================================================

def catalog_inventory(ws, wvn_lo=2350.0, wvn_hi=2600.0):
    """Per-line catalogue data of every CO2 band, plus the summary numbers.

    Returns the sorted line arrays (F0, wavenumber, I0, isotopologue name /
    mass / terrestrial ratio, band T0, band id), the number of bands and lines
    that have at least one line in [wvn_lo, wvn_hi] cm^-1, and the edges of
    the band that carries the strongest line of the region (the CO2-626 nu3
    fundamental).
    """
    c = 100.0 * scc.c                      # cm^-1 <-> Hz
    ratios = ws.isotopologue_ratios.value.data

    def isotopologue(index):
        """(name, mass g/mol, terrestrial ratio) of an isotopologue index."""
        rec = arts.SpeciesIsotopeRecord(index)
        return (str(rec.name), float(rec.mass), float(ratios[index]))

    f0, i0, isoi, t0b, bid = [], [], [], [], []
    n_bands, n_lines = 0, 0
    bands_in_range = 0
    lines_in_range = 0

    for grp in ws.abs_lines_per_species.value:
        for band in grp:
            qi = band.quantumidentity
            fl = np.array([ln.F0 for ln in band.lines], dtype=float)
            il = np.array([ln.I0 for ln in band.lines], dtype=float)
            n_bands += 1
            n_lines += fl.size
            inside = (fl > wvn_lo * c) & (fl < wvn_hi * c)
            if inside.any():
                bands_in_range += 1
                lines_in_range += int(inside.sum())

            f0.append(fl)
            i0.append(il)
            isoi.append(np.full(fl.size, int(qi.isotopologue_index), dtype=int))
            t0b.append(np.full(fl.size, float(band.T0), dtype=float))
            bid.append(np.full(fl.size, n_bands - 1, dtype=int))

    f0 = np.concatenate(f0)
    i0 = np.concatenate(i0)
    isoi = np.concatenate(isoi)
    t0b = np.concatenate(t0b)
    bid = np.concatenate(bid)

    # isotopologue bookkeeping: index -> (name, mass, ratio)
    uniq = np.unique(isoi)
    info = {int(u): isotopologue(int(u)) for u in uniq}
    name_of = {int(u): info[int(u)][0] for u in uniq}
    iso_names = np.array([name_of[int(i)] for i in isoi], dtype=object)
    masses = np.array([info[int(i)][1] for i in isoi], dtype=float)
    iso_rat = np.array([info[int(i)][2] for i in isoi], dtype=float)

    k = np.argsort(f0)
    f0, i0 = f0[k], i0[k]
    bid = bid[k]
    iso_names, masses, iso_rat, t0b = iso_names[k], masses[k], iso_rat[k], t0b[k]
    wvn = f0 / c

    in_range = (wvn > wvn_lo) & (wvn < wvn_hi)
    j_max_i0 = int(np.argmax(np.where(in_range, i0, -1.0)))

    # the band that carries the strongest line of the region is the CO2-626
    # nu3 fundamental: its upper frequency edge is the band head beyond which
    # only hot bands and line wings can contribute
    head_band = int(bid[j_max_i0])
    sel = bid == head_band

    return {"f0": f0, "wvn": wvn, "i0": i0,
            "iso": iso_names, "mass": masses, "iso_ratio": iso_rat,
            "band_T0": t0b,
            "n_bands": n_bands, "n_lines": n_lines,
            "bands_in_range": bands_in_range, "lines_in_range": lines_in_range,
            "range_cm": (wvn_lo, wvn_hi),
            "max_i0_wvn": float(wvn[j_max_i0]),
            "max_i0": float(i0[j_max_i0]),
            "band_iso": str(iso_names[j_max_i0]),
            "band_n_lines": int(sel.sum()),
            "band_lo_cm": float(wvn[sel].min()),
            "band_head_cm": float(wvn[sel].max()),
            "iso_info": info}


def pick_isolated_line(inv, wvn_lo=2340.0, wvn_hi=2520.0):
    """Strongest catalog line with no neighbour within g2_min_line_gap_cm."""
    wvn, i0, f0 = inv["wvn"], inv["i0"], inv["f0"]
    idx = np.where((wvn > wvn_lo) & (wvn < wvn_hi))[0]
    gap = np.minimum(f0[idx] - f0[idx - 1], f0[idx + 1] - f0[idx]) / (100 * scc.c)
    ok = np.flatnonzero(gap > g2_min_line_gap_cm)
    j = int(idx[ok[np.argmax(i0[idx[ok]])]])
    return {"i": j, "f0": float(f0[j]), "wvn": float(wvn[j]),
            "i0": float(i0[j]), "gap_cm": float(gap[ok[np.argmax(i0[idx[ok]])]]),
            "iso": str(inv["iso"][j]), "mass_gmol": float(inv["mass"][j]),
            "iso_ratio": float(inv["iso_ratio"][j]),
            "T0": float(inv["band_T0"][j])}


# =============================================================================
# %% gates
# =============================================================================

def gate_g1(ws, wvn, sigma_ref):
    """G1: LUT route vs. direct on-the-fly alpha at the same state.

    Deliberately the only use of abs_lookup in this script.  Two things are
    needed to make the comparison exact rather than interpolating:
      * the state pressure must sit on a lookup node -- abs_lookupSetupBatch
        builds abs_p as a log grid starting at max(p) of the batch fields, so a
        test profile with its surface at exactly P_state puts the state on
        node 0 (nearest-node interpolation cannot shift it);
      * the profile temperature is constant, so the node reference temperature
        is T_state and the t_pert book at 0 is hit exactly.
    The H2O group is required (abs_lookupSetupBatch bails out without it) but
    gets an EMPTY band list, so it costs nothing and contributes nothing.
    Note: reading lut.xsec pages by index is NOT safe (nonlinear species
    duplicate pages) -- always go through propmat_clearskyAddFromLookup.
    """
    alpha_expected = sigma_ref * vmr_co2 * P_state / (scc.k * T_state)

    ws_l = ws                                  # reuse the CO2 catalogue
    ws_l.atmosphere_dim = 1
    ws_l.f_grid = arts.convert.kaycm2freq(np.asarray(wvn, dtype=float))

    co2_bands = ws_l.abs_lines_per_species.value[0]
    ws_l.abs_species = [[arts.SpeciesTag("H2O")], [arts.SpeciesTag("CO2")]]
    ws_l.abs_lines_per_species = [arts.ArrayOfAbsorptionLines(), co2_bands]

    # one compact 3-level field: p strictly decreasing, z increasing
    field = arts.GriddedField4()
    p_prof = np.array([P_state, 0.5 * (P_state + g1_p_top), g1_p_top])
    field.set_grid(0, ["T", "z", "abs_species-H2O", "abs_species-CO2"])
    field.set_grid(1, p_prof.astype(np.float64))
    data = np.zeros((4, p_prof.size, 1, 1))
    data[0, :, 0, 0] = T_state
    data[1, :, 0, 0] = [0.0, 5.0e3, 1.2e4]
    data[2, :, 0, 0] = 0.0                      # H2O: no catalog, no effect
    data[3, :, 0, 0] = vmr_co2
    field.data = data
    ws_l.batch_atm_fields_compact = [field]

    ws_l.propmat_clearsky_agendaAuto(lines_speedup_option="None")
    ws_l.lbl_checkedCalc()
    ws_l.abs_lookupSetupBatch(p_step=g1_p_step10)
    ws_l.abs_t_pert = np.asarray(g1_t_pert, dtype=float)
    ws_l.abs_lines_per_speciesCutoff(
        option="ByLine", value=arts.convert.kaycm2freq(reference_cutoff_cm))
    t0 = time.time()
    ws_l.abs_lookupCalc()
    t_lut = time.time() - t0

    ws_l.rtp_pressure = P_state
    ws_l.rtp_temperature = T_state
    ws_l.rtp_vmr = [0.0, vmr_co2]
    ws_l.rtp_nlte = arts.EnergyLevelMap()
    ws_l.propmat_clearskyInit()
    ws_l.propmat_clearskyAddFromLookup()
    alpha_lut = np.array(ws_l.propmat_clearsky.value.data).squeeze()

    rel = np.abs(alpha_lut - alpha_expected) / np.maximum(alpha_expected, 1e-300)
    ok = bool(rel.max() < g1_tol_rel)
    print(f"G1 two-path (abs_lookup vs on-the-fly, {wvn.size} points, "
          f"LUT {len(np.array(ws_l.abs_p.value))} p nodes x "
          f"{len(g1_t_pert)} T books, {t_lut:.1f} s): "
          f"max rel diff {rel.max():.3e} (median {np.median(rel):.3e}) "
          f"-> {'PASS' if ok else 'FAIL'} (tol {g1_tol_rel:.0%})")
    return ok


def gate_g2(ws, inv):
    """G2: Doppler-limit normalisation of one isolated nu3 line."""
    line = pick_isolated_line(inv)
    T = line["T0"]                     # catalog reference temperature: the LTE
    #                                    population factor is exactly 1 there
    m_kg = line["mass_gmol"] * 1e-3 / scc.N_A
    gamma_D = line["f0"] * np.sqrt(2 * scc.k * T * np.log(2.0) / (m_kg * scc.c ** 2))
    GD = gamma_D / np.sqrt(np.log(2.0))          # Doppler 1/e width used by ARTS
    peak = (line["i0"] * line["iso_ratio"]       # the GROUP cross section carries
            * np.sqrt(np.log(2.0) / np.pi) / gamma_D)      # the isotopologue ratio

    offs = np.linspace(-g2_grid_halfwidth_GD, g2_grid_halfwidth_GD,
                       g2_grid_points) * GD / (100.0 * scc.c)
    g2_wvn = line["wvn"] + offs
    ws.rtp_temperature = T
    ws.rtp_vmr = [vmr_co2]
    ws.rtp_nlte = arts.EnergyLevelMap()

    ratios = {}
    for P in (g2_pressure, g2_pressure_pure):
        ws.rtp_pressure = P
        alpha = lbl_absorption(ws, None, g2_wvn)
        sig = alpha / (vmr_co2 * P / (scc.k * T))
        ratios[P] = float(sig[np.argmin(np.abs(g2_wvn - line["wvn"]))] / peak)

    ok = all(abs(r - 1.0) < g2_tol_rel for r in ratios.values())
    print(f"G2 Doppler limit, isolated line {line['wvn']:.5f} cm^-1 "
          f"({line['iso']}, I0={line['i0']:.3e} m^2 Hz, gap {line['gap_cm']:.4f} "
          f"cm^-1, m={line['mass_gmol']:.5f} g/mol): "
          f"sigma_peak / [I0*sqrt(ln2/pi)/gamma_D * iso-ratio] = "
          f"{ratios[g2_pressure]:.5f} at P={g2_pressure:g} Pa, "
          f"{ratios[g2_pressure_pure]:.5f} at P={g2_pressure_pure:g} Pa "
          f"-> {'PASS' if ok else 'FAIL'} (tol {g2_tol_rel:.0%}; the small "
          "shortfall at 1 Pa is the Voigt Lorentz admixture)")
    ws.rtp_pressure = P_state
    ws.rtp_temperature = T_state
    return ok, line


def gate_g3(sigma, cuts_cm):
    """G3: sigma is pointwise monotonic in the cutoff (larger cutoff = larger
    line support; inside the support nothing is renormalised).

    Arms sorted by cutoff: sigma[:, a] must never exceed sigma[:, b] for a
    smaller cutoff a and a larger cutoff b -- a positive difference is the
    violation.
    """
    order = sorted(range(len(cuts_cm)),
                   key=lambda k: np.inf if cuts_cm[k] is None else cuts_cm[k])
    tol = 1e-12
    worst, worst_ij = 0.0, (cuts_cm[order[0]], cuts_cm[order[1]])
    for a, b in zip(order[:-1], order[1:]):
        d = sigma[:, a] - sigma[:, b]          # must be <= 0
        scale = np.maximum(np.maximum(np.abs(sigma[:, a]), np.abs(sigma[:, b])),
                           xsec_floor)
        viol = float(np.max(d / scale))
        if viol > worst:
            worst, worst_ij = viol, (cuts_cm[a], cuts_cm[b])
    ok = worst < tol
    return ok, worst, worst_ij


def gate_g4(ws, wvn, sigma_ref):
    """G4: resolution -- recompute the reference arm at dwvn/g4_refine.

    A pure line-by-line calculation has no grid-dependent approximation, so the
    values at coincident grid points agree to round-off (printed as context --
    it proves that switching the LBL speedups off really gives grid-independent
    numbers).  The *resolution* question is answered by how well the coarse
    grid interpolates the fine one: linear interpolation of the coarse spectrum
    onto the midpoints of the fine grid, compared with the exact fine values.
    """
    n_fine = int(round((wvn_max - wvn_min) / (dwvn / g4_refine))) + 1
    wvn_fine = np.linspace(wvn_min, wvn_max, n_fine)
    sig_fine = crosssections(lbl_absorption(ws, reference_cutoff_cm, wvn_fine))

    # (a) coincident points: grid independence of the on-the-fly calculation
    rel_same = np.abs(sig_fine[::2] - sigma_ref) / np.maximum(
        np.maximum(np.abs(sigma_ref), np.abs(sig_fine[::2])), xsec_floor)

    # (b) midpoints: how much of the fine structure the coarse grid misses
    mid = wvn_fine[1::2]
    sig_mid_interp = np.interp(mid, wvn, sigma_ref)
    sig_mid_true = sig_fine[1::2]
    rel_mid = np.abs(sig_mid_interp - sig_mid_true) / np.maximum(
        np.maximum(np.abs(sig_mid_true), np.abs(sig_mid_interp)), xsec_floor)
    j = int(np.argmax(rel_mid))

    ok = bool(rel_mid.max() < g4_tol_rel)
    print(f"G4 resolution, {arm_label(reference_cutoff_cm)} cm^-1 arm at "
          f"dwvn/{g4_refine} = {dwvn / g4_refine} cm^-1 ({n_fine} points): "
          f"coarse-vs-exact at the {mid.size} midpoints max rel diff "
          f"{rel_mid.max():.3e} at {mid[j]:.2f} cm^-1, 99.9% percentile "
          f"{np.percentile(rel_mid, 99.9):.3e} "
          f"-> {'PASS' if ok else 'FAIL'} (tol {g4_tol_rel:.0%}); "
          f"coincident points agree to {rel_same.max():.1e} "
          "(no LBL speedup in use)")
    return ok


def measure_mtckd_context(wvn_probe=(2350.0, 2400.0, 2450.0, 2500.0, 2550.0)):
    """MT-CKD CO2 continuum (CO2-CKDMT252) effective cross section, for context
    only -- it is NOT part of the line cross sections of this script (a
    continuum model has no line-wing cutoff)."""
    ws = Workspace(verbosity=0)
    ws.stokes_dim = 1
    ws.f_grid = arts.convert.kaycm2freq(np.asarray(wvn_probe, dtype=float))
    ws.abs_species = [[arts.SpeciesTag("CO2-CKDMT252")]]
    ws.abs_lines_per_species = [arts.ArrayOfAbsorptionLines()]
    ws.propmat_clearsky_agendaAuto()
    ws.jacobian_quantities = []
    ws.rtp_pressure = P_state
    ws.rtp_temperature = T_state
    ws.rtp_vmr = [vmr_co2]
    ws.propmat_clearskyInit()
    ws.propmat_clearskyAddPredefined()
    alpha = np.array(ws.propmat_clearsky.value.data).squeeze()
    sig = alpha / (vmr_co2 * P_state / (scc.k * T_state))
    del ws
    return np.asarray(wvn_probe, dtype=float), sig


# =============================================================================
# %% console summary
# =============================================================================

def print_summary(wvn, sigma, cuts_cm, integral):
    ref = cuts_cm.index(reference_cutoff_cm)

    print("\n" + "=" * 100)
    print(f"CO2 cross sections at T = {T_state:g} K, P = {P_state / 100:g} hPa, "
          f"vmr(CO2) = {vmr_co2:g}  ({wvn.size} points, {dwvn} cm^-1)")
    print("=" * 100)
    head = "cutoff cm^-1 |" + "".join(f"{w:>10.0f}" for w in report_wvn)
    print("  sigma / 1e-29 m^2 molec.^-1 at")
    print(head + " | integral 2398-2520 / 1e-29")
    print("-" * len(head + " | ---------------------------"))
    scale = 1e29
    for k, cut in enumerate(cuts_cm):
        vals = []
        for w in report_wvn:
            j = int(np.argmin(np.abs(wvn - w)))
            vals.append(f"{sigma[j, k] * scale:10.3f}")
        print(f"{arm_label(cut):>12} |" + "".join(vals)
              + f" | {integral[k] * scale:14.3f}")
    print("\n  ratios vs the "
          f"{arm_label(reference_cutoff_cm)} cm^-1 arm (750 GHz = the "
          "FluxSimulator/benchmark cutoff)")
    print(head + " | integral 2398-2520")
    print("-" * len(head + " | ---------------------------"))
    for k, cut in enumerate(cuts_cm):
        vals = []
        for w in report_wvn:
            j = int(np.argmin(np.abs(wvn - w)))
            r = (sigma[j, k] / sigma[j, ref] if sigma[j, ref] > 0 else np.nan)
            vals.append(f"{r:10.3g}")
        print(f"{arm_label(cut):>12} |" + "".join(vals)
              + f" | {integral[k] / integral[ref]:18.4g}")
    print("=" * 100)


# =============================================================================
# %% figures
# =============================================================================

def _mark_deficit(ax, band_head_cm, label_y=0.98, xytext=(6, 0), arrow=False,
                  ha="left"):
    """Shade the deficit window and mark the CO2 nu3 band head."""
    ax.axvspan(*deficit_range_cm, color=ccow.palette["shade"], zorder=0)
    ax.axvline(band_head_cm, color=ccow.palette["feature"], linewidth=0.5,
               linestyle=(0, (1, 3)), zorder=1)
    kwargs = {}
    if arrow:
        kwargs["arrowprops"] = dict(arrowstyle="-",
                                    color=ccow.palette["feature"],
                                    linewidth=0.5, shrinkA=0, shrinkB=1)
    ax.annotate(rf"CO$_2$ $\nu_3$ band head, {band_head_cm:.1f} cm$^{{-1}}$",
                xy=(band_head_cm, label_y),
                xycoords=("data", "axes fraction"),
                xytext=xytext, textcoords="offset points",
                ha=ha, va="top", fontsize=6.5,
                color=ccow.palette["feature"], zorder=4, **kwargs)


def figure_spectra(wvn, sigma, cuts_cm, band_head_cm):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for k, cut in enumerate(cuts_cm):
        ax.semilogy(wvn, np.maximum(sigma[:, k], xsec_floor),
                    color=cutoff_colors[cut], lw=ccow.line_width
                    + (0.6 if cut == reference_cutoff_cm else 0.0),
                    label=arm_legend(cut))
    ax.set_xlim(wvn[0], wvn[-1])
    # one extra decade of headroom so that the legend stays clear of the curves
    top = 10.0 ** (np.ceil(np.log10(np.nanmax(sigma))) + 1.0)
    ax.set_ylim(xsec_floor, top)
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(r"CO$_2$ absorption cross section $\sigma$ / "
                  r"m$^2$ molec.$^{-1}$")
    _mark_deficit(ax, band_head_cm, label_y=0.995, xytext=(-6, 0), ha="right")
    ccow.annotate_spectral_features(ax, shade_window=False)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(r"CO$_2$ line-by-line cross sections at "
                 f"{T_state:g} K, {P_state / 100:.0f} hPa, "
                 f"Voigt + SFS, no line mixing, no MT-CKD continuum",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = plot_folder / "fig_co2_xsec_spectra.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def figure_ratios(wvn, sigma, cuts_cm, ratio, band_head_cm):
    ref = cuts_cm.index(reference_cutoff_cm)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for k, cut in enumerate(cuts_cm):
        if k == ref:
            continue
        ax.semilogy(wvn, np.maximum(ratio[:, k], 1.0),
                    color=cutoff_colors[cut], lw=ccow.line_width,
                    label=f"{arm_label(cut)} / {arm_label(reference_cutoff_cm)}"
                          " cm$^{-1}$")
    ax.axhline(1.0, color="0.5", lw=0.8, zorder=2)
    ax.set_xlim(wvn[0], wvn[-1])
    # monotonic in the cutoff (gate G3), so the ratio is >= 1 everywhere; the
    # empty decade band below 1 carries the band-head label
    top = 10.0 ** (np.ceil(np.nanmax(np.log10(np.maximum(ratio, 1.0)))) + 0.5)
    ax.set_ylim(1e-6, top)
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(r"$\sigma$(cutoff) / $\sigma$($25$ cm$^{-1}$) / --")
    _mark_deficit(ax, band_head_cm, label_y=0.45)
    ccow.annotate_spectral_features(ax, shade_window=False)
    ax.legend(loc="lower left", ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(r"CO$_2$ cross-section gain when the line-wing cutoff is "
                 f"enlarged (state {T_state:g} K, {P_state / 100:.0f} hPa)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = plot_folder / "fig_co2_xsec_ratios.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def figure_cutoff_vs_wvn(wvn, sigma, cuts_cm):
    """sigma at the reporting wavenumbers as a function of the cutoff."""
    x = [float(c) if c is not None else 1.4 * max(c for c in cuts_cm if c)
         for c in cuts_cm]
    xt = ["untruncated" if c is None else f"{c:.0f}\n(cm$^{-1}$)"
          for c in cuts_cm]
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, len(report_wvn)))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for m, w in enumerate(report_wvn):
        j = int(np.argmin(np.abs(wvn - w)))
        ax.semilogy(x, np.maximum(sigma[j, :], xsec_floor), color=cmap[m],
                    marker="o", ms=3.5, lw=ccow.line_width + 0.3,
                    label=f"{w:.0f} cm$^{{-1}}$")
    ax.set_xticks(x)
    ax.set_xticklabels(xt)
    ax.set_xlim(min(x) * 0.8, max(x) * 1.15)
    ax.set_xlabel("Line-wing cutoff / cm$^{-1}$")
    ax.set_ylabel(r"CO$_2$ absorption cross section $\sigma$ / "
                  r"m$^2$ molec.$^{-1}$")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(r"CO$_2$ cross sections vs. line-wing cutoff "
                 f"({T_state:g} K, {P_state / 100:.0f} hPa); last point: "
                 "untruncated Voigt+SFS", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = plot_folder / "fig_co2_xsec_cutoff_vs_wvn.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =============================================================================
# %% NetCDF export
# =============================================================================

def export_netcdf(wvn, sigma, alpha, ratio, cuts_cm, integral, inv, g2_line):
    cutoff_cm = np.array([untrunc_marker_cm if c is None else float(c)
                          for c in cuts_cm])
    cutoff_hz = np.array([0.0 if c is None
                          else arts.convert.kaycm2freq(float(c))
                          for c in cuts_cm])

    ds = xr.Dataset(
        {
            "crosssection": (["wavenumber", "cutoff"], sigma,
                             {"units": "m2 molecule^-1",
                              "note": "CO2 species group (all isotopologues, "
                                      "terrestrial ratios included)"}),
            "crosssection_cm2": (["wavenumber", "cutoff"], sigma * 1e4,
                                 {"units": "cm2 molecule^-1"}),
            "absorption_coefficient": (["wavenumber", "cutoff"], alpha,
                                       {"units": "m^-1"}),
            "crosssection_ratio_to_25cm":
                (["wavenumber", "cutoff"], ratio,
                 {"units": "-",
                  "note": f"sigma / sigma({reference_cutoff_cm:g} cm^-1 arm)"}),
            "window_integral": (["cutoff"], integral,
                                {"units": "m2 molecule^-1 cm^-1",
                                 "note": f"integral of sigma over "
                                         f"{deficit_range_cm[0]:.0f}-"
                                         f"{deficit_range_cm[1]:.0f} cm^-1"}),
        },
        coords={
            "wavenumber": (["wavenumber"], wvn, {"units": "cm^-1"}),
            "frequency": (["wavenumber"], arts.convert.kaycm2freq(wvn),
                          {"units": "Hz"}),
            "cutoff": (["cutoff"], cutoff_cm,
                       {"units": "cm^-1",
                        "note": f"{untrunc_marker_cm:g} marks the "
                                "untruncated reference arm"}),
            "cutoff_hz": (["cutoff"], cutoff_hz, {"units": "Hz"}),
        },
    )

    ds.attrs["state"] = (f"T = {T_state} K, total p = {P_state} Pa, "
                         f"vmr(CO2) = {vmr_co2}, CO2 dilute in air "
                         "(ARTS AIR broadener bath = 1 - vmr(CO2))")
    ds.attrs["spectral_region_cm-1"] = f"{wvn_min}-{wvn_max}"
    ds.attrs["dwvn_cm-1"] = dwvn
    ds.attrs["deficit_range_cm-1"] = (f"{deficit_range_cm[0]}-"
                                      f"{deficit_range_cm[1]}")
    ds.attrs["cutoff_semantics"] = ("abs_lines_per_speciesCutoff ByLine: a "
                                    "line contributes only inside "
                                    "F0 +/- (cutoff + D0); cutoff_cm = -1 -> "
                                    "option 'None' (no truncation). 25.01 "
                                    "cm^-1 = 750 GHz = FluxSimulator/benchmark "
                                    "cutoff.")
    ds.attrs["method"] = ("ARTS 2.6 on-the-fly line-by-line absorption "
                          "(propmat_clearskyInit + propmat_clearskyAddLines, "
                          "lines_speedup_option='None'); sigma = alpha / "
                          "(vmr * p/(kB T)), the same convention as the "
                          "abs_lookup cross sections used by FluxSimulator")
    ds.attrs["lineshape"] = ("catalog default: Voigt (VP) with SFS far-wing "
                             "normalization, LTE populations, T0 = 296 K")
    ds.attrs["population_and_mixing"] = ("LTE, NO line mixing (catalog "
                                         "population is plain LTE) -- same "
                                         "spectroscopy as the production "
                                         "benchmark")
    ds.attrs["continuum"] = ("CO2 MT-CKD continuum (CO2-CKDMT252) EXCLUDED "
                             "from these cross sections (no cutoff applies to "
                             "a continuum)")
    iso_table = sorted({(v[0], v[2]) for v in inv["iso_info"].values()})
    ds.attrs["isotope_ratios"] = "; ".join(f"{nm} {r:.6g}" for nm, r in iso_table)
    ds.attrs["catalog_datapath"] = "; ".join(str(p)
                                             for p in arts.globals.parameters.datapath)
    ds.attrs["catalog_summary"] = (f"{inv['n_bands']} CO2 bands / "
                                   f"{inv['n_lines']} lines read; "
                                   f"{inv['bands_in_range']} bands with at "
                                   f"least one line in "
                                   f"{inv['range_cm'][0]:.0f}-"
                                   f"{inv['range_cm'][1]:.0f} cm^-1 "
                                   f"({inv['lines_in_range']} lines)")
    ds.attrs["nu3_band"] = (
        f"strongest line of the region: {inv['max_i0']:.3e} m2/Hz at "
        f"{inv['max_i0_wvn']:.2f} cm^-1; it belongs to the {inv['band_iso']} "
        f"band spanning {inv['band_lo_cm']:.1f}-{inv['band_head_cm']:.1f} cm^-1 "
        f"({inv['band_n_lines']} lines) -- beyond the upper edge only hot bands "
        "and the wings of those cold lines can contribute")
    ds.attrs["gate_G2_line"] = (f"isolated line {g2_line['wvn']:.5f} cm^-1 "
                                f"({g2_line['iso']}, I0 = {g2_line['i0']:.4e} "
                                f"m2 Hz, mass {g2_line['mass_gmol']:.5f} "
                                f"g/mol) matches I0*sqrt(ln2/pi)/gamma_D at "
                                "the Doppler limit")
    ds.attrs["pyarts_version"] = str(pyarts.__version__)
    ds.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out = results_folder / "co2_xsection_280K_1000hPa_cutoffs.nc"
    ds.to_netcdf(str(out))
    return out


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_folder.mkdir(parents=True, exist_ok=True)
    ccow.apply_plot_style()

    n_grid = int(round((wvn_max - wvn_min) / dwvn)) + 1
    wvn = np.linspace(wvn_min, wvn_max, n_grid)

    print("=" * 100)
    print("CO2 absorption cross sections vs. line-wing cutoff "
          "(on-the-fly LBL, no FluxSimulator, no abs_lookup)")
    print(f"  state       : T = {T_state:g} K, p = {P_state / 100:g} hPa, "
          f"vmr(CO2) = {vmr_co2:g} (CO2 dilute in air)")
    print(f"  spectrum    : {wvn[0]:.0f}-{wvn[-1]:.0f} cm^-1, "
          f"dwvn = {dwvn} cm^-1 ({n_grid} points)")
    print(f"  deficit     : {deficit_range_cm[0]:.0f}-"
          f"{deficit_range_cm[1]:.0f} cm^-1")
    print(f"  cutoff arms : {', '.join(arm_label(c) for c in cutoffs_cm)} "
          f"cm^-1  (25.01 cm^-1 = 750 GHz = benchmark)")
    print(f"  ARTS threads: {n_threads}")
    print(f"  results     : {results_folder}")
    print(f"  figures     : {plot_folder}")
    print("=" * 100)

    cat.download.retrieve(verbose=False)
    print("ARTS datapath:", [str(p) for p in arts.globals.parameters.datapath])

    # ------------------------------------------------------------------------
    # the line-by-line calculation, one arm per cutoff
    # ------------------------------------------------------------------------
    t0 = time.time()
    ws = open_lbl_workspace(wvn)
    print(f"  catalog read + agenda: {time.time() - t0:.1f} s "
          f"({sum(len(g) for g in ws.abs_lines_per_species.value)} bands)")

    inv = catalog_inventory(ws)
    print(f"  catalog: {inv['n_bands']} bands / {inv['n_lines']} lines; "
          f"{inv['bands_in_range']} bands have at least one line in "
          f"{inv['range_cm'][0]:.0f}-{inv['range_cm'][1]:.0f} cm^-1 "
          f"({inv['lines_in_range']} lines)")
    print(f"  strongest line of that range: {inv['max_i0']:.3e} m^2/Hz at "
          f"{inv['max_i0_wvn']:.2f} cm^-1; it belongs to the {inv['band_iso']} "
          f"band {inv['band_lo_cm']:.1f}-{inv['band_head_cm']:.1f} cm^-1 "
          f"({inv['band_n_lines']} lines) -- above that band head only hot "
          f"bands and the wings of the cold nu3 lines can contribute, which is "
          f"what the cutoff arms below fight over")
    band_head_cm = inv["band_head_cm"]

    sigma = np.zeros((wvn.size, len(cutoffs_cm)))
    alpha = np.zeros_like(sigma)
    for k, cut in enumerate(cutoffs_cm):
        t1 = time.time()
        alpha[:, k] = lbl_absorption(ws, cut)
        sigma[:, k] = crosssections(alpha[:, k])
        j2450 = int(np.argmin(abs(wvn - 2450)))
        j2519 = int(np.argmin(abs(wvn - 2519)))
        print(f"  arm {arm_label(cut):>8}: {time.time() - t1:6.2f} s   "
              f"sigma(2450) = {sigma[j2450, k]:.3e}   "
              f"sigma(2519) = {sigma[j2519, k]:.3e} m^2 molec.^-1")

    # ratio to the benchmark arm and the window integral
    ref = cutoffs_cm.index(reference_cutoff_cm)
    ratio = np.divide(sigma, sigma[:, [ref]],
                      out=np.full_like(sigma, np.nan),
                      where=sigma[:, [ref]] > 0)
    m_win = (wvn >= deficit_range_cm[0]) & (wvn <= deficit_range_cm[1])
    integral = np.trapezoid(sigma[m_win, :], wvn[m_win], axis=0)

    # ------------------------------------------------------------------------
    # context: the MT-CKD CO2 continuum (NOT included in the cross sections)
    # ------------------------------------------------------------------------
    probe_wvn, sig_mtckd = measure_mtckd_context()
    j2450 = int(np.argmin(np.abs(probe_wvn - 2450.0)))
    j2450_main = int(np.argmin(np.abs(wvn - 2450.0)))
    print(f"\n  context, MT-CKD CO2 continuum (CO2-CKDMT252, excluded above): "
          f"sigma = {sig_mtckd[j2450]:.2e} m^2 at 2450 cm^-1 vs. "
          f"{sigma[j2450_main, ref]:.2e} m^2 for the CO2 LINES with the "
          f"{arm_label(reference_cutoff_cm)} cm^-1 cutoff "
          f"({sig_mtckd[j2450] / sigma[j2450_main, ref]:.1f}x)")

    # ------------------------------------------------------------------------
    # summary table
    # ------------------------------------------------------------------------
    print_summary(wvn, sigma, cutoffs_cm, integral)

    # ------------------------------------------------------------------------
    # gates
    # ------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("gates / self-checks")
    print("=" * 100)
    ok_g4 = gate_g4(ws, wvn, sigma[:, ref])
    ok_g2, g2_line = gate_g2(ws, inv)

    ok_g3, worst_g3, pair_g3 = gate_g3(sigma, cutoffs_cm)
    print(f"G3 monotonicity of sigma in the cutoff, pointwise over "
          f"{wvn.size} x {len(cutoffs_cm)} points: worst violation "
          f"{worst_g3:.3e} (at the step {arm_label(pair_g3[0])} -> "
          f"{arm_label(pair_g3[1])} cm^-1) -> {'PASS' if ok_g3 else 'FAIL'} "
          f"(tol 1e-12); inventory: {inv['bands_in_range']} bands / "
          f"{inv['lines_in_range']} lines in "
          f"{inv['range_cm'][0]:.0f}-{inv['range_cm'][1]:.0f} cm^-1, the "
          f"strongest {inv['band_iso']} nu3 lines end at "
          f"{band_head_cm:.2f} cm^-1")

    ok_g1 = gate_g1(ws, wvn, sigma[:, ref])
    all_ok = all([ok_g1, ok_g2, ok_g3, ok_g4])
    verdict = ("all gates passed" if all_ok
               else "GATE FAILURE -- results not trustworthy")
    print(f"gate summary: G1 {'PASS' if ok_g1 else 'FAIL'}, "
          f"G2 {'PASS' if ok_g2 else 'FAIL'}, "
          f"G3 {'PASS' if ok_g3 else 'FAIL'}, "
          f"G4 {'PASS' if ok_g4 else 'FAIL'}  ->  {verdict}")
    print("=" * 100)

    # ------------------------------------------------------------------------
    # products
    # ------------------------------------------------------------------------
    out_nc = export_netcdf(wvn, sigma, alpha, ratio, cutoffs_cm, integral,
                           inv, g2_line)
    print(f"\n  exported {out_nc}")

    figs = [figure_spectra(wvn, sigma, cutoffs_cm, band_head_cm),
            figure_ratios(wvn, sigma, cutoffs_cm, ratio, band_head_cm),
            figure_cutoff_vs_wvn(wvn, sigma, cutoffs_cm)]
    for f in figs:
        print(f"  wrote {f}")

    print(f"\ntotal wall time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
