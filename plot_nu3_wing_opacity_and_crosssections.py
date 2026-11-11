#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Opacity profiles and per-species cross sections at the tau = 1 level of the
CO2 nu3 wing (2398-2520 cm^-1) for a single atmosphere column.

Diagnostic for the smooth T_B deficit of the simulation in 2398-2520 cm^-1
(see plot_co2_nu3_wing_analysis.py).  For ONE column of data/atmdata.nc
(default idx 12345) it answers two questions:

1. How large is the one-way vertical opacity of this column at *every*
   wavenumber of the region, and where does the opacity accumulated from the
   TOA downwards reach tau = 1?
   -> fig_opacity_profile_idx<N>.png / .pdf
      (a) the column profile, (b) 2D opacity field tau(wavenumber, pressure)
      with iso-opacity contours including tau = 1, (c) the tau = 1 level over
      the region together with the total column opacity, (d) which molecules
      build up that opacity

2. What are the absorption cross sections of all absorption species at exactly
   those tau = 1 conditions (pressure, temperature and VMRs of the level where
   tau = 1 is reached)?  Isotopologues are not distinguished -- all catalogue
   entries of one molecule (all isotopologue blocks, XFIT cross sections, CIA /
   continuum models) are summed into a single panel per molecule.  Six panels
   per A4 portrait page, ordered by decreasing contribution to the opacity of
   the region; molecules without any line there are marked as such.
   -> fig_tau1_crosssections_idx<N>.pdf (six molecules per page)
      fig_tau1_crosssections_idx<N>_pXX.png (one PNG per page)

Two opacity estimates are computed from the same line-resolved cross sections:
the *line-resolved* one (fine grid, individual line cores included) and the
*band-mean* one, averaged over `bin_width_cm` wide wavenumber bins.  Which of
the two defines the state of the cross-section panels is set by
`use_line_resolved_state_for_crosssections`; `use_binned_for_line_resolved`
combines the lookup-node bracket of the band mean with the state of the
line-resolved opacity.  Where the opacity stays below one all the way down to
the surface there is no tau = 1 level at all, and the surface state is used
instead (the affected wavenumbers are marked grey).  For this column that is
the case over the whole region: it is optically thin, the emission comes from
the surface, and the CO2 wing only modulates how much of it is absorbed on the
way out -- which is why the simulated T_B there is too warm.

Method
------
The cross sections are recomputed with ARTS 2.6 *abs_lookupCalc* on a fine
wavenumber grid over the region of interest, with exactly the species groups,
line-catalog cutoff (ByLine 750 GHz) and MT-CKD setup that FluxSimulator uses
in the radiance benchmark (fmip_radiance_batch.py).  The lookup pressure grid
is the log-pressure grid that *abs_lookupSetupBatch* derives from this one
atmosphere (log10 step `p_step10`); with abs_t_pert = [0] and abs_nls_pert =
[0] every lookup pressure node carries exactly the temperature and VMRs
interpolated from the profile -- one page per species group, no interpolation
in temperature needed.

The absorption coefficient of a page (a group of tags sharing one VMR) at a
node is
    k = sigma * VMR_ref * n_tot,        n_tot = p / (kB T),
which is the scaling GasAbsLookup::Extract applies inside the RTE.  Layer
optical depths are k * dz with dz from the profile altitude interpolated onto
the lookup pressure grid; the cumulative opacity is summed from the TOA
downwards.  Note that the 750 GHz line cutoff truncates the CO2 nu3 wing at
25 cm^-1, so this is the opacity of the *benchmark setup*, not a full-wing
calculation.

CPU: the ARTS workspace is capped at `n_threads` (<= 10) threads so that the
machine stays usable.

Environment (mamba/conda): pyarts (pyarts 2.6 + FluxSimulator).  Run from the
scripts/ directory:
    $ python plot_nu3_wing_opacity_and_crosssections.py
The ARTS data path must contain lines/, xsec/ and model/ (default
~/.cache/arts/arts-cat-data-2.6.20).
"""

import time
import pathlib

import numpy as np
import xarray as xr
import scipy.constants as scc

import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from pyarts.workspace import Workspace
from pyarts.arts import SpeciesTag
import FluxSimulator as fsm

import fmip_radiance_batch as frb
import compare_convolved_with_obs as ccow   # plot style / palette


# =============================================================================
# %% paths / run constants  (edit here)
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
results_folder = script_dir.parent / "results" / "OPACITY"
plot_folder = script_dir.parent / "plot" / "opacity"

atmdata_file = data_dir / "atmdata.nc"

# column of data/atmdata.nc to analyse
atm_index = 12345

# spectral region of the deficit (cm^-1) and frequency step
wvn_min, wvn_max = 2398.0, 2520.0
dwvn = 0.01                   # cm^-1  ->  12201 grid points

# lookup pressure grid: log10(p/Pa) step passed to abs_lookupSetupBatch
p_step10 = 0.05

# spectroscopy options -- identical to the radiance benchmark
line_cutoff_hz = 750e9        # FluxSimulator get_lookuptableBatch(cutoff=True)
mtckd_version = "4.3"

# FluxSimulator setup name used only to derive the abs_species list
species_setup_name = "fmip_radiance_batch"

# hard cap on the threads ARTS may use (the machine must stay usable)
n_threads = 10

# -- state of the cross-section panels (task 2) -------------------------------
# The line-resolved tau = 1 level exists inside individual line cores, where it
# is a perfectly well defined state; averaged over a wavenumber band, tau < 1
# may hold everywhere and the level is undefined.  Both are reported:
#   use_line_resolved_state_for_crosssections
#       True  -> panels use the line-resolved tau = 1 state
#       False -> panels use the band-mean tau = 1 state
#   use_binned_for_line_resolved
#       True  -> the line-resolved cross sections borrow the lookup-node
#                bracket of the band-mean opacity (tau = 1 of the *band*), but
#                the state itself (p, T, VMR) is taken from the line-resolved
#                opacity -> no node interpolation error
bin_width_cm = 20.0              # cm^-1, IASI channel spacing
use_line_resolved_state_for_crosssections = True
use_binned_for_line_resolved = False

# cross-section panels: 6 rows x 1 column per A4 portrait page
panels_per_page = 6
a4_portrait_inch = (21.0 / 2.54, 29.7 / 2.54)

# colour scale of the 2D opacity plot (log scale, floor for the transparent part)
tau_color_floor = 1e-8
tau_contour_levels = [1e-2, 1e-1, 1.0]
tau_contour_style = {      # level: (colour, linestyle, width, contour label)
    1e-3: ("0.3", "dotted", 0.5, r"$10^{-3}$"),
    1e-2: ("0.3", "dashed", 0.6, r"$10^{-2}$"),
    1e-1: ("0.2", "solid", 0.8, r"$10^{-1}$"),
    1.0: ("white", "solid", 1.8, r"$\tau = 1$"),
}

# cross sections below this value are floored in the log plots
xsec_floor = 1e-33


# =============================================================================
# %% lookup table of one profile (ARTS)
# =============================================================================

def build_abs_species_groups(species_list_of_data):
    """Species groups exactly as the radiance benchmark defines them.

    FluxSimulator defaults (H2O + MT-CKD continuums, O2/N2 CIA, CO2-CKDMT252,
    O3, O3-XFIT) plus one group per data species found in the line or
    cross-section catalogues (fmip_radiance_batch.define_abs_species).
    """
    Flx = fsm.FluxSimulator(species_setup_name)
    Flx.add_species(frb.define_abs_species(Flx, species_list_of_data))
    return [str(grp) for grp in Flx.species]


def calc_lookup_table(atm_field, groups_str, wvn_grid):
    """Compute the per-node absorption cross sections of one profile.

    Returns a dict with the lookup pressure grid (decreasing, surface first),
    the reference temperature / VMR states of the nodes, the cross-section
    pages (m2 molecule-1) and the species-group tags.
    """
    ws = Workspace(verbosity=1)
    ws.SetNumberOfThreads(n_threads)
    ws.atmosphere_dim = 1
    ws.stokes_dim = 1
    ws.abs_p_interp_order = 0     # nearest neighbour in p: the node state is
                                  # exactly the profile state at that pressure

    ws.f_grid = 100.0 * scc.c * np.asarray(wvn_grid, dtype=float)
    ws.abs_species = [[SpeciesTag(t) for t in grp.split(",")]
                      for grp in groups_str]

    # same catalogues, line cutoff and MT-CKD model as FluxSimulator
    ws.ReadXsecData(basename="xsec/")
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=line_cutoff_hz)
    ws.ReadXML(ws.predefined_model_data,
               f"model/mt_ckd_{mtckd_version}/H2O.xml")

    ws.batch_atm_fields_compact = [atm_field]
    ws.abs_lookupSetupBatch(p_step=p_step10)

    # one temperature book at the node reference temperature, one VMR book for
    # the nonlinear (continuum / CIA) species -> exactly one page per group
    ws.abs_t_pert = [0.0]
    ws.abs_nls_pert = [0.0]

    ws.propmat_clearsky_agendaAuto(lines_speedup_option="None")
    ws.lbl_checked = 1

    t_lut = time.time()
    ws.abs_lookupCalc()
    print(f"  abs_lookupCalc: {time.time() - t_lut:.1f} s")

    lut = ws.abs_lookup.value
    xsec = np.array(lut.xsec).astype(np.float32)
    p_nodes = np.array(lut.p_grid, dtype=np.float64)
    t_nodes = np.array(lut.t_ref, dtype=np.float64)
    vmrs = np.array(lut.vmrs, dtype=np.float64)
    groups = [[str(t) for t in grp] for grp in lut.species]
    t_pert = np.array(lut.t_pert, dtype=np.float64)
    del lut, ws            # the line catalogues are large -- free them

    if xsec.shape[0] != 1 or not np.allclose(t_pert, [0.0]):
        raise AssertionError(f"expected a single temperature book, got "
                             f"books={xsec.shape[0]}, t_pert={t_pert}")
    if xsec.shape[1] != len(groups):
        raise AssertionError(f"{len(groups)} species groups but {xsec.shape[1]} "
                             f"cross-section pages -- nonlinear species would "
                             f"need one page per VMR perturbation")

    return {"xsec": xsec[0], "p": p_nodes, "t": t_nodes,
            "vmrs": vmrs, "groups": groups}


# =============================================================================
# %% opacity algebra
# =============================================================================

def group_labels(groups):
    """Molecule name of every lookup page (isotopologues / XFIT merged)."""
    return [grp[0].split("-")[0] for grp in groups]


def cumulative_opacity(k, z_nodes):
    """Cumulative layer opacity from the TOA downwards.

    k is the absorption coefficient (n_f, n_p) in 1/m on the lookup node grid
    (node 0 = surface, last node = TOA).  tau[:, i] is the opacity between the
    TOA and node i, so tau[:, 0] is the total column opacity.
    """
    dz = np.diff(z_nodes)                            # node i+1 above node i
    tau_layer = 0.5 * (k[:, :-1] + k[:, 1:]) * dz[None, :]
    tau = np.zeros_like(k)
    tau[:, :-1] = np.cumsum(tau_layer[:, ::-1], axis=1)[:, ::-1]
    return tau


def gather_to_nodes(field, node_index):
    """Take (n_f, n_p, ...) along the node axis at per-frequency indices."""
    n_f = field.shape[0]
    idx = np.asarray(node_index).reshape((n_f, 1) + (1,) * (field.ndim - 2))
    return np.take_along_axis(field, idx, axis=1)[:, 0, ...]


def find_tau1_level(tau_cum, p_nodes, z_nodes, t_nodes):
    """Per-frequency level where the TOA-downwards opacity reaches tau = 1.

    The cumulative opacity grows towards the surface (decreasing node index),
    so the crossing lies between node i_lo (opaque) and node i_hi = i_lo + 1
    (still transparent).  Frequencies that stay transparent down to the ground
    get np.nan for the level and fall back to the surface node.
    """
    n_f, n_p = tau_cum.shape
    reached = (tau_cum >= 1.0).any(axis=1)

    i_lo = np.argmax(tau_cum >= 1.0, axis=1)         # first opaque node at sfc
    i_lo[~reached] = 0
    i_hi = np.minimum(i_lo + 1, n_p - 1)

    tau_lo = tau_cum[np.arange(n_f), i_lo]
    tau_hi = tau_cum[np.arange(n_f), i_hi]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (1.0 - tau_hi) / (tau_lo - tau_hi)    # 1 at i_lo, 0 at i_hi
    frac = np.where(reached, np.nan_to_num(frac, nan=0.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)

    p1 = np.exp(frac * np.log(p_nodes[i_lo]) + (1 - frac) * np.log(p_nodes[i_hi]))
    z1 = frac * z_nodes[i_lo] + (1 - frac) * z_nodes[i_hi]
    T1 = frac * t_nodes[i_lo] + (1 - frac) * t_nodes[i_hi]

    return {"reached": reached,
            "p": np.where(reached, p1, np.nan),
            "z": np.where(reached, z1, np.nan),
            "T": np.where(reached, T1, np.nan),
            "i_lo": i_lo, "i_hi": i_hi, "frac": frac,
            "tau_total": tau_cum[:, 0]}


def bin_field(field, edges, wvn_grid):
    """Wavenumber-average a (n_f, n_p) field into the bin edges."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    starts = np.searchsorted(wvn_grid, edges[:-1], side="left")
    starts = np.clip(starts, 0, field.shape[0] - 1)
    ends = np.append(starts[1:], field.shape[0])
    counts = np.maximum(ends - starts, 1)
    binned = np.add.reduceat(field, starts, axis=0) / counts[:, None]
    return centers, binned


def per_molecule_sigma_and_opacity(xsec, vmrs, p_nodes, t_nodes, z_nodes,
                                   labels, tau1_state, wvn_grid):
    """Cross sections and opacity shares at the tau = 1 level.

    `tau1_state` is the tau = 1 level of either the line-resolved or the
    band-mean opacity, defined per wavenumber of `wvn_grid`; frequencies where
    the opacity stays below one down to the surface fall back to the surface
    node.  Works molecule by molecule to keep the memory footprint small (the
    full per-page (n_pages, n_f, n_p) array is never built); the cross sections
    are interpolated between the two lookup nodes that bracket the crossing.

    Returns molecule labels, sigma at the tau = 1 state, sigma at the surface
    state, and the opacity of every molecule between the TOA and the tau = 1
    level (all on the given wavenumber grid).
    """
    molecules = sorted(set(labels))
    n_f = wvn_grid.size
    n_tot = p_nodes / (scc.k * t_nodes)              # molecules / m3
    sfc = np.zeros(n_f, dtype=np.int64)              # node index 0 = surface

    sigma_tau1 = np.zeros((n_f, len(molecules)))
    sigma_sfc = np.zeros((n_f, len(molecules)))
    tau_share = np.zeros((n_f, len(molecules)))

    for m, mol in enumerate(molecules):
        pages = [i for i, lab in enumerate(labels) if lab == mol]
        sig = xsec[pages].sum(axis=0).astype(np.float64)     # (n_f, n_p)

        # all pages of one molecule share the same reference VMR profile
        k = sig * vmrs[pages[0]][None, :] * n_tot[None, :]
        tau = cumulative_opacity(k, z_nodes)

        tau_share[:, m] = (tau1_state["frac"]
                           * gather_to_nodes(tau, tau1_state["i_lo"])
                           + (1 - tau1_state["frac"])
                           * gather_to_nodes(tau, tau1_state["i_hi"]))

        sigma_tau1[:, m] = (tau1_state["frac"]
                            * gather_to_nodes(sig, tau1_state["i_lo"])
                            + (1 - tau1_state["frac"])
                            * gather_to_nodes(sig, tau1_state["i_hi"]))
        sigma_sfc[:, m] = gather_to_nodes(sig, sfc)

    return molecules, sigma_tau1, sigma_sfc, tau_share


# =============================================================================
# %% figures
# =============================================================================

def opacity_profile_figure(wvn, p_prof, t_prof, z_prof, p_nodes, t_nodes,
                           tau_cum, tau1, bin_wvn, tau1_binned, bin_tau_total,
                           molecules, tau_share):
    """Four-panel A4 portrait figure -- the opacity of the column (task 1)."""
    fig = plt.figure(figsize=a4_portrait_inch)
    spec = gridspec.GridSpec(4, 1, figure=fig,
                             height_ratios=[1.0, 1.8, 1.0, 1.2], hspace=0.62)

    p_hPa = p_nodes / 100.0
    tau_max = float(tau_cum[:, 0].max())
    tau1_found = bool(tau_cum.max() >= 1.0)

    # --- (a) the atmospheric column ------------------------------------------
    ax = fig.add_subplot(spec[0])
    ax.plot(t_prof, p_prof / 100.0, color=ccow.palette["sim"],
            lw=ccow.line_width + 0.3, label="column profile")
    ax.plot(t_nodes, p_hPa, marker=".", ms=2.5, ls="none",
            color="0.4", label="lookup nodes")
    ax.set_yscale("log")
    ax.set_ylim(p_hPa[0], p_hPa[-1])                 # surface at the bottom
    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Pressure / hPa")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left")
    ax.set_title(f"Atmospheric column idx {atm_index} "
                 f"(surface {z_prof[0] / 1000:.1f} km ... TOA "
                 f"{z_prof[-1] / 1000:.1f} km)", fontsize=9)

    # --- (b) 2D opacity field tau(wavenumber, pressure) ----------------------
    ax = fig.add_subplot(spec[1])
    tau_plot = np.maximum(tau_cum, tau_color_floor)
    mesh = ax.pcolormesh(wvn, p_hPa, tau_plot.T, shading="auto",
                         norm=LogNorm(vmin=tau_color_floor,
                                      vmax=max(1.0, tau_plot.max())),
                         cmap=plt.cm.viridis)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02,
                        label=r"Opacity from the TOA downwards / --")
    cbar.ax.tick_params(labelsize=7)

    cs = ax.contour(wvn, p_hPa, tau_plot.T, levels=tau_contour_levels,
                    colors=[tau_contour_style[lv][0] for lv in tau_contour_levels],
                    linestyles=[tau_contour_style[lv][1]
                                for lv in tau_contour_levels],
                    linewidths=[tau_contour_style[lv][2]
                                for lv in tau_contour_levels])
    ax.clabel(cs, fmt={lv: tau_contour_style[lv][3]
                       for lv in tau_contour_levels},
              fontsize=7, inline=True)

    ax.set_yscale("log")
    ax.set_ylim(p_hPa[0], p_hPa[-1])                 # surface at the bottom
    ax.set_xlim(wvn[0], wvn[-1])
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel("Pressure / hPa")
    if tau1_found:
        sub = r"the $\tau = 1$ contour marks where it reaches 1"
    else:
        sub = (r"no $\tau = 1$ contour: the maximum opacity of the whole "
               rf"column is {tau_max:.2f}")
    ax.set_title(r"Opacity of every wavenumber from the TOA downwards -- " + sub,
                 fontsize=9)

    # --- (c) the tau = 1 level and how close the column comes to it ----------
    ax = fig.add_subplot(spec[2])
    r, rb = tau1["reached"], tau1_binned["reached"]
    ax.fill_between(bin_wvn, p_hPa[-1], p_hPa[0], where=~rb, step="mid",
                    color=ccow.palette["shade"], alpha=0.35, zorder=0,
                    label=rf"band-mean $\tau < 1$ everywhere "
                          rf"({bin_width_cm:.0f} cm$^{{-1}}$ bins)")
    ax.plot(bin_wvn[rb], tau1_binned["p"][rb] / 100.0,
            color=ccow.palette["sim"], lw=ccow.line_width + 0.5,
            label=r"band-mean $\tau = 1$ level")
    ax.plot(wvn[r], np.asarray(tau1["p"])[r] / 100.0,
            ls="none", marker=".", ms=2.0, color="0.45",
            label=r"line-resolved $\tau = 1$ level")
    ax.set_yscale("log")
    ax.set_ylim(p_hPa[0], p_hPa[-1])
    ax.set_xlim(wvn[0], wvn[-1])
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(r"Pressure of the $\tau = 1$ level / hPa")
    ax.grid(alpha=0.3, which="both")

    axx = ax.twinx()
    axx.axhline(1.0, color=ccow.palette["diff"], lw=1.0, ls="--")
    axx.annotate(r"$\tau = 1$", xy=(wvn[0] + 0.02 * np.ptp(wvn), 1.03),
                 fontsize=7.5, color=ccow.palette["diff"], ha="left")
    axx.plot(wvn, tau_cum[:, 0], color=ccow.palette["obs"], alpha=0.5,
             lw=0.6, label=r"line-resolved total $\tau$ "
                           rf"(TOA $\rightarrow$ surface)")
    axx.plot(bin_wvn, bin_tau_total, color=ccow.palette["obs"],
             lw=ccow.line_width + 0.4,
             label=rf"band-mean total $\tau$ ({bin_width_cm:.0f} cm$^{{-1}}$)")
    axx.set_ylim(0, 1.15)
    axx.set_ylabel(r"Total column opacity / --", color=ccow.palette["obs"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axx.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5)
    ax.set_title(r"$\tau = 1$ level (the emission level) and how close the "
                 r"whole column comes to it", fontsize=9)

    # --- (d) molecule contributions to the opacity ---------------------------
    ax = fig.add_subplot(spec[3])
    order = list(np.argsort(-tau_share.max(axis=0)))
    keep = [m for m in order if tau_share[:, m].max() > 1e-3]
    other = [m for m in order if m not in keep]
    stack = [tau_share[:, m] for m in keep]
    names = [molecules[m] for m in keep]
    if other:
        stack.append(tau_share[:, other].sum(axis=1))
        names.append(f"other ({len(other)})")
    cmap = plt.cm.tab20(np.linspace(0, 0.9, max(len(stack), 3)))
    ax.stackplot(wvn, stack, labels=names, colors=cmap[:len(stack)], alpha=0.9)
    ax.axhline(1.0, color=ccow.palette["diff"], lw=1.0, ls="--")
    ax.plot(bin_wvn, bin_tau_total, color="0.15", lw=ccow.line_width + 0.4,
            label="total (band mean)")
    ax.set_xlim(wvn[0], wvn[-1])
    ax.set_ylim(0, max(1.05, 1.05 * tau_cum[:, 0].max()))
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(r"Opacity TOA $\rightarrow$ $\tau = 1$ level / --")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(r"Opacity split by molecule (down to the $\tau = 1$ level, or "
                 r"to the surface where $\tau < 1$)", fontsize=9)

    fig.suptitle(r"CO$_2$ $\nu_3$ wing 2398--2520 cm$^{-1}$: column opacity "
                 rf"and its $\tau = 1$ level (column idx {atm_index})",
                 fontsize=11)
    fig.savefig(plot_folder / f"fig_opacity_profile_idx{atm_index}.png")
    fig.savefig(plot_folder / f"fig_opacity_profile_idx{atm_index}.pdf")
    plt.close(fig)


def crosssection_figures(wvn, molecules, sigma_tau1, sigma_sfc, tau1_state,
                         state_kind):
    """Cross sections at the tau = 1 state -- six panels per A4 portrait page."""
    n_mol = len(molecules)
    n_pages = int(np.ceil(n_mol / panels_per_page))

    pos = sigma_tau1[sigma_tau1 > 0]
    y_lo = np.power(10.0, max(np.floor(np.log10(pos.min())) - 1.0,
                              np.floor(np.log10(xsec_floor))))

    no_tau1 = ~tau1_state["reached"]      # surface state used at these wvn

    with PdfPages(plot_folder /
                  f"fig_tau1_crosssections_idx{atm_index}.pdf") as pdf:
        for ip in range(n_pages):
            sel = molecules[ip * panels_per_page:(ip + 1) * panels_per_page]
            fig = plt.figure(figsize=a4_portrait_inch)
            spec = gridspec.GridSpec(panels_per_page, 1, figure=fig, hspace=0.45,
                                     left=0.10, right=0.97, top=0.885,
                                     bottom=0.045)
            for k, mol in enumerate(sel):
                m = molecules.index(mol)
                ax = fig.add_subplot(spec[k])
                ax.semilogy(wvn, np.maximum(sigma_tau1[:, m], y_lo),
                            color=ccow.palette["sim"],
                            lw=ccow.line_width + 0.2, zorder=3)
                ax.semilogy(wvn, np.maximum(sigma_sfc[:, m], y_lo),
                            color="0.55", lw=ccow.line_width, ls=":", zorder=2)
                # one decade margin above the peak of this molecule; molecules
                # without any line in the region keep the default decade
                peak = sigma_tau1[:, m].max()
                y_hi = (np.power(10.0, np.ceil(np.log10(peak)) + 0.5)
                        if peak > 0 else y_lo * 10.0)
                ax.set_ylim(y_lo, max(y_hi, y_lo * 10.0))
                if peak == 0:
                    ax.annotate("no lines in this region",
                                xy=(0.5, 0.5), xycoords="axes fraction",
                                ha="center", fontsize=7.5, color="0.4")
                ax.set_xlim(wvn[0], wvn[-1])
                if no_tau1.any() and not no_tau1.all():
                    # wavenumbers whose opacity stays below 1 down to the ground
                    ax.fill_between(wvn, y_lo, y_hi, where=no_tau1, step="mid",
                                    color=ccow.palette["shade"], alpha=0.45,
                                    zorder=0)
                ax.set_ylabel("$\\sigma$ / m$^2$ molec.$^{-1}$", fontsize=7.5)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3, which="both")
                ax.set_title(mol, fontsize=9.5, loc="left")
                if k == len(sel) - 1:
                    ax.set_xlabel("Wavenumber / cm$^{-1}$")

            r = tau1_state["reached"]
            if np.any(r):
                cond = (f"solid: {state_kind} $\\tau = 1$ conditions "
                        f"($p = {np.nanmin(tau1_state['p'][r]) / 100:.0f}$--"
                        f"{np.nanmax(tau1_state['p'][r]) / 100:.0f}$ hPa, "
                        f"$T = {np.nanmin(tau1_state['T'][r]):.0f}$--"
                        f"{np.nanmax(tau1_state['T'][r]):.0f}$ K)")
            else:
                cond = f"solid: no {state_kind} $\\tau = 1$ level -- identical " \
                       "to the surface state"
            fig.suptitle(
                "Absorption cross sections at the "
                r"$\tau = 1$ level, CO$_2$ $\nu_3$ wing "
                f"2398--2520 cm$^{{-1}}$, column idx {atm_index}\n"
                f"{cond}   dotted: surface conditions   "
                "grey: opacity $< 1$ down to the surface "
                "(surface state used)",
                fontsize=8.5)
            fig.savefig(plot_folder /
                        f"fig_tau1_crosssections_idx{atm_index}"
                        f"_p{ip + 1:02d}.png")
            pdf.savefig(fig)
            plt.close(fig)


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # the column
    # -------------------------------------------------------------------------
    atmdata = xr.open_dataset(atmdata_file)
    species_list = [spc for spc in atmdata.data_vars
                    if spc not in ("p", "t", "alt")]

    if atm_index >= int(atmdata.sizes["idx"]):
        raise ValueError(f"atm_index {atm_index} outside the input file "
                         f"({atmdata.sizes['idx']} columns)")

    # input files are TOA-first, ARTS compact fields are surface-first
    p_prof = atmdata["p"].values[atm_index][::-1]
    t_prof = atmdata["t"].values[atm_index][::-1]
    z_prof = atmdata["alt"].values[atm_index][::-1]
    vmr_prof = {spc: np.clip(atmdata[spc].values[atm_index][::-1], 0.0, None)
                for spc in species_list}

    wvn = np.arange(wvn_min, wvn_max + dwvn / 2, dwvn)
    bin_edges = np.arange(wvn_min, wvn_max + bin_width_cm, bin_width_cm)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    print("=" * 70)
    print("CO2 nu3 wing: opacity profile and tau = 1 cross sections")
    print(f"  column      : idx {atm_index}  "
          f"(p_sfc {p_prof[0] / 100:.1f} hPa, T_sfc {t_prof[0]:.1f} K, "
          f"TOA {z_prof[-1] / 1000:.1f} km)")
    print(f"  spectrum    : {wvn[0]:.0f} - {wvn[-1]:.0f} cm^-1, "
          f"step {dwvn} cm^-1 ({wvn.size} points)")
    print(f"  band bins   : {bin_width_cm} cm^-1 "
          f"({bin_edges.size - 1} bins)")
    print(f"  lookup p    : log10 step {p_step10}")
    print(f"  cutoff      : ByLine {line_cutoff_hz / 1e9:.0f} GHz, "
          f"MT-CKD {mtckd_version}")
    print(f"  ARTS threads: {n_threads}")
    print(f"  results     : {results_folder}")
    print(f"  figures     : {plot_folder}")
    print("=" * 70)

    groups_str = build_abs_species_groups(species_list)
    print(f"  species groups: {len(groups_str)}")

    atm_field = frb.make_gridded_field(p_prof, t_prof, z_prof,
                                       vmr_prof, species_list)

    # -------------------------------------------------------------------------
    # 1) cross sections per lookup node and the opacity of the column
    # -------------------------------------------------------------------------
    lut = calc_lookup_table(atm_field, groups_str, wvn)
    labels = group_labels(lut["groups"])
    p_nodes, t_nodes, vmrs = lut["p"], lut["t"], lut["vmrs"]
    print(f"  lookup nodes: {p_nodes.size} "
          f"({p_nodes[0] / 100:.1f} hPa ... {p_nodes[-1]:.2f} Pa)")
    print(f"  cross-section pages {lut['xsec'].shape[1:]}, "
          f"{len(set(labels))} molecules")

    # profile altitude interpolated onto the lookup pressure grid (log-p)
    z_nodes = np.interp(np.log(p_nodes), np.log(p_prof[::-1]), z_prof[::-1])

    # line-resolved opacity
    n_tot = p_nodes / (scc.k * t_nodes)
    k_total = (np.einsum("gfp,gp->fp", lut["xsec"].astype(np.float64), vmrs)
               * n_tot[None, :])
    tau_cum = cumulative_opacity(k_total, z_nodes)
    tau1 = find_tau1_level(tau_cum, p_nodes, z_nodes, t_nodes)

    # band-mean opacity (band width of the instrument footprint)
    bin_centers, k_bin = bin_field(k_total, bin_edges, wvn)
    tau_bin = cumulative_opacity(k_bin, z_nodes)
    tau1_bin = find_tau1_level(tau_bin, p_nodes, z_nodes, t_nodes)

    # the state used for the cross-section panels (task 2)
    if use_line_resolved_state_for_crosssections:
        tau1_state, state_kind = tau1, "line-resolved"
    else:
        tau1_state, state_kind = tau1_bin, "band-mean"
    if use_binned_for_line_resolved:
        # node brackets from the band mean, state from the line-resolved opacity
        bin_of_wvn = np.clip(np.searchsorted(bin_edges, wvn, side="right") - 1,
                             0, bin_centers.size - 1)
        tau1_state = dict(tau1)
        for key in ("i_lo", "i_hi", "frac"):
            tau1_state[key] = tau1_bin[key][bin_of_wvn]

    molecules, sigma_tau1, sigma_sfc, tau_share = per_molecule_sigma_and_opacity(
        lut["xsec"], vmrs, p_nodes, t_nodes, z_nodes, labels, tau1_state, wvn)
    lut = None

    # order the molecules by importance (largest opacity first) -- the figures
    # and the NetCDF molecule axis share this order, so the key species of the
    # region (CO2, N2, N2O) come first and empty panels last
    mol_order = np.argsort(-(tau_share.mean(axis=0)
                             + 1e-30 * sigma_tau1.max(axis=0)), kind="stable")
    molecules = [molecules[m] for m in mol_order]
    sigma_tau1 = sigma_tau1[:, mol_order]
    sigma_sfc = sigma_sfc[:, mol_order]
    tau_share = tau_share[:, mol_order]

    for tag, tau, t1 in (("line-resolved", tau_cum, tau1),
                         (f"band-mean ({bin_width_cm:.0f} cm^-1 bins)",
                          tau_bin, tau1_bin)):
        print(f"\n  {tag} opacity TOA -> surface: "
              f"{tau[:, 0].min():.3f} ... {tau[:, 0].max():.3f} "
              f"(mean {tau[:, 0].mean():.3f})")
        if np.any(t1["reached"]):
            r = t1["reached"]
            n_r = int(r.sum())
            grid = wvn if tau is tau_cum else bin_centers
            print(f"    tau = 1 reached at {n_r} of {grid.size} grid points "
                  f"({grid[r].min():.0f} - {grid[r].max():.0f} cm^-1)")
            print(f"    tau = 1 level : p "
                  f"{np.nanmin(t1['p'][r]) / 100:.1f} - "
                  f"{np.nanmax(t1['p'][r]) / 100:.1f} hPa, "
                  f"z {np.nanmin(t1['z'][r]) / 1000:.1f} - "
                  f"{np.nanmax(t1['z'][r]) / 1000:.1f} km, "
                  f"T {np.nanmin(t1['T'][r]):.1f} - {np.nanmax(t1['T'][r]):.1f} K")
        else:
            print("    tau = 1 is NEVER reached -- the column is transparent "
                  "down to the surface (the surface state is used for the "
                  "cross sections)")

    top = np.argsort(-tau_share.mean(axis=0))[:6]
    print(f"\n  mean opacity of the molecules above the {state_kind} tau = 1 "
          "level: " +
          ", ".join(f"{molecules[m]} {tau_share[:, m].mean():.4f}"
                    for m in top if tau_share[:, m].mean() > 1e-6))

    # -------------------------------------------------------------------------
    # 2) products
    # -------------------------------------------------------------------------
    ds = xr.Dataset(
        {
            "absorption_coefficient": (["wavenumber", "node"], k_total,
                                       {"units": "m^-1",
                                        "note": "sum over all species"}),
            "optical_depth_TOA_to_node":
                (["wavenumber", "node"], tau_cum,
                 {"units": "-",
                  "note": "cumulative downwards from the TOA; node 0 = surface"}),
            "absorption_coefficient_bandmean":
                (["wavenumber_bin", "node"], k_bin, {"units": "m^-1"}),
            "optical_depth_TOA_to_node_bandmean":
                (["wavenumber_bin", "node"], tau_bin, {"units": "-"}),
            "crosssection_tau1": (["wavenumber", "molecule"], sigma_tau1,
                                  {"units": "m2 molecule^-1",
                                   "note": f"at the p/T/VMR state of the "
                                           f"{state_kind} tau = 1 level; "
                                           "surface state where tau < 1"}),
            "crosssection_surface": (["wavenumber", "molecule"], sigma_sfc,
                                     {"units": "m2 molecule^-1"}),
            "opacity_share_tau1": (["wavenumber", "molecule"], tau_share,
                                   {"units": "-",
                                    "note": "opacity TOA -> tau = 1 level; "
                                            "sums to the total opacity there"}),
            "p_tau1": (["wavenumber"], tau1["p"], {"units": "Pa"}),
            "z_tau1": (["wavenumber"], tau1["z"], {"units": "m"}),
            "T_tau1": (["wavenumber"], tau1["T"], {"units": "K"}),
            "tau_total": (["wavenumber"], tau1["tau_total"], {"units": "-"}),
            "tau_reached": (["wavenumber"], tau1["reached"]),
            "p_tau1_bandmean": (["wavenumber_bin"], tau1_bin["p"],
                                {"units": "Pa"}),
            "z_tau1_bandmean": (["wavenumber_bin"], tau1_bin["z"], {"units": "m"}),
            "T_tau1_bandmean": (["wavenumber_bin"], tau1_bin["T"], {"units": "K"}),
            "tau_total_bandmean": (["wavenumber_bin"], tau1_bin["tau_total"],
                                   {"units": "-"}),
            "tau_reached_bandmean": (["wavenumber_bin"], tau1_bin["reached"]),
            "node_pressure": (["node"], p_nodes, {"units": "Pa"}),
            "node_altitude": (["node"], z_nodes, {"units": "m"}),
            "node_temperature": (["node"], t_nodes, {"units": "K"}),
        },
        coords={
            "wavenumber": (["wavenumber"], wvn, {"units": "cm^-1"}),
            "wavenumber_bin": (["wavenumber_bin"], bin_centers,
                               {"units": "cm^-1"}),
            "node": (["node"], np.arange(p_nodes.size, dtype=np.int64)),
            "molecule": (["molecule"], np.array(molecules, dtype=object)),
        },
    )
    ds.attrs["column_idx"] = atm_index
    ds.attrs["spectral_region_cm-1"] = f"{wvn_min}-{wvn_max}"
    ds.attrs["dwvn_cm-1"] = dwvn
    ds.attrs["bin_width_cm"] = bin_width_cm
    ds.attrs["p_step10"] = p_step10
    ds.attrs["line_cutoff_GHz"] = line_cutoff_hz / 1e9
    ds.attrs["mtckd_version"] = mtckd_version
    ds.attrs["species_groups"] = "; ".join(groups_str)
    ds.attrs["molecules"] = ", ".join(molecules)
    ds.attrs["method"] = ("ARTS 2.6 abs_lookupCalc, one lookup node per "
                          "profile state; absorption coefficient k = sigma * "
                          "VMR_ref * p/(kB T); opacity integrated over the "
                          "profile altitude from the TOA downwards")
    ds.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_nc = (results_folder /
              f"opacity_idx{atm_index}_Nf{wvn.size}"
              f"_{wvn_min:.0f}-{wvn_max:.0f}cm-1.nc")
    ds.to_netcdf(str(out_nc))
    print(f"\n  exported {out_nc}")

    # -------------------------------------------------------------------------
    # 3) figures
    # -------------------------------------------------------------------------
    opacity_profile_figure(wvn, p_prof, t_prof, z_prof, p_nodes, t_nodes,
                           tau_cum, tau1, bin_centers, tau1_bin, tau_bin[:, 0],
                           molecules, tau_share)
    crosssection_figures(wvn, molecules, sigma_tau1, sigma_sfc, tau1_state,
                         state_kind)
    print(f"  figures written to {plot_folder}")

    print(f"\ntotal wall time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
