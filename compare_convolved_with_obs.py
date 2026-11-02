#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare SRF-convolved simulated radiances / brightness temperatures with IASI
satellite observations (satdata.nc).

This script reads the convolved output of convolve_sensor_response.py and the
observed brightness temperatures from data/satdata.nc, and computes
per-wavenumber statistics of the difference (simulation - observation):

    - mean difference
    - standard deviation of the difference
    - RMSE

for three quantities:
    1. absolute radiance          [W m^-2 Hz^-1 sr^-1]
       (observed T_b is converted to radiance via the frequency-domain
       Planck function, matching the ARTS/DISORT output units)
    2. relative radiance          [% of the observed Planck radiance]
    3. brightness temperature     [K]

In addition, the distribution of the differences is accumulated as a
per-wavenumber histogram and plotted as pcolormesh (x = wavenumber,
y = difference bin, colour = probability density), and the mean of the
simulation and of the observation themselves is accumulated per wavenumber
together with the standard deviation across columns -- plotted as one figure
per quantity with both mean spectra and their +-1 std bands.

One simulation run is compared per invocation, selected by the run_name
constant (results/RADIANCE/<run_name>/), mirroring convolve_sensor_response.py.
Only convolved files belonging to that run are discovered (file-name patterns
pinned to run_name).  convolve_sensor_response.py writes its products on the
sensor (IASI) channel grid, so they match the 8461-channel observation grid
(645-2760 cm^-1, 0.25 cm^-1 spacing) exactly and are compared point by point;
older products still on a finer simulation frequency grid (e.g. the merged
SLURM run of an earlier convolve run, 0.02 cm^-1 spacing) are collocated to the
observation grid by nearest simulation channel - an offset of at most half the
simulation grid spacing, far below the SRF width.  A run whose simulation
range does not cover the SRF width beyond the satellite range is missing its
edge channels (the convolve step drops channels whose SRF support is not
covered); the collocation tolerance check reports that.  Files that carry only
brightness_temperature (convolve run with export_convolved_radiance = False)
are compared in brightness temperature only; radiance quantities are skipped.
All chunks carry the global column index (idx 0-19705), so columns match the
observations exactly.

Environment (mamba/conda):    pyarts
                              (same as convolve_sensor_response.py)

Input data:
    data/satdata.nc                                   (observed T_b)
    results/RADIANCE/<run_name>/Reference_convolved_*.nc
    (searched recursively within the run sub folder, file names pinned to
    run_name)

Output:
    results/COMPARISON/comparison_stats_<run_name>.nc  aggregate statistics
    plot/comparison/fig_<...>_<run_name>.png / .pdf    figures
    The statistics file holds, per compared quantity and wavenumber, the
    difference statistics (diff_mean / diff_std / diff_rmse), the histogram
    counts, and the mean and standard deviation of the simulation and of the
    observation themselves (sim_mean_*, sim_std_*, obs_mean_*, obs_std_*).
    Figures: one mean-spectra figure per quantity (mean simulation vs mean
    observation, each with its +-1 std band, plus the difference strip), the
    three-panel difference statistics, and the difference-distribution
    pcolormesh.

Example
-------
Set the run_name constant to the run sub folder to compare, then run the
script; it discovers the run's convolved chunks automatically.

    $ python compare_convolved_with_obs.py

Notes
-----
* All statistics and histograms are computed on the observation channel grid;
  the convolved products are written on that grid by convolve_sensor_response.py
  (products of older convolve runs on a finer simulation grid are collocated to
  it by nearest simulation channel) before the streaming accumulation starts.
* The +-1 std bands of the mean-spectra figure are the across-column (global)
  variability of the scene/atmosphere at a given channel -- dominated by the
  surface and lower-troposphere emission in the window channels -- not an
  uncertainty of the mean itself (that would be std / sqrt(N)).
* Axis labels use the "quantity / unit" notation; styling (font, palette,
  spectral-region annotations) is set once by apply_plot_style below.
* The observed T_b fields contain fill values (-777); they are masked via the
  valid-range check (bt_valid_range) and excluded from all statistics.
* Processing runs in two streaming passes over the chunks per setup: pass 1
  computes the running sums for mean/std/RMSE (plus a column subsample used
  to set the fixed radiance histogram range), pass 2 accumulates the
  per-wavenumber histogram counts. Peak memory stays moderate.
* Pcolormesh plots are rasterized (rasterized=True) so the vector PDFs stay
  small; line plots stay vector.
* Plot formatting follows default_plot_format / default_figure_gridspec of
  opacity_helper_function.py (opacity_rule project); the helpers are copied
  locally so this script is self-contained.
"""

import time
import pathlib
import glob
import re

import numpy as np
import xarray as xr
import scipy.constants as scc

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

# Location of this script
script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
results_root = script_dir.parent / "results" / "RADIANCE"
stats_folder = script_dir.parent / "results" / "COMPARISON"
plot_folder = script_dir.parent / "plot" / "comparison"

# run sub folder of results/RADIANCE/ to compare against the observations
# (run_name = <setup_name>_Nf<N_wvn>_Nza<N_za>, same constant as in
# convolve_sensor_response.py; only that run's convolved files are used)
run_name = "fmip_radiance_batch_slurm_Nf108951_Nza20"
results_folder = results_root / run_name

# Input data files
satdata_file = data_dir / "satdata.nc"

# Collocation tolerance [cm^-1]: maximum distance of the nearest simulation
# grid point to an observation channel centre (grid-coarser-than-obs grids
# are rejected by this check; exact or finer grids pass trivially)
collocation_tol_cm = 0.125

# Observed T_b validity: values outside this range are treated as fill values
# (satdata.nc uses -777 as fill value)
bt_valid_range = (50.0, 400.0)

# Histogram configuration for the pcolormesh distribution plots
hist_n_bins = 101
bt_hist_range = (-5.0, 5.0)            # K, symmetric
rel_hist_range = (-2.0, 2.0)           # %, symmetric
abs_hist_range = None                  # W m^-2 Hz^-1 sr^-1; None -> auto
abs_hist_percentile = 99.0             # percentile of |diff| for auto range
sample_column_stride = 10              # column subsampling for auto range

# Export / plotting options
export_stats = True
make_plots = True

# Plot styling ("quantity / unit" axis labels throughout)
plot_font = "Helvetica Neue"       # falls back to Arial / DejaVu Sans if absent
palette = {
    "obs": "#1b5e8a",              # observation curves
    "sim": "#b3352b",              # simulation curves
    "diff": "#3d3d3d",             # difference curves
    "band_obs": "#1b5e8a",
    "band_sim": "#b3352b",
    "shade": "#d9d9d9",            # atmospheric-window shading
    "feature": "#7f7f7f",          # absorption-feature markers
}
line_width = 0.7

# Atmospheric window (shaded) and major absorption features (dotted markers)
# in the IASI range -- (wavenumber / cm^-1, label)
window_range_cm = (800.0, 1200.0)
spectral_features = [
    (667.4, r"CO$_2$ $\nu_2$"),
    (1040.0, r"O$_3$"),
    (1284.0, r"N$_2$O"),
    (1306.0, r"CH$_4$"),
    (1594.0, r"H$_2$O $\nu_2$"),
    (2349.0, r"CO$_2$ $\nu_3$"),
]

# Plain-text spellings of the mathtext units used in axis labels: for inline
# annotations (plain_units) and for the probability-density colour bars,
# which carry the reciprocal unit (inverse_units).
plain_units = {
    "K": "K",
    "W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$": "W m^-2 Hz^-1 sr^-1",
    "%": "%",
}
inverse_units = {
    "K": "K$^{-1}$",
    "W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$": "W$^{-1}$ m$^2$ Hz sr$",
    "%": "%$^{-1}$",
}

# =============================================================================
# %% plot formatting (adapted from opacity_helper_function.py)
# =============================================================================

def apply_plot_style():
    """
    Set the global matplotlib style of the comparison figures.

    Prefers plot_font, falls back through the available fonts; compact
    sans-serif labels, thin spines, dotted grids, mathtext in the same
    sans family so that the "quantity / unit" labels look consistent.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in (plot_font, "Helvetica Neue", "Helvetica", "Arial",
                          "DejaVu Sans") if f in available] or ["DejaVu Sans"]

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family + ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.linewidth": 0.3,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.fontsize": 7.5,
        "legend.framealpha": 0.9,
        "legend.borderpad": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


apply_plot_style()


def default_figure_gridspec(rows=1, columns=1, width_in_cm=21.0,
                            height_in_cm=15.0, constrained_layout=True,
                            hspace=0.35, wspace=0.2, height_ratios=None):
    """Create a figure with cm-based size and a gridspec."""
    fig = plt.figure(constrained_layout=constrained_layout)
    fig.set_size_inches(width_in_cm / 2.54, height_in_cm / 2.54)
    spec = gridspec.GridSpec(ncols=columns, nrows=rows, figure=fig,
                             hspace=hspace, wspace=wspace,
                             height_ratios=height_ratios)

    return fig, spec


def default_plot_format(ax, font_name=None, grid_on=True, minor_ticks=True):
    """Apply consistent formatting to matplotlib axes."""
    font = FontProperties()
    if font_name is not None:
        font.set_name(font_name)

    if grid_on:
        ax.grid(which="major", linestyle=":", linewidth=0.3, alpha=0.6)
        if minor_ticks:
            ax.grid(which="minor", linestyle=":", linewidth=0.2, alpha=0.25)
    for spine in ("right", "top"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction="in", which="both")
    if minor_ticks and grid_on:
        ax.minorticks_on()
    ax.set_axisbelow(True)

    return ax, font


# =============================================================================
# %% helpers
# =============================================================================

def planck_frequency(nu, T):
    """
    Frequency-domain Planck function (spectral radiance per Hz).

    Parameters
    ----------
    nu : ndarray, shape (N_wvn,)
        Frequency [Hz].
    T : ndarray
        Temperature [K], broadcastable against nu on the last axis.

    Returns
    -------
    ndarray
        Spectral radiance [W m^-2 Hz^-1 sr^-1].
    """
    h = scc.h   # Planck constant [J s]
    c = scc.c   # Speed of light [m/s]
    k_B = scc.k  # Boltzmann constant [J/K]

    shape = (np.asarray(nu).size,) + (1,) * (np.asarray(T).ndim - 1)
    c2_nu = (2.0 * h * np.asarray(nu) ** 3 / c**2).reshape(shape)
    h_nu = (h * np.asarray(nu)).reshape(shape)

    with np.errstate(over="ignore"):
        return c2_nu / np.expm1(h_nu / (k_B * T))


def discover_convolved_chunks(results_folder, run_name):
    """
    Sorted list of convolved chunk NetCDF files of one run, pinned to run_name.

    The search covers the run sub folder selected by the run_name constant
    (results/RADIANCE/<run_name>/) recursively; only files whose names belong
    to run_name are returned.  convolve_sensor_response.py mirrors the input
    radiance names (radiance_TOA -> convolved), so this run's convolved files
    either contain the full run_name (merged runs,
    Reference_convolved_merged_<run_name>_...) or the setup-name part plus
    the _Nf<..>_/ _Nza<..>_ tags of run_name (
    Reference_convolved_Nf*_Nza*_<setup_name>_...).
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

    pattern = str(results_folder / "**" / "Reference_convolved_*.nc")
    files = [
        pathlib.Path(f)
        for f in glob.glob(pattern, recursive=True)
        if setup_name in pathlib.Path(f).name
        and (nf_tag is None or nf_tag in pathlib.Path(f).name)
        and (nza_tag is None or nza_tag in pathlib.Path(f).name)
    ]
    return sorted(files)


def collocation_indices(sim_wvn, wvn_obs, tol_cm):
    """
    Index of the nearest simulation grid point for each observation channel.

    Grids that already match return exact indices (zero distance).

    Parameters
    ----------
    sim_wvn, wvn_obs : ndarray
        Simulation and observation wavenumber grids [cm^-1], ascending.
    tol_cm : float
        Maximum allowed distance of the nearest simulation point to an
        observation channel centre.

    Returns
    -------
    ndarray
        Index into sim_wvn for each observation channel.
    """
    pos = np.clip(np.searchsorted(sim_wvn, wvn_obs), 1, sim_wvn.size - 1)
    left, right = pos - 1, pos
    sel = np.where(np.abs(sim_wvn[left] - wvn_obs)
                   <= np.abs(sim_wvn[right] - wvn_obs), left, right)
    dist = np.abs(sim_wvn[sel] - wvn_obs)
    if dist.max() > tol_cm:
        raise ValueError(
            f"simulation wavenumber grid too coarse/misaligned: observation "
            f"channel {wvn_obs[int(dist.argmax())]:.2f} cm^-1 is "
            f"{dist.max():.3f} cm^-1 from the nearest simulation point "
            f"(tolerance {tol_cm} cm^-1)."
        )
    return sel


def extract_chunk_arrays(convolved_ds):
    """
    Load the (frequency, idx) arrays of one convolved chunk.

    spectral_radiance_convolved is optional (runs convolved with
    export_convolved_radiance = False carry only brightness_temperature);
    radiance is None in that case.
    """
    wvn = convolved_ds.coords["wavenumber"].values
    freq = convolved_ds.coords["frequency"].values
    idx = convolved_ds.coords["idx"].values
    if "spectral_radiance_convolved" in convolved_ds:
        radiance = convolved_ds["spectral_radiance_convolved"].values
    else:
        radiance = None
    bt = convolved_ds["brightness_temperature"].values
    return wvn, freq, idx, radiance, bt


def compute_differences(sim_radiance, sim_bt, obs_tb, freq):
    """
    Compute sim - obs differences for all three quantities.

    Invalid entries (obs fill values, non-finite sim values, non-positive
    observed radiance) are set to NaN.

    Returns
    -------
    d_rad, d_rel, d_bt : ndarray, shape (N_wvn, N_idx)
        Absolute radiance difference [W m^-2 Hz^-1 sr^-1],
        relative radiance difference [%], brightness temperature difference [K].
        d_rad and d_rel are None if sim_radiance is None (BT-only run).
    fields : dict
        The collocated simulation and observation fields themselves, masked
        with the same validity mask as the differences, so that the mean and
        standard deviation of each side can be accumulated alongside the
        difference statistics: {"bt": (sim, obs)} plus {"rad": (sim, obs)} for
        radiance runs.
    """
    with np.errstate(invalid="ignore"):
        obs_valid = (obs_tb > bt_valid_range[0]) & (obs_tb < bt_valid_range[1])

    # obs radiance from the observed brightness temperature
    obs_tb_safe = np.where(obs_valid, obs_tb, np.nan)

    if sim_radiance is None:
        valid = obs_valid & np.isfinite(sim_bt)
        d_bt = sim_bt - obs_tb_safe
        d_bt = np.where(~valid, np.nan, d_bt)
        fields = {"bt": (np.where(valid, sim_bt, np.nan),
                         np.where(valid, obs_tb_safe, np.nan))}
        return None, None, d_bt, fields

    obs_radiance = planck_frequency(freq, obs_tb_safe)

    valid = obs_valid & np.isfinite(sim_radiance) & np.isfinite(sim_bt)

    with np.errstate(invalid="ignore", divide="ignore"):
        d_rad = sim_radiance - obs_radiance
        d_rel = 100.0 * d_rad / obs_radiance
        d_bt = sim_bt - obs_tb_safe

    bad = ~valid | ~np.isfinite(d_rad) | ~np.isfinite(d_rel)
    d_rad = np.where(bad, np.nan, d_rad)
    d_rel = np.where(bad, np.nan, d_rel)
    d_bt = np.where(bad, np.nan, d_bt)

    valid_bt = valid & np.isfinite(d_bt)
    valid_rad = valid & np.isfinite(d_rad)
    fields = {
        "bt": (np.where(valid_bt, sim_bt, np.nan),
               np.where(valid_bt, obs_tb_safe, np.nan)),
        "rad": (np.where(valid_rad, sim_radiance, np.nan),
                np.where(valid_rad, obs_radiance, np.nan)),
    }

    return d_rad, d_rel, d_bt, fields


def accumulate_stats(counts, sums, sum_sqs, d):
    """Accumulate running per-wavenumber sum / sum of squares / counts."""
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d)
        d_clean = np.where(valid, d, 0.0)
    counts += valid.sum(axis=1)
    sums += d_clean.sum(axis=1)
    sum_sqs += (d_clean**2).sum(axis=1)


def accumulate_field_stats(counts, sums, sum_sqs, values):
    """
    Accumulate the running statistics of one side of the comparison.

    Same running sums as accumulate_stats, but applied to the simulation or
    observation field itself (already masked to NaN for the invalid pairs) so
    that the mean spectrum and the across-column standard deviation of each
    side can be reported next to the difference statistics.
    """
    accumulate_stats(counts, sums, sum_sqs, values)


def finalize_stats(counts, sums, sum_sqs):
    """Mean, standard deviation and RMSE from the running accumulators."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
        var = sum_sqs / counts - mean**2
        std = np.sqrt(np.maximum(var, 0.0))
        rmse = np.sqrt(sum_sqs / counts)
    return mean, std, rmse


def accumulate_histogram(hist, edges, d):
    """
    Accumulate per-wavenumber histogram counts (vectorised).

    Values outside [edges[0], edges[-1]] are ignored; NaN values are ignored.
    hist : ndarray, shape (N_wvn, N_bins)
    edges : ndarray, shape (N_bins + 1,)
    """
    n_wvn, n_bins = hist.shape
    lo, hi = edges[0], edges[-1]

    inside = np.isfinite(d) & (d >= lo) & (d <= hi)
    if not inside.any():
        return

    d_inside = d[inside]
    row = np.broadcast_to(np.arange(n_wvn)[:, None], d.shape)[inside]

    # bin index: value == hi goes into the last bin
    bin_idx = np.searchsorted(edges, d_inside, side="right") - 1
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

    flat = row * n_bins + bin_idx
    hist += np.bincount(flat, minlength=n_wvn * n_bins).reshape(n_wvn, n_bins)


def normalize_histogram(hist, bin_width):
    """Normalise each wavenumber column to a probability density."""
    density = np.zeros_like(hist)
    with np.errstate(invalid="ignore", divide="ignore"):
        totals = hist.sum(axis=1)
        valid = totals > 0
        density[valid] = hist[valid] / (totals[valid, None] * bin_width)
    return density


# =============================================================================
# %% plotting
# =============================================================================

def annotate_spectral_features(ax, shade_window=True, label=True,
                               headroom=0.06):
    """
    Shade the atmospheric window and mark the major absorption features.

    Purely decorative context for the IASI range: a light band over
    window_range_cm and dotted markers at spectral_features, with the feature
    labels placed inside the top of the axes (two staggered rows, so that
    close-lying features like N2O / CH4 stay readable) after adding a little
    headroom to the y limits.
    """
    if shade_window:
        ax.axvspan(*window_range_cm, color=palette["shade"], alpha=0.35,
                   linewidth=0.0, zorder=0)
        if label:
            ax.annotate("window",
                        xy=(0.5 * sum(window_range_cm), 0.015),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7,
                        color="0.4", style="italic")
    if not label:
        return

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + headroom * (ymax - ymin))
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.015 * (ymax - ymin)
    x_lo, x_hi = ax.get_xlim()
    margin = 0.02 * (x_hi - x_lo)
    for k, (wvn0, feat_label) in enumerate(spectral_features):
        ax.axvline(wvn0, color=palette["feature"], linewidth=0.5,
                   linestyle=(0, (1, 3)), zorder=1)
        # keep labels of edge features inside the axes
        if wvn0 < x_lo + margin:
            ha = "left"
        elif wvn0 > x_hi - margin:
            ha = "right"
        else:
            ha = "center"
        ax.annotate(feat_label, xy=(wvn0, y_text), xycoords="data",
                    xytext=(0, -9 * (k % 2)), textcoords="offset points",
                    ha=ha, va="top", fontsize=6.5,
                    color=palette["feature"], zorder=4)


def plot_stats_vs_wvn(wvn, mean, std, rmse, symbol, ylabel_unit,
                      quantity_label, out_base, ylim=None):
    """
    Three stacked panels: mean difference, std and RMSE vs wavenumber.

    Labels use the "quantity / unit" notation (e.g. "T_B difference / K").
    """
    fig, spec = default_figure_gridspec(
        rows=3, columns=1, width_in_cm=21.0, height_in_cm=18.0, hspace=0.45
    )

    panels = [
        (mean, f"{symbol} / {ylabel_unit}", "mean"),
        (std, f"standard deviation of {symbol} / {ylabel_unit}", "level"),
        (rmse, f"root mean square of {symbol} / {ylabel_unit}", "level"),
    ]

    for k, (y, ylab, kind) in enumerate(panels):
        ax, font = default_plot_format(fig.add_subplot(spec[k, :]))
        if kind == "mean":
            # zero line plus a fill coloured by the sign of the difference
            ax.axhline(0.0, color="0.5", linewidth=0.6, zorder=2)
            ax.fill_between(wvn, 0.0, y, where=np.isfinite(y) & (y >= 0),
                            color=palette["sim"], alpha=0.18,
                            interpolate=True, zorder=1)
            ax.fill_between(wvn, 0.0, y, where=np.isfinite(y) & (y < 0),
                            color=palette["obs"], alpha=0.18,
                            interpolate=True, zorder=1)
            ax.plot(wvn, y, color=palette["diff"], linewidth=line_width,
                    zorder=3)
        else:
            ax.plot(wvn, y, color=palette["diff"], linewidth=line_width)
            avg = np.nanmean(y)
            ax.axhline(avg, color=palette["feature"], linewidth=0.6,
                       linestyle="--", zorder=3)
            ax.annotate(f"mean {avg:.2f} {plain_units[ylabel_unit]}",
                        xy=(0.985, avg), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points",
                        ha="right", fontsize=7, color=palette["feature"])
        ax.set_ylabel(ylab, fontproperties=font)
        if ylim is not None:
            ax.set_ylim(ylim)
        if k == 0:
            ax.set_title(quantity_label, fontproperties=font)
            annotate_spectral_features(ax, headroom=0.0 if ylim else 0.06)
        if k == len(panels) - 1:
            ax.set_xlabel("Wavenumber / cm$^{-1}$", fontproperties=font)
        ax.set_xlim(wvn.min(), wvn.max())

    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


def plot_mean_spectra(wvn, sim_mean, sim_std, obs_mean, obs_std, symbol,
                      ylabel_unit, quantity_label, out_base):
    """
    Mean simulation and observation spectra, each with its +-1 std band.

    Upper panel: the two mean spectra (across all compared columns) with the
    +-1 standard-deviation band of each scene-to-scene spread drawn around
    them.  Lower strip: the difference of the two mean spectra, which is the
    bias part of the mismatch (the difference statistics of
    plot_stats_vs_wvn also contain the column-to-column scatter).
    """
    fig, spec = default_figure_gridspec(
        rows=2, columns=1, width_in_cm=21.0, height_in_cm=14.5,
        constrained_layout=False, hspace=0.06, height_ratios=[2.7, 1.0],
    )
    fig.subplots_adjust(top=0.945, bottom=0.075, left=0.085, right=0.995)

    ax, font = default_plot_format(fig.add_subplot(spec[0, :]))
    for mean, std, color, side in (
        (obs_mean, obs_std, palette["obs"], "IASI observation"),
        (sim_mean, sim_std, palette["sim"], "simulation (SRF-convolved)"),
    ):
        ax.fill_between(wvn, mean - std, mean + std, color=color, alpha=0.16,
                        linewidth=0.0, zorder=1)
        ax.plot(wvn, mean - std, color=color, linewidth=0.35, alpha=0.5,
                zorder=2)
        ax.plot(wvn, mean + std, color=color, linewidth=0.35, alpha=0.5,
                zorder=2)
        ax.plot(wvn, mean, color=color, linewidth=1.1, zorder=3,
                label=f"{side} mean")
    annotate_spectral_features(ax)
    ax.set_ylabel(f"{symbol} / {ylabel_unit}", fontproperties=font)
    ax.set_title(f"{quantity_label}: mean over all compared columns "
                 f"(bands: \u00b11 std)", fontproperties=font)
    ax.legend(loc="lower right", handlelength=1.6)
    ax.set_xlim(wvn.min(), wvn.max())
    ax.set_xticklabels([])

    ax_diff, font = default_plot_format(fig.add_subplot(spec[1, :]))
    diff = sim_mean - obs_mean
    ax_diff.axhline(0.0, color="0.5", linewidth=0.6, zorder=2)
    ax_diff.fill_between(wvn, 0.0, diff, where=np.isfinite(diff) & (diff >= 0),
                         color=palette["sim"], alpha=0.25, interpolate=True)
    ax_diff.fill_between(wvn, 0.0, diff, where=np.isfinite(diff) & (diff < 0),
                         color=palette["obs"], alpha=0.25, interpolate=True)
    ax_diff.plot(wvn, diff, color=palette["diff"], linewidth=line_width,
                 zorder=3)
    ax_diff.set_ylabel(f"difference of means / {ylabel_unit}",
                       fontproperties=font)
    ax_diff.set_xlabel("Wavenumber / cm$^{-1}$", fontproperties=font)
    ax_diff.set_xlim(wvn.min(), wvn.max())

    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


def plot_histogram_pcolormesh(wvn, bin_centers, density, symbol, ylabel_unit,
                              quantity_label, out_base, mean=None, std=None):
    """
    Pcolormesh of the per-wavenumber difference distribution.

    Optionally overplots the mean difference (white line) and the +-1 std
    envelope (dotted white lines) from the aggregate statistics.
    """
    fig, spec = default_figure_gridspec(
        rows=1, columns=1, width_in_cm=21.0, height_in_cm=10.0
    )
    ax, font = default_plot_format(fig.add_subplot(spec[:, :]), grid_on=False)

    mesh = ax.pcolormesh(
        wvn, bin_centers, density.T,
        shading="auto", cmap="inferno",
        norm=LogNorm(vmin=max(np.nanmin(density[density > 0]), 1e-6)),
        rasterized=True,
    )

    if mean is not None:
        ax.plot(wvn, mean, color="white", linewidth=0.8, zorder=3)
    if std is not None:
        ax.plot(wvn, mean + std, color="white", linewidth=0.5,
                linestyle=(0, (1, 2)), zorder=3)
        ax.plot(wvn, mean - std, color="white", linewidth=0.5,
                linestyle=(0, (1, 2)), zorder=3)
        ax.annotate("solid: mean,  dotted: \u00b11 std",
                    xy=(0.5, 0.95), xycoords="axes fraction", ha="center",
                    fontsize=7, color="white", zorder=4,
                    path_effects=[patheffects.withStroke(linewidth=2.0,
                                                         foreground="0.15")])

    ax.set_xlabel("Wavenumber / cm$^{-1}$", fontproperties=font)
    ax.set_ylabel(f"{symbol} / {ylabel_unit}", fontproperties=font)
    ax.set_xlim(wvn.min(), wvn.max())
    ax.set_title(quantity_label, fontproperties=font)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(f"probability density / {inverse_units[ylabel_unit]}",
                   fontproperties=font)

    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


# =============================================================================
# %% comparison for one setup
# =============================================================================

def run_comparison(chunk_files, obs_Tb_values, wvn_obs, tag):
    """
    Run the streaming comparison for one simulation run.

    Chunk arrays are collocated to the observation channel grid; radiance
    quantities are only computed if the chunks carry spectral_radiance_convolved.

    Returns
    -------
    dict with the compared quantities (quantities), per-quantity
    mean/std/rmse/density arrays, histogram counts, bin edges, per-wavenumber
    valid column counts, and the per-quantity field statistics ("fields":
    mean / std of the simulation and of the observation themselves).
    """
    print("\n" + "=" * 70)
    print(f"Run {tag}: {len(chunk_files)} convolved chunk(s)")
    print("=" * 70)

    counts = {}
    sums = {}
    sum_sqs = {}
    field_counts = {}
    field_sums = {}
    field_sum_sqs = {}
    d_rad_samples = []
    quantities = None
    sel = None
    wvn_ref = None

    # Pass 1: running statistics + subsample for the radiance hist range
    print("  Pass 1: accumulating mean / std / RMSE statistics...")
    for ck, chunk_file in enumerate(chunk_files):
        ds = xr.open_dataset(chunk_file)
        wvn, freq, idx, sim_rad, sim_bt = extract_chunk_arrays(ds)
        ds.close()

        if quantities is None:
            # grid alignment / collocation is set from the first chunk
            wvn_ref = wvn
            sel = collocation_indices(wvn, wvn_obs, collocation_tol_cm)
            if np.array_equal(wvn[sel], wvn_obs):
                print("  Simulation grid matches the observation grid exactly.")
            else:
                print(f"  Collocating simulation grid ({wvn.size} points) to "
                      f"the observation grid ({wvn_obs.size} channels): "
                      f"max offset "
                      f"{np.abs(wvn[sel] - wvn_obs).max():.4f} cm^-1")
            quantities = ("rad", "rel", "bt") if sim_rad is not None else ("bt",)
            if sim_rad is None:
                print("  No spectral_radiance_convolved in the chunks: "
                      "comparing brightness temperature only.")
            for q in quantities:
                counts[q] = np.zeros(wvn_obs.size, dtype=np.int64)
                sums[q] = np.zeros(wvn_obs.size)
                sum_sqs[q] = np.zeros(wvn_obs.size)
                if q in ("bt", "rad"):
                    # mean / std of each side separately (mean-spectra figure)
                    for side in ("sim", "obs"):
                        key = f"{q}_{side}"
                        field_counts[key] = np.zeros(wvn_obs.size, dtype=np.int64)
                        field_sums[key] = np.zeros(wvn_obs.size)
                        field_sum_sqs[key] = np.zeros(wvn_obs.size)
        elif not np.array_equal(wvn, wvn_ref):
            raise ValueError(f"{tag}: wavenumber grid mismatch in "
                             f"{chunk_file.name}")

        wvn_c = wvn[sel]
        freq_c = freq[sel]
        if sim_rad is not None:
            sim_rad = sim_rad[sel, :]
        sim_bt = sim_bt[sel, :]

        obs_tb = obs_Tb_values[idx, :]  # (N_idx, N_wvn)
        d_rad, d_rel, d_bt, fields = compute_differences(
            sim_rad, sim_bt, obs_tb.T, freq_c
        )

        diffs = {"bt": d_bt}
        if d_rad is not None:
            diffs["rad"] = d_rad
            diffs["rel"] = d_rel
        for q in quantities:
            accumulate_stats(counts[q], sums[q], sum_sqs[q], diffs[q])
            if q in ("bt", "rad"):
                for side, values in zip(("sim", "obs"), fields[q]):
                    key = f"{q}_{side}"
                    accumulate_field_stats(field_counts[key], field_sums[key],
                                           field_sum_sqs[key], values)

        if d_rad is not None:
            d_rad_samples.append(
                d_rad[:, ::sample_column_stride].ravel().astype(np.float32)
            )

        print(f"    chunk {ck + 1}/{len(chunk_files)}: idx {idx.min()}–"
              f"{idx.max()}, {d_bt.shape[1]} columns")

    print(f"  Total columns compared: {counts['bt'].sum()}")

    # Histogram bin configuration
    if "rad" in quantities:
        d_rad_all = np.concatenate(d_rad_samples)
        d_rad_all = d_rad_all[np.isfinite(d_rad_all)]
        if abs_hist_range is None:
            p = np.percentile(np.abs(d_rad_all), abs_hist_percentile)
            rad_range = (-p, p)
            print(f"  Auto radiance histogram range (P{abs_hist_percentile:g} of "
                  f"|diff|): ±{p:.3e} W m^-2 Hz^-1 sr^-1")
        else:
            rad_range = abs_hist_range
        del d_rad_all, d_rad_samples
    else:
        rad_range = (0.0, 1.0)  # unused

    bin_configs = {
        "bt": np.linspace(bt_hist_range[0], bt_hist_range[1], hist_n_bins + 1),
    }
    if "rad" in quantities:
        bin_configs["rad"] = np.linspace(rad_range[0], rad_range[1],
                                         hist_n_bins + 1)
        bin_configs["rel"] = np.linspace(rel_hist_range[0], rel_hist_range[1],
                                         hist_n_bins + 1)
    hists = {q: np.zeros((wvn_obs.size, hist_n_bins), dtype=np.int64)
             for q in quantities}

    # Pass 2: per-wavenumber histogram accumulation
    print("  Pass 2: accumulating difference histograms...")
    for ck, chunk_file in enumerate(chunk_files):
        ds = xr.open_dataset(chunk_file)
        wvn, freq, idx, sim_rad, sim_bt = extract_chunk_arrays(ds)
        ds.close()

        freq_c = freq[sel]
        if sim_rad is not None:
            sim_rad = sim_rad[sel, :]
        sim_bt = sim_bt[sel, :]

        obs_tb = obs_Tb_values[idx, :]
        d_rad, d_rel, d_bt, _ = compute_differences(sim_rad, sim_bt,
                                                    obs_tb.T, freq_c)

        diffs = {"bt": d_bt}
        if d_rad is not None:
            diffs["rad"] = d_rad
            diffs["rel"] = d_rel
        for q in quantities:
            accumulate_histogram(hists[q], bin_configs[q], diffs[q])

        print(f"    chunk {ck + 1}/{len(chunk_files)} done")

    # Finalize statistics
    stats = {"tag": tag, "quantities": quantities, "bins": bin_configs,
             "hist": hists, "n_columns": counts["bt"], "fields": {}}
    for q in quantities:
        mean, std, rmse = finalize_stats(counts[q], sums[q], sum_sqs[q])
        stats[q] = {
            "mean": mean, "std": std, "rmse": rmse,
            "density": normalize_histogram(
                hists[q].astype(float), bin_configs[q][1] - bin_configs[q][0]
            ),
        }
    for key, f_counts in field_counts.items():
        mean, std, _ = finalize_stats(f_counts, field_sums[key],
                                      field_sum_sqs[key])
        stats["fields"][key] = {"mean": mean, "std": std, "n": f_counts}
    return stats


def export_stats(stats, wvn_obs, out_file):
    """Write the aggregate statistics of one run to NetCDF."""
    quantities = stats["quantities"]
    units = {"rad": "W m^-2 Hz^-1 sr^-1", "rel": "%", "bt": "K"}
    names = {"rad": "radiance", "rel": "radiance_rel", "bt": "bt"}

    data_vars = {}
    for q in quantities:
        nm = names[q]
        data_vars[f"diff_mean_{nm}"] = (["wavenumber"], stats[q]["mean"],
                                        {"units": units[q]})
        data_vars[f"diff_std_{nm}"] = (["wavenumber"], stats[q]["std"],
                                       {"units": units[q]})
        data_vars[f"diff_rmse_{nm}"] = (["wavenumber"], stats[q]["rmse"],
                                        {"units": units[q]})
        data_vars[f"hist_counts_{nm}"] = (["wavenumber", "diff_bin"],
                                          stats["hist"][q])
        data_vars[f"hist_bin_{nm}"] = (
            ["diff_bin"],
            0.5 * (stats["bins"][q][:-1] + stats["bins"][q][1:]),
            {"units": units[q]},
        )
    for key, field in stats.get("fields", {}).items():
        q, side = key.split("_")
        nm = names[q]
        data_vars[f"{side}_mean_{nm}"] = (
            ["wavenumber"], field["mean"],
            {"units": units[q],
             "description": f"mean over all valid columns of the {side}"},
        )
        data_vars[f"{side}_std_{nm}"] = (
            ["wavenumber"], field["std"],
            {"units": units[q],
             "description": f"standard deviation across columns of the {side} "
                            f"(scene variability, not an uncertainty of the mean)"},
        )
    data_vars["n_columns"] = (["wavenumber"], stats["n_columns"],
                              {"description": "valid columns per wavenumber"})

    out_ds = xr.Dataset(
        data_vars,
        coords={
            "wavenumber": ("wavenumber", wvn_obs, {"units": "cm^-1"}),
            "diff_bin": ("diff_bin", 0.5 * (stats["bins"]["bt"][:-1]
                                            + stats["bins"]["bt"][1:])),
        },
    )
    out_ds.attrs["difference"] = "simulation - observation"
    out_ds.attrs["setup"] = stats["tag"]
    out_ds.attrs["obs_file"] = satdata_file.name
    out_ds.to_netcdf(str(out_file))
    print(f"  Exported {out_file}")


def make_setup_plots(stats, wvn_obs):
    """
    Create the figures of one run: mean spectra, difference statistics,
    difference distributions (per compared quantity).
    """
    bin_centers = {q: 0.5 * (stats["bins"][q][:-1] + stats["bins"][q][1:])
                   for q in stats["quantities"]}
    tag = stats["tag"]
    fields = stats.get("fields", {})

    unit_rad = "W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$"

    # --- mean spectra: simulation vs observation, +-1 std bands ------------
    if "rad" in stats["quantities"] and "rad_sim" in fields:
        plot_mean_spectra(
            wvn_obs, fields["rad_sim"]["mean"], fields["rad_sim"]["std"],
            fields["rad_obs"]["mean"], fields["rad_obs"]["std"],
            "spectral radiance L$_\\nu$", unit_rad,
            "TOA spectral radiance",
            plot_folder / f"fig_mean_radiance_{tag}",
        )
    if "bt_sim" in fields:
        plot_mean_spectra(
            wvn_obs, fields["bt_sim"]["mean"], fields["bt_sim"]["std"],
            fields["bt_obs"]["mean"], fields["bt_obs"]["std"],
            "brightness temperature T$_B$", "K",
            "IASI brightness temperature",
            plot_folder / f"fig_mean_bt_{tag}",
        )

    # --- difference statistics ---------------------------------------------
    if "rad" in stats["quantities"]:
        plot_stats_vs_wvn(
            wvn_obs, stats["rad"]["mean"], stats["rad"]["std"],
            stats["rad"]["rmse"],
            "radiance difference", unit_rad,
            "Spectral radiance difference (absolute, sim − obs)",
            plot_folder / f"fig_radiance_abs_stats_{tag}",
        )
        plot_stats_vs_wvn(
            wvn_obs, stats["rel"]["mean"], stats["rel"]["std"],
            stats["rel"]["rmse"],
            "radiance difference", "%",
            "Spectral radiance difference (relative to observation)",
            plot_folder / f"fig_radiance_rel_stats_{tag}",
        )

    plot_stats_vs_wvn(
        wvn_obs, stats["bt"]["mean"], stats["bt"]["std"], stats["bt"]["rmse"],
        "T$_B$ difference", "K",
        "Brightness temperature difference (sim − obs)",
        plot_folder / f"fig_bt_stats_{tag}",
    )

    # --- difference distributions ------------------------------------------
    if "rad" in stats["quantities"]:
        plot_histogram_pcolormesh(
            wvn_obs, bin_centers["rad"], stats["rad"]["density"],
            "radiance difference", unit_rad,
            "Radiance difference distribution (absolute, sim − obs)",
            plot_folder / f"fig_radiance_abs_hist_{tag}",
            mean=stats["rad"]["mean"], std=stats["rad"]["std"],
        )
        plot_histogram_pcolormesh(
            wvn_obs, bin_centers["rel"], stats["rel"]["density"],
            "radiance difference", "%",
            "Radiance difference distribution (relative, sim − obs)",
            plot_folder / f"fig_radiance_rel_hist_{tag}",
            mean=stats["rel"]["mean"], std=stats["rel"]["std"],
        )

    plot_histogram_pcolormesh(
        wvn_obs, bin_centers["bt"], stats["bt"]["density"],
        "T$_B$ difference", "K",
        "Brightness temperature difference distribution (sim − obs)",
        plot_folder / f"fig_bt_hist_{tag}",
        mean=stats["bt"]["mean"], std=stats["bt"]["std"],
    )


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("Simulation vs. observation comparison (SRF-convolved products)")
    print("=" * 70)

    stats_folder.mkdir(parents=True, exist_ok=True)
    plot_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load observations
    # -------------------------------------------------------------------------
    print("\nLoading observation data...")
    satdata = xr.open_dataset(satdata_file)
    # NB: the spectral axis of T_b is mislabelled 'lev' in satdata.nc
    obs_Tb = satdata["T_b"].rename({"lev": "wvn_obs"})
    wvn_obs = satdata.coords["wvn"].values
    n_fill = int(((obs_Tb < bt_valid_range[0]) | (obs_Tb > bt_valid_range[1])).sum())
    print(f"  Observation shape: {obs_Tb.shape} (idx, wvn)")
    print(f"  Wavenumber range: {wvn_obs.min():.2f} to {wvn_obs.max():.2f} cm^-1")
    print(f"  Fill values masked (outside {bt_valid_range}): {n_fill}")
    obs_Tb_values = obs_Tb.values.astype(np.float32)  # (idx, wvn)

    # -------------------------------------------------------------------------
    # Discover and compare the convolved chunks of the selected run
    # -------------------------------------------------------------------------
    if not results_folder.is_dir():
        available = sorted(
            p.name for p in results_root.iterdir() if p.is_dir()
        ) if results_root.is_dir() else []
        raise FileNotFoundError(
            f"Run sub folder {results_folder} does not exist. "
            f"Set run_name to one of the available runs: "
            f"{', '.join(available) if available else '(none)'}"
        )
    chunk_files = discover_convolved_chunks(results_folder, run_name)
    if not chunk_files:
        raise FileNotFoundError(
            f"No convolved output files (Reference_convolved_* belonging to "
            f"'{run_name}') found in {results_folder}. "
            f"Run convolve_sensor_response.py first."
        )

    stats = run_comparison(chunk_files, obs_Tb_values, wvn_obs, run_name)

    if export_stats:
        export_stats(stats, wvn_obs,
                     stats_folder / f"comparison_stats_{run_name}.nc")

    if make_plots:
        make_setup_plots(stats, wvn_obs)
        print(f"  Figures written to {plot_folder}")

    satdata.close()

    print(f"\nTotal processing time: {time.time() - t_start:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
