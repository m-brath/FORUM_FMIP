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
y = difference bin, colour = normalised frequency / probability density).

Two simulation setups are compared independently (one output set each):
    - Nf42301  : Reference_convolved_Nf42301_Nza20_Naa1_chunk*.nc
    - Nf109000 : Reference_convolved_Nf109000_Nza20_fmip_radiance_batch_chunk*.nc
Both convolve to the same 8461-channel observation grid (645-2760 cm^-1,
0.25 cm^-1 spacing) and carry the global column index (idx 0-19705), so the
comparison is an exact collocation - no interpolation needed.

Environment (mamba/conda):    pyarts
                              (same as convolve_sensor_response.py)

Input data:
    data/satdata.nc                                   (observed T_b)
    results/RADIANCE/**/Reference_convolved_Nf42301_*_chunk*.nc
    results/RADIANCE/**/Reference_convolved_Nf109000_*_chunk*.nc
    (the setup patterns are searched recursively, so the per-run sub folders
    results/RADIANCE/<run_name>/ are included)

Output:
    results/COMPARISON/comparison_stats_<setup>.nc    aggregate statistics
    plot/comparison/fig_<...>_<setup>.png / .pdf      figures

Example
-------
Simply run the script; it discovers all convolved chunks automatically.

    $ python compare_convolved_with_obs.py

Notes
-----
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

import numpy as np
import xarray as xr
import scipy.constants as scc

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

# Location of this script
script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
results_folder = script_dir.parent / "results" / "RADIANCE"
stats_folder = script_dir.parent / "results" / "COMPARISON"
plot_folder = script_dir.parent / "plot" / "comparison"

# Input data files
satdata_file = data_dir / "satdata.nc"

# Simulation setups to compare (file pattern within results_folder, tag used
# in output file names). The convolved products of both setups live on the
# observation channel grid.
setups = [   
    {"tag": "Nf109000",
     "pattern": "Reference_convolved_Nf109000_Nza20_fmip_radiance_batch_chunk*.nc"},
    {"tag": "Nf42301",
     "pattern": "Reference_convolved_Nf42301_Nza20_Naa1_chunk*.nc"}
]

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

# =============================================================================
# %% plot formatting (adapted from opacity_helper_function.py)
# =============================================================================

def default_figure_gridspec(rows=1, columns=1, width_in_cm=21.0,
                            height_in_cm=15.0, constrained_layout=True):
    """Create a figure with cm-based size and a gridspec."""
    fig = plt.figure(constrained_layout=constrained_layout)
    fig.set_size_inches(width_in_cm / 2.54, height_in_cm / 2.54)
    spec = gridspec.GridSpec(ncols=columns, nrows=rows, figure=fig, hspace=0.35)

    return fig, spec


def default_plot_format(ax, font_name=None, grid_on=True):
    """Apply consistent formatting to matplotlib axes."""
    font = FontProperties()
    if font_name is not None:
        font.set_name(font_name)

    if grid_on:
        ax.grid(which='both', linestyle=':', linewidth=0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axes.tick_params(direction='in', which='both')

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


def discover_convolved_chunks(results_folder, pattern):
    """
    Sorted list of convolved chunk NetCDF files matching pattern.

    The search is recursive, so the per-run sub folders of results/RADIANCE/
    (results/RADIANCE/<run_name>/) are included.
    """
    files = sorted(glob.glob(str(results_folder / "**" / pattern),
                             recursive=True))
    return [pathlib.Path(f) for f in files]


def extract_chunk_arrays(convolved_ds):
    """Load the (frequency, idx) arrays of one convolved chunk."""
    wvn = convolved_ds.coords["wavenumber"].values
    freq = convolved_ds.coords["frequency"].values
    idx = convolved_ds.coords["idx"].values
    radiance = convolved_ds["spectral_radiance_convolved"].values
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
    """
    with np.errstate(invalid="ignore"):
        obs_valid = (obs_tb > bt_valid_range[0]) & (obs_tb < bt_valid_range[1])

    # obs radiance from the observed brightness temperature
    obs_tb_safe = np.where(obs_valid, obs_tb, np.nan)
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

    return d_rad, d_rel, d_bt


def accumulate_stats(counts, sums, sum_sqs, d):
    """Accumulate running per-wavenumber sum / sum of squares / counts."""
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d)
        d_clean = np.where(valid, d, 0.0)
    counts += valid.sum(axis=1)
    sums += d_clean.sum(axis=1)
    sum_sqs += (d_clean**2).sum(axis=1)


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

def plot_stats_vs_wvn(wvn, mean, std, rmse, ylabel_unit, quantity_label,
                      out_base, ylim=None):
    """Three stacked panels: mean difference, std, RMSE vs wavenumber."""
    fig, spec = default_figure_gridspec(
        rows=3, columns=1, width_in_cm=21.0, height_in_cm=18.0
    )

    panels = [
        (mean, f"Mean difference (sim − obs)\n[{ylabel_unit}]"),
        (std, f"Std. of difference\n[{ylabel_unit}]"),
        (rmse, f"RMSE\n[{ylabel_unit}]"),
    ]

    for k, (y, ylab) in enumerate(panels):
        ax, font = default_plot_format(fig.add_subplot(spec[k, :]))
        ax.plot(wvn, y, color="0.2", linewidth=0.6)
        ax.set_ylabel(ylab, fontproperties=font)
        if ylim is not None:
            ax.set_ylim(ylim)
        if k == 0:
            ax.set_title(quantity_label, fontproperties=font)
        if k == len(panels) - 1:
            ax.set_xlabel("Wavenumber [cm$^{-1}$]", fontproperties=font)
        ax.set_xlim(wvn.min(), wvn.max())

    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


def plot_histogram_pcolormesh(wvn, bin_centers, density, ylabel, quantity_label,
                              out_base):
    """Pcolormesh of the per-wavenumber difference distribution."""
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

    ax.set_xlabel("Wavenumber [cm$^{-1}$]", fontproperties=font)
    ax.set_ylabel(ylabel, fontproperties=font)
    ax.set_xlim(wvn.min(), wvn.max())
    ax.set_title(quantity_label, fontproperties=font)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Normalised frequency [density]", fontproperties=font)

    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


# =============================================================================
# %% comparison for one setup
# =============================================================================

def run_comparison(chunk_files, obs_Tb_values, wvn_obs, tag):
    """
    Run the streaming comparison for one simulation setup.

    Returns
    -------
    dict with per-quantity mean/std/rmse/density arrays, histogram counts,
    bin edges, and per-wavenumber valid column counts.
    """
    print("\n" + "=" * 70)
    print(f"Setup {tag}: {len(chunk_files)} convolved chunk(s)")
    print("=" * 70)

    quantities = ("rad", "rel", "bt")
    counts = {q: None for q in quantities}
    sums = {q: None for q in quantities}
    sum_sqs = {q: None for q in quantities}
    d_rad_samples = []
    n_wvn = None

    # Pass 1: running statistics + subsample for the radiance hist range
    print("  Pass 1: accumulating mean / std / RMSE statistics...")
    for ck, chunk_file in enumerate(chunk_files):
        ds = xr.open_dataset(chunk_file)
        wvn, freq, idx, sim_rad, sim_bt = extract_chunk_arrays(ds)
        ds.close()

        if n_wvn is None:
            n_wvn = wvn.size
            if not np.allclose(wvn, wvn_obs):
                raise ValueError(
                    f"{tag}: simulation and observation wavenumber grids do "
                    f"not match."
                )
            for q in quantities:
                counts[q] = np.zeros(n_wvn, dtype=np.int64)
                sums[q] = np.zeros(n_wvn)
                sum_sqs[q] = np.zeros(n_wvn)
        elif not np.array_equal(wvn, wvn_obs):
            raise ValueError(f"{tag}: wavenumber grid mismatch in "
                             f"{chunk_file.name}")

        obs_tb = obs_Tb_values[idx, :]  # (N_idx, N_wvn)
        d_rad, d_rel, d_bt = compute_differences(sim_rad, sim_bt, obs_tb.T, freq)

        for q, d in zip(quantities, (d_rad, d_rel, d_bt)):
            accumulate_stats(counts[q], sums[q], sum_sqs[q], d)

        d_rad_samples.append(
            d_rad[:, ::sample_column_stride].ravel().astype(np.float32)
        )

        print(f"    chunk {ck + 1}/{len(chunk_files)}: idx {idx.min()}–"
              f"{idx.max()}, {d_rad.shape[1]} columns")

    print(f"  Total columns compared: {counts['rad'].sum()}")

    # Histogram bin configuration
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

    bin_configs = {
        "rad": np.linspace(rad_range[0], rad_range[1], hist_n_bins + 1),
        "rel": np.linspace(rel_hist_range[0], rel_hist_range[1], hist_n_bins + 1),
        "bt": np.linspace(bt_hist_range[0], bt_hist_range[1], hist_n_bins + 1),
    }
    hists = {q: np.zeros((n_wvn, hist_n_bins), dtype=np.int64)
             for q in quantities}

    # Pass 2: per-wavenumber histogram accumulation
    print("  Pass 2: accumulating difference histograms...")
    for ck, chunk_file in enumerate(chunk_files):
        ds = xr.open_dataset(chunk_file)
        wvn, freq, idx, sim_rad, sim_bt = extract_chunk_arrays(ds)
        ds.close()

        obs_tb = obs_Tb_values[idx, :]
        d_rad, d_rel, d_bt = compute_differences(sim_rad, sim_bt, obs_tb.T, freq)

        for q, d in zip(quantities, (d_rad, d_rel, d_bt)):
            accumulate_histogram(hists[q], bin_configs[q], d)

        print(f"    chunk {ck + 1}/{len(chunk_files)} done")

    # Finalize statistics
    stats = {"tag": tag, "bins": bin_configs, "hist": hists,
             "n_columns": counts["bt"]}
    for q in quantities:
        mean, std, rmse = finalize_stats(counts[q], sums[q], sum_sqs[q])
        stats[q] = {
            "mean": mean, "std": std, "rmse": rmse,
            "density": normalize_histogram(
                hists[q].astype(float), bin_configs[q][1] - bin_configs[q][0]
            ),
        }
    return stats


def export_stats(stats, wvn_obs, out_file):
    """Write the aggregate statistics of one setup to NetCDF."""
    quantities = ("rad", "rel", "bt")
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
    """Create the stats and histogram figures of one setup."""
    bin_centers = {q: 0.5 * (stats["bins"][q][:-1] + stats["bins"][q][1:])
                   for q in ("rad", "rel", "bt")}
    tag = stats["tag"]

    plot_stats_vs_wvn(
        wvn_obs, stats["rad"]["mean"], stats["rad"]["std"], stats["rad"]["rmse"],
        "W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$",
        "Spectral radiance difference (absolute)",
        plot_folder / f"fig_radiance_abs_stats_{tag}",
    )
    plot_stats_vs_wvn(
        wvn_obs, stats["rel"]["mean"], stats["rel"]["std"], stats["rel"]["rmse"],
        "%",
        "Spectral radiance difference (relative to observation)",
        plot_folder / f"fig_radiance_rel_stats_{tag}",
    )
    plot_stats_vs_wvn(
        wvn_obs, stats["bt"]["mean"], stats["bt"]["std"], stats["bt"]["rmse"],
        "K",
        "Brightness temperature difference",
        plot_folder / f"fig_bt_stats_{tag}",
    )

    plot_histogram_pcolormesh(
        wvn_obs, bin_centers["rad"], stats["rad"]["density"],
        "Difference [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]",
        "Radiance difference distribution (absolute, sim − obs)",
        plot_folder / f"fig_radiance_abs_hist_{tag}",
    )
    plot_histogram_pcolormesh(
        wvn_obs, bin_centers["rel"], stats["rel"]["density"],
        "Difference [%]",
        "Radiance difference distribution (relative, sim − obs)",
        plot_folder / f"fig_radiance_rel_hist_{tag}",
    )
    plot_histogram_pcolormesh(
        wvn_obs, bin_centers["bt"], stats["bt"]["density"],
        "Difference [K]",
        "Brightness temperature difference distribution (sim − obs)",
        plot_folder / f"fig_bt_hist_{tag}",
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
    # Compare each setup separately
    # -------------------------------------------------------------------------
    for setup in setups:
        chunk_files = discover_convolved_chunks(results_folder,
                                                setup["pattern"])
        if not chunk_files:
            print(f"\nWARNING: no convolved files matching "
                  f"'{setup['pattern']}' in {results_folder} - skipped.")
            continue

        stats = run_comparison(chunk_files, obs_Tb_values, wvn_obs,
                               setup["tag"])

        if export_stats:
            export_stats(stats, wvn_obs,
                         stats_folder / f"comparison_stats_{setup['tag']}.nc")

        if make_plots:
            make_setup_plots(stats, wvn_obs)
            print(f"  Figures written to {plot_folder}")

    satdata.close()

    print(f"\nTotal processing time: {time.time() - t_start:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
