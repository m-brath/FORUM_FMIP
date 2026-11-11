"""
plot_lmix_report.py -- report-grade figures for the CO2 line-mixing test
(co2_lmix_rosenkranz_test.py, FMIP column idx 12345).

Reads the final-resolution arms
    results/LMIX_TEST/co2_lmix_{none,ra0,ra1}_Nf4001_Nza20/
        Reference_radiance_TOA_*.nc
and adds the IASI context that the test script itself does not produce:
the SRF-convolved model channel radiances vs. the observed T_b from
data/satdata.nc (SRF operator from frtmip_aux_functions, exactly as in
convolve_sensor_response.py), so the model-minus-obs deficit in the
2398-2520 cm^-1 window can be shown for all three line-mixing arms on
the sensor grid.

Figures -> plot/lmix_test/:
    fig_report_lmix_spectra.png   monochromatic T_B, overview + band-edge zoom
    fig_report_lmix_deltas.png    T_B line-mixing increments, full + window zoom
    fig_report_channel_deficit.png  SRF-convolved T_B vs. IASI obs, the
                                  per-channel model-minus-obs deficit, and
                                  the line-mixing shift of that deficit

Run from the scripts/ directory:
    /Users/u242031/miniforge3/envs/pyarts/bin/python plot_lmix_report.py
"""

# =============================================================================
# imports
# =============================================================================

import pathlib

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyarts import arts

import frtmip_aux_functions as fax
import convolve_sensor_response as csr
import compare_convolved_with_obs as ccw

# =============================================================================
# %% constants  (edit here)
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
lmix_root = script_dir.parent / "results" / "LMIX_TEST"
plot_folder = script_dir.parent / "plot" / "lmix_test"

# final-resolution arms produced by co2_lmix_rosenkranz_test.py
arms = {
    "none": "co2_lmix_none_Nf4001_Nza20",
    "ra0": "co2_lmix_ra0_Nf4001_Nza20",
    "ra1": "co2_lmix_ra1_Nf4001_Nza20",
}

# spectral regions of interest [cm^-1]
deficit_window = (2398.0, 2520.0)     # the deficit to explain
lmix_cluster = (2381.0, 2398.0)       # dense CO2 nu3 R-branch line cluster
channel_plot_range = (2360.0, 2540.0) # x range of the channel panels

# colors for the three arms (red = production-style baseline, matching the
# existing fig_lmix_*.png; green/purple for the two ECS adaptation modes)
arm_colors = {
    "none": ccw.palette["sim"],       # #b3352b
    "ra0": "#2e7d32",
    "ra1": "#7e57c2",
}

# =============================================================================
# %% data loading
# =============================================================================

def load_arm(folder):
    """Read one arm's TOA nadir spectrum: wavenumber + spectral radiance."""
    ncs = sorted(lmix_root.joinpath(folder).glob("Reference_radiance_TOA_*.nc"))
    if len(ncs) != 1:
        raise FileNotFoundError(
            f"expected exactly one Reference_radiance_TOA_*.nc in {folder}, "
            f"found {len(ncs)}")
    ds = xr.open_dataset(ncs[0])
    wvn = ds["wavenumber"].values
    rad = ds["spectral_radiance_TOA"].values[:, 0]   # single column
    return wvn, rad


def main():
    ccw.apply_plot_style()
    plot_folder.mkdir(parents=True, exist_ok=True)

    print(f"loading arms: {', '.join(arms)}")
    wvn = None
    bt, dI = {}, {}
    for arm, folder in arms.items():
        w_a, rad_a = load_arm(folder)
        if wvn is None:
            wvn = w_a
        elif not np.allclose(w_a, wvn):
            raise ValueError(f"frequency grids of the arms differ ({arm})")
        # Planck inversion of the full-resolution spectrum -> monochromatic BT
        bt[arm] = fax.radiance2brightness_temp(rad_a, wvn)
        dI[arm] = rad_a

    # ---- statistics over the deficit window --------------------------------
    wsel = (wvn >= deficit_window[0]) & (wvn <= deficit_window[1])
    csel = (wvn >= lmix_cluster[0]) & (wvn < lmix_cluster[1])
    print(f"\nwindow statistics over {deficit_window[0]:.0f}-"
          f"{deficit_window[1]:.0f} cm^-1 ({int(wsel.sum())} points):")
    stats = {}
    for arm in ("ra0", "ra1"):
        d = bt[arm] - bt["none"]
        stats[arm] = dict(mean=float(d[wsel].mean()),
                          max=float(d[wsel].max()),
                          min=float(d[wsel].min()),
                          cluster_min=float(d[csel].min()),
                          cluster_max=float(d[csel].max()))
        print(f"  {arm} - none : mean {stats[arm]['mean']:+.3f} K, "
              f"range {stats[arm]['min']:+.3f} .. {stats[arm]['max']:+.3f} K | "
              f"cluster {stats[arm]['cluster_min']:+.2f} .. "
              f"{stats[arm]['cluster_max']:+.2f} K")

    # ========================================================================
    # SRF convolution onto the IASI channel grid (sensor-level comparison)
    # ========================================================================
    f_grid = arts.convert.kaycm2freq(wvn)
    sensor_response = xr.open_dataset(data_dir / "sensor_response.nc")
    srf_wvn, srf_response, _ = csr.validate_and_normalize_srf(
        sensor_response["wvn"].values, sensor_response["response"].values)

    satdata = xr.open_dataset(data_dir / "satdata.nc")
    channel_wvn = satdata["wvn"].values
    tb_obs = satdata["T_b"].isel(idx=12345).values

    operator, chan = fax.build_srf_channel_operator(
        f_grid, srf_wvn, srf_response, channel_wvn)

    # keep only channels whose full SRF support lies inside the simulated
    # window (partial-support edge channels would be renormalized -> wrong)
    # and whose observation is valid (satdata fill value -777)
    srf_lo, srf_hi = srf_wvn.min(), srf_wvn.max()
    covered = ((chan >= wvn.min() - srf_lo) & (chan <= wvn.max() - srf_hi))
    chan_obs_idx = np.searchsorted(channel_wvn, chan)   # chan is a subset of
    tb_chan_obs = tb_obs[chan_obs_idx]                  # the satdata grid
    keep = covered & (tb_chan_obs > 50.0) & (tb_chan_obs < 400.0)
    chan_k = chan[keep]
    tb_obs_k = tb_chan_obs[keep]
    print(f"channel panels: {int(keep.sum())} fully-covered, obs-valid "
          f"channels ({chan_k.min():.2f}-{chan_k.max():.2f} cm^-1)")

    tb_chan = {}
    for arm in arms:
        rad_c = fax.convolve_with_srf_to_channels(dI[arm], operator)[keep]
        tb_chan[arm] = fax.radiance2brightness_temp(rad_c, chan_k)

    dsel = (chan_k >= deficit_window[0]) & (chan_k <= deficit_window[1])
    print(f"\nchannel deficit (model - obs, fully-covered channels inside "
          f"the {deficit_window[0]:.0f}-{deficit_window[1]:.0f} cm^-1 window):")
    ch_stats = {}
    for arm in arms:
        dd = tb_chan[arm] - tb_obs_k
        ch_stats[arm] = float(dd[dsel].mean())
        print(f"  {arm:4s}: mean deficit {ch_stats[arm]:+.3f} K")
    print(f"  ra1 - none : {ch_stats['ra1'] - ch_stats['none']:+.3f} K "
          f"(line-mixing shift of the deficit)")
    print(f"  ra0 - none : {ch_stats['ra0'] - ch_stats['none']:+.3f} K")

    # ========================================================================
    # %% figure 1: monochromatic BT, overview + band-edge zoom
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.0, 6.4), height_ratios=[1.15, 1.0], sharex=False)

    ax1.axvspan(*lmix_cluster, color=ccw.palette["shade"], alpha=0.45, lw=0)
    ax1.axvspan(*deficit_window, color=ccw.palette["shade"], alpha=0.85, lw=0)
    for arm in arms:
        ax1.plot(wvn, bt[arm], color=arm_colors[arm], lw=ccw.line_width,
                 label=f"{arm}")
    ax1.set_xlim(wvn.min(), wvn.max())
    ax1.set_ylim(190, 292)
    ax1.set_ylabel("$T_B$ (monochromatic) / K")
    ax1.legend(loc="lower right", ncol=2)
    ax1.set_title("CO$_2$ ECS line mixing, FMIP column idx 12345: "
                  "monochromatic TOA $T_B$")

    zsel = (wvn >= 2378) & (wvn <= 2402)
    for arm in arms:
        ax2.plot(wvn[zsel], bt[arm][zsel], color=arm_colors[arm],
                 lw=1.0, label=arm)
    ax2.axvspan(*lmix_cluster, color=ccw.palette["shade"], alpha=0.45, lw=0)
    ax2.set_xlim(2378, 2402)
    ax2.set_ylabel("$T_B$ (monochromatic) / K")
    ax2.set_xlabel("Wavenumber / cm$^{-1}$")
    ax2.legend(loc="upper left")
    ax2.set_title("zoom: CO$_2$ $\\nu_3$ R-branch line cluster "
                  "(where line mixing acts)")

    fig.tight_layout()
    out = plot_folder / "fig_report_lmix_spectra.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    # ========================================================================
    # %% figure 2: line-mixing increments, full window + zoom inside window
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.0, 6.4), height_ratios=[1.15, 1.0])

    ax1.axvspan(*lmix_cluster, color=ccw.palette["shade"], alpha=0.45, lw=0)
    ax1.axvspan(*deficit_window, color=ccw.palette["shade"], alpha=0.85, lw=0)
    for arm, ls in (("ra0", "-"), ("ra1", "-")):
        ax1.plot(wvn, bt[arm] - bt["none"], color=arm_colors[arm],
                 lw=1.0, linestyle=ls, label=f"{arm} $-$ none")
    ax1.plot(wvn, bt["ra1"] - bt["ra0"], color=arm_colors["ra1"],
             lw=0.7, linestyle="--", alpha=0.7, label="ra1 $-$ ra0")
    ax1.axhline(0.0, color="black", lw=0.6)
    ax1.set_xlim(wvn.min(), wvn.max())
    ax1.set_ylabel("$T_B$ difference / K")
    ax1.legend(loc="lower left")
    ax1.set_title("$T_B$ (monochromatic) line-mixing increments, "
                  "FMIP column idx 12345")

    ax2.axvspan(*deficit_window, color=ccw.palette["shade"], alpha=0.85, lw=0)
    z2 = (wvn >= 2394) & (wvn <= 2524)
    for arm in ("ra0", "ra1"):
        ax2.plot(wvn[z2], (bt[arm] - bt["none"])[z2],
                 color=arm_colors[arm], lw=0.8, label=f"{arm} $-$ none")
        ax2.axhline(stats[arm]["mean"], color=arm_colors[arm], lw=0.8,
                    linestyle=":",
                    label=f"window mean {stats[arm]['mean']:+.3f} K")
    ax2.axhline(0.0, color="black", lw=0.6)
    ax2.set_xlim(2394, 2524)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_xlabel("Wavenumber / cm$^{-1}$")
    ax2.set_ylabel("$T_B$ difference / K")
    ax2.legend(loc="upper right")
    ax2.set_title("zoom: increments INSIDE the deficit window "
                  "(note the 15x finer $\\Delta T_B$ scale)")

    fig.tight_layout()
    out = plot_folder / "fig_report_lmix_deltas.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    # ========================================================================
    # %% figure 3: sensor-level channel comparison, deficit, lmix shift
    # ========================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(9.0, 8.6), height_ratios=[1.0, 1.0, 0.7], sharex=True)

    ax1.axvspan(*deficit_window, color=ccw.palette["shade"], alpha=0.85, lw=0)
    for arm in ("ra0", "ra1"):
        ax1.plot(chan_k, tb_chan[arm], color=arm_colors[arm],
                 lw=0.6, alpha=0.8, label=f"model {arm}")
    ax1.plot(chan_k, tb_chan["none"], color=arm_colors["none"], lw=1.1,
             label="model none")
    ax1.plot(chan_k, tb_obs_k, color=ccw.palette["obs"], lw=0.9, alpha=0.9,
             label="IASI obs $T_B$")
    ax1.set_xlim(*channel_plot_range)
    ax1.set_ylabel("$T_B$ / K")
    ax1.legend(loc="lower right", ncol=2)
    ax1.set_title("IASI channels, FMIP column idx 12345: SRF-convolved "
                  "model vs. observation")

    ax2.axvspan(*deficit_window, color=ccw.palette["shade"], alpha=0.85, lw=0)
    dd_none = tb_chan["none"] - tb_obs_k
    ax2.plot(chan_k, dd_none, color=arm_colors["none"], lw=0.9,
             label="model none $-$ obs")
    ax2.axhline(ch_stats["none"], color=arm_colors["none"], lw=0.9,
                linestyle=":",
                label=f"window mean {ch_stats['none']:+.2f} K")
    ax2.axhline(0.0, color="black", lw=0.6)
    ax2.set_ylim(-1.5, 6.0)
    ax2.set_ylabel("model $-$ obs / K")
    ax2.legend(loc="upper right")
    ax2.set_title("the deficit: $\\approx$ +1.5 K mean over the window, "
                  "up to +4.5 K at the band edge")

    ax3.axvspan(*deficit_window, color=ccw.palette["shade"], alpha=0.85, lw=0)
    for arm in ("ra0", "ra1"):
        dd = tb_chan[arm] - tb_chan["none"]
        ax3.plot(chan_k, dd, color=arm_colors[arm], lw=0.9,
                 label=f"{arm} $-$ none")
        ax3.axhline(ch_stats[arm] - ch_stats["none"], color=arm_colors[arm],
                    lw=0.8, linestyle=":",
                    label=f"window mean shift "
                          f"{ch_stats[arm] - ch_stats['none']:+.3f} K")
    ax3.axhline(0.0, color="black", lw=0.6)
    ax3.set_ylim(-0.7, 0.7)
    ax3.set_xlabel("Wavenumber / cm$^{-1}$")
    ax3.set_ylabel("$\\Delta T_B$ shift / K")
    ax3.legend(loc="upper right", ncol=2)
    ax3.set_title("line-mixing shift of the deficit: "
                  "$\\leq$0.1 K inside the window (fine scale)")

    fig.tight_layout()
    out = plot_folder / "fig_report_channel_deficit.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# =============================================================================
# %% main
# =============================================================================

if __name__ == "__main__":
    main()
