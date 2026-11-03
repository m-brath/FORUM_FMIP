#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnosis of the simulated-vs-IASI T_B bias at the CO2 nu3 (4.3 micron) band edge.

Three stacked panels over 2200-2560 cm-1:

A  Mean nadir T_B spectra, simulation (sensor-convolved) vs IASI observations.
B  T_B difference (sim - obs) with +/-1 std band over all scenes; shades mark
   the negative lobe (sim too cold, 2380-2396 cm-1) and the smooth positive
   deficit (sim too warm, 2398-2520 cm-1).
C  CO2-626 line-strength census of the ARTS catalog (HITRAN2020 compilation)
   in 20 cm-1 bins, total and restricted to cold lower states (E_low < 900 K),
   with the strength density required to close the smooth T_B deficit from
   panel B (grey dashed line).

Data sources:
  results/COMPARISON/comparison_stats_<run>.nc   (mean/std spectra)
  plot/lut_crosssections/_co2_line_lists.npz     (parsed catalog line lists)

Output: plot/comparison/fig_co2_nu3_wing_analysis.png
"""

# === Imports and run constants =================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import compare_convolved_with_obs as ccow

run_name = "fmip_radiance_batch_slurm_Nf108951_Nza20"
stats_file = f"../results/COMPARISON/comparison_stats_{run_name}.nc"
lines_file = "../plot/lut_crosssections/_co2_line_lists.npz"
fig_file = "../plot/comparison/fig_co2_nu3_wing_analysis.png"

wvn_min, wvn_max = 2200.0, 2560.0

# Band-edge lobes identified in the fine difference profile (cm-1)
lobe_range = (2380.0, 2396.0)
deficit_range = (2398.0, 2520.0)

# CO2 line-strength census bins (cm-1)
census_edges = np.arange(2320.0, 2561.0, 20.0)

# Column estimate for the "required strength" line:
# a smooth extra band-mean optical depth dtau needs summed line strength
#   dS/dwvn = dtau * (c[Hz/cm-1]) / N_CO2      [m2 Hz / molec / cm-1]
# with N_CO2 = VMR * dry molecular column.
dtau_required = 0.12          # gray-layer estimate, see analysis note
N_dry = 2.15e29               # molecules / m2 (p_s / g scaled)
co2_vmr = 4.0e-4              # mean VMR in data/atmdata.nc
S_needed_per_cm = dtau_required * 2.99792458e10 / (co2_vmr * N_dry)

# === Load data =================================================================

ds = xr.open_dataset(stats_file)
w = ds["wavenumber"].values
dT = ds["diff_mean_bt"].values
sT = ds["diff_std_bt"].values
sim_bt = ds["sim_mean_bt"].values
obs_bt = ds["obs_mean_bt"].values

lines = np.load(lines_file)
w626, i626, e626 = (lines["626"].reshape(3, -1)
                    if lines["626"].ndim == 2 else
                    (lines["626"][0], lines["626"][1], lines["626"][2]))

# Line-strength census in 20 cm-1 bins
cen_all = np.zeros(len(census_edges) - 1)
cen_cold = np.zeros(len(census_edges) - 1)
for k, (lo, hi) in enumerate(zip(census_edges[:-1], census_edges[1:])):
    m = (w626 >= lo) & (w626 < hi)
    cen_all[k] = i626[m].sum()
    cen_cold[k] = i626[m & (e626 < 900)].sum()

# === Figure ====================================================================

ccow.apply_plot_style()
fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.4))

ax = axes[0]
sel = (w >= wvn_min) & (w <= wvn_max)
ax.plot(w[sel], sim_bt[sel], color=ccow.palette["sim"], lw=ccow.line_width,
        label="simulation")
ax.plot(w[sel], obs_bt[sel], color=ccow.palette["obs"], lw=ccow.line_width,
        ls="--", label="IASI obs")
ax.axvspan(*deficit_range, color="0.85", zorder=0)
ax.annotate(r"CO$_2$ $\nu_3$ band origin" + "\n" + r"2349.2 cm$^{-1}$",
            xy=(2349.2, 234), xytext=(2245, 258),
            arrowprops=dict(arrowstyle="->", color="0.4"), fontsize=8,
            color="0.35")
ax.set_ylabel("Mean $T_B$ / K")
ax.set_xlim(wvn_min, wvn_max)
ax.legend(loc="upper left")

ax = axes[1]
ax.plot(w[sel], dT[sel], color=ccow.palette["diff"], lw=ccow.line_width)
ax.fill_between(w[sel], dT[sel] - sT[sel], dT[sel] + sT[sel],
                color=ccow.palette["diff"], alpha=0.25,
                label=r"$\pm$1 std over scenes")
ax.axhline(0.0, color="0.5", lw=0.8)
ax.axvspan(*lobe_range, color="0.85", zorder=0)
ax.axvspan(*deficit_range, color="0.85", zorder=0)
ax.annotate("sim too cold", xy=(2388, -9.5), ha="center", fontsize=8,
            color="0.3")
ax.annotate("smooth opacity deficit\n(sim too warm)", xy=(2459, 3.2),
            ha="center", fontsize=8, color="0.3")
ax.set_ylabel("$T_B$ difference / K")
ax.set_xlabel("Wavenumber / cm$^{-1}$")
ax.set_xlim(wvn_min, wvn_max)
ax.set_ylim(-12, 6.5)
ax.legend(loc="lower right")

ax = axes[2]
xc = 0.5 * (census_edges[:-1] + census_edges[1:])
ax.bar(xc, cen_all, width=18, color=ccow.palette["sim"], alpha=0.85,
       label=r"$\Sigma S$, catalog total")
ax.bar(xc, cen_cold, width=8, color="0.3", alpha=0.9,
       label=r"$\Sigma S$, $E_{low}$ < 900 K")
ax.axhline(S_needed_per_cm * 20, color="0.4", ls="--", lw=1.2,
           label=(f"needed if $\\Delta\\tau$={dtau_required} were\n"
                  "line strength in this bin"))
ax.set_yscale("log")
ax.set_ylabel(r"CO$_2$-626 $\Sigma S$ / m$^2$ Hz (20 cm$^{-1}$ bins)")
ax.set_xlabel("Wavenumber / cm$^{-1}$")
ax.set_xlim(wvn_min, wvn_max)
ax.legend(loc="upper right", fontsize=8)

for a in axes:
    a.grid(alpha=0.3)

fig.suptitle(r"CO$_2$ $\nu_3$ 4.3 $\mu$m band edge: sim$-$obs bias vs catalog "
             r"line-strength census", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(fig_file, dpi=200)
print(f"wrote {fig_file}")

print(f"required band-mean line-strength density: "
      f"{S_needed_per_cm:.2e} m2 Hz / molec / cm-1")
for k, (lo, hi) in enumerate(zip(census_edges[:-1], census_edges[1:])):
    print(f"  {lo:.0f}-{hi:.0f}: total {cen_all[k]:.2e}  cold {cen_cold[k]:.2e}")
