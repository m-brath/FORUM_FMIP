#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity of the CO2 nu3 wing opacity to the line-wing cutoff.

The radiance benchmark (and FluxSimulator's default) truncates every line at
750 GHz = 25.01 cm^-1 (abs_lines_per_speciesCutoff ByLine).  In the deficit
region 2398-2520 cm^-1 this means that the strong cold lines of the nu3 R
branch (2340-2390 cm^-1, where the catalog is complete) can only contribute
their far wings up to ~2415 cm^-1 -- everything beyond is exactly zero.  The
line shape there is a plain Voigt with simple frequency scaling of the far
wings (SFS, see the CO2 catalog header); real CO2 air-broadened wings are known
to be *super*-Lorentzian, i.e. a Voigt+SFS model underestimates absorption at
tens of cm^-1 from the line centre.

This probe quantifies the *truncation* part of that argument for ONE column:
it recomputes the line-resolved cross sections and the total column opacity
for several cutoffs and prints the opacity of the whole column at the deficit
wavenumbers.  Everything else (species groups, frequency grid, lookup pressure
grid, profile) is kept identical to plot_nu3_wing_opacity_and_crosssections.py,
which is imported for the machinery.

Interpretation guide: tau(TOA -> surface) in this region is far below 1, so the
simulated T_B is essentially surface emission attenuated by exp(-tau); a missing
opacity of d(tau) ~ 0.1 near 2405 cm^-1 corresponds to the ~4 K sim-minus-obs
deficit (see results/COMPARISON/analysis_co2_nu3_wing_2026-11-03.md, sec. 4).

Run from the scripts/ directory (mamba env pyarts):
    $ python probe_co2_wing_cutoff_sensitivity.py
"""

import time
import numpy as np

import plot_nu3_wing_opacity_and_crosssections as pnw


# =============================================================================
# %% run constants  (edit here)
# =============================================================================

# line-wing cutoffs to compare (Hz).  750 GHz == 25.01 cm^-1 is the benchmark
# value; ByLine truncation is |f - f0| > value, so 12 THz (400 cm^-1) covers
# essentially the whole nu3 band of the cold lines.
cutoffs_hz = [750e9, 1500e9, 3000e9, 12000e9]

# wavenumbers whose total column opacity is reported (cm^-1)
report_wvn = [2398.0, 2405.0, 2410.0, 2420.0, 2440.0, 2460.0, 2480.0,
              2500.0, 2520.0]

# opacity that the observations demand in addition to the benchmark value
# (analysis note sec. 4, gray-layer estimate at 2405 cm^-1)
tau_required = 0.12

# cheap settings for the probe -- coarse grid / coarse node spacing is enough to
# resolve the smooth band-mean wing level that matters here
probe_dwvn = 0.05             # cm^-1
probe_p_step10 = 0.15


# =============================================================================
# %% main
# =============================================================================

def main():
    import xarray as xr
    import scipy.constants as scc

    import fmip_radiance_batch as frb

    pnw.dwvn = probe_dwvn
    pnw.p_step10 = probe_p_step10

    # --- the column (identical to the opacity script) ------------------------
    atmdata = xr.open_dataset(pnw.atmdata_file)
    species_list = [spc for spc in atmdata.data_vars
                    if spc not in ("p", "t", "alt")]

    p_prof = atmdata["p"].values[pnw.atm_index][::-1]
    t_prof = atmdata["t"].values[pnw.atm_index][::-1]
    z_prof = atmdata["alt"].values[pnw.atm_index][::-1]
    vmr_prof = {spc: np.clip(atmdata[spc].values[pnw.atm_index][::-1], 0.0, None)
                for spc in species_list}
    atm_field = frb.make_gridded_field(p_prof, t_prof, z_prof,
                                       vmr_prof, species_list)

    wvn = np.arange(pnw.wvn_min, pnw.wvn_max + pnw.dwvn / 2, pnw.dwvn)
    groups_str = pnw.build_abs_species_groups(species_list)

    print("=" * 78)
    print("CO2 nu3 wing opacity vs. line-wing cutoff "
          f"(column idx {pnw.atm_index}, {wvn.size} points at "
          f"{pnw.dwvn} cm^-1, lookup log10-p step {pnw.p_step10})")
    print("=" * 78)

    results = {}
    for cut in cutoffs_hz:
        pnw.line_cutoff_hz = cut
        t0 = time.time()
        lut = pnw.calc_lookup_table(atm_field, groups_str, wvn)
        labels = pnw.group_labels(lut["groups"])

        p_nodes, t_nodes, vmrs = lut["p"], lut["t"], lut["vmrs"]
        z_nodes = np.interp(np.log(p_nodes), np.log(p_prof[::-1]), z_prof[::-1])
        n_tot = p_nodes / (scc.k * t_nodes)

        k_total = (np.einsum("gfp,gp->fp", lut["xsec"].astype(np.float64), vmrs)
                   * n_tot[None, :])
        tau = pnw.cumulative_opacity(k_total, z_nodes)[:, 0]      # total column

        # CO2-only share (all CO2 pages, isotopologues and continuum merged)
        co2_pages = [i for i, lab in enumerate(labels) if lab == "CO2"]
        sig_co2 = lut["xsec"][co2_pages].sum(axis=0).astype(np.float64)
        k_co2 = sig_co2 * vmrs[co2_pages[0]][None, :] * n_tot[None, :]
        tau_co2 = pnw.cumulative_opacity(k_co2, z_nodes)[:, 0]

        results[cut] = (tau, tau_co2)
        print(f"\ncutoff {cut / 1e9:7.0f} GHz = {cut / (100.0 * scc.c):6.2f} "
              f"cm^-1   ({time.time() - t0:.0f} s)")
        print("   wavenumber     tau total    tau(CO2)   tau - tau(750 GHz)")
        base = results.get(750e9)
        for w in report_wvn:
            j = int(np.argmin(np.abs(wvn - w)))
            d = tau[j] - base[0][j] if base is not None and cut != 750e9 else 0.0
            print(f"   {wvn[j]:8.1f}   {tau[j]:10.4f}  {tau_co2[j]:9.4f}"
                  f"      {d:+9.4f}")
        del lut

    # --- verdict --------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"The observations need about tau + {tau_required:.2f} near 2405 cm^-1 "
          "(analysis note sec. 4).")
    j = int(np.argmin(np.abs(wvn - 2405.0)))
    for cut in cutoffs_hz:
        tau = results[cut][0][j]
        print(f"  cutoff {cut / 1e9:7.0f} GHz : tau(2405 cm^-1) = {tau:.4f}"
              f"   -> {tau / tau_required * 100:5.1f} % of the missing opacity "
              "supplied by untruncated Voigt+SFS wings")
    print("\nNOTE: this is the *cutoff* effect only.  Even an untruncated Voigt "
          "misses the\nsuper-Lorentzian excess and the line-mixing redistribution "
          "of the real band.")


if __name__ == "__main__":
    main()
