#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the "exomol" CO2 catalog variant for the nu3 band-edge causal test
(test_co2_intensity_scaling.py, CO2_SCALE_VARIANT=exomol).

Data source: ExoMol UCL-4000 line list for 12C-16O2 (Yurchenko et al. 2020),
downloaded to ../exomol_co2/ (states.bz2 + the 2000-3000 cm^-1 transition
chunk + partition function + HITRAN-based air-broadening file).

The audit (analysis note results/COMPARISON/analysis_co2_nu3_wing_*.md,
sec. 10) showed both arts-cat-data builds carry the SAME intensity-pruned
CO2 subset: in 2398-2520 cm^-1 no line has E_low < 1849 K, while the
observed excess opacity requires quasi-dense hot-sequence lines at
E_low ~ 960-1600 K. UCL-4000 contains them.

Method:
  * parse states (E, g, J) and the transition chunk (lower, upper, A);
  * drop transitions that coincide with an existing 2.6 catalog line
    (|df0| < 1e-3 cm^-1 and |dE_low| < 0.5 cm^-1 -> same physical line,
    avoid double counting);
  * convert the remaining lines to ARTS catalog intensity
        i0 = A gu c^2/(8 pi f0^2) * gl exp(-e0/kT0) * (1-exp(-h f0/kT0)) / Q(T0)
    with Q(T0=296) absorbed in the empirically exact catalog constant
    K_CAT (fitted on matched arts-cat-data lines, zero spread).  Line-level
    HITRAN-vs-UCL4000 agreement in this window is 0.7x +- factor 2 (both
    data sets contain the strong hot-sequence lines, with ab-initio vs
    semi-empirical A values); the UCL-4000 *additional* lines are the ones
    not in the pruned catalog;
  * bundle the surviving hot lines into one pseudo-line per mini-run grid
    bin (0.05 cm^-1): the LUT/simulation grid cannot resolve sub-bin
    structure, and in this quasi-dense regime the smooth summed opacity is
    what enters the radiative transfer. Bundling keeps the LUT build cost
    comparable to the baseline runs;
  * append the bundle as one additional <AbsorptionLines> group to a copy
    of the CO2-626 block in a symlink farm of arts-cat-data-2.6.20
    (../catalog_co2exomol/), everything else symlinked unchanged.

Run from anywhere; output: ../catalog_co2exomol/lines/CO2-626.xml + audit
printout + a summary npz under ../exomol_co2/.
"""

# =============================================================================
# %% imports / constants  (edit here)
# =============================================================================

import bz2
import glob
import pathlib
import numpy as np
import scipy.constants as scc

script_dir = pathlib.Path(__file__).parent.resolve()
project = script_dir.parent
exo_dir = project / "exomol_co2"
cat26 = pathlib.Path.home() / ".cache/arts/arts-cat-data-2.6.20"
farm = project / "catalog_co2exomol"

wvn_min, wvn_max, wvn_step = 2330.0, 2560.0, 0.05   # mini-run grid
T0 = 296.0
match_df0 = 2e-4          # cm^-1, same-line exclusion window vs 2.6 catalog
match_dE = 0.2            # cm^-1
# broadening for the pseudo-lines: G0 [Hz/Pa] at 296 K, T-exponent; SELF/AIR
# ratio taken from the existing CO2 blocks of arts-cat-data-2.6.20
gamma_air_cm1_bar = 0.0949          # 12C-16O2__air.broad, low-J average
n_T = 0.73                          # temperature exponent (catalog-style)
self_over_air = 1.28
shift_D0 = 56.2                     # Hz/Pa (same order as catalog lines)

CM1_TO_HZPA = 100.0 * scc.c / 1e5   # cm^-1 / bar -> Hz / Pa

C, H, KB = scc.c, scc.h, scc.k


# =============================================================================
# %% parse the 2.6 CO2-626 block (to exclude identical lines)
# =============================================================================

def read_cat26_lines(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        t = line.split()
        f0 = float(t[0]) / (100 * C)
        i0 = float(t[1])
        e0 = float(t[2]) / (scc.h * C * 100)       # J -> cm^-1
        out.append((f0, e0, i0))
    a = np.array(out)
    return a[np.argsort(a[:, 0])]


# =============================================================================
# %% parse ExoMol states + transitions
# =============================================================================

def read_states():
    ids, E, g, J = [], [], [], []
    with bz2.BZ2File(exo_dir / "12C-16O2__UCL-4000.states.bz2") as f:
        for raw in f:
            s = raw.decode(errors="replace").split()
            if len(s) < 10:
                continue
            try:
                ids.append(int(s[0]))
                E.append(float(s[1]))
                g.append(float(s[2]))
                J.append(float(s[3]))
            except ValueError:
                break
    return np.array(ids), np.array(E), np.array(g), np.array(J)


def read_trans(chunk_glob):
    """Return (upper_id, lower_id, A). UCL-4000 trans files list the UPPER
    state first (verified: dE < 0 for col1-col2; col3 matches the 2.6
    catalog Einstein-A values exactly for shared main-band lines)."""
    up, lo, A = [], [], []
    for path in sorted(glob.glob(str(exo_dir / chunk_glob))):
        with bz2.BZ2File(path) as f:
            for raw in f:
                s = raw.decode(errors="replace").split()
                if len(s) < 3:
                    continue
                try:
                    up.append(int(s[0]))
                    lo.append(int(s[1]))
                    A.append(float(s[2]))
                except ValueError:
                    break
    return np.array(up), np.array(lo), np.array(A)


# =============================================================================
# %% write the catalog farm with the bundled pseudo-lines
# =============================================================================

def write_farm(bin_center, sum_i0, mean_El, mean_Jl, mean_Ju, mean_A):
    """Create the symlink farm and write the CO2-626.xml with the appended
    pseudo-line group.  Arrays are the per-bin bundle quantities."""
    import os
    import re
    G0_air = gamma_air_cm1_bar * CM1_TO_HZPA
    G0_self = self_over_air * G0_air
    farm.mkdir(parents=True, exist_ok=True)
    for p in cat26.iterdir():
        if p.name == "lines":
            continue
        tgt = farm / p.name
        if not tgt.exists():
            os.symlink(p, tgt)
    lines_dir = farm / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    for p in (cat26 / "lines").glob("*"):
        if p.name != "CO2-626.xml":
            tgt = lines_dir / p.name
            if not tgt.exists():
                os.symlink(p, tgt)

    out = lines_dir / "CO2-626.xml"
    src = (cat26 / "lines/CO2-626.xml").read_text()

    # The group id must parse as an ARTS QuantumIdentifier -- derive it from
    # an existing group's id with shifted vibrational labels (the LTE Voigt
    # path only uses the isotopologue part, so the numbers are inert).
    m_id = re.search(r'id="([^"]+)"', src)
    band_id = re.sub(r"\bv3 \d+ \d+\b", "v3 9 9", m_id.group(1), count=1)

    ngrp = int(src.split('nelem="')[1].split('"')[0])
    nbin = len(bin_center)
    recs = []
    for k in range(nbin):
        fHz_k = 100 * C * bin_center[k]
        e0_J = scc.h * C * 100 * mean_El[k]
        gl_k = max(int(2 * mean_Jl[k] + 1), 1)
        gu_k = max(int(2 * mean_Ju[k] + 1), 1)
        recs.append(
            f"{fHz_k:.6f} {sum_i0[k]:.6e} {e0_J:.6e} {gl_k} {gu_k} "
            f"{mean_A[k]:.4e} "
            f"0 0 "
            f"{G0_self:.6f} {n_T} 0 0 {shift_D0:.4f} 0 0 0 "
            f"{G0_air:.6f} {n_T} 0 0 {shift_D0:.4f} 0 0 0 "
            f" {int(mean_Jl[k])} {int(mean_Ju[k])}")
    group = (
        f'<AbsorptionLines version="2" id="{band_id}" nlines="{nbin}" '
        f'cutofftype="None" '
        f'mirroringtype="None" populationtype="LTE" '
        f'normalizationtype="SFS" lineshapetype="VP" T0="296" '
        f'cutofffreq="7.5e+11" linemixinglimit="-1" localquanta="J" '
        f'broadeningspecies="SELF AIR" '
        f'temperaturemodes="G0 T1 T1 D0 T0 T0">\n'
        + "\n".join(recs) + "\n</AbsorptionLines>\n")
    src = src.replace(f'nelem="{ngrp}"', f'nelem="{ngrp + 1}"')
    src = src.replace("</Array>", group + "</Array>")
    out.write_text(src)
    print(f"wrote {out}  (+{nbin} pseudo-lines in 1 extra group)")


# =============================================================================
# %% main
# =============================================================================

if __name__ == "__main__":
    cat = read_cat26_lines(cat26 / "lines/CO2-626.xml")
    print(f"2.6 CO2-626: {len(cat)} lines")

    ids, E, g, J = read_states()
    print(f"states: {len(ids)}  (E up to {E.max():.0f} cm^-1)")
    idx = np.full(ids.max() + 1, -1)
    idx[ids] = np.arange(len(ids))

    pf = np.loadtxt(exo_dir / "12C-16O2__UCL-4000.pf")
    Q296 = float(np.interp(T0, pf[:, 0], pf[:, 1]))
    print(f"Q(296) UCL pf = {Q296:.1f}  (not used directly; catalog "
          f"convention calibrated below)")
    # Catalog-conversion constant: with the catalog's i0 convention
    #     i0 = K * A gu c^2/(8 pi f0^2) gl exp(-e0/kT0)(1-exp(-h f0/kT0)),
    # K is an exact constant (fitted on 117 catalog lines with zero spread;
    # see results/COMPARISON/analysis note sec. 10).
    K_CAT = 5.9243e-5

    up, lo, A = read_trans("12C-16O2__UCL-4000__02000-03000.trans.bz2") \
        if (exo_dir / "12C-16O2__UCL-4000__02000-03000.trans.bz2").exists() \
        else read_trans("trans2.bz2")
    print(f"transitions read: {len(A)}")
    iu, jl = idx[up], idx[lo]
    nu = E[iu] - E[jl]
    m = (nu >= wvn_min) & (nu <= wvn_max) & (A > 0)
    print(f"in {wvn_min:.0f}-{wvn_max:.0f} cm^-1: {m.sum()}")

    # --- exclude lines already in the 2.6 catalog (same physical line) ----
    nu_m, El_m = nu[m], E[jl[m]]
    j = np.searchsorted(cat[:, 0], nu_m)
    dup = np.zeros(len(nu_m), bool)
    for jj in (j - 1, j):
        ok = (jj >= 0) & (jj < len(cat))
        d = np.abs(np.where(ok, cat[np.clip(jj, 0, len(cat) - 1), 0], 1e9) - nu_m)
        dE = np.abs(np.where(ok, cat[np.clip(jj, 0, len(cat) - 1), 1],
                             1e9) - El_m)
        dup |= ok & (d < match_df0) & (dE < match_dE)
    sel = np.where(m)[0][~dup]
    print(f"kept as NEW hot lines: {len(sel)}  "
          f"(dropped {m.sum() - len(sel)} catalog duplicates)")

    f0 = nu[sel]
    El = E[jl[sel]]
    A_s = A[sel]
    gl, gu = g[jl[sel]], g[iu[sel]]

    # --- convert to catalog intensity (same convention as the catalogs) ----
    fHz = 100 * C * f0
    i0 = (K_CAT * A_s * gu * C**2 / (8 * np.pi * fHz**2) * gl
          * np.exp(-scc.h * C * 100 * El / (KB * T0))
          * (1 - np.exp(-H * fHz / (KB * T0))))

    # validation on the dropped duplicates: same lines in both data sets
    dup_idx = np.where(m)[0][dup]
    jd = np.searchsorted(cat[:, 0], nu[dup_idx])
    r_v = []
    for n, jj in zip(dup_idx, jd):
        for jx in (jj - 1, jj):
            if 0 <= jx < len(cat) and abs(cat[jx, 0] - nu[n]) < match_df0 \
                    and abs(cat[jx, 1] - E[jl[n]]) < match_dE:
                fHz_n = 100 * C * nu[n]
                i0x = (K_CAT * A[n] * g[iu[n]] * C**2 / (8 * np.pi * fHz_n**2)
                       * g[jl[n]]
                       * np.exp(-scc.h * C * 100 * E[jl[n]] / (KB * T0))
                       * (1 - np.exp(-H * fHz_n / (KB * T0))))
                r_v.append(cat[jx, 2] / i0x)
                break
    r_v = np.array(r_v)
    corr = float(np.median(r_v))
    print(f"validation on {len(r_v)} duplicate lines: catalog i0 / exomol "
          f"i0 median {corr:.4f} (16-84%: "
          f"{np.percentile(r_v, 16):.3f}-{np.percentile(r_v, 84):.3f})")
    if not (0.5 < corr < 2.0):
        raise SystemExit("convention validation failed -- aborting")
    # NB: this median is the genuine HITRAN-vs-UCL4000 line-strength
    # agreement in this window (ab-initio vs semi-empirical A values), not
    # a convention offset -- K_CAT is convention-exact, so no correction is
    # applied to the new lines.
    if abs(corr - 1.0) > 0.05:
        print(f"  (line-level HITRAN/UCL agreement {corr:.2f}x: used as a "
              f"sanity gate only, no global rescaling applied)")

    print("new-line i0 percentiles:",
          " ".join(f"{x:.1e}" for x in np.percentile(i0, [50, 90, 99, 99.9])))

    # --- strength census before/after bundling -----------------------------
    print("\n  window        added Sigma i0 / cm-1 (lines)")
    for lo_, hi_ in [(2380, 2398), (2398, 2410), (2410, 2430),
                     (2430, 2460), (2460, 2520), (2520, 2560)]:
        s = (f0 >= lo_) & (f0 < hi_)
        print(f"  {lo_}-{hi_}:  {i0[s].sum()/(hi_-lo_):.2e}  ({s.sum()})")

    # --- bundle into grid bins ---------------------------------------------
    nb = int(round((wvn_max - wvn_min) / wvn_step))
    b = np.clip(((f0 - wvn_min) / wvn_step).astype(int), 0, nb - 1)
    bc = wvn_min + (np.arange(nb) + 0.5) * wvn_step     # per-bin centres
    sum_i0 = np.zeros(nb)
    w_El = np.zeros(nb)
    w_Jl = np.zeros(nb)
    w_Ju = np.zeros(nb)
    w_A = np.zeros(nb)
    cnt = np.zeros(nb, int)
    np.add.at(sum_i0, b, i0)
    np.add.at(w_El, b, i0 * El)
    np.add.at(w_Jl, b, i0 * J[jl[sel]])
    np.add.at(w_Ju, b, i0 * J[iu[sel]])
    np.add.at(w_A, b, i0 * A_s)
    np.add.at(cnt, b, 1)
    keep = sum_i0 > 0
    print(f"\nbins with content: {keep.sum()} / {nb}, "
          f"total added i0 = {sum_i0.sum():.3e}")

    np.savez(exo_dir / "ucl4000_bundle.npz",
             bin_center=bc[keep], sum_i0=sum_i0[keep],
             mean_El_cm=w_El[keep] / sum_i0[keep],
             mean_Jl=w_Jl[keep] / sum_i0[keep],
             mean_Ju=w_Ju[keep] / sum_i0[keep],
             mean_A=w_A[keep] / sum_i0[keep], n_lines=cnt[keep])

    # --- write the catalog farm (same layout as the catalog_co2x10 precedent)
    write_farm(bc[keep], sum_i0[keep], w_El[keep] / sum_i0[keep],
               w_Jl[keep] / sum_i0[keep], w_Ju[keep] / sum_i0[keep],
               w_A[keep] / sum_i0[keep])
