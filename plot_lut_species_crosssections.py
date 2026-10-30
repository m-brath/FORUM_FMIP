#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Absorption cross sections of all LUT species at 500 hPa / 260 K.

Diagnostic for the T_B comparison (compare_convolved_with_obs.py): plots the
gas absorption cross sections of every species group stored in the
FluxSimulator absorption lookup table (LUT) at the LUT state closest to
P = 500 hPa and T = 260 K.

Two data sources are handled:

1. The *stored* LUT binaries in cache/<setup_name>/LUT.xml(+.bin), read back
   with ReadXML. The cross sections are plotted at the nearest LUT pressure
   node and the nearest temperature-perturbation book. This is exactly what
   the RTE actually used (coarse LUT frequency grid, ~200 points for the
   fmip_radiance_batch cache).

2. A *fresh single-state recomputation* with abs_lookupCalc: one LUT node at
   500 hPa / 260 K on a fine wavenumber grid, using the same species groups,
   line-cutoff option (ByLine 750 GHz) and MT-CKD loading as
   FluxSimulator.get_lookuptableBatch. This resolves the line structure and
   shows whether any species group is missing/zero in the pipeline (CO is
   added as an extra probe group because it is present in atmdata.nc but was
   skipped by define_abs_species and therefore never entered the LUT).

Outputs (plot_folder):
    fig_lut_crosssections_stored.png/.pdf    stored LUT, per-species panels
    fig_lut_crosssections_recalc.png/.pdf    fine-grid recomputation
    fig_lut_crosssections_recalc_coarse.png  same, log-linear overview
    results/COMPARISON/lut_crosssections_recalc_500hPa_260K.nc

Environment (mamba/conda): pyarts. The ARTS data/catalog search path must be
provided via the ARTS_DATA_PATH environment variable (this script checks and
aborts if the catalogs cannot be found).

Run from the scripts/ directory:
    $ ARTS_DATA_PATH=<path-to-arts-cat-data> python plot_lut_species_crosssections.py
"""

import os
import time
import pathlib

import numpy as np
import xarray as xr
import scipy.constants as scc

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

from pyarts.workspace import Workspace
from pyarts.arts import SpeciesTag


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
cache_root = script_dir / "cache"
stats_folder = script_dir.parent / "results" / "COMPARISON"
plot_folder = script_dir.parent / "plot" / "lut_crosssections"

# stored LUT setups to inspect (cache subfolder names)
stored_lut_setups = ["fmip_radiance_batch"]

# recomputation state: nearest requested LUT node
P_recalc = 500.0e2          # Pa (500 hPa)
T_recalc = 260.0            # K
wvn_min, wvn_max = 620.0, 2810.0    # cm^-1 (covers satdata + SRF extension)
dwvn = 0.05                 # cm^-1 fine grid
dwvn_coarse = 0.0           # 0 -> reuse fine grid for the NC export

# stored-LUT plotting node: nearest state to these values
P_plot = 500.0e2
T_plot = 260.0

# same line cutoff option as FluxSimulator.get_lookuptableBatch(cutoff=True)
line_cutoff_hz = 750e9

# MT-CKD version for the recomputation -- matched to the catalog that
# FluxSimulator actually uses (~/.cache/arts/arts-cat-data-2.6.20 contains
# mt_ckd_4.3).  The older arts-cat-data-2.6.16 only ships mt_ckd_4.0.
mtckd_version = "4.3"

# flag whether to run the (slower) fine-grid recomputation
run_recalc = True

atmdata_file = data_dir / "atmdata.nc"


# =============================================================================
# %% plot formatting (same helpers as compare_convolved_with_obs.py)
# =============================================================================

def default_figure_gridspec(rows=1, columns=1, width_in_cm=21.0,
                            height_in_cm=15.0, constrained_layout=True):
    """Create a figure with cm-based size and a gridspec."""
    fig = plt.figure(constrained_layout=constrained_layout)
    fig.set_size_inches(width_in_cm / 2.54, height_in_cm / 2.54)
    spec = gridspec.GridSpec(ncols=columns, nrows=rows, figure=fig, hspace=0.55)

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

def read_stored_lut(lut_xml_path):
    """Read a stored FluxSimulator LUT and return python arrays."""
    ws = Workspace()
    ws.ReadXML(ws.abs_lookup, str(lut_xml_path))
    lut = ws.abs_lookup.value
    return {
        "f": np.array(lut.f_grid),
        "p": np.array(lut.p_grid),
        "t_pert": np.array(lut.t_pert),
        "t_ref": np.array(lut.t_ref),
        "vmrs": np.array(lut.vmrs),
        "xsec": np.array(lut.xsec),
        "tags": [str(t) for grp in lut.species for t in grp],
        "groups": [[str(t) for t in grp] for grp in lut.species],
    }


def typical_vmr_at_p(atmdata, p_target):
    """Column-mean VMR profile of atmdata.nc interpolated to pressure level.

    atmdata.nc is TOA-first; per column the level nearest p_target is picked,
    VMRs are averaged over all columns. Returns dict species -> vmr.
    """
    p = atmdata["p"].values              # (n_idx, n_lev)
    i_lev = np.abs(p - p_target).argmin(axis=1)
    out = {}
    for spc in atmdata.data_vars:
        if spc in ("p", "t", "alt"):
            continue
        v = atmdata[spc].values[np.arange(p.shape[0]), i_lev]
        out[spc] = float(np.nanmean(v))
    # temperature at the same levels (for reference / refinement)
    t = atmdata["t"].values[np.arange(p.shape[0]), i_lev]
    out["_T"] = float(np.nanmean(t))
    return out


def make_atm_field(p, t, vmr_dict):
    """Compact GriddedField4 with explicit grids (T, z, species) for the
    single LUT state (surface-first, one level)."""
    from pyarts.arts import GriddedField4
    spcs = [k for k in vmr_dict if not k.startswith("_")]
    gf = GriddedField4()
    gf.set_grid(0, ["T", "z"] + [f"abs_species-{s}" for s in spcs])
    gf.set_grid(1, np.asarray(p, dtype=np.float64))
    # NB: grids 2/3 (lat/lon) stay empty -- exactly as the batch script's
    # make_gridded_field, which is accepted by abs_lookupSetupBatch with the
    # default plane-parallel handling.
    data = np.zeros((3 + len(spcs), len(p), 1, 1))
    data[0, :, 0, 0] = t
    data[1, :, 0, 0] = 0.0
    for i, s in enumerate(spcs):
        data[i + 2, :, 0, 0] = vmr_dict[s]
    gf.data = data
    return gf


def recalc_crosssections(groups_str, vmr_dict, wvn_grid):
    """Recompute a single-state (500 hPa / 260 K) lookup table with the same
    setup options as FluxSimulator.get_lookuptableBatch."""
    ws = Workspace()
    ws.atmosphere_dim = 1
    ws.stokes_dim = 1

    f_grid = 100.0 * scc.c * wvn_grid       # cm^-1 -> Hz

    ws.f_grid = f_grid
    ws.abs_species = [[SpeciesTag(t) for t in grp] for grp in groups_str]

    # spectroscopic catalogs (same calls as FluxSimulator)
    ws.ReadXsecData(basename="xsec/")
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=line_cutoff_hz)

    # MT-CKD H2O continuum model (FluxSimulator _load_ckdmt analogue)
    ws.ReadXML(ws.predefined_model_data,
               f"model/mt_ckd_{mtckd_version}/H2O.xml")

    # single-state atmosphere at P_recalc / T_recalc
    # (ARTS requires >= 2 levels in the compact grid -- use a thin layer
    # around the requested pressure, both levels at T_recalc, and pick the
    # closest LUT pressure node afterwards)
    atm = make_atm_field([1.1 * P_recalc, 0.9 * P_recalc], [T_recalc] * 2,
                         {s: v for s, v in vmr_dict.items()})
    ws.batch_atm_fields_compact = [atm]
    ws.abs_lookupSetupBatch(p_step=0.05)

    # single temperature book at exactly the reference temperature
    ws.abs_t_pert = [0.0]

    # FluxSimulator H2O-vmr patch (continuum species rows of abs_vmrs)
    abs_vmrs = ws.abs_vmrs.value
    h2o_rows = [i for i, grp in enumerate(ws.abs_species.value)
                if "H2O" in str(grp[0])]
    if len(h2o_rows) > 1:
        vmr0 = abs_vmrs[h2o_rows[0], 0]
        for i in h2o_rows[1:]:
            if abs_vmrs[i, 0] == 0:
                abs_vmrs[i, 0] = vmr0
        ws.abs_vmrs.value = abs_vmrs

    ws.propmat_clearsky_agendaAuto(lines_speedup_option="None")
    ws.lbl_checked = 1

    ws.abs_lookupCalc()

    lut = ws.abs_lookup.value
    xsec = np.array(lut.xsec)                   # (nbooks, npages, n_f, n_p)
    pnode = int(np.argmin(np.abs(np.array(lut.p_grid) - P_recalc)))
    tags = [str(t) for grp in lut.species for t in grp]
    return {
        "xsec": xsec[0, :, :, pnode],           # single book, closest p node
        "p_node_Pa": float(np.array(lut.p_grid)[pnode]),
        "tags": tags,
        "groups": [[str(t) for t in grp] for grp in lut.species],
    }


def plot_per_species_panels(wvn, sigmas, tags, out_base, n_col=4,
                            yfloor=1e-33, title_extra=""):
    """One pcolormesh-free panel per species: log10 cross section vs wvn."""
    n = len(tags)
    n_row = int(np.ceil(n / n_col))
    fig, spec = default_figure_gridspec(
        rows=n_row, columns=n_col,
        width_in_cm=1.0 + 4.6 * n_col, height_in_cm=2.9 * n_row)

    for k, (sig, tag) in enumerate(zip(sigmas, tags)):
        ax, font = default_plot_format(fig.add_subplot(spec[k // n_col, k % n_col]))
        with np.errstate(divide="ignore"):
            log_sig = np.log10(np.maximum(sig, 1e-40))
        ax.plot(wvn, log_sig, color="0.2", linewidth=0.7)
        ax.set_ylim(np.log10(yfloor), -14)
        ax.set_xlim(wvn[0], wvn[-1])
        ax.set_title(tag.split("-")[0] + (" (xsec)" if "XFIT" in tag else
                                          (" (cont/CIA)" if "Cont" in tag or "CIA" in tag else "")),
                     fontsize=8, fontproperties=font)
        ax.tick_params(labelsize=6)
        if k // n_col == n_row - 1:
            ax.set_xlabel("Wavenumber [cm$^{-1}$]", fontsize=7,
                          fontproperties=font)
        if k % n_col == 0:
            ax.set_ylabel("log$_{10}$ σ [m$^2$ molecule$^{-1}$]", fontsize=7,
                          fontproperties=font)

    if title_extra:
        fig.suptitle(title_extra, fontsize=10)

    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


# =============================================================================
# %% main
# =============================================================================

def main():
    t_start = time.time()
    plot_folder.mkdir(parents=True, exist_ok=True)
    stats_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LUT species absorption cross sections at "
          f"P={P_plot/100:.0f} hPa, T nearest {T_plot:.0f} K")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) stored LUT(s): exact cross sections used by the RTE
    # ------------------------------------------------------------------
    for setup in stored_lut_setups:
        lut_file = cache_root / setup / "LUT.xml"
        if not lut_file.exists():
            print(f"\nSKIP {setup}: no {lut_file}")
            continue
        print(f"\nStored LUT: {lut_file}")
        lut = read_stored_lut(lut_file)
        wvn = lut["f"] / (100.0 * scc.c)
        node = int(np.argmin(np.abs(lut["p"] - P_plot)))
        book_ts = lut["t_ref"][node] + lut["t_pert"]
        book = int(np.argmin(np.abs(book_ts - T_plot)))
        print(f"  nearest state: p={lut['p'][node]/100:.1f} hPa, "
              f"T={book_ts[book]:.2f} K (ref {lut['t_ref'][node]:.2f} K "
              f"+ pert {lut['t_pert'][book]:+.2f} K)")
        sig = lut["xsec"][book, :, :, node]
        print(f"  {sig.shape[0]} cross-section pages, "
              f"{len(lut['tags'])} species tags")
        if sig.shape[0] != len(lut["tags"]):
            print(f"  WARNING: {len(lut['tags'])} tags but {sig.shape[0]} "
                  f"pages -> at least one tag has NO cross sections stored!")

        zero = [i for i in range(sig.shape[0]) if lut["xsec"][:, i, :, :].max() == 0]
        tiny = [i for i in range(sig.shape[0])
                if 0 < lut["xsec"][:, i, :, :].max() < 1e-30]
        print("  ALL-ZERO pages: ",
              [(i, lut["tags"][i]) for i in zero])
        print("  near-zero pages (<1e-30 m2): ",
              [(i, lut["tags"][i], f"{lut['xsec'][:,i,:,:].max():.1e}")
               for i in tiny])

        plot_per_species_panels(
            wvn, sig, lut["tags"][:sig.shape[0]],
            plot_folder / f"fig_lut_crosssections_stored_{setup}",
            title_extra=(f"Stored LUT '{setup}' at "
                         f"{lut['p'][node]/100:.0f} hPa / {book_ts[book]:.1f} K "
                         f"(coarse {wvn.size}-point LUT grid)"))

    # ------------------------------------------------------------------
    # 2) fine-grid recomputation at exactly 500 hPa / 260 K
    # ------------------------------------------------------------------
    if run_recalc:
        print("\nRecomputation of cross sections (fine grid)...")
        if not os.environ.get("ARTS_DATA_PATH"):
            raise SystemExit(
                "Set ARTS_DATA_PATH to the arts-cat-data root "
                "(dir containing lines/, xsec/, model/).")

        groups_str = [
            ["H2O-*-*-*", f"H2O-SelfContCKDMT{('400' if mtckd_version=='4.0' else '430')}-*-*",
             f"H2O-ForeignContCKDMT{('400' if mtckd_version=='4.0' else '430')}-*-*"],
            ["O2-*-1000000000000-1e+99", "O2-CIAfunCKDMT100-*-*"],
            ["N2-*-*-*", "N2-CIAfunCKDMT252-*-*", "N2-CIArotCKDMT252-*-*"],
            ["CO2-*-*-*", "CO2-CKDMT252-*-*"],
            ["CH4-*-*-*"],
            ["O3-*-*-*"],
            ["O3-XFIT"],
            ["N2O-*-*-*"],
            ["CO-*-*-*"],          # probe group: in atmdata but NOT in the LUT
            ["NO-*-*-*"],
            ["SO2-*-*-*"],
            ["NO2-*-*-*"],
            ["NH3-*-*-*"],
            ["HNO3-*-*-*"],
            ["OH-*-*-*"],
            ["HF-*-*-*"],
            ["HCl-*-*-*"],
            ["HBr-*-*-*"],
            ["HI-*-*-*"],
            ["ClO-*-*-*"],
            ["OCS-*-*-*"],
            ["H2CO-*-*-*"],
            ["HOCl-*-*-*"],
            ["HCN-*-*-*"],
            ["CH3Cl-*-*-*"],
            ["H2O2-*-*-*"],
            ["C2H2-*-*-*"],
            ["C2H6-*-*-*"],
            ["PH3-*-*-*"],
            ["CFC11-XFIT"],
            ["CFC12-XFIT"],
            ["CF4-XFIT"],
            ["CCl4-XFIT"],
        ]

        atmdata = xr.open_dataset(atmdata_file)
        vmr = typical_vmr_at_p(atmdata, P_recalc)
        print(f"  typical VMRs at the 500 hPa level (col-mean), "
              f"T_colmean {vmr['_T']:.1f} K")

        wvn_grid = np.arange(wvn_min, wvn_max + dwvn / 2, dwvn)
        res = recalc_crosssections(groups_str, vmr, wvn_grid)

        sig = res["xsec"]                       # (npages, n_f)
        tags = res["tags"]
        if sig.ndim == 2 and sig.shape[0] != len(wvn_grid) and sig.shape[1] == len(wvn_grid):
            sig = sig.T                         # -> (n_f, npages)
        npage = sig.shape[1] if sig.ndim == 2 else sig.shape[0]
        flat = [t for grp in groups_str for t in grp]
        print(f"  recomputed: {len(flat)} species tags but {npage} "
              f"cross-section pages")
        for i in range(npage):
            col = sig[:, i] if sig.ndim == 2 else sig[i, :]
            if col.max() == 0:
                print(f"  recomputed ALL-ZERO page {i}: {tags[i]}")

        # export
        page_tags = tags[:npage]
        ds = xr.Dataset(
            {"crosssection": (["wavenumber", "page"], sig[:, :npage]
                              if sig.ndim == 2 else sig),
             "page_tag": ("page", np.array(page_tags, dtype=object))},
            coords={"wavenumber": wvn_grid},
        )
        ds.attrs["pressure_Pa"] = res.get("p_node_Pa", P_recalc)
        ds.attrs["temperature_K"] = T_recalc
        ds.attrs["mtckd_version"] = mtckd_version
        out_nc = stats_folder / "lut_crosssections_recalc_500hPa_260K.nc"
        ds.to_netcdf(str(out_nc))
        print(f"  exported {out_nc}")

        plot_per_species_panels(
            wvn_grid, (sig[:, :npage] if sig.ndim == 2 else sig).T,
            tags[:npage],
            plot_folder / "fig_lut_crosssections_recalc",
            title_extra=f"Recomputed cross sections at {P_recalc/100:.0f} hPa"
                        f" / {T_recalc:.0f} K (MT-CKD {mtckd_version})")

    print(f"\ntotal wall time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
