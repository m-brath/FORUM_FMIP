#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the frequency-chunk radiance files of fmip_radiance_slurm_main.py /
fmip_radiance_slurm_sub.py back into one file per column subset.

Input (results/RADIANCE/<run_name>/):
    Reference_radiance_TOA_Nf*_Nza*_<setup_base>_fchunk*_fmin*_fmax*GHz_N*_
    [chunk###].nc   (one file per frequency chunk and column chunk)

Output (same run sub folder):
    Reference_radiance_TOA_merged_<setup_base>[_chunk###].nc
    -- one file per column chunk group (no column chunk suffix if the jobs
    used a single column chunk), containing the full-frequency
    spectral_radiance_TOA on the assembled frequency grid.

Notes
-----
* Frequency chunks are sorted by their minimum frequency before concatenation
  along "frequency"; the concatenated grid is checked to be strictly
  increasing and free of duplicated coordinates.
* altitude_min / altitude_max are identical in every frequency chunk, so they
  are taken from the first file instead of stacked.
* Only run parameters (the file glob) are constants at the top of this file,
  consistent with the other scripts.
"""

import re
import glob
import pathlib

import numpy as np
import xarray as xr


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()

# run sub folder of results/RADIANCE/ written by one
# fmip_radiance_slurm_main.py launch
# (run_name = <setup_name>_Nf<N_wvn>_Nza<N_za>)
run_name = "fmip_radiance_batch_slurm_Nf200_Nza20"
results_folder = script_dir.parent / "results" / "RADIANCE" / run_name

# glob for the frequency-chunk files of one setup (merged files are excluded
# automatically since they start with "..._merged_")
chunk_pattern = "Reference_radiance_TOA_Nf*_fmip_radiance_batch_slurm_fchunk*.nc"

# setup name base used in the merged output file name
setup_base = "fmip_radiance_batch_slurm"


# =============================================================================
# %% main
# =============================================================================

def main():
    files = sorted(glob.glob(str(results_folder / chunk_pattern)))
    if not files:
        raise FileNotFoundError(
            f"no radiance chunk files matching "
            f"{results_folder / chunk_pattern}"
        )

    print("=" * 70)
    print("Merging frequency-chunk radiance files")
    print(f"  pattern : {chunk_pattern}")
    print(f"  found   : {len(files)} file(s)")
    print("=" * 70)

    # group the files by column chunk id (the trailing "_chunk###" suffix of
    # the sub script when the column subset was split into several chunks)
    col_chunk_re = re.compile(r"_chunk(\d+)\.nc$")
    groups = {}
    for f in files:
        m = col_chunk_re.search(f)
        key = m.group(1) if m else ""
        groups.setdefault(key, []).append(f)

    for key, group in sorted(groups.items()):
        # sort the frequency chunks by their minimum frequency
        dss = []
        for f in sorted(group,
                        key=lambda p: float(
                            xr.open_dataset(p)["frequency"].values.min())):
            dss.append(xr.open_dataset(f))

        fmins = [float(ds["frequency"].values.min()) for ds in dss]
        fmaxs = [float(ds["frequency"].values.max()) for ds in dss]
        print(f"\ncolumn chunk '{key or '-'}': {len(dss)} frequency chunk(s), "
              f"{sum(len(ds['frequency']) for ds in dss)} frequencies total")
        print(f"  frequency ranges (GHz): "
              + ", ".join(f"{a / 1e9:.3f}-{b / 1e9:.3f}"
                          for a, b in zip(fmins, fmaxs)))

        # no gaps/overlaps between the (sorted) chunk ranges
        for i in range(len(dss) - 1):
            if fmaxs[i] >= fmins[i + 1]:
                raise AssertionError(
                    f"frequency chunks overlap or are not sorted: "
                    f"{fmaxs[i]:e} >= {fmins[i + 1]:e}"
                )

        # concatenate the radiance (and the frequency-dependent coords)
        merged = xr.concat(
            [ds[["spectral_radiance_TOA", "frequency", "wavenumber"]]
             for ds in dss],
            dim="frequency",
        )

        # check the assembled grid
        f_all = merged["frequency"].values
        if np.any(np.diff(f_all) <= 0):
            raise AssertionError("merged frequency grid is not strictly "
                                 "increasing")

        # per-column scalars are identical across frequency chunks -> take
        # them from the first file
        for var in ("altitude_min", "altitude_max"):
            if var in dss[0]:
                merged[var] = dss[0][var]

        # sanity: every column chunk group covers the same columns
        idx_all = merged["idx"].values
        print(f"  columns   : {idx_all.min()} .. {idx_all.max()} "
              f"({len(idx_all)} columns)")

        suffix = f"_chunk{key}" if key else ""
        out_file = (results_folder /
                    f"Reference_radiance_TOA_merged_{setup_base}{suffix}.nc")
        merged.attrs["history"] = (
            f"merged {len(dss)} frequency-chunk file(s) "
            f"matching '{chunk_pattern}'")
        merged.to_netcdf(str(out_file))
        print(f"  exported {out_file}")

    print("\ndone")


if __name__ == "__main__":
    main()
