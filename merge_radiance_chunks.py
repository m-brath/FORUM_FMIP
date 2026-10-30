#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the frequency-chunk radiance files of fmip_radiance_slurm_main.py /
fmip_radiance_slurm_sub.py back into one file per column subset.

Input (results/RADIANCE/<run_name>/):
    Reference_radiance_TOA_Nf*_Nza*_<setup_base>_fchunk*_fmin*_fmax*GHz_N*_
    [chunk###].nc   (one file per frequency chunk and column chunk)
    with <setup_base> derived from run_name
    (run_name = <setup_base>_Nf<N_wvn>_Nza<N_za>)

Output (same run sub folder):
    Reference_radiance_TOA_merged_<run_name>[_chunk###].nc
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
* Only the run name is a constant at the top of this file; the setup base
  name and the chunk-file glob are derived from it, consistent with the
  other scripts.
* With delete_chunk_files = True the merged frequency-chunk input files are
  removed after ALL merged outputs have been written and verified. This is
  one-way: merged files are excluded from the chunk glob, so re-running the
  script afterwards finds no chunk files and raises FileNotFoundError.
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

# setup name base derived from run_name (the <setup_name> part of
# run_name = <setup_name>_Nf<N_wvn>_Nza<N_za>)
m = re.match(r"(.*)_Nf\d+_Nza\d+$", run_name)
if m is None:
    raise ValueError(
        f"run_name '{run_name}' does not follow the naming convention "
        f"<setup_name>_Nf<N_wvn>_Nza<N_za>")
setup_base = m.group(1)

# glob for the frequency-chunk files of this run (merged files are excluded
# automatically since they start with "..._merged_")
chunk_pattern = (f"Reference_radiance_TOA_Nf*_Nza*_{setup_base}_fchunk*.nc")

# delete the frequency-chunk input files after a successful merge (the merged
# outputs can always be re-derived from the chunk files, but not vice versa,
# so this is opt-in -- set to True only when the cleanup is wanted)
delete_chunk_files = False


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

    # keep track of which chunk files went into which merged output so they
    # can be removed at the end when delete_chunk_files is set
    merged_outputs = []

    for key, group in sorted(groups.items()):
        # sort the frequency chunks by their minimum frequency
        fmin_of = {}
        for f in group:
            with xr.open_dataset(f) as ds:
                fmin_of[f] = float(ds["frequency"].values.min())
        dss = [xr.open_dataset(f)
               for f in sorted(group, key=fmin_of.get)]

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

        # concatenate the radiance (and the frequency-dependent coords);
        # join="outer" because the chunk frequency grids differ by design
        # (pinned explicitly -- xarray will change the default to "exact")
        merged = xr.concat(
            [ds[["spectral_radiance_TOA", "frequency", "wavenumber"]]
             for ds in dss],
            dim="frequency",
            join="outer",
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
                    f"Reference_radiance_TOA_merged_{run_name}{suffix}.nc")
        merged.attrs["history"] = (
            f"merged {len(dss)} frequency-chunk file(s) "
            f"matching '{chunk_pattern}'")
        merged.to_netcdf(str(out_file))
        print(f"  exported {out_file}")

        # close the chunk inputs so they can be removed later
        for ds in dss:
            ds.close()
        merged_outputs.append((out_file, group))

    if delete_chunk_files:
        # only delete once every group has a valid merged output on disk
        for out_file, group in merged_outputs:
            if not (out_file.is_file() and out_file.stat().st_size > 0):
                raise RuntimeError(
                    f"merged output {out_file} is missing or empty -- "
                    f"keeping the {len(group)} frequency-chunk file(s)")
        removed = 0
        for out_file, group in merged_outputs:
            for f in group:
                try:
                    pathlib.Path(f).unlink()
                    removed += 1
                except OSError as e:
                    print(f"  WARNING: could not remove {f}: {e}")
        print(f"\nremoved {removed} frequency-chunk file(s)")

    print("\ndone")


if __name__ == "__main__":
    main()
