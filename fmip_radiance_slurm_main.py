#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMIP batch spectral-radiance benchmark -- SLURM launcher (frequency chunks).

Companion of fmip_radiance_slurm_sub.py, following the two-file pattern of
fast_scheme/scripts/flux_simulator_slurm_main.py + _slurm_sub.py:

    this script  -> slices the FULL simulation frequency grid into frequency
                    chunks, writes one JSON config per chunk, and dispatches
                    each chunk as one job,
    _sub.py      -> reads one config and runs the radiance_simulator_batch
                    simulation for that frequency chunk (all requested columns
                    are looped inside the job, in column chunks).

Frequency chunking vs. the plain fmip_radiance_batch.py
-------------------------------------------------------
* Job dimension is the FREQUENCY grid: one job per frequency chunk. The
  column subset / column chunking loop (as in fmip_radiance_batch.py) stays
  INSIDE each job (constants below are passed through the config).
* LUT: each frequency chunk gets its OWN FluxSimulator setup name
  (f"{setup_name}_{identifier}") and therefore its own LUT cache dir
  <cwd>/cache/<setup_name>/. No LUT state is shared between parallel jobs.
  The per-chunk LUT is built by the job itself, from the columns that job
  simulates, on the job's sub-frequency grid.
* Output: results/RADIANCE/<run_name>/Reference_radiance_TOA_Nf<N_chunk>_...
  -- one file per (frequency chunk [, column chunk]), all frequency chunks of
  one launch sharing the same run sub folder (run_name is passed through the
  config). convolve_sensor_response.py discovers them recursively, and
  merge_radiance_chunks.py re-assembles the full-frequency files.

On levante (SLURM)
------------------
`sbatch_simple` is installed there; the launcher dispatches with

    sbatch_simple <partition> <jobname> <mem> <runtime> \
        python fmip_radiance_slurm_sub.py <config.json>

Slurm writes the job log files (*.out / *.err) into the submission
directory, so the launcher submits from slurm_logs_dir
(results/RADIANCE/<run_name>/logs) instead of the scripts folder. The job
itself is unaffected: fmip_radiance_slurm_sub.py pins its cwd to the
scripts/ folder (LUT cache location), and the config / script paths passed
to sbatch_simple are absolute.

On this machine `sbatch_simple` does not exist and the launcher instead runs
the sub-scripts directly and sequentially with the current python
(the pyarts env); no log files are written (direct console output).
Set use_slurm to force either mode.

Environment (mamba/conda): pyarts (pyarts 2.6 + FluxSimulator, NOT pyarts3).
Run from the scripts/ directory (LUT cache: <cwd>/cache/<setup_name>/).

LUT cache note: cache dirs are keyed by the per-chunk setup name, which
contains the frequency range and chunk count. Changing N_wvn or
freq_per_chunk changes the identifiers -> new cache dirs. Delete
scripts/cache/<setup>*/ manually if the spectrum or species setup changed.
"""

import os
import sys
import json
import shutil
import pathlib
import subprocess

import numpy as np

from fmip_radiance_batch import define_frequency_grid


# =============================================================================
# %% paths / constants  (edit here)
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()

# where the per-chunk JSON configs are written; like the results, every
# run gets its own sub folder configs/<run_name>/ (see run_name below)
config_path = script_dir.parent / "configs"

# -----------------------------------------------------------------------------
# full longwave (wavenumber) grid -- same definition as fmip_radiance_batch.py
# -----------------------------------------------------------------------------

N_wvn = 200          # total number of wavenumbers of the full grid
include_sensor_response = True  # extend range by the sensor response wvn axis

# -----------------------------------------------------------------------------
# frequency chunking (the SLURM job dimension)
# -----------------------------------------------------------------------------
freq_per_chunk = 50  # number of frequencies per frequency chunk / job

# -----------------------------------------------------------------------------
# column subset / column chunking (looped INSIDE each job, as in
# fmip_radiance_batch.py)
# -----------------------------------------------------------------------------
start_column = 0     # first (file order) column to process
n_columns = 20       # number of columns to process
chunk_size = 20      # number of columns per radiance_simulator_batch call

# angular grid, shared by every profile of every batch call
N_za = 20            # number of zenith angles
N_aa = 1             # number of azimuth angles

# profile refinement (see fmip_radiance_batch.py)
refine = False
refinement_steps = 1
points_added_per_step = 3

# If True, FORCE recalculation of the per-chunk LUT even if a stored one
# exists in the chunk's cache dir.
recalc_lut = False

# output options (passed through to the sub script)
export_full_field = False   # also store the full (frequency, idx, lev, za, aa)
                            # cube in the chunk files

# FluxSimulator setup name BASE. Each frequency chunk uses
# f"{setup_name}_{identifier}", so every chunk has its own LUT cache dir.
setup_name = "fmip_radiance_batch_slurm"

# Run identity: every launch writes its results into the sub folder
# results/RADIANCE/<run_name>/ (shared by all frequency chunks of this
# launch), keeping different runs/configurations apart. Passed to the
# sub-scripts through the config.
run_name = f"{setup_name}_Nf{N_wvn}_Nza{N_za}"

# SLURM job log files (*.out / *.err): Slurm writes them into the directory
# the submission runs from, so sbatch_simple is dispatched with this folder
# as cwd (keeps the scripts/ folder clean). Defaults to a logs sub folder of
# the run's results folder. Ignored in direct (non-SLURM) mode.
slurm_logs_dir = (script_dir.parent / "results" / "RADIANCE"
                  / run_name / "logs")

# -----------------------------------------------------------------------------
# SLURM dispatch
# -----------------------------------------------------------------------------
# None -> auto: use sbatch_simple when it is found on PATH (levante),
#          otherwise run the sub-scripts directly and sequentially (local).
# True/False -> force SLURM / direct mode.
use_slurm = None

# arguments for sbatch_simple <partition> <jobname> <mem> <runtime>
slurm_partition = "compute"
slurm_mem = "128"            # GB per job
slurm_runtime = "08:00:00"   # HH:MM:SS

# write the configs but do not start the jobs
only_configs = False


# =============================================================================
# %% main
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # full frequency grid and its chunking
    # -------------------------------------------------------------------------
    f_grid, wvn_grid = define_frequency_grid(
        N_wvn, include_sensor_response=include_sensor_response)

    n_f_chunks = int(np.ceil(len(f_grid) / freq_per_chunk))

    # -------------------------------------------------------------------------
    # column subset (same for every job; looped inside the sub script)
    # -------------------------------------------------------------------------
    n_to_run = n_columns

    # -------------------------------------------------------------------------
    # run summary
    # -------------------------------------------------------------------------
    slurm_mode = (shutil.which("sbatch_simple") is not None
                  if use_slurm is None else bool(use_slurm))

    print("=" * 70)
    print("FMIP batch spectral-radiance -- frequency-chunk launcher")
    print(f"  spectrum    : {wvn_grid.min():.1f} - {wvn_grid.max():.1f} cm^-1, "
          f"N_wvn = {N_wvn}")
    print(f"  freq chunks : {n_f_chunks} chunk(s) of <= {freq_per_chunk} "
          f"frequencies ({len(f_grid)} total)")
    print(f"  columns     : {start_column} .. {start_column + n_to_run - 1} "
          f"({n_to_run} columns, column chunks of <= {chunk_size} inside jobs)")
    print(f"  angular     : N_za = {N_za}, N_aa = N_aa = {N_aa}")
    print(f"  setup base  : {setup_name}")
    print(f"  run folder  : results/RADIANCE/{run_name}/")
    print(f"  configs     : {config_path / run_name}")
    print(f"  slurm logs  : {slurm_logs_dir}")
    print(f"  dispatch    : "
          f"{'sbatch_simple (' + slurm_partition + ')' if slurm_mode else 'direct sequential run'}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # one config + one job per frequency chunk
    # -------------------------------------------------------------------------
    config_dir = config_path / run_name
    config_dir.mkdir(parents=True, exist_ok=True)

    if slurm_mode:
        # submission directory for sbatch_simple -> Slurm log files land here
        slurm_logs_dir.mkdir(parents=True, exist_ok=True)

    joblist = []
    for i in range(n_f_chunks):
        i0 = int(i * freq_per_chunk)
        i1 = min(int((i + 1) * freq_per_chunk), len(f_grid))
        sub_f_grid = f_grid[i0:i1]

        fmin = int(np.min(sub_f_grid) / 1e9)
        fmax = int(np.max(sub_f_grid) / 1e9)
        identifier = (f"fchunk{i:03d}"
                      f"_fmin{fmin:07d}_fmax{fmax:07d}GHz"
                      f"_N{len(sub_f_grid)}")

        config = {
            "f_grid": [float(f) for f in sub_f_grid],
            "identifier": identifier,
            # FluxSimulator setup of this chunk -> own LUT cache dir
            "flxsim_setup": f"{setup_name}_{identifier}",
            "start_column": int(start_column),
            "n_columns": int(n_to_run),
            "chunk_size": int(chunk_size),
            "N_za": int(N_za),
            "N_aa": int(N_aa),
            "refine": bool(refine),
            "refinement_steps": int(refinement_steps),
            "points_added_per_step": int(points_added_per_step),
            "recalc_lut": bool(recalc_lut),
            "export_full_field": bool(export_full_field),
            # results sub folder name shared by all chunks of this run
            "run_name": run_name,
        }

        config_filename = config_dir / f"config_{identifier}.json"
        with open(config_filename, "w") as fout:
            json.dump(config, fout)

        jobname = f"rad_chk_{i:05d}_{setup_name}"

        python_str = [sys.executable,
                      str(script_dir / "fmip_radiance_slurm_sub.py"),
                      str(config_filename)]

        if slurm_mode:
            command_str = (["sbatch_simple", slurm_partition, jobname,
                            slurm_mem, slurm_runtime] + python_str)
        else:
            command_str = python_str

        print("sending to queue:" if slurm_mode else "starting job:")
        for cmd_str_i in command_str:
            print(f"  {cmd_str_i}")

        joblist.append({"job": i, "config": str(config_filename)})

        if not only_configs:
            if slurm_mode:
                # submit from the log folder: Slurm writes the job *.out /
                # *.err files into the submission directory, and the job's
                # own cwd is pinned to scripts/ by the sub script (LUT
                # cache), so the log location has no side effects
                subprocess.run(command_str, cwd=slurm_logs_dir)
            else:
                # run sequentially and fail loudly on a broken chunk
                subprocess.run(command_str, check=True)

        print("  ")

    print("Jobs send to queue:" if slurm_mode else "Jobs run:")
    for job in joblist:
        print(f"  job: {job['job']}; {job['config']}")
    print("=======================================================================")
    print(f"{len(joblist)} jobs "
          f"{'send to the queue' if slurm_mode else 'run'}")
    print("  ")

    return joblist


if __name__ == "__main__":
    main()
