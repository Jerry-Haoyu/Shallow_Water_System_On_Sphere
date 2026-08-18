#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Generate slurm scripts to batch numerical simulation with different initial
conditions.

Given a dataset_name (see `make download_era5` / reanalysis_data/<name>/data.nc)
and the solver configuration, this enumerates every time point along the
``time_dim`` coordinate and submits one independent sbatch job per time point. Each job gets its own small YAML config (just a
``run_solver:`` section, identical in shape to config.yml's) with ``ic_time``
set to that job's time point, and invokes ``run_solver.py`` with it - mirroring
how `make run_solver` itself is invoked (see makefile), since run_solver.py
now reads its configuration from a YAML file rather than CLI flags.

Since each job is its own independent sbatch process, `run_solver.py` cannot
share an in-memory dataset across jobs. Instead, this script opens and loads
the (potentially large, multi-year) ERA5 dataset exactly once here, slices
out just the one time point each job needs, and writes that slice to a small
per-job NetCDF file (``ic_data_dir``). Each job's config points at its own
slice via ``ic_time`` / ``ic_data_path``, so `run_solver.py` only ever opens
that tiny file instead of re-loading the full dataset per job.

Configuration is read from the ``batch_simulation`` section of config.yml
(pass an alternative path as the first CLI argument), matching the
run_solver.py / inference.py convention.

Example
-------
    python -m src.entries.batch_simulation config.yml

@author Haoyu Tang hytang2@illinois.edu
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
import yaml
import subprocess

# project root = .../Shallow_Water_System_On_Sphere (this file lives in src/entries/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = "/data/gzhang13/a/hytang2/envs/swe/bin/python"


DEFAULT_CONFIG = {
    # ------------------------- data / time selection ------------------------- #
    "dataset_name": None,            # required, e.g. "1980_2025_odd_month" -> reanalysis_data/<name>/data.nc
    "time_dim": "valid_time",
    "engine": None,                  # xarray engine (e.g. 'cfgrib' for grib); default: auto-detect
    "stride": 1,                     # use every Nth time point
    "limit": None,                   # only submit the first N (post-stride) time points

    # ------------------------- solver configuration --------------------------- #
    # forwarded into each job's own run_solver.py config
    "lmax": 64,
    "tau": [30000, 30000, 30],
    "cfl": 0.25,
    "grid": "equiangular",           # equiangular | legendre-gauss | lobatto
    "semi_implicit": True,
    "dealias": True,
    "non_dimensional": True,         # rescale by dataset-wide U=sqrt(g*h_avg), T=radius/U
    "duration": 5,                   # days
    "save_interval_minutes": 30,
    "pressure": None,                # required (for naming), e.g. "500" or "(100,1000)"
    "rad": False,                    # forwarded into each job's run_solver config (see run_solver.py)
    "tau_rad": None,
    "rad_smooth_fraction": 0.5,

    # ----------------------------- slurm knobs --------------------------------- #
    "partition": "gpu",
    "account": "gzhang13-group",
    "time_limit": "12:00:00",
    "gres": "gpu:1",                 # pass "" to omit (CPU-only)
    "mem": "32G",
    "cpus": 4,
    "job_name_prefix": "swe_rw",
    "python": DEFAULT_PYTHON,
    "script_dir": "slurm_scripts",
    "config_dir": "slurm_scripts/configs",
    "ic_data_dir": "slurm_scripts/ic_data",
    "log_dir": "slurm_logs",
    "dry_run": False,                # generate scripts/configs without calling sbatch
}


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) or {}

    raw = DEFAULT_CONFIG | config.get("batch_simulation", {})

    # normalize types regardless of YAML formatting
    raw["lmax"] = int(raw["lmax"])
    raw["tau"] = [int(t) for t in raw["tau"]]
    raw["cfl"] = float(raw["cfl"])
    raw["duration"] = float(raw["duration"])
    raw["save_interval_minutes"] = float(raw["save_interval_minutes"])
    raw["stride"] = int(raw["stride"])
    raw["limit"] = int(raw["limit"]) if raw["limit"] is not None else None
    raw["cpus"] = int(raw["cpus"])
    raw["rad"] = bool(raw["rad"])
    raw["tau_rad"] = float(raw["tau_rad"]) if raw["tau_rad"] is not None else None
    raw["rad_smooth_fraction"] = float(raw["rad_smooth_fraction"])

    cfg = SimpleNamespace(**raw)

    if not cfg.dataset_name:
        raise ValueError("batch_simulation.dataset_name is required.")
    if cfg.pressure is None:
        raise ValueError("batch_simulation.pressure is required (used for naming), e.g. 500.")
    if cfg.rad and cfg.tau_rad is None:
        raise ValueError("batch_simulation.tau_rad is required when rad=True.")
    return cfg


def prepare_ic_slices(dataset_name, time_dim, engine, stride, limit, ic_data_dir, job_name_prefix):
    """Open + load the (potentially large) ERA5 dataset exactly once, then
    write one small per-job NetCDF slice (one time point each) per selected
    job. Returns a list of (ic_time, ic_data_path) pairs, in job order.

    Slicing keeps ``time_dim`` as a length-1 dimension (``isel(..., [idx])``,
    not squeezed away), so run_solver.py's existing
    ``era5_dataset.sel(valid_time=ic_time, method='nearest').squeeze()`` call
    works unchanged against each tiny slice file.
    """
    netcdf_path = PROJECT_ROOT / "reanalysis_data" / dataset_name / "data.nc"
    ds = xr.open_dataset(netcdf_path, engine=engine)
    if time_dim not in ds.coords and time_dim not in ds.dims:
        raise KeyError(
            f"'{time_dim}' not found in {netcdf_path}. Available coords: {list(ds.coords)}"
        )

    all_times = np.atleast_1d(ds[time_dim].values)
    idxs = np.arange(len(all_times))[::stride]
    if limit is not None:
        idxs = idxs[:limit]

    # single full-dataset load, shared by every job's slice below - this is
    # the overhead that used to be repeated once per sbatch job.
    ds = ds.load()

    ic_data_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, idx in enumerate(idxs):
        # second-resolution ISO string (e.g. 2000-01-01T00:00:00); xarray's
        # sel(time=..., method='nearest') parses these unambiguously. Cast
        # off numpy.str_ (PyYAML's safe_dump can't represent it) to a plain str.
        ic_time = str(np.datetime_as_string(all_times[idx], unit="s"))

        job_name = f"{job_name_prefix}_{i:04d}"
        ic_data_path = ic_data_dir / f"{job_name}_ic.nc"
        ds.isel(**{time_dim: [idx]}).to_netcdf(ic_data_path)

        jobs.append((ic_time, ic_data_path))

    ds.close()
    return jobs


def build_run_solver_config(ic_time, ic_data_path, cfg):
    """The ``run_solver:`` section for this job (see run_solver.py's DEFAULT_CONFIG)."""
    return {
        "run_solver": {
            "lmax": cfg.lmax,
            "tau": list(cfg.tau),
            "cfl": cfg.cfl,
            "grid": cfg.grid,
            "semi_implicit": cfg.semi_implicit,
            "dealias": cfg.dealias,
            "non_dimensional": cfg.non_dimensional,
            "duration": cfg.duration,
            "save_interval_minutes": cfg.save_interval_minutes,
            "ic": "real_world",
            "dataset_name": cfg.dataset_name,
            "ic_time": ic_time,
            "ic_data_path": str(ic_data_path),
            "pressure": cfg.pressure,
            "rad": cfg.rad,
            "tau_rad": cfg.tau_rad,
            "rad_smooth_fraction": cfg.rad_smooth_fraction,
            "checkpoint_only": False,
        }
    }


def build_script(ic_time, ic_data_path, index, cfg, abs_config_dir, abs_log_dir):
    """Write this job's run_solver.py config and render its sbatch script text."""
    job_name = f"{cfg.job_name_prefix}_{index:04d}"

    job_config_path = abs_config_dir / f"{job_name}.yml"
    job_config_path.write_text(
        yaml.safe_dump(build_run_solver_config(ic_time, ic_data_path, cfg), sort_keys=False)
    )

    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={cfg.partition}",
        f"#SBATCH --account={cfg.account}",
        f"#SBATCH --time={cfg.time_limit}",
        f"#SBATCH --mem={cfg.mem}",
        f"#SBATCH --cpus-per-task={cfg.cpus}",
        f"#SBATCH --output={abs_log_dir}/{job_name}_%j.out",
        f"#SBATCH --error={abs_log_dir}/{job_name}_%j.err",
    ]
    if cfg.gres:
        sbatch_lines.insert(6, f"#SBATCH --gres={cfg.gres}")

    body = [
        "",
        "set -euo pipefail",
        f"cd {PROJECT_ROOT}",
        f'echo "Running real-world simulation for ic_time={ic_time}"',
        # mirrors `make run_solver` (see makefile): run_solver.py is invoked as a
        # module and takes its config as a single positional YAML path.
        f"{cfg.python} -m src.entries.run_solver {job_config_path}",
        "",
    ]
    return job_config_path, "\n".join(sbatch_lines + body)


def main():
    cfg = load_config()

    script_dir = (PROJECT_ROOT / cfg.script_dir).resolve()
    log_dir = (PROJECT_ROOT / cfg.log_dir).resolve()
    config_dir = (PROJECT_ROOT / cfg.config_dir).resolve()
    ic_data_dir = (PROJECT_ROOT / cfg.ic_data_dir).resolve()
    for d in (script_dir, log_dir, config_dir, ic_data_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{cfg.dataset_name}' once and slicing per-job initial conditions...")
    jobs = prepare_ic_slices(
        cfg.dataset_name, cfg.time_dim, cfg.engine, cfg.stride, cfg.limit,
        ic_data_dir, cfg.job_name_prefix,
    )
    if not jobs:
        sys.exit("No time points selected; nothing to submit.")

    print(f"Selected {len(jobs)} time point(s); "
          f"{'generating (dry run)' if cfg.dry_run else 'submitting'} jobs...")

    submitted = 0
    for i, (ic_time, ic_data_path) in enumerate(jobs):
        job_config_path, script_text = build_script(ic_time, ic_data_path, i, cfg, config_dir, log_dir)
        script_path = script_dir / f"{cfg.job_name_prefix}_{i:04d}.slurm"
        script_path.write_text(script_text)

        if cfg.dry_run:
            print(f"[dry-run] {script_path}  (ic_time={ic_time}, config={job_config_path})")
            continue

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[FAILED] ic_time={ic_time}: {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"[ok] ic_time={ic_time} -> {result.stdout.strip()}")
            submitted += 1

    if cfg.dry_run:
        print(f"Wrote {len(jobs)} script(s) to {script_dir}")
    else:
        print(f"Submitted {submitted}/{len(jobs)} job(s).")


if __name__ == "__main__":
    main()
