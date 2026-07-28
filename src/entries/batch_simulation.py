#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Generate slurm scripts to batch numerical simulation with different initial
conditions.

Given a netcdf (ERA5-style) file and the solver configuration, this enumerates
every time point along the ``time`` coordinate and submits one independent
sbatch job per time point. Each job gets its own small YAML config (just a
``run_solver:`` section, identical in shape to config.yml's) with ``ic_time``
set to that job's time point, and invokes ``run_solver.py`` with it - mirroring
how `make run_solver` itself is invoked (see makefile), since run_solver.py
now reads its configuration from a YAML file rather than CLI flags.

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
    "netcdf_path": None,             # required
    "time_dim": "time",
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
    "duration": 5,                   # days
    "save_interval_minutes": 30,
    "pressure": None,                # required (for naming), e.g. "500" or "(100,1000)"

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

    cfg = SimpleNamespace(**raw)

    if not cfg.netcdf_path:
        raise ValueError("batch_simulation.netcdf_path is required.")
    if cfg.pressure is None:
        raise ValueError("batch_simulation.pressure is required (used for naming), e.g. 500.")
    return cfg


def enumerate_times(netcdf_path, time_dim, engine, stride, limit):
    """Return ISO-8601 second-resolution strings for the selected time points."""
    ds = xr.open_dataset(netcdf_path, engine=engine)
    if time_dim not in ds.coords and time_dim not in ds.dims:
        raise KeyError(
            f"'{time_dim}' not found in {netcdf_path}. Available coords: {list(ds.coords)}"
        )
    times = np.atleast_1d(ds[time_dim].values)
    ds.close()

    times = times[::stride]
    if limit is not None:
        times = times[:limit]

    # second-resolution ISO strings (e.g. 2000-01-01T00:00:00); xarray's
    # sel(time=..., method='nearest') parses these unambiguously. Cast off
    # numpy.str_ (PyYAML's safe_dump can't represent it) to a plain str.
    return [str(np.datetime_as_string(t, unit="s")) for t in times]


def build_run_solver_config(ic_time, cfg):
    """The ``run_solver:`` section for this job (see run_solver.py's DEFAULT_CONFIG)."""
    return {
        "run_solver": {
            "lmax": cfg.lmax,
            "tau": list(cfg.tau),
            "cfl": cfg.cfl,
            "grid": cfg.grid,
            "semi_implicit": cfg.semi_implicit,
            "dealias": cfg.dealias,
            "duration": cfg.duration,
            "save_interval_minutes": cfg.save_interval_minutes,
            "ic": "real_world",
            "netcdf_path": str(Path(cfg.netcdf_path).resolve()),
            "ic_time": ic_time,
            "pressure": cfg.pressure,
            "checkpoint_only": False,
        }
    }


def build_script(ic_time, index, cfg, abs_config_dir, abs_log_dir):
    """Write this job's run_solver.py config and render its sbatch script text."""
    job_name = f"{cfg.job_name_prefix}_{index:04d}"

    job_config_path = abs_config_dir / f"{job_name}.yml"
    job_config_path.write_text(yaml.safe_dump(build_run_solver_config(ic_time, cfg), sort_keys=False))

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

    times = enumerate_times(
        cfg.netcdf_path, cfg.time_dim, cfg.engine, cfg.stride, cfg.limit
    )
    if not times:
        sys.exit("No time points selected; nothing to submit.")

    script_dir = (PROJECT_ROOT / cfg.script_dir).resolve()
    log_dir = (PROJECT_ROOT / cfg.log_dir).resolve()
    config_dir = (PROJECT_ROOT / cfg.config_dir).resolve()
    for d in (script_dir, log_dir, config_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(times)} time point(s); "
          f"{'generating (dry run)' if cfg.dry_run else 'submitting'} jobs...")

    submitted = 0
    for i, ic_time in enumerate(times):
        job_config_path, script_text = build_script(ic_time, i, cfg, config_dir, log_dir)
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
        print(f"Wrote {len(times)} script(s) to {script_dir}")
    else:
        print(f"Submitted {submitted}/{len(times)} job(s).")


if __name__ == "__main__":
    main()
