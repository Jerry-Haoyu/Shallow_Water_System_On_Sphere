"""
Generate slurm scripts to batch numerical simulation with different initial
conditions.

Two modes (``mode`` in config, see DEFAULT_CONFIG):

  "real_world" (default): given a dataset_name (see `make download_era5` /
    reanalysis_data/<name>/data.nc) and the solver configuration, this
    enumerates every time point along the ``time_dim`` coordinate and submits
    one independent sbatch job per time point.

  "galewsky": no dataset - instead samples ``n_jobs`` random combinations of
    galewsky_initial_condition's own parameters (umax, jet placement,
    perturbation location/amplitude/noise - see the *_range knobs below) so
    the resulting trajectories form a varied TRAINING SET (many distinct
    barotropically-unstable jets) rather than the one canned test case.

Either way, each job gets its own small YAML config (just a ``run_solver:``
section, identical in shape to config.yml's) and invokes ``run_solver.py``
with it - mirroring how `make run_solver` itself is invoked (see makefile),
since run_solver.py now reads its configuration from a YAML file rather than
CLI flags.

Since each job is its own independent sbatch process, `run_solver.py` cannot
share an in-memory dataset across jobs. Instead, this script opens and loads
the (potentially large, multi-year) ERA5 dataset exactly once here (real_world
mode only), slices out just the one time point each job needs, and writes
that slice to a small per-job NetCDF file (``ic_data_dir``). Each job's config
points at its own slice via ``ic_time`` / ``ic_data_path``, so `run_solver.py`
only ever opens that tiny file instead of re-loading the full dataset per job.

Configuration is read from the ``batch_simulation`` section of config.yml
(pass an alternative path as the first CLI argument), matching the
run_solver.py / inference.py convention.

The generated sbatch scripts are cluster-agnostic by construction: every
``#SBATCH`` directive and every ``module load`` line comes from config
(the ``slurm knobs`` block below), not from anything hardcoded here. To
migrate to a different slurm cluster, only config.yml's batch_simulation
block needs to change (account/partition/gpu names, modules, python
path) -- this file itself has no cluster-specific assumptions. The
defaults here match NCSA Delta's convention (see .claude/skills/request-gpu).

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

from src.helpers.config import load_raw_config

# project root = .../Shallow_Water_System_On_Sphere (this file lives in src/entries/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# prefer this project's own venv (portable across clusters/checkouts); fall
# back to whatever python is currently running this script.
DEFAULT_PYTHON = (
    str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if (PROJECT_ROOT / ".venv" / "bin" / "python").exists()
    else sys.executable
)


DEFAULT_CONFIG = {
    # "real_world" (iterate an ERA5 dataset's own time points) or "galewsky"
    # (sample random galewsky_initial_condition parameter combinations) - see
    # module docstring.
    "mode": "real_world",

    # ------------------------- data / time selection (real_world only) -------- #
    "dataset_name": None,            # required, e.g. "1980_2025_odd_month" -> reanalysis_data/<name>/data.nc
    "time_dim": "valid_time",
    "engine": None,                  # xarray engine (e.g. 'cfgrib' for grib); default: auto-detect
    "time_filter": None,             # null | "first_of_month" (see _TIME_FILTERS); applied before stride/limit
    "stride": 1,                     # use every Nth (post-filter) time point
    "limit": None,                   # only submit the first N (post-filter, post-stride) time points

    # ------------------------- galewsky parameter sampling (galewsky only) ---- #
    "n_jobs": 100,                    # number of random parameter combinations to submit
    "seed": 0,                        # RNG seed - reruns regenerate the identical sweep
    "galewsky_umax_range": [50., 110.],
    "galewsky_usouth_range": [0.08, 0.20],
    "galewsky_unorth_range": [0.30, 0.45],
    "galewsky_perturb_loc_range": [0.10, 0.40],
    "galewsky_perturb_amp_range": [0.5, 1.5],
    "galewsky_noise_level_range": [0.5, 1.5],

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
    # exponential spectral filter, forwarded into each job's run_solver config (see
    # run_solver.py / initial_condition.rw_initial_condition): sigma=exp(-a*(l/lmax))^p
    "a": 2,
    "p": 16,
    "rad": False,                    # forwarded into each job's run_solver config (see run_solver.py)
    "tau_rad": None,
    "rad_smooth_fraction": 0.5,

    # ----------------------------- slurm knobs --------------------------------- #
    # Every one of these maps directly onto a generic sbatch flag or a raw
    # extra line (see `modules` / `extra_sbatch`); there is nothing
    # cluster-specific hardcoded in build_script(). To move to a different
    # cluster, change only the values below.
    "partition": "gpuA100x4",
    "account": "bgvu-delta-gpu",
    "time_limit": "00:01:00",        # #SBATCH --time; wall-clock budget per job, "HH:MM:SS"
    "gpus_per_node": 1,               # #SBATCH --gpus-per-node; null/0 to omit (CPU-only)
    "mem": "32G",
    "cpus": 4,
    "mail_user": None,                # #SBATCH --mail-user; null to omit
    "mail_type": None,                # #SBATCH --mail-type, e.g. "BEGIN,END,FAIL"; null to omit
    "modules": [],                    # `module load <name>` lines run before the job, e.g. ["cuda/12.4"]
    "extra_sbatch": [],               # raw extra "#SBATCH ..." lines for anything cluster-specific
    "job_name_prefix": "swe_rw",
    "python": DEFAULT_PYTHON,
    "script_dir": "slurm_scripts",
    "config_dir": "slurm_scripts/configs",
    "ic_data_dir": "slurm_scripts/ic_data",
    "log_dir": "slurm_logs",
    "dry_run": False,                # generate scripts/configs without calling sbatch
}


def load_config():
    raw = load_raw_config("batch_simulation", DEFAULT_CONFIG)

    # normalize types regardless of YAML formatting
    raw["mode"] = raw["mode"] or "real_world"
    raw["lmax"] = int(raw["lmax"])
    raw["tau"] = [int(t) for t in raw["tau"]]
    raw["cfl"] = float(raw["cfl"])
    raw["duration"] = float(raw["duration"])
    raw["save_interval_minutes"] = float(raw["save_interval_minutes"])
    raw["stride"] = int(raw["stride"])
    raw["limit"] = int(raw["limit"]) if raw["limit"] is not None else None
    raw["n_jobs"] = int(raw["n_jobs"])
    raw["seed"] = int(raw["seed"])
    for key in ("galewsky_umax_range", "galewsky_usouth_range", "galewsky_unorth_range",
                "galewsky_perturb_loc_range", "galewsky_perturb_amp_range",
                "galewsky_noise_level_range"):
        raw[key] = [float(x) for x in raw[key]]
    raw["cpus"] = int(raw["cpus"])
    raw["a"] = float(raw["a"])
    raw["p"] = float(raw["p"])
    raw["rad"] = bool(raw["rad"])
    raw["tau_rad"] = float(raw["tau_rad"]) if raw["tau_rad"] is not None else None
    raw["rad_smooth_fraction"] = float(raw["rad_smooth_fraction"])
    raw["gpus_per_node"] = int(raw["gpus_per_node"]) if raw["gpus_per_node"] else None
    raw["modules"] = list(raw["modules"] or [])
    raw["extra_sbatch"] = list(raw["extra_sbatch"] or [])
    raw["python"] = raw["python"] or DEFAULT_PYTHON

    cfg = SimpleNamespace(**raw)

    if cfg.mode not in ("real_world", "galewsky"):
        raise ValueError(f"batch_simulation.mode must be 'real_world' or 'galewsky', got {cfg.mode!r}.")
    if cfg.mode == "real_world":
        if not cfg.dataset_name:
            raise ValueError("batch_simulation.dataset_name is required when mode == 'real_world'.")
        if cfg.pressure is None:
            raise ValueError("batch_simulation.pressure is required (used for naming), e.g. 500.")
        if cfg.time_filter is not None and cfg.time_filter not in _TIME_FILTERS:
            raise ValueError(
                f"batch_simulation.time_filter must be one of {list(_TIME_FILTERS)} or null, "
                f"got {cfg.time_filter!r}."
            )
    if cfg.rad and cfg.tau_rad is None:
        raise ValueError("batch_simulation.tau_rad is required when rad=True.")
    return cfg


_TIME_FILTERS = {
    # calendar-aware pre-filters, applied before stride/limit. Add more here
    # as needed (keep them dataset-agnostic - they only look at the
    # calendar fields of time_dim's own values).
    "first_of_month": lambda t: (t.dt.day == 1) & (t.dt.hour == 0),
}


def prepare_ic_slices(dataset_name, time_dim, engine, stride, limit, ic_data_dir,
                       job_name_prefix, time_filter=None):
    """Open + load the (potentially large) ERA5 dataset exactly once, then
    write one small per-job NetCDF slice (one time point each) per selected
    job. Returns a list of (ic_time, ic_data_path) pairs, in job order.

    Slicing keeps ``time_dim`` as a length-1 dimension (``isel(..., [idx])``,
    not squeezed away), so run_solver.py's existing
    ``era5_dataset.sel(valid_time=ic_time, method='nearest').squeeze()`` call
    works unchanged against each tiny slice file.

    ``time_filter``, if given, is a key into ``_TIME_FILTERS`` (e.g.
    "first_of_month") applied to narrow the candidate time points down by
    calendar fields before ``stride``/``limit`` are applied.
    """
    netcdf_path = PROJECT_ROOT / "reanalysis_data" / dataset_name / "data.nc"
    ds = xr.open_dataset(netcdf_path, engine=engine)
    if time_dim not in ds.coords and time_dim not in ds.dims:
        raise KeyError(
            f"'{time_dim}' not found in {netcdf_path}. Available coords: {list(ds.coords)}"
        )

    # time_dim may be a non-dimension coordinate (e.g. ERA5's "valid_time",
    # indexed along a differently-named "time" dim) - isel() needs the
    # actual dimension name, while sel()/values lookups work fine off
    # time_dim itself either way.
    time_index_dim = ds[time_dim].dims[0] if time_dim not in ds.dims else time_dim

    all_times = np.atleast_1d(ds[time_dim].values)
    idxs = np.arange(len(all_times))
    if time_filter is not None:
        mask = _TIME_FILTERS[time_filter](xr.DataArray(all_times)).values
        idxs = idxs[mask]
    idxs = idxs[::stride]
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
        ds.isel(**{time_index_dim: [idx]}).to_netcdf(ic_data_path)

        jobs.append((ic_time, ic_data_path))

    ds.close()
    return jobs


def build_run_solver_config_real_world(ic_time, ic_data_path, cfg):
    """The ``run_solver:`` section for this real_world-mode job (see
    run_solver.py's DEFAULT_CONFIG)."""
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
            "a": cfg.a,
            "p": cfg.p,
            "rad": cfg.rad,
            "tau_rad": cfg.tau_rad,
            "rad_smooth_fraction": cfg.rad_smooth_fraction,
            "checkpoint_only": False,
        }
    }


def sample_galewsky_params(n_jobs, seed, cfg):
    """``n_jobs`` random galewsky_initial_condition parameter combinations,
    uniformly sampled within cfg's own *_range knobs - a fixed seed makes the
    sweep reproducible (re-running the config regenerates the identical set).
    usouth_range's max is kept below unorth_range's min (DEFAULT_CONFIG's
    defaults do this) so the jet band (usouth, unorth) is always a valid,
    non-empty latitude band regardless of what's independently sampled for
    each."""
    rng = np.random.default_rng(seed)
    keys = ("umax", "usouth", "unorth", "perturb_loc", "perturb_amp", "noise_level")
    sampled = {
        key: rng.uniform(*getattr(cfg, f"galewsky_{key}_range"), size=n_jobs)
        for key in keys
    }
    return [{key: float(sampled[key][i]) for key in keys} for i in range(n_jobs)]


def build_run_solver_config_galewsky(params, variant_tag, cfg):
    """The ``run_solver:`` section for this galewsky-mode job."""
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
            "ic": "galewsky",
            "galewsky_umax": params["umax"],
            "galewsky_usouth": params["usouth"],
            "galewsky_unorth": params["unorth"],
            "galewsky_perturb_loc": params["perturb_loc"],
            "galewsky_perturb_amp": params["perturb_amp"],
            "galewsky_noise_level": params["noise_level"],
            "variant_tag": variant_tag,
            "checkpoint_only": False,
        }
    }


def render_sbatch_script(job_name, job_config_path, echo_line, cfg, abs_log_dir):
    """Render this job's sbatch script text (shared by both modes). Every
    directive here comes straight from config (cfg.*) -- nothing
    cluster-specific is baked into this function itself, so retargeting the
    pipeline at a different slurm cluster only means editing config.yml."""
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
    if cfg.gpus_per_node:
        sbatch_lines.append(f"#SBATCH --gpus-per-node={cfg.gpus_per_node}")
    if cfg.mail_user:
        sbatch_lines.append(f"#SBATCH --mail-user={cfg.mail_user}")
    if cfg.mail_type:
        sbatch_lines.append(f"#SBATCH --mail-type={cfg.mail_type}")
    for line in cfg.extra_sbatch:
        sbatch_lines.append(line if line.startswith("#SBATCH") else f"#SBATCH {line}")

    body = [
        "",
        "set -euo pipefail",
    ]
    if cfg.modules:
        body.append("module purge")
        body += [f"module load {name}" for name in cfg.modules]
    body += [
        f"cd {PROJECT_ROOT}",
        f'echo "{echo_line}"',
        # mirrors `make run_solver` (see makefile): run_solver.py is invoked as a
        # module and takes its config as a single positional YAML path.
        f"{cfg.python} -m src.entries.run_solver {job_config_path}",
        "",
    ]
    return "\n".join(sbatch_lines + body)


def main():
    cfg = load_config()

    script_dir = (PROJECT_ROOT / cfg.script_dir).resolve()
    log_dir = (PROJECT_ROOT / cfg.log_dir).resolve()
    config_dir = (PROJECT_ROOT / cfg.config_dir).resolve()
    ic_data_dir = (PROJECT_ROOT / cfg.ic_data_dir).resolve()
    for d in (script_dir, log_dir, config_dir, ic_data_dir):
        d.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "real_world":
        print(f"Loading '{cfg.dataset_name}' once and slicing per-job initial conditions...")
        raw_jobs = prepare_ic_slices(
            cfg.dataset_name, cfg.time_dim, cfg.engine, cfg.stride, cfg.limit,
            ic_data_dir, cfg.job_name_prefix, time_filter=cfg.time_filter,
        )
        if not raw_jobs:
            sys.exit("No time points selected; nothing to submit.")
        jobs = [
            (f"{cfg.job_name_prefix}_{i:04d}",
             build_run_solver_config_real_world(ic_time, ic_data_path, cfg),
             f"Running real-world simulation for ic_time={ic_time}")
            for i, (ic_time, ic_data_path) in enumerate(raw_jobs)
        ]
    else:
        print(f"Sampling {cfg.n_jobs} galewsky parameter combination(s) (seed={cfg.seed})...")
        param_sets = sample_galewsky_params(cfg.n_jobs, cfg.seed, cfg)
        jobs = [
            (f"{cfg.job_name_prefix}_{i:04d}",
             build_run_solver_config_galewsky(params, f"{cfg.job_name_prefix}_{i:04d}", cfg),
             f"Running galewsky simulation: {params}")
            for i, params in enumerate(param_sets)
        ]

    print(f"Selected {len(jobs)} job(s); "
          f"{'generating (dry run)' if cfg.dry_run else 'submitting'} jobs...")

    submitted = 0
    for job_name, run_solver_config, echo_line in jobs:
        job_config_path = config_dir / f"{job_name}.yml"
        job_config_path.write_text(yaml.safe_dump(run_solver_config, sort_keys=False))

        script_text = render_sbatch_script(job_name, job_config_path, echo_line, cfg, log_dir)
        script_path = script_dir / f"{job_name}.slurm"
        script_path.write_text(script_text)

        if cfg.dry_run:
            print(f"[dry-run] {script_path}  (config={job_config_path})")
            continue

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[FAILED] {job_name}: {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"[ok] {job_name} -> {result.stdout.strip()}")
            submitted += 1

    if cfg.dry_run:
        print(f"Wrote {len(jobs)} script(s) to {script_dir}")
    else:
        print(f"Submitted {submitted}/{len(jobs)} job(s).")


if __name__ == "__main__":
    main()
