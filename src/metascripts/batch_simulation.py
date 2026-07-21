#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Generate slurm scripts to batch numerical simulation with different initial conditions.

Given a netcdf (ERA5-style) file and the solver configuration, this enumerates every
time point along the ``time`` coordinate and submits one independent sbatch job per
time point. Each job invokes ``ps_solver_entry.py`` with ``--ic real_world`` and the
corresponding ``--ic_time`` so that every simulation starts from the real-world state
at a different date.

Example
-------
    python -m src.metascripts.batch_simulation \
        --netcdf_path reanalysis_data/2000_2025_odd_months.nc \
        --lmax 64 --tau 30000 30000 5 --duration 20 --semi_implicit \
        --output_dir simulation_output/training_data \
        --partition gpu --account gzhang13-group --gres gpu:1

Add ``--dry_run`` to generate the scripts without submitting them.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# project root = .../Shallow_Water_System_On_Sphere (this file lives in src/metascripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = "/data/gzhang13/a/hytang2/envs/swe/bin/python"


def parse_args():
    p = argparse.ArgumentParser(
        prog="batch_simulation",
        description="Submit one SWE simulation per time point of a netcdf file.",
    )

    # ------------------------- data / time selection ------------------------- #
    p.add_argument("--netcdf_path", type=str, required=True,
                   help="Path to the (pressure-averaged) netcdf file whose time points seed the runs.")
    p.add_argument("--engine", type=str, default=None,
                   help="xarray engine to open the file (e.g. 'cfgrib' for grib). Default: auto-detect.")
    p.add_argument("--time_dim", type=str, default="time",
                   help="Name of the time coordinate to enumerate. Default: time.")
    p.add_argument("--stride", type=int, default=1,
                   help="Use every Nth time point. Default: 1 (all).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only submit the first N (post-stride) time points. Default: all.")

    # ------------------------- solver configuration -------------------------- #
    # forwarded verbatim to ps_solver_entry.py
    p.add_argument("--lmax", type=int, required=True)
    p.add_argument("--tau", type=int, nargs=3, required=True,
                   help="Hyperdiffusion e-fold times (tau2 tau4 tau8), in hours.")
    p.add_argument("--cfl", type=float, default=0.25)
    p.add_argument("--grid", type=str, default="legendre-gauss")
    p.add_argument("--semi_implicit", action="store_true")
    p.add_argument("--dealias", action="store_true")
    p.add_argument("--duration", type=float, default=5)
    p.add_argument("--save_interval_minutes", type=float, default=30,
                   help="Simulated-time interval (minutes) between saved trajectory frames.")
    p.add_argument("--output_dir", type=str, default="simulation_output/psuedo_spectral/rw",
                   help="Directory the solver writes trajectory .pt files to.")

    # ----------------------------- slurm knobs ------------------------------- #
    p.add_argument("--partition", type=str, default="gpu")
    p.add_argument("--account", type=str, default="gzhang13-group")
    p.add_argument("--time_limit", type=str, default="12:00:00",
                   help="Per-job wallclock limit (SBATCH --time). Default: 12:00:00.")
    p.add_argument("--gres", type=str, default="gpu:1",
                   help="SBATCH --gres. Pass '' to omit (CPU-only). Default: gpu:1.")
    p.add_argument("--mem", type=str, default="32G")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--job_name_prefix", type=str, default="swe_rw")
    p.add_argument("--python", type=str, default=DEFAULT_PYTHON,
                   help="Python interpreter used inside the job.")
    p.add_argument("--script_dir", type=str, default="slurm_scripts",
                   help="Where generated .slurm scripts are written.")
    p.add_argument("--log_dir", type=str, default="slurm_logs",
                   help="Where sbatch stdout/stderr logs are written.")
    p.add_argument("--dry_run", action="store_true",
                   help="Generate the scripts but do not call sbatch.")

    return p.parse_args()


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
    # sel(time=..., method='nearest') parses these unambiguously.
    return [np.datetime_as_string(t, unit="s") for t in times]


def build_script(ic_time, index, args, abs_output_dir, abs_log_dir):
    """Render the sbatch script text for a single time point."""
    job_name = f"{args.job_name_prefix}_{index:04d}"

    solver_flags = [
        "--ic", "real_world",
        "--netcdf_path", str(Path(args.netcdf_path).resolve()),
        "--ic_time", ic_time,
        "--lmax", str(args.lmax),
        "--tau", *[str(t) for t in args.tau],
        "--cfl", str(args.cfl),
        "--grid", args.grid,
        "--duration", str(args.duration),
        "--save_interval_minutes", str(args.save_interval_minutes),
        "--output_dir", str(abs_output_dir),
    ]
    if args.semi_implicit:
        solver_flags.append("--semi_implicit")
    if args.dealias:
        solver_flags.append("--dealias")
    solver_cmd = " ".join(solver_flags)

    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --account={args.account}",
        f"#SBATCH --time={args.time_limit}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --output={abs_log_dir}/{job_name}_%j.out",
        f"#SBATCH --error={abs_log_dir}/{job_name}_%j.err",
    ]
    if args.gres:
        sbatch_lines.insert(6, f"#SBATCH --gres={args.gres}")

    body = [
        "",
        "set -euo pipefail",
        f"cd {PROJECT_ROOT}",
        f'echo "Running real-world simulation for ic_time={ic_time}"',
        f"{args.python} ps_solver_entry.py {solver_cmd}",
        "",
    ]
    return "\n".join(sbatch_lines + body)


def main():
    args = parse_args()

    times = enumerate_times(
        args.netcdf_path, args.time_dim, args.engine, args.stride, args.limit
    )
    if not times:
        sys.exit("No time points selected; nothing to submit.")

    script_dir = (PROJECT_ROOT / args.script_dir).resolve()
    log_dir = (PROJECT_ROOT / args.log_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    for d in (script_dir, log_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(times)} time point(s); "
          f"{'generating (dry run)' if args.dry_run else 'submitting'} jobs...")

    submitted = 0
    for i, ic_time in enumerate(times):
        script_text = build_script(ic_time, i, args, output_dir, log_dir)
        script_path = script_dir / f"{args.job_name_prefix}_{i:04d}.slurm"
        script_path.write_text(script_text)

        if args.dry_run:
            print(f"[dry-run] {script_path}  (ic_time={ic_time})")
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

    if args.dry_run:
        print(f"Wrote {len(times)} script(s) to {script_dir}")
    else:
        print(f"Submitted {submitted}/{len(times)} job(s).")


if __name__ == "__main__":
    main()
