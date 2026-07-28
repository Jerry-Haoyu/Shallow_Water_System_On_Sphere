#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Solve the SWE on the sphere with the psuedospectral method.

Configuration is read from the ``run_solver`` section of config.yml (pass an
alternative path as the first CLI argument). The entry point follows the
data/model organization convention (README.md):

  1. Resolve where the trajectory would live per the convention. If it already
     exists, there is nothing to do.
  2. Otherwise resolve the solver checkpoint. If it does not exist, initialize
     the solver and save it (a design choice to mirror the ML checkpointing,
     even though re-initializing a numerical solver is cheap).
  3. Generate the initial condition and call run() in src/helpers/run_model.py,
     which simulates and stores the trajectory following the convention.

@author Haoyu Tang hytang2@illinois.edu
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.numerical_solver.initial_condition import *
from src.helpers.run_model import (
    run,
    numerical_checkpoint_path,
    numerical_data_path,
)
from src.helpers.print import print_in_box


DEFAULT_CONFIG = {
    # solver / resolution configuration
    "lmax": 64,
    "tau": [30000, 30000, 30],
    "cfl": 0.25,
    "grid": "equiangular",          # equiangular | legendre-gauss | lobatto
    "semi_implicit": True,
    "dealias": True,
    # simulation
    "duration": 5,                  # days
    "save_interval_minutes": 30,
    # initial condition
    "ic": "galewsky",               # galewsky | real_world
    "netcdf_path": None,            # required when ic == real_world
    "ic_time": None,                # required when ic == real_world
    "pressure": None,               # for naming (real_world), e.g. 500
    # only initialize + save the solver checkpoint, skip the simulation
    "checkpoint_only": False,
}


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) or {}

    raw = DEFAULT_CONFIG | config.get("run_solver", {})

    # normalize types regardless of YAML formatting
    raw["lmax"] = int(raw["lmax"])
    raw["tau"] = [int(t) for t in raw["tau"]]
    raw["cfl"] = float(raw["cfl"])
    raw["duration"] = float(raw["duration"])
    raw["save_interval_minutes"] = float(raw["save_interval_minutes"])

    cfg = SimpleNamespace(**raw)

    if cfg.ic == "real_world":
        if not cfg.ic_time:
            raise ValueError("run_solver.ic_time is required when ic == real_world.")
        if not cfg.netcdf_path:
            raise ValueError("run_solver.netcdf_path is required when ic == real_world.")
        if cfg.pressure is None:
            raise ValueError("run_solver.pressure is required when ic == real_world (used for naming).")
    return cfg


def main():
    cfg = load_config()

    # ------------------------------------------------------------------ #
    # (1) Where would the trajectory live? If it already exists, stop.
    # ------------------------------------------------------------------ #
    data_dir, data_file = numerical_data_path(
        lmax=cfg.lmax, tau=cfg.tau, grid=cfg.grid, semi_implicit=cfg.semi_implicit,
        duration=cfg.duration, ic=cfg.ic, pressure=cfg.pressure, ic_time=cfg.ic_time,
    )
    data_path = Path(data_dir) / data_file

    print_in_box({
        "title": "Run Solver",
        "lines": [
            f"lmax = {cfg.lmax} | tau = {cfg.tau} | grid = {cfg.grid} | semi_implicit = {cfg.semi_implicit}",
            f"duration = {cfg.duration} days | save_interval = {cfg.save_interval_minutes} min",
            f"ic = {cfg.ic}"
            # f"data -> {data_path}",
        ],
    })

    if not cfg.checkpoint_only and data_path.is_file():
        print(f"✅ Trajectory already exists at {data_path}; nothing to do.")
        return

    # ------------------------------------------------------------------ #
    # (2) Fall back to the solver checkpoint. Initialize + save if absent.
    # ------------------------------------------------------------------ #
    ckpt_dir, ckpt_file = numerical_checkpoint_path(
        lmax=cfg.lmax, tau=cfg.tau, grid=cfg.grid, semi_implicit=cfg.semi_implicit,
    )
    ckpt_path = Path(ckpt_dir) / ckpt_file

    if ckpt_path.is_file():
        print(f"📦 Loading existing solver checkpoint {ckpt_path}")
        solver = torch.load(ckpt_path, weights_only=False)
        solver.to(solver.device)
    else:
        print(f"🛠  Initializing solver checkpoint -> {ckpt_path}")
        solver = ShallowWaterSolver(
            cfg.lmax, cfg.tau, cfg.cfl, grid=cfg.grid,
            dealias=cfg.dealias, semi_implicit=cfg.semi_implicit,
        )
        solver.to(solver.device)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(solver, ckpt_path)

    if cfg.checkpoint_only:
        print(f"✅ checkpoint_only set; saved solver checkpoint at {ckpt_path}.")
        return

    # ------------------------------------------------------------------ #
    # (3) Generate the initial condition and simulate the trajectory.
    # ------------------------------------------------------------------ #
    if cfg.ic == "galewsky":
        phivrtdivspec_0 = galewsky_initial_condition(model=solver)
    elif cfg.ic == "real_world":
        phivrtdivspec_0 = rw_initial_condition(
            model=solver, netcdf_path=cfg.netcdf_path, ic_time=cfg.ic_time)
    else:
        raise ValueError(f"Unknown ic '{cfg.ic}' (expected galewsky or real_world).")

    run(
        model_checkpoint=str(ckpt_path),
        initial_condition=phivrtdivspec_0,
        output_dir=data_dir,
        file_name=data_file,
        duration=cfg.duration,
        save_interval_minutes=cfg.save_interval_minutes,
    )


if __name__ == "__main__":
    main()
