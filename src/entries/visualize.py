#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Visualization entry point: renders animations/comparison plots from saved SWE
trajectory ``.pt`` files (the ``{"metadata", "trajectory"}`` dict written by
``src/helpers/run_model.py``'s ``run()``) via ``src/analyze/visualization.py``.

Three independent tasks, each configured by its own top-level config.yml
section (so they can be tuned/run independently) and its own `make` target:

    make animate_trajectory        -> animate_trajectory config block
    make plot_rollout_comparison   -> plot_rollout_comparison config block
    make animate_spectrum          -> animate_spectrum config block

Usage: python -m src.entries.visualize <task> [config_path]
    <task>      : animate_trajectory | plot_rollout_comparison | animate_spectrum
    config_path : defaults to config.yml

@author Haoyu Tang hytang2@illinois.edu
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

from types import SimpleNamespace

import torch
import yaml

from src.analyze.visualization import (
    animate_swe_on_sphere,
    animate_swe_on_box,
    plot_sphere_comparison,
    plot_box_comparison,
    animate_spectrum as _animate_spectrum,
)
from src.helpers.print import print_in_box

_PROJECTIONS = ("sphere", "box", "both")

DEFAULT_CONFIG = {
    "animate_trajectory": {
        "trajectory_path": None,      # required: a saved trajectory .pt (see run_model.run())
        "projection": "sphere",       # sphere | box | both
        "variables": ["phi", "vorticity", "divergence"],  # see visualization._VAR_INFO
        "output_dir": "visualization_output/animate_trajectory",
        "fps": 15,
        "coarsen_factor": 1,          # use every Nth frame
        "coastline": False,           # box projection only: overlay cartopy coastlines
    },
    "plot_rollout_comparison": {
        "ref_trajectory_path": None,  # required: reference/ground-truth trajectory .pt
        "inf_trajectory_path": None,  # required: comparison (e.g. SFNO inference) trajectory .pt
        "projection": "both",         # sphere | box | both
        "vars": ["pv"],               # see visualization._VAR_INFO; one output file per var
        "hours": [0, 1, 2, 4, 5],     # hours since each trajectory's own frame 0
        "ref_label": "Ground Truth",
        "inf_label": "Inference",
        "output_dir": "visualization_output/plot_rollout_comparison",
        "elev": 15,                   # sphere projection only (view angle)
        "azim": 35,                   # sphere projection only (view angle)
        "figsize": None,              # [width, height]; null = auto-sized
    },
    "animate_spectrum": {
        "trajectory_path": None,      # required: a saved trajectory .pt (see run_model.run())
        "output_dir": "visualization_output/animate_spectrum",
        "time_coarsen_factor": 2,     # use every Nth frame
    },
}


def load_config(task, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    raw = DEFAULT_CONFIG[task] | (config.get(task) or {})
    cfg = SimpleNamespace(**raw)
    if getattr(cfg, "projection", "both") not in _PROJECTIONS:
        raise ValueError(f"{task}.projection must be one of {_PROJECTIONS}, got {cfg.projection!r}")
    return cfg


def _load_trajectory(path, label):
    if not path:
        raise ValueError(f"visualize.py: '{label}' is required in the config.")
    return torch.load(path, map_location="cpu", weights_only=False)


def run_animate_trajectory(cfg):
    print_in_box({
        "title": "Animate Trajectory",
        "lines": [
            f"trajectory = {cfg.trajectory_path}",
            f"projection = {cfg.projection} | variables = {cfg.variables}",
            f"fps = {cfg.fps} | coarsen_factor = {cfg.coarsen_factor} | coastline = {cfg.coastline}",
            f"output_dir = {cfg.output_dir}",
        ],
    })
    data = _load_trajectory(cfg.trajectory_path, "trajectory_path")

    if cfg.projection in ("sphere", "both"):
        animate_swe_on_sphere(data, cfg.variables, cfg.output_dir,
                              fps=cfg.fps, coarsen_factor=cfg.coarsen_factor)
    if cfg.projection in ("box", "both"):
        animate_swe_on_box(data, cfg.variables, cfg.output_dir,
                           fps=cfg.fps, coarsen_factor=cfg.coarsen_factor, coastline=cfg.coastline)


def run_plot_rollout_comparison(cfg):
    print_in_box({
        "title": "Plot Rollout Comparison",
        "lines": [
            f"ref = {cfg.ref_trajectory_path}",
            f"inf = {cfg.inf_trajectory_path}",
            f"projection = {cfg.projection} | vars = {cfg.vars} | hours = {cfg.hours}",
            f"output_dir = {cfg.output_dir}",
        ],
    })
    ref_data = _load_trajectory(cfg.ref_trajectory_path, "ref_trajectory_path")
    inf_data = _load_trajectory(cfg.inf_trajectory_path, "inf_trajectory_path")
    figsize = tuple(cfg.figsize) if cfg.figsize else None

    for var in cfg.vars:
        if cfg.projection in ("sphere", "both"):
            plot_sphere_comparison(
                ref_data=ref_data, inf_data=inf_data, var=var, hours=cfg.hours,
                output_path=f"{cfg.output_dir}/{var}_sphere_comparison.png",
                ref_label=cfg.ref_label, inf_label=cfg.inf_label,
                elev=cfg.elev, azim=cfg.azim, figsize=figsize,
            )
        if cfg.projection in ("box", "both"):
            plot_box_comparison(
                ref_data=ref_data, inf_data=inf_data, var=var, hours=cfg.hours,
                output_path=f"{cfg.output_dir}/{var}_box_comparison.png",
                ref_label=cfg.ref_label, inf_label=cfg.inf_label, figsize=figsize,
            )


def run_animate_spectrum(cfg):
    print_in_box({
        "title": "Animate Spectrum",
        "lines": [
            f"trajectory = {cfg.trajectory_path}",
            f"time_coarsen_factor = {cfg.time_coarsen_factor}",
            f"output_dir = {cfg.output_dir}",
        ],
    })
    data = _load_trajectory(cfg.trajectory_path, "trajectory_path")
    _animate_spectrum(data, cfg.output_dir, time_coarsen_factor=cfg.time_coarsen_factor)


RUNNERS = {
    "animate_trajectory": run_animate_trajectory,
    "plot_rollout_comparison": run_plot_rollout_comparison,
    "animate_spectrum": run_animate_spectrum,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in RUNNERS:
        sys.exit(f"Usage: python -m src.entries.visualize <task> [config_path]\n"
                 f"  <task> must be one of {tuple(RUNNERS)}")
    task = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yml"
    cfg = load_config(task, config_path)
    RUNNERS[task](cfg)


if __name__ == "__main__":
    main()
