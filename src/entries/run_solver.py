#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Solve the SWE on the sphere with the psuedospectral method.

Configuration is read from the ``run_solver`` section of config.yml (pass an
alternative path as the first CLI argument). The entry point follows the
data/model organization convention (README.md):

  1. Resolve the solver checkpoint. If it does not exist, initialize the
     solver and save it (a design choice to mirror the ML checkpointing, even
     though re-initializing a numerical solver is cheap).
  2. Generate the initial condition and call run() in src/helpers/run_model.py.
     run() derives where the trajectory lives entirely from the checkpoint's
     own path (the data tree is isomorphic to the checkpoint tree) plus
     (duration, ic, ...), and skips the simulation if it's already there.

@author Haoyu Tang hytang2@illinois.edu
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml
from torch_harmonics.sht import RealVectorSHT

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.numerical_solver.initial_condition import *
from src.helpers.run_model import (
    run,
    numerical_checkpoint_path,
    save_numerical_checkpoint,
    load_numerical_checkpoint,
    load_h_stats,
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
    "non_dimensional": True,        # rescale by dataset-wide U=sqrt(g*h_avg), T=radius/U
    # simulation
    "duration": 5,                  # days
    "save_interval_minutes": 30,
    # initial condition
    "ic": "galewsky",               # galewsky | real_world
    "dataset_name": None,           # required when ic == real_world, e.g. "1980_2025_odd_month"
    "ic_time": None,                # required when ic == real_world
    "ic_data_path": None,           # optional (real_world): pre-sliced NetCDF (see batch_simulation.py)
                                     # to open instead of the full dataset_name/data.nc
    "pressure": None,               # for naming (real_world), e.g. 500
    # radiative relaxation of geopotential (real_world ic only, see ShallowWaterSolver's
    # rad/tau_rad and initial_condition.radiative_equilibrium_geopotential)
    "rad": False,
    "tau_rad": None,                # relaxation e-fold time (days); required when rad=True
    "rad_smooth_fraction": 0.5,     # keep spherical-harmonic degrees l <= this*lmax in phi_eq's u profile
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
    raw["rad"] = bool(raw["rad"])
    raw["tau_rad"] = float(raw["tau_rad"]) if raw["tau_rad"] is not None else None
    raw["rad_smooth_fraction"] = float(raw["rad_smooth_fraction"])

    cfg = SimpleNamespace(**raw)

    if cfg.ic == "real_world":
        if not cfg.ic_time:
            raise ValueError("run_solver.ic_time is required when ic == real_world.")
        if not cfg.dataset_name:
            raise ValueError("run_solver.dataset_name is required when ic == real_world.")
        if cfg.pressure is None:
            raise ValueError("run_solver.pressure is required when ic == real_world (used for naming).")
    if cfg.rad:
        if cfg.ic != "real_world":
            raise ValueError("run_solver.rad=True requires ic == real_world (needs the ERA5 climatology).")
        if cfg.tau_rad is None:
            raise ValueError("run_solver.tau_rad is required when rad=True.")
    return cfg


# fixed categorical order (dataviz skill): slot 1/2/3 hold up all-pairs CVD checks
_DIAG_COLORS = {"geopotential": "#2a78d6", "vorticity": "#eb6834", "divergence": "#1baf7a"}


def plot_trajectory_diagnostics(save_path, output_dir):
    """Line plot of the spatial average (mean +/- amplitude band) of geopotential,
    vorticity and divergence across a saved trajectory.

    Both the mean and the amplitude (RMS deviation around the mean) are read
    directly off each frame's Y_0^0 / degree-l spherical-harmonic coefficients
    (orthonormal real SHT: l=0,m=0 is the domain mean up to sqrt(4*pi), and
    Parseval's theorem gives the mean-square from the coefficient magnitudes),
    not by reconstructing the grid and integrating - same technique as
    src/analyze/statistics.py's compute_h_stats_ic.
    """
    data = torch.load(save_path, weights_only=False)
    trajectory = data["trajectory"]  # (N, 3, lmax, mmax) complex: phi, vrt, div
    save_interval_minutes = data["metadata"]["save_interval_minutes"]

    sqrt_4pi = float(4.0 * np.pi) ** 0.5
    n_frames = trajectory.shape[0]
    hours = np.arange(n_frames) * save_interval_minutes / 60.0

    fig, ax = plt.subplots(3, figsize=(8, 4.5))
    for i, name in enumerate(["geopotential", "vorticity", "divergence"]):
        coeff = trajectory[:, i]  # (N, lmax, mmax) complex
        mean = coeff[:, 0, 0].real / sqrt_4pi
        meansq = (coeff[:, :, 0].abs() ** 2).sum(dim=-1) + 2.0 * (coeff[:, :, 1:].abs() ** 2).sum(dim=(-2, -1))
        meansq = meansq / (4.0 * np.pi)
        amp = torch.sqrt((meansq - mean ** 2).clamp_min(0))

        mean_np, amp_np = mean.numpy(), amp.numpy()
        color = _DIAG_COLORS[name]
        ax[i].plot(hours, mean_np, label=name, color=color, linewidth=2)
        ax[i].fill_between(hours, mean_np - amp_np, mean_np + amp_np, color=color, alpha=0.15, linewidth=0)
        ax[i].set_title(f"Trajectory diagnostics: {name}")

        ax2 = ax[i].secondary_yaxis('right')
        ax2.set_ylabel('Change(%)')
        ax[i].plot(hours, np.abs(mean_np - mean_np[0])/mean_np[0], color=color, linestyle='dashed', label=r'$\frac{\mu_{t}-\mu_0}{\mu_0}$')
        ax[i].axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
        ax[i].grid(True, color="#e1e0d9", linewidth=0.8)
        ax[i].legend(frameon=False, loc='upper right')
        
    fig.supxlabel('Time (hours)')
    fig.supylabel('Diagnostic Distribution (μ±σ)')

    fig.tight_layout()

    out_path = Path(output_dir) / f"{Path(save_path).stem}_diagnostics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"📈 Saved trajectory diagnostics plot -> {out_path}")
    return out_path


def main():
    cfg = load_config()

    # real-world runs derive h_avg/h_amp from the dataset, which feeds into the
    # solver's non-dimensionalization (dt, hyperdiffusion, Coriolis) - so the
    # checkpoint (and, isomorphically, the data run() derives from it) is
    # keyed by dataset_name too (see numerical_checkpoint_path).
    dataset_name = cfg.dataset_name if cfg.ic == "real_world" else None
    netcdf_path = Path("reanalysis_data") / cfg.dataset_name / "data.nc" if cfg.ic == "real_world" else None

    h_avg = h_amp = None
    if cfg.ic == "real_world":
        h_avg, h_amp = load_h_stats(cfg.dataset_name)
        print(f"    Using dataset-wide h_avg={h_avg:.2f} m, h_amp={h_amp:.2f} m from '{cfg.dataset_name}'")

    print_in_box({
        "title": "Run Solver",
        "lines": [
            f"lmax = {cfg.lmax} | tau = {cfg.tau} | grid = {cfg.grid} | semi_implicit = {cfg.semi_implicit}",
            f"duration = {cfg.duration} days | save_interval = {cfg.save_interval_minutes} min",
            f"ic = {cfg.ic} | non_dimensional = {cfg.non_dimensional}",
            f"rad = {cfg.rad}" + (f" | tau_rad = {cfg.tau_rad} days | rad_smooth_fraction = {cfg.rad_smooth_fraction}" if cfg.rad else ""),
        ],
    })

    # ------------------------------------------------------------------ #
    # (1) Resolve the solver checkpoint. Initialize + save if absent.
    #     Trajectory-path deduction and reuse (is the output already there?)
    #     are entirely delegated to run() below - it derives the data path
    #     straight from ckpt_dir itself (see run_model.py's
    #     _data_path_from_checkpoint).
    # ------------------------------------------------------------------ #
    ckpt_dir = numerical_checkpoint_path(
        lmax=cfg.lmax, tau=cfg.tau, grid=cfg.grid, semi_implicit=cfg.semi_implicit,
        rad=cfg.rad, dataset_name=dataset_name,
    )
    info_path = Path(ckpt_dir) / "model_info.json"

    if info_path.is_file():
        print(f"📦 Loading existing solver checkpoint {ckpt_dir}")
        solver, _ = load_numerical_checkpoint(ckpt_dir)
        solver.to(solver.device)
    else:
        print(f"🛠  Initializing solver checkpoint -> {ckpt_dir}")
        solver = ShallowWaterSolver(
            cfg.lmax, cfg.tau, cfg.cfl, grid=cfg.grid,
            dealias=cfg.dealias, semi_implicit=cfg.semi_implicit,
            h_avg=h_avg, h_amp=h_amp, non_dimensional=cfg.non_dimensional,
            rad=cfg.rad, tau_rad=cfg.tau_rad,
        )
        solver.to(solver.device)
        save_numerical_checkpoint(
            solver, ckpt_dir, dataset_name=dataset_name,
            pressure=cfg.pressure if cfg.ic == "real_world" else None,
        )

    if cfg.checkpoint_only:
        print(f"✅ checkpoint_only set; saved solver checkpoint at {ckpt_dir}.")
        return

    # ------------------------------------------------------------------ #
    # (2) Generate the initial condition and simulate the trajectory.
    # ------------------------------------------------------------------ #
    # phi_eq_spec is only ever set for cfg.rad (real_world) below; run() needs it
    # explicitly (not via solver.set_equilibrium_geopotential here) since it
    # reconstructs its own solver from ckpt_dir internally - see run_model.run().
    phi_eq_spec = None
    if cfg.ic == "galewsky":
        phivrtdivspec_0 = galewsky_initial_condition(model=solver)
    elif cfg.ic == "real_world":
        # a pre-sliced per-job file (see batch_simulation.py) avoids opening
        # the full (potentially large, multi-year) ERA5 dataset in every job.
        ic_source_path = Path(cfg.ic_data_path) if cfg.ic_data_path else netcdf_path

        ds = xr.open_dataset(ic_source_path)
        nlat_data, nlon_data = ds.sizes['latitude'], ds.sizes['longitude']
        ds.close()
        vSHT = RealVectorSHT(nlat=nlat_data, nlon=nlon_data, lmax=nlat_data// 2, mmax=nlat_data // 2,
                            grid='equiangular', csphase=False).to(solver.device)

        start_load_time = time.perf_counter()
        era5_dataset = xr.open_dataset(ic_source_path).load()
        finish_load_time = time.perf_counter()
        print(f"Finished loading ERA5 data in {(finish_load_time - start_load_time):2f} seconds")
        phivrtdivspec_0 = rw_initial_condition(
            model=solver,
            vSHT=vSHT,
            era5_dataset=era5_dataset,
            ic_time=cfg.ic_time)

        if cfg.rad:
            # the day-of-year climatology needs the FULL multi-year dataset, but
            # era5_dataset above may be a single-time-point per-job slice (see
            # batch_simulation.py's ic_data_path) - open the full dataset
            # separately for the climatology in that case.
            climatology_dataset = era5_dataset if not cfg.ic_data_path else xr.open_dataset(netcdf_path).load()
            phi_eq_spec = radiative_equilibrium_geopotential(
                model=solver, vSHT=vSHT, clim_ds=day_of_year_climatology(climatology_dataset), ic_time=cfg.ic_time,
                smooth_fraction=cfg.rad_smooth_fraction, log=True,
            )
            if cfg.ic_data_path:
                climatology_dataset.close()
    else:
        raise ValueError(f"Unknown ic '{cfg.ic}' (expected galewsky or real_world).")

    save_path = run(
        model_checkpoint=str(ckpt_dir),
        initial_condition=phivrtdivspec_0,
        duration=cfg.duration,
        ic=cfg.ic,
        pressure=cfg.pressure,
        ic_time=cfg.ic_time,
        dataset_name=dataset_name,
        save_interval_minutes=cfg.save_interval_minutes,
        phi_eq_spec=phi_eq_spec,
    )

    plot_trajectory_diagnostics(save_path, Path(save_path).parent)


if __name__ == "__main__":
    main()
