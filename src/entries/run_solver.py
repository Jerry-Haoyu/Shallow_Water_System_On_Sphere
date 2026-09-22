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
import xarray as xr
from torch_harmonics.sht import RealVectorSHT

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.numerical_solver.initial_condition import *
from src.analyze.visualization import plot_trajectory_diagnostics
from src.helpers.config import load_raw_config
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
    # galewsky ic only - overrides for galewsky_initial_condition's own defaults
    # (see initial_condition.py), e.g. for sweeping jet placement/strength/
    # perturbation across many jobs (see batch_simulation.py's galewsky mode)
    # rather than always the one canned test case.
    "galewsky_umax": 80.,
    "galewsky_usouth": 1 / 7,
    "galewsky_unorth": 5 / 14,
    "galewsky_perturb_loc": 0.25,
    "galewsky_perturb_amp": 1.,
    "galewsky_noise_level": 1,
    # galewsky ic only - distinguishes this run's output filename from another
    # galewsky run off the same checkpoint (see run_model.run()'s variant_tag);
    # galewsky otherwise has no ic_time of its own to make the filename unique.
    "variant_tag": None,
    # exponential spectral filter applied by rw_initial_condition in place of a hard
    # truncation when bringing real-world data down to lmax: sigma=exp(-a*(l/lmax))^p
    "a": 2,
    "p": 16,
    # radiative relaxation of geopotential (real_world ic only, see ShallowWaterSolver's
    # rad/tau_rad and initial_condition.radiative_equilibrium_geopotential)
    "rad": False,
    "tau_rad": None,                # relaxation e-fold time (days); required when rad=True
    "rad_smooth_fraction": 0.5,     # keep spherical-harmonic degrees l <= this*lmax in phi_eq's u profile
    # only initialize + save the solver checkpoint, skip the simulation
    "checkpoint_only": False,
}


def load_config():
    raw = load_raw_config("run_solver", DEFAULT_CONFIG)

    # normalize types regardless of YAML formatting
    raw["lmax"] = int(raw["lmax"])
    raw["tau"] = [int(t) for t in raw["tau"]]
    raw["cfl"] = float(raw["cfl"])
    raw["duration"] = float(raw["duration"])
    raw["save_interval_minutes"] = float(raw["save_interval_minutes"])
    raw["rad"] = bool(raw["rad"])
    raw["tau_rad"] = float(raw["tau_rad"]) if raw["tau_rad"] is not None else None
    raw["rad_smooth_fraction"] = float(raw["rad_smooth_fraction"])
    raw["a"] = float(raw["a"])
    raw["p"] = float(raw["p"])
    raw["galewsky_umax"] = float(raw["galewsky_umax"])
    raw["galewsky_usouth"] = float(raw["galewsky_usouth"])
    raw["galewsky_unorth"] = float(raw["galewsky_unorth"])
    raw["galewsky_perturb_loc"] = float(raw["galewsky_perturb_loc"])
    raw["galewsky_perturb_amp"] = float(raw["galewsky_perturb_amp"])
    raw["galewsky_noise_level"] = float(raw["galewsky_noise_level"])

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
            f"ic = {cfg.ic} | non_dimensional = {cfg.non_dimensional}"
            + (f" | filter a = {cfg.a} | filter p = {cfg.p}" if cfg.ic == "real_world" else ""),
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
        phivrtdivspec_0 = galewsky_initial_condition(
            model=solver,
            umax=cfg.galewsky_umax,
            usouth=cfg.galewsky_usouth,
            unorth=cfg.galewsky_unorth,
            perturb_loc=cfg.galewsky_perturb_loc,
            perturb_amp=cfg.galewsky_perturb_amp,
            noise_level=cfg.galewsky_noise_level,
        )
    elif cfg.ic == "real_world":
        # a pre-sliced per-job file (see batch_simulation.py) avoids opening
        # the full (potentially large, multi-year) ERA5 dataset in every job.
        ic_source_path = Path(cfg.ic_data_path) if cfg.ic_data_path else netcdf_path

        start_load_time = time.perf_counter()
        era5_dataset = xr.open_dataset(ic_source_path).load()
        finish_load_time = time.perf_counter()
        print(f"Finished loading ERA5 data in {(finish_load_time - start_load_time):2f} seconds")

        # gridded ERA5 (u, v on a latitude/longitude grid) needs a RealVectorSHT sized to
        # its own resolution (see rw_initial_condition's grid branch); spectral ERA5
        # (native spherical-harmonic vo/d/z, no lat/lon dims) needs none - rw_initial_condition
        # remaps its coefficients directly and auto-detects which of the two this is.
        vSHT = None
        if 'latitude' in era5_dataset.dims and 'longitude' in era5_dataset.dims:
            nlat_data, nlon_data = era5_dataset.sizes['latitude'], era5_dataset.sizes['longitude']
            vSHT = RealVectorSHT(nlat=nlat_data, nlon=nlon_data, lmax=nlat_data // 2, mmax=nlat_data // 2,
                                grid='equiangular', csphase=False).to(solver.device)
        phivrtdivspec_0 = rw_initial_condition(
            model=solver,
            vSHT=vSHT,
            era5_dataset=era5_dataset,
            ic_time=cfg.ic_time,
            a=cfg.a,
            p=cfg.p)

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
        filter_a=cfg.a if cfg.ic == "real_world" else None,
        filter_p=cfg.p if cfg.ic == "real_world" else None,
        variant_tag=cfg.variant_tag if cfg.ic == "galewsky" else None,
    )

    plot_trajectory_diagnostics(save_path, Path(save_path).parent)


if __name__ == "__main__":
    main()
