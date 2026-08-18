#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Run neural-operator (SFNO) inference and store the rollout following the
data/model organization convention (README.md). This parallels run_solver.py:

  1. Resolve the trained model directory. Unlike the numerical solver (which
     can be initialized on the fly) a neural operator must already be
     trained, so a missing checkpoint is a hard error (train it first).
     Likewise resolve/initialize the reference numerical solver's checkpoint.
  2. Generate the initial condition (grid space) and call run() in
     src/helpers/run_model.py for both the SFNO rollout and the reference
     numerical run. run() derives where each trajectory lives entirely from
     its own model_checkpoint path (the data tree is isomorphic to the
     checkpoint tree) plus (duration, ic, ...), and skips the
     simulation/inference if it's already there.

Configuration is read from the ``inference`` section of config.yml (pass an
alternative path as the first CLI argument).

@author Haoyu Tang hytang2@illinois.edu
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

from pathlib import Path
from types import SimpleNamespace

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import xarray as xr
import yaml
import tqdm
import json
from datetime import datetime, timedelta
import re
from torch_harmonics.sht import RealVectorSHT

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.numerical_solver.initial_condition import *
from src.neural_operator.loss import LOSS_FUNCTIONS
from src.analyze.visualization import plot_sphere_comparison
from src.helpers.run_model import (
    run,
    neural_model_path,
    numerical_checkpoint_path,
    save_numerical_checkpoint,
    load_numerical_checkpoint,
    load_h_stats,
    physical_to_nondim,
)
from src.helpers.print import print_in_box




DEFAULT_CONFIG = {
    # which trained model (locates the checkpoint via the convention)
    "nlat": 128,
    "nlon": 256,
    "n_future": 1,
    "num_layers": 4,
    "embed_dim": 16,
    "pos_embed": "learnable lat",
    "trainData": "equiangular",
    "grid": "equiangular",          # equiangular | legendre-gauss | lobatto
    "normalization_layer": "none",  # must match the trained run's own value
    "loss_type": "spectral",        # must match the trained run's own value
    "index": 0,                     # 0/, 1/, ... training-config slot
    "single_step": True,
    # rollout
    "duration": 20,                 # days
    "save_interval_minutes": 30,
    # initial condition
    "ic": "galewsky",               # galewsky | real_world
    "dataset_name": None,           # required when ic == real_world, e.g. "1980_2025_odd_month"
    "ic_time": None,                # required when ic == real_world
    "pressure": None,               # required when ic == real_world (for naming)
    "spinup_days": 1.0,
}


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) or {}

    raw = DEFAULT_CONFIG | config.get("inference", {})

    # normalize types regardless of YAML formatting
    raw["nlat"] = int(raw["nlat"])
    raw["nlon"] = int(raw["nlon"])
    raw["n_future"] = int(raw["n_future"])
    raw["num_layers"] = int(raw["num_layers"])
    raw["embed_dim"] = int(raw["embed_dim"])
    raw["index"] = int(raw["index"])
    raw["duration"] = float(raw["duration"])
    raw["save_interval_minutes"] = float(raw["save_interval_minutes"])
    raw["spinup_days"] = int(raw['spinup_days'])


    cfg = SimpleNamespace(**raw)

    if cfg.ic == "real_world":
        if not cfg.ic_time:
            raise ValueError("inference.ic_time is required when ic == real_world.")
        if not cfg.dataset_name:
            raise ValueError("inference.dataset_name is required when ic == real_world.")
        if cfg.pressure is None:
            raise ValueError("inference.pressure is required when ic == real_world (used for naming).")
    return cfg

def numerical_model_info_from_neural_opeartor_model_info(neural_model_dir):
    """
    Recover numerical model info(tau, semi_implicit, radiation) from the save neural operator model_info.json
    
    Args:
        neural_model_dir : The directory of the neural operator model
    """
    json_path = f"{neural_model_dir}/model_info.json"
    with open(json_path) as f:
        model_info = json.load(f)
    
    training_data_path = model_info['training_data']
    # print(f"The training_data_path is {training_data_path}")
    tau_2, tau_4, tau_8 = re.findall(r'tau_\((\d+),(\d+),(\d+)\)', training_data_path)[0]
    method = re.findall(r'method_(\w+)', training_data_path)[0]
    radiation = re.findall(r'radiation_(\w+)', training_data_path)[0]
    numerical_model_info = { 
        "tau" : (float(tau_2), float(tau_4), float(tau_8)),
        "semi_implicit" : True if method == "implicit" else False ,
        "rad": True if radiation == 'rad' else False
    }
    return numerical_model_info 

def main():
    cfg = load_config()

    model_kwargs = dict(
        resol=(cfg.nlat, cfg.nlon),
        n_future=cfg.n_future,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        pos_embed=cfg.pos_embed,
        trainData_path=cfg.trainData,
        grid=cfg.grid,
        normalization_layer=cfg.normalization_layer,
        loss_type=cfg.loss_type,
        index=cfg.index,
    )

    print_in_box({
        "title": "Neural Operator Inference",
        "lines": [
            f"resol = ({cfg.nlat}, {cfg.nlon}) | nfuture = {cfg.n_future} | grid = {cfg.grid}",
            f"nlayer = {cfg.num_layers} | embed = {cfg.embed_dim} | posEmbed = {cfg.pos_embed}",
            f"trainData = {cfg.trainData} | index = {cfg.index} | single_step = {cfg.single_step}",
            f"normalization_layer = {cfg.normalization_layer} | loss_type = {cfg.loss_type}",
            f"duration = {cfg.duration} days | save_interval = {cfg.save_interval_minutes} min | ic = {cfg.ic}",
        ],
    })


    dataset_name = cfg.dataset_name if cfg.ic == "real_world" else None

    # dataset-wide h_avg/h_amp (real-world only) feed the IC/reference solvers'
    # own non-dimensionalization; run() reads its own T/U back off each
    # checkpoint's model_info.json (see run_model.run()'s docstring).
    h_avg = h_amp = None
    if cfg.ic == "real_world":
        h_avg, h_amp = load_h_stats(cfg.dataset_name)
        print(f"    Using dataset-wide h_avg={h_avg:.2f} m, h_amp={h_amp:.2f} m from '{cfg.dataset_name}'")
        
    # Resolve the trained model directory; it must already exist.
    model_dir, single_name, multi_name, info_name = neural_model_path(**model_kwargs)
    ckpt_name = single_name if cfg.single_step else multi_name
    ckpt_path = Path(model_dir) / ckpt_name
    info_path = Path(model_dir) / info_name

    if not ckpt_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(
            f"No trained model found at {model_dir} (expected {ckpt_name} and {info_name}). "
            f"Train it first with `make train_single`."
        )
    print(f"📦 Using trained model {ckpt_path}")
    
    ref_num_model_info = numerical_model_info_from_neural_opeartor_model_info(neural_model_dir=model_dir)
    ref_tau, ref_semi_implicit, ref_rad = ref_num_model_info['tau'], ref_num_model_info['semi_implicit'], ref_num_model_info['rad']

    ref_ckpt_dir = numerical_checkpoint_path(
        lmax=cfg.nlat // 2, tau=ref_tau, grid=cfg.grid, semi_implicit=ref_semi_implicit, rad=ref_rad,
        dataset_name=dataset_name,
    )
    ref_info_path = Path(ref_ckpt_dir) / "model_info.json"


    if ref_info_path.is_file():
        print(f"📦 Loading existing reference solver checkpoint {ref_ckpt_dir}")
        ref_solver, _ = load_numerical_checkpoint(ref_ckpt_dir)
        ref_solver.to(ref_solver.device)
    else:
        print(f"🛠  Initializing reference solver checkpoint -> {ref_ckpt_dir}")
        ref_solver = ShallowWaterSolver(
            lmax=cfg.nlat // 2, tau=ref_tau, grid=cfg.grid,
            dealias=False, semi_implicit=ref_semi_implicit,
            rad=ref_rad,h_avg=h_avg, h_amp=h_amp, non_dimensional=True,
        )
        ref_solver.to(ref_solver.device)
        save_numerical_checkpoint(
            ref_solver, ref_ckpt_dir, dataset_name=dataset_name,
            pressure=cfg.pressure if cfg.ic == "real_world" else None,
        )


    if cfg.ic == "galewsky":
        print("Initial conidtion is galewsky")
        phivrtdivspec_0 = galewsky_initial_condition(model=ref_solver)
    elif cfg.ic == "real_world":
        print("Initial condition is real-world")
        netcdf_path = Path("reanalysis_data") / cfg.dataset_name / "data.nc"
        ds = xr.open_dataset(netcdf_path)
        nlat_data, nlon_data = ds.sizes['latitude'], ds.sizes['longitude']
        ds.close()
        vSHT = RealVectorSHT(nlat=nlat_data, nlon=nlon_data, lmax=nlat_data // 2, mmax=nlat_data // 2,
                            grid='equiangular', csphase=False).to(ref_solver.device)

        start_load_time = time.perf_counter()
        era5_dataset = xr.open_dataset(netcdf_path).load()
        print(f"Finished loading ERA5 data in {(time.perf_counter() - start_load_time):2f} seconds")
        
        phivrtdivspec_0 = rw_initial_condition(
            model=ref_solver, vSHT=vSHT, era5_dataset=era5_dataset, ic_time=cfg.ic_time
            )
        print("Preparing solver for grid->spec transformation and initial spin-up...")
        # spin up the initial condition using numerical solver
        state = phivrtdivspec_0
        with torch.no_grad():
            for i in tqdm.trange(int(86400*cfg.spinup_days//(ref_solver.dt * ref_solver.T)), desc=f"🐺->...->🦮  Spinning up the initial condition for {cfg.spinup_days} day"):
                state = ref_solver.timestep(uspec=state, nsteps=1)
        phivrtdivspec_0_spinned_up = state
    else:
        raise ValueError(f"Unknown ic '{cfg.ic}' (expected galewsky or real_world).")

    # ------------------------------------------------------------------ #
    # (2) Run the SFNO rollout and the reference numerical solver from the
    #     same IC. Trajectory path deduction and reuse are entirely
    #     delegated to run() - it derives where each trajectory lives
    #     straight from the model_checkpoint path it's given (see
    #     run_model.py's _data_path_from_checkpoint) and skips straight to
    #     returning that path if a matching trajectory is already there.
    # ------------------------------------------------------------------ #
    ic_time = (datetime.fromisoformat(cfg.ic_time) + timedelta(days=cfg.spinup_days)).isoformat()
    neural_save_path = run(
        model_checkpoint=str(model_dir),
        initial_condition=phivrtdivspec_0_spinned_up.clone(),
        duration=cfg.duration,
        ic=cfg.ic,
        pressure=cfg.pressure,
        ic_time=ic_time,
        dataset_name=dataset_name,
        save_interval_minutes=cfg.save_interval_minutes,
        single_step=cfg.single_step,
    )
    
    
    ref_save_path = run(
        model_checkpoint=str(ref_ckpt_dir),
        initial_condition=phivrtdivspec_0_spinned_up.clone(),
        duration=cfg.duration,
        ic=cfg.ic,
        pressure=cfg.pressure,
        ic_time=ic_time,
        dataset_name=dataset_name,
        save_interval_minutes=cfg.save_interval_minutes,
    )

    data_dir = Path(neural_save_path).parent
    data_file = Path(neural_save_path).name

    # ------------------------------------------------------------------ #
    # (3) Per-step relative spectral L2 loss of the SFNO rollout against
    #     the reference numerical trajectory, plotted next to the .pt file.
    # ------------------------------------------------------------------ #
    plot_per_step_loss(
        neural_traj_path=Path(neural_save_path),
        ref_traj_path=Path(ref_save_path),
        info_path=info_path,
        lmax=cfg.nlat // 2,
        grid=cfg.grid,
        save_interval_minutes=cfg.save_interval_minutes,
        output_path=data_dir / f"{Path(data_file).stem}_l2_spectral_loss.png",
    )

    # ------------------------------------------------------------------ #
    # (4) 2x4 sphere-projection comparison: rows = (reference, SFNO), columns
    #     = t=0/1/2/6 hours since each trajectory's own frame 0 (i.e. since
    #     the end of the warm-up spin-up). ref_traj is offset by
    #     warmup_frames so both trajectories' frame 0 is the same absolute time.
    # ------------------------------------------------------------------ #
    neural_data = torch.load(neural_save_path, weights_only=False)
    ref_data = torch.load(ref_save_path, weights_only=False)

    plot_sphere_comparison(
        ref_data=ref_data,
        inf_data=neural_data,
        var="pv",
        hours=[0, 6, 12, 24],
        output_path=str(data_dir / f"{Path(data_file).stem}_sphere_comparison.png"),
        ref_label="Numerical (ground truth)",
        inf_label="SFNO (inference)",
    )


def plot_per_step_loss(neural_traj_path, ref_traj_path, info_path, lmax, grid,
                        save_interval_minutes, output_path):
    with open(info_path, "r") as f:
        model_info = json.load(f)

    solver = ShallowWaterSolver(lmax=lmax, grid=grid, dealias=False, non_dimensional=False)
    solver.to(solver.device)

    T, U = model_info["T"], model_info["U"]
    loss_fn = LOSS_FUNCTIONS[model_info.get("loss_type", "spectral")]

    neural_traj = torch.load(neural_traj_path, weights_only=False)["trajectory"].to(solver.device)
    ref_traj = torch.load(ref_traj_path, weights_only=False)["trajectory"].to(solver.device)

    n_steps = min(neural_traj.shape[0], ref_traj.shape[0])
    if neural_traj.shape[0] != ref_traj.shape[0]:
        print(f"⚠️  Trajectory lengths differ after the warm-up offset (neural={neural_traj.shape[0]}, "
              f"ref={ref_traj.shape[0]}); comparing the first {n_steps} steps.")

    losses = []
    with torch.no_grad():
        for t in range(n_steps):
            prd = solver.spec2grid(neural_traj[t].to(torch.complex64))
            tar = solver.spec2grid(ref_traj[t].to(torch.complex64))
            prd_n = physical_to_nondim(prd, T, U)
            tar_n = physical_to_nondim(tar, T, U)
            loss = loss_fn(
                solver, prd_n.unsqueeze(0), tar_n.unsqueeze(0), relative=True, squared=False,
            )
            losses.append(loss.item())

    hours = [t * save_interval_minutes / 60.0 for t in range(n_steps)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(hours, losses, marker=".", markersize=3, linewidth=1)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("relative spectral L2 loss")
    ax.set_title("SFNO rollout vs. numerical reference")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"📈 Saved per-step L2 loss plot -> {output_path}")


if __name__ == "__main__":
    main()
