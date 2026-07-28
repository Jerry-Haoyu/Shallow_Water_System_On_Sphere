#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Run neural-operator (SFNO) inference and store the rollout following the
data/model organization convention (README.md). This parallels run_solver.py:

  1. Resolve where the inference trajectory would live per the convention. If
     it already exists, there is nothing to do.
  2. Otherwise resolve the trained model directory. Unlike the numerical solver
     (which can be initialized on the fly) a neural operator must already be
     trained, so a missing checkpoint is a hard error (train it first).
  3. Generate the initial condition (grid space) and call run() in
     src/helpers/run_model.py, which infers and stores the trajectory.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.numerical_solver.initial_condition import *
from src.neural_operator.loss import spectral_l2loss_sphere
from src.analyze.visualization import plot_sphere_comparison
from src.helpers.run_model import (
    run,
    neural_model_path,
    neural_data_path,
    numerical_checkpoint_path,
    numerical_data_path,
)
from src.helpers.print import print_in_box

# Must match the hardcoded spin-up length in run_model.py's neural branch
# ("Spinning up the initial condition" loop, run_model.py:320).
SPINUP_DAYS = 1.0

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
    "index": 0,                     # 0/, 1/, ... training-config slot
    "single_step": True,
    # rollout
    "duration": 20,                 # days
    "save_interval_minutes": 30,
    # initial condition
    "ic": "galewsky",               # galewsky | real_world
    "netcdf_path": None,            # required when ic == real_world
    "ic_time": None,                # required when ic == real_world
    "pressure": None,               # required when ic == real_world (for naming)
    # reference numerical solver run (ground truth for the per-step loss plot)
    "ref_tau": [30000, 30000, 30],
    "ref_cfl": 0.25,
    "ref_semi_implicit": True,
    "ref_dealias": True,
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
    raw["ref_tau"] = [int(t) for t in raw["ref_tau"]]
    raw["ref_cfl"] = float(raw["ref_cfl"])

    cfg = SimpleNamespace(**raw)

    if cfg.ic == "real_world":
        if not cfg.ic_time:
            raise ValueError("inference.ic_time is required when ic == real_world.")
        if not cfg.netcdf_path:
            raise ValueError("inference.netcdf_path is required when ic == real_world.")
        if cfg.pressure is None:
            raise ValueError("inference.pressure is required when ic == real_world (used for naming).")
    return cfg


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
        index=cfg.index,
    )

    # ------------------------------------------------------------------ #
    # (1) Where would the inference trajectory live? If present, stop.
    # ------------------------------------------------------------------ #
    data_dir, data_file = neural_data_path(
        duration=cfg.duration, ic=cfg.ic, pressure=cfg.pressure, ic_time=cfg.ic_time,
        **model_kwargs,
    )
    data_path = Path(data_dir) / data_file

    print_in_box({
        "title": "Neural Operator Inference",
        "lines": [
            f"resol = ({cfg.nlat}, {cfg.nlon}) | nfuture = {cfg.n_future} | grid = {cfg.grid}",
            f"nlayer = {cfg.num_layers} | embed = {cfg.embed_dim} | posEmbed = {cfg.pos_embed}",
            f"trainData = {cfg.trainData} | index = {cfg.index} | single_step = {cfg.single_step}",
            f"duration = {cfg.duration} days | save_interval = {cfg.save_interval_minutes} min | ic = {cfg.ic}"
            # f"data -> {data_path}",
        ],
    })

    # ------------------------------------------------------------------ #
    # (2) Resolve the reference numerical solver's checkpoint/data paths
    #     (same tau/cfl/etc convention as run_solver.py) - this is the
    #     ground truth the SFNO rollout gets compared against.
    #
    #     run_model.py's neural branch spins the IC up for SPINUP_DAYS with
    #     the numerical solver *before* saving frame 0 (see run_model.py's
    #     "Spinning up the initial condition" loop), so the SFNO trajectory's
    #     frame t is really at absolute time SPINUP_DAYS + t*save_interval.
    #     The reference run covers that same head start so frames line up;
    #     plot_per_step_loss() offsets into it by warmup_frames.
    # ------------------------------------------------------------------ #
    ref_ckpt_dir, ref_ckpt_file = numerical_checkpoint_path(
        lmax=cfg.nlat // 2, tau=cfg.ref_tau, grid=cfg.grid, semi_implicit=cfg.ref_semi_implicit,
    )
    ref_ckpt_path = Path(ref_ckpt_dir) / ref_ckpt_file

    ref_duration = cfg.duration + SPINUP_DAYS
    warmup_frames = round(SPINUP_DAYS * 1440.0 / cfg.save_interval_minutes)

    ref_data_dir, ref_data_file = numerical_data_path(
        lmax=cfg.nlat // 2, tau=cfg.ref_tau, grid=cfg.grid, semi_implicit=cfg.ref_semi_implicit,
        duration=ref_duration, ic=cfg.ic, pressure=cfg.pressure, ic_time=cfg.ic_time,
    )
    ref_data_path = Path(ref_data_dir) / ref_data_file

    need_neural = not data_path.is_file()
    need_ref = not ref_data_path.is_file()

    if not need_neural:
        print(f"✅ Inference trajectory already exists at {data_path}.")
    if not need_ref:
        print(f"✅ Reference numerical trajectory already exists at {ref_data_path}.")

    # ------------------------------------------------------------------ #
    # (3) Resolve the trained model directory; it must already exist.
    # ------------------------------------------------------------------ #
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

    print("Preparing solver for computing initial condition...")
    ic_solver = ShallowWaterSolver(lmax=cfg.nlat//2, grid=cfg.grid, dealias=False)
    ic_solver.to(ic_solver.device)

    if cfg.ic == "galewsky":
        print("Initial conidtion is galewsky")
        phivrtdivspec_0 = galewsky_initial_condition(model=ic_solver)
    elif cfg.ic == "real_world":
        print("Initial condition is real-world")
        phivrtdivspec_0 = rw_initial_condition(
            model=ic_solver, netcdf_path=cfg.netcdf_path, ic_time=cfg.ic_time)
    else:
        raise ValueError(f"Unknown ic '{cfg.ic}' (expected galewsky or real_world).")

    if need_neural:
        run(
            model_checkpoint=str(model_dir),
            initial_condition=phivrtdivspec_0.clone(),
            output_dir=data_dir,
            file_name=data_file,
            duration=cfg.duration,
            save_interval_minutes=cfg.save_interval_minutes,
            single_step=cfg.single_step,
        )

    # ------------------------------------------------------------------ #
    # (4) Run the reference numerical solver alongside, from the same IC.
    # ------------------------------------------------------------------ #
    if need_ref:
        if ref_ckpt_path.is_file():
            print(f"📦 Loading existing reference solver checkpoint {ref_ckpt_path}")
            ref_solver = torch.load(ref_ckpt_path, weights_only=False)
            ref_solver.to(ref_solver.device)
        else:
            print(f"🛠  Initializing reference solver checkpoint -> {ref_ckpt_path}")
            ref_solver = ShallowWaterSolver(
                cfg.nlat // 2, cfg.ref_tau, cfg.ref_cfl, grid=cfg.grid,
                dealias=cfg.ref_dealias, semi_implicit=cfg.ref_semi_implicit,
            )
            ref_solver.to(ref_solver.device)
            ref_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ref_solver, ref_ckpt_path)

        run(
            model_checkpoint=str(ref_ckpt_path),
            initial_condition=phivrtdivspec_0.clone(),
            output_dir=ref_data_dir,
            file_name=ref_data_file,
            duration=ref_duration,
            save_interval_minutes=cfg.save_interval_minutes,
        )

    # ------------------------------------------------------------------ #
    # (5) Per-step relative spectral L2 loss of the SFNO rollout against
    #     the reference numerical trajectory, plotted next to the .pt file.
    # ------------------------------------------------------------------ #
    plot_per_step_loss(
        neural_traj_path=data_path,
        ref_traj_path=ref_data_path,
        ref_warmup_frames=warmup_frames,
        info_path=info_path,
        lmax=cfg.nlat // 2,
        grid=cfg.grid,
        save_interval_minutes=cfg.save_interval_minutes,
        output_path=Path(data_dir) / f"{Path(data_file).stem}_l2_spectral_loss.png",
    )

    # ------------------------------------------------------------------ #
    # (6) 2x4 sphere-projection comparison: rows = (reference, SFNO), columns
    #     = t=0/1/2/6 hours since each trajectory's own frame 0 (i.e. since
    #     the end of the warm-up spin-up). ref_traj is offset by
    #     warmup_frames so both trajectories' frame 0 is the same absolute time.
    # ------------------------------------------------------------------ #
    neural_data = torch.load(data_path, weights_only=False)
    ref_data_raw = torch.load(ref_data_path, weights_only=False)
    ref_data_aligned = {
        "metadata": ref_data_raw["metadata"],
        "trajectory": ref_data_raw["trajectory"][warmup_frames:],
    }
    plot_sphere_comparison(
        ref_data=ref_data_aligned,
        inf_data=neural_data,
        var="pv",
        hours=[0, 1, 6, 24],
        output_path=str(Path(data_dir) / f"{Path(data_file).stem}_sphere_comparison.png"),
        ref_label="Numerical (ground truth)",
        inf_label="SFNO (inference)",
    )


def plot_per_step_loss(neural_traj_path, ref_traj_path, ref_warmup_frames, info_path, lmax, grid,
                        save_interval_minutes, output_path):
    """Relative spectral L2 loss of the SFNO rollout vs. the reference
    numerical trajectory, one value per saved step, saved as a line plot.

    Both trajectories are converted to grid space and normalized with the
    model's own (mean, std) before the loss is computed, so the resulting
    numbers are directly comparable to the training/validation loss.
    ``ref_traj``'s first ``ref_warmup_frames`` frames are the SPINUP_DAYS
    head start that ``neural_traj`` skips past before its own frame 0, so
    ``ref_traj[ref_warmup_frames + t]`` is what lines up with ``neural_traj[t]``.
    """
    import json

    with open(info_path, "r") as f:
        model_info = json.load(f)

    solver = ShallowWaterSolver(lmax=lmax, grid=grid, dealias=False)
    solver.to(solver.device)
    
    mean = torch.tensor(
        [model_info["inp_mean_gp"], model_info["inp_mean_zeta"], model_info["inp_mean_delta"]],
    ).reshape(3, 1, 1).to(solver.device)
    std = torch.tensor(
        [model_info["inp_std_gp"], model_info["inp_std_zeta"], model_info["inp_std_delta"]],
    ).reshape(3, 1, 1).to(solver.device)


    neural_traj = torch.load(neural_traj_path, weights_only=False)["trajectory"].to(solver.device)
    ref_traj = torch.load(ref_traj_path, weights_only=False)["trajectory"][ref_warmup_frames:].to(solver.device)

    n_steps = min(neural_traj.shape[0], ref_traj.shape[0])
    if neural_traj.shape[0] != ref_traj.shape[0]:
        print(f"⚠️  Trajectory lengths differ after the warm-up offset (neural={neural_traj.shape[0]}, "
              f"ref={ref_traj.shape[0]}); comparing the first {n_steps} steps.")

    losses = []
    with torch.no_grad():
        for t in range(n_steps):
            prd = solver.spec2grid(neural_traj[t].to(torch.complex64))
            tar = solver.spec2grid(ref_traj[t].to(torch.complex64))
            prd_n = (prd - mean) / std
            tar_n = (tar - mean) / std
            loss = spectral_l2loss_sphere(
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
