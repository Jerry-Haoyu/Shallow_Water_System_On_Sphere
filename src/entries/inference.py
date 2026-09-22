#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Run neural-operator (SFNO) inference and store the rollout following the
data/model organization convention (README.md). This parallels run_solver.py:

  1. Resolve the trained model directory. Unlike the numerical solver (which
     can be initialized on the fly) a neural operator must already be
     trained, so a missing checkpoint is a hard error (train it first).
     Likewise resolve/initialize the reference numerical solver's checkpoint.
  2. For ic == "real_world", loop over every time point in the evaluation
     dataset's own reanalysis_data/<dataset_name>/data.nc (produced by
     src/entries/download_era5_spectral.py) - there is no single
     initial-condition time/date to configure any more, the whole dataset IS
     the evaluation set. For ic == "galewsky" (no associated dataset) this is
     just the one canned test case. For each time point, generate the initial
     condition (grid space) and call run() in src/helpers/run_model.py for
     both the SFNO rollout and the reference numerical run. run() derives
     where each trajectory lives entirely from its own model_checkpoint path
     (the data tree is isomorphic to the checkpoint tree) plus
     (duration, ic, ...), and skips the simulation/inference if it's already
     there.

This script itself only computes and saves trajectories (.pt files under
model_output/) - it does not plot anything directly. Once every trajectory has
been computed - the whole real_world evaluation dataset, or (for galewsky) the
one canned trajectory, treated as a singleton evaluation set - main()
automatically shells out to task/batch_inference/run_batch_inference.py
against the same config, which produces the aggregate diagnostics (per-step
spectral loss - combined and per-channel - plus sphere/box comparisons for a
random sample of trajectories) - see run_batch_inference().

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
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import tqdm
import json
from datetime import datetime, timedelta
import re
from torch_harmonics.sht import RealVectorSHT

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.numerical_solver.initial_condition import *
from src.neural_operator.loss import LOSS_FUNCTIONS
from src.analyze.visualization import plot_sphere_comparison, plot_box_comparison
from src.helpers.config import load_raw_config
from src.helpers.run_model import (
    run,
    neural_model_path,
    numerical_checkpoint_path,
    save_numerical_checkpoint,
    load_numerical_checkpoint,
    load_h_stats,
    load_model_info,
    physical_to_nondim,
    compute_step_per_save,
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
    # initial condition
    "ic": "galewsky",               # galewsky | real_world
    # required when ic == real_world: reanalysis_data/<dataset_name>/data.nc
    # - the EVALUATION dataset (see src/entries/download_era5_spectral.py).
    # There is no separate ic_time knob any more: every valid_time in that
    # dataset's data.nc is run as its own initial condition (see
    # resolve_eval_times/main below).
    "dataset_name": None,
    "pressure": None,               # required when ic == real_world (for naming)
    "spinup_days": 2.0,
    # how many randomly-chosen evaluation trajectories
    # task/batch_inference/run_batch_inference.py renders box/sphere
    # comparison plots for - not used by this script itself.
    "sample_number_to_plot": 5,
    # which saved-frame steps (0 = the IC itself, 1 = one save_interval_minutes
    # later, ...) task/batch_inference/run_batch_inference.py renders box/
    # sphere comparison columns for, on each of the sampled trajectories -
    # not used by this script itself.
    "comparison_steps": [0, 1, 2, 4, 5],
}


def load_config():
    raw = load_raw_config("inference", DEFAULT_CONFIG)

    # normalize types regardless of YAML formatting
    raw["nlat"] = int(raw["nlat"])
    raw["nlon"] = int(raw["nlon"])
    raw["n_future"] = int(raw["n_future"])
    raw["num_layers"] = int(raw["num_layers"])
    raw["embed_dim"] = int(raw["embed_dim"])
    raw["index"] = int(raw["index"])
    raw["duration"] = float(raw["duration"])
    # float, not int: fractional spin-ups (e.g. 0.5 days) are a normal input -
    # see different_spin_up/task.md, which sweeps spinup_days over
    # [0.5, 1.0, ..., 4.0]. spinup_frames (below) is what actually rounds this
    # to a whole number of training frames.
    raw["spinup_days"] = float(raw['spinup_days'])
    raw["sample_number_to_plot"] = int(raw["sample_number_to_plot"])
    raw["comparison_steps"] = [int(s) for s in raw["comparison_steps"]]
    # not a config input - see the comment on DEFAULT_CONFIG's rollout section.
    # Popped rather than left in raw so a stray inference.save_interval_minutes
    # in a config file is silently overridden here, not silently kept.
    # save_interval_minutes itself is set in main(), once the trained model's
    # own recorded true_interval_minutes is known - see the comment there.
    raw.pop("save_interval_minutes", None)
    # no longer a config input (see DEFAULT_CONFIG's dataset_name comment) -
    # popped so a stale ic_time left over in an old config.yml is silently
    # ignored rather than silently doing nothing while looking configured.
    raw.pop("ic_time", None)

    cfg = SimpleNamespace(**raw)

    if cfg.ic == "real_world":
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


def dataset_tag(cfg):
    """Which value identifies "this run's h_avg/h_amp source" for reference-
    checkpoint / trajectory-path disambiguation (see run_model.py's
    numerical_checkpoint_path and _data_path_from_checkpoint): the EVALUATION
    dataset for ic == "real_world" (cfg.dataset_name, see load_h_stats), or -
    since galewsky has no dataset of its own - the trained neural model's OWN
    training dataset (cfg.trainData), which is where the h_avg/h_amp its
    reference solver reuses (see run_rollout_pair / galewsky_initial_condition,
    which non-dimensionalizes its umax/noise_level using this same h_avg)
    actually came from. A single source of truth used by
    resolve_checkpoint_dirs, run_rollout_pair, and
    task/batch_inference/run_batch_inference.py's resolve_trajectory_dirs, so
    all three agree on the exact same paths - a mismatch here (e.g. one of
    them falling back to the stale/irrelevant cfg.dataset_name for a galewsky
    run) surfaces as _data_path_from_checkpoint's "does not match the dataset
    baked into checkpoint" ValueError.
    """
    return cfg.dataset_name if cfg.ic == "real_world" else cfg.trainData


def resolve_checkpoint_dirs(cfg):
    """(model_dir, ref_ckpt_dir, ref_num_model_info): the trained SFNO run
    directory, its paired reference numerical-solver checkpoint directory, and
    the (tau, semi_implicit, rad) recovered from the SFNO's own model_info.json
    (see numerical_model_info_from_neural_opeartor_model_info) - resolved
    purely from cfg's model-identity fields (see neural_model_path /
    numerical_checkpoint_path). Factored out of run_rollout_pair so
    task/batch_inference/run_batch_inference.py can resolve the exact same
    directories `make inference` wrote its trajectories under, without
    touching the trained model/solver themselves (it only reads already-saved
    .pt trajectories back off disk - see run_model.py's
    _data_path_from_checkpoint for the further duration/ic/pressure/dataset
    nodes appended on top of these two directories to reach a trajectory's
    own leaf directory).
    """
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
    model_dir, single_name, multi_name, info_name = neural_model_path(**model_kwargs)
    ckpt_name = single_name if cfg.single_step else multi_name
    ckpt_path = Path(model_dir) / ckpt_name
    info_path = Path(model_dir) / info_name

    if not ckpt_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(
            f"No trained model found at {model_dir} (expected {ckpt_name} and {info_name}). "
            f"Train it first with `make train_single`."
        )

    ref_num_model_info = numerical_model_info_from_neural_opeartor_model_info(neural_model_dir=model_dir)

    ref_ckpt_dir = numerical_checkpoint_path(
        lmax=cfg.nlat // 2, tau=ref_num_model_info['tau'], grid=cfg.grid,
        semi_implicit=ref_num_model_info['semi_implicit'], rad=ref_num_model_info['rad'],
        dataset_name=dataset_tag(cfg),
    )
    return model_dir, ref_ckpt_dir, ref_num_model_info


def resolve_eval_times(dataset_name):
    """Every valid_time in the evaluation dataset's own data.nc (see
    src/entries/download_era5_spectral.py) - one inference run happens per
    entry, in file order."""
    nc_path = Path("reanalysis_data") / dataset_name / "data.nc"
    with xr.open_dataset(nc_path) as ds:
        return np.atleast_1d(ds["valid_time"].values).copy()


def run_rollout_pair(cfg):
    """Resolve the trained model, build the spun-up initial condition, and run
    both the SFNO rollout and the reference numerical rollout from it.

    Factored out of main() so a batch driver (see different_spin_up/scripts,
    or main()'s own per-evaluation-time loop below) can generate many
    (ic_time, spinup_days, duration) combinations without going through
    config.yml/CLI for each one - `cfg.ic_time` is set by the caller before
    invoking this (galewsky ignores it; real_world reads it as the IC's own
    valid_time).

    Returns a SimpleNamespace with: model_dir, info_path, neural_save_path,
    ref_save_path, neural_data, ref_data, true_interval_minutes.
    """
    # also doubles as the reference-checkpoint's own recorded provenance (see
    # save_numerical_checkpoint below) - harmless for the trajectory-output
    # path calls further down (dataset_name only affects
    # _data_path_from_checkpoint's ic == "real_world" branch, never galewsky's).
    dataset_name = dataset_tag(cfg)

    model_dir, ref_ckpt_dir, ref_num_model_info = resolve_checkpoint_dirs(cfg)
    ckpt_name = "checkpoints_single.pt" if cfg.single_step else "checkpoints_multi.pt"
    ckpt_path = Path(model_dir) / ckpt_name
    info_path = Path(model_dir) / "model_info.json"
    print(f"📦 Using trained model {ckpt_path}")

    # model_info's own recorded true_interval_minutes (the REAL elapsed time
    # per training frame, already float-rounded once at data-generation time -
    # see run_model.py's run()) is the golden-standard source for how much
    # real time this model's own n_future forecast-jump spans - no more
    # assuming a nominal 30-min cadence and re-deriving the true value from it
    # (see stage0.md Findings: frame-cadence rounding drift). This single
    # value now drives everything below: number_of_frames for both branches,
    # the reference solver's own substep count, the spin-up length, and all
    # hour labels.
    model_info = load_model_info(str(model_dir))
    cfg.save_interval_minutes = model_info["n_future"] * model_info["true_interval_minutes"]

    # h_avg/h_amp feed the reference solver's own non-dimensionalization (only
    # used below when no cached ref_ckpt_dir checkpoint exists yet - run()
    # itself always reads its own T/U back off each checkpoint's model_info.json,
    # see run_model.run()'s docstring). real_world: the EVALUATION dataset's
    # own h_stats.npz (which may differ from the model's training dataset).
    # galewsky has no dataset of its own, so it instead reuses the trained
    # neural model's OWN recorded h_avg/h_amp (this same model_info, just
    # loaded above) - this is what lets galewsky_initial_condition
    # non-dimensionalize its (literal, Galewsky-2004) umax/noise_level using
    # the checkpoint's own physical scale, rather than a generic 10 km/80 m/s
    # default unrelated to it.
    if cfg.ic == "real_world":
        h_avg, h_amp = load_h_stats(cfg.dataset_name)
        print(f"    Using dataset-wide h_avg={h_avg:.2f} m, h_amp={h_amp:.2f} m from '{cfg.dataset_name}'")
    else:
        h_avg, h_amp = model_info["h_avg"], model_info["h_amp"]
        print(f"    Using trained model's own h_avg={h_avg:.2f} m, h_amp={h_amp:.2f} m "
              f"(model_info.json, trainData='{cfg.trainData}')")

    print_in_box({
        "title": "Neural Operator Inference",
        "lines": [
            f"resol = ({cfg.nlat}, {cfg.nlon}) | nfuture = {cfg.n_future} | grid = {cfg.grid}",
            f"nlayer = {cfg.num_layers} | embed = {cfg.embed_dim} | posEmbed = {cfg.pos_embed}",
            f"trainData = {cfg.trainData} | index = {cfg.index} | single_step = {cfg.single_step}",
            f"normalization_layer = {cfg.normalization_layer} | loss_type = {cfg.loss_type}",
            f"duration = {cfg.duration} days | ic = {cfg.ic}"
            + (f" | ic_time = {cfg.ic_time}" if cfg.ic == "real_world" else ""),
            f"save_interval (golden, = n_future * true_interval_minutes) = "
            f"{cfg.save_interval_minutes:.2f} min ({cfg.save_interval_minutes/60:.2f} h)",
        ],
    })

    if Path(ref_ckpt_dir, "model_info.json").is_file():
        print(f"📦 Loading existing reference solver checkpoint {ref_ckpt_dir}")
        ref_solver, _ = load_numerical_checkpoint(ref_ckpt_dir)
        ref_solver.to(ref_solver.device)
    else:
        print(f"🛠  Initializing reference solver checkpoint -> {ref_ckpt_dir}")
        ref_solver = ShallowWaterSolver(
            lmax=cfg.nlat // 2, tau=ref_num_model_info['tau'], grid=cfg.grid,
            dealias=False, semi_implicit=ref_num_model_info['semi_implicit'],
            rad=ref_num_model_info['rad'], h_avg=h_avg, h_amp=h_amp, non_dimensional=True,
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
        # a spectral (native spherical-harmonic) ERA5 dataset has a flat 'values'
        # dim and no lat/lon, needing no vector transform - see
        # rw_initial_condition's own auto-detection/docstring, mirrored here.
        is_spectral = 'values' in ds.dims
        vSHT = None
        if not is_spectral:
            nlat_data, nlon_data = ds.sizes['latitude'], ds.sizes['longitude']
            vSHT = RealVectorSHT(nlat=nlat_data, nlon=nlon_data, lmax=nlat_data // 2, mmax=nlat_data // 2,
                                grid='equiangular', csphase=False).to(ref_solver.device)
        ds.close()

        start_load_time = time.perf_counter()
        era5_dataset = xr.open_dataset(netcdf_path).load()
        print(f"Finished loading ERA5 data in {(time.perf_counter() - start_load_time):2f} seconds")

        phivrtdivspec_0 = rw_initial_condition(
            model=ref_solver, vSHT=vSHT, era5_dataset=era5_dataset, ic_time=cfg.ic_time
            )
    else:
        raise ValueError(f"Unknown ic '{cfg.ic}' (expected galewsky or real_world).")

    # Spin up the initial condition using the numerical solver, for BOTH ic
    # branches - galewsky's analytic balanced-jet-plus-perturbation state
    # needs several days of free integration for the barotropic instability
    # to develop before it's a meaningful rollout start (the classic Galewsky
    # et al. 2004 protocol), exactly as real_world's raw reanalysis state
    # needs spinning up to the model's own dynamical balance. cfg.spinup_days
    # == 0 (e.g. task/galewsky_inference/make_inference_configs.py's galewsky
    # configs) skips this entirely, so it stays a no-op unless spin-up is
    # actually requested.
    if cfg.spinup_days > 0:
        print(f"Preparing solver for grid->spec transformation and initial spin-up ({cfg.ic})...")
        # Substep count is derived the same way SWEDataset's own warmup_steps
        # is (a frame count at the training data's own true_interval_minutes,
        # times compute_step_per_save's substeps-per-frame) rather than
        # independently from wall-clock seconds or an assumed nominal cadence,
        # so the state handed to the model below lands on the exact same real
        # time "frame 96" of a training trajectory represents - see
        # stage0.md Findings (frame-cadence rounding drift) for why an
        # independent seconds-based derivation silently drifted from that by
        # several hours. Reading model_info["true_interval_minutes"] directly
        # (rather than assuming 30 min) is what makes this exact even if the
        # model was trained on data with a different native cadence (e.g.
        # ERA5-direct).
        train_frame_minutes = model_info["true_interval_minutes"]
        spinup_frames = max(1, round(cfg.spinup_days * 24 * 60 / train_frame_minutes))
        n_spinup_substeps = spinup_frames * compute_step_per_save(train_frame_minutes, ref_solver.dt, ref_solver.T)
        state = phivrtdivspec_0
        with torch.no_grad():
            for i in tqdm.trange(n_spinup_substeps, desc=f"🐺->...->🦮  Spinning up the initial condition for {cfg.spinup_days} day"):
                state = ref_solver.timestep(uspec=state, nsteps=1)
        phivrtdivspec_0_spinned_up = state
    else:
        phivrtdivspec_0_spinned_up = phivrtdivspec_0

    # ------------------------------------------------------------------ #
    # (2) Run the SFNO rollout and the reference numerical solver from the
    #     same IC. Trajectory path deduction and reuse are entirely
    #     delegated to run() - it derives where each trajectory lives
    #     straight from the model_checkpoint path it's given (see
    #     run_model.py's _data_path_from_checkpoint) and skips straight to
    #     returning that path if a matching trajectory is already there.
    # ------------------------------------------------------------------ #
    ic_time = ((datetime.fromisoformat(cfg.ic_time) + timedelta(days=cfg.spinup_days)).isoformat()
               if cfg.ic == "real_world" else None)
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
        # galewsky only (see _data_path_from_checkpoint's docstring):
        # real_world already bakes spinup_days into the shifted ic_time above,
        # so this only changes the cache key for galewsky, where otherwise a
        # re-run with a different spinup_days would silently reuse a
        # trajectory spun up under a different value.
        spinup_days=cfg.spinup_days,
    )


    ref_save_path = run(
        model_checkpoint=str(ref_ckpt_dir),
        initial_condition=phivrtdivspec_0_spinned_up.clone(),
        duration=cfg.duration,
        ic=cfg.ic,
        pressure=cfg.pressure,
        ic_time=ic_time,
        dataset_name=dataset_name,
        # cfg.save_interval_minutes IS the golden true_interval_minutes-derived
        # cadence (see its computation above) - passing it here directly is
        # what keeps both branches frame-aligned AND makes the reference
        # trajectory advance by the real elapsed time an n_future model
        # forecast-jump represents; no separate nominal/true split needed.
        save_interval_minutes=cfg.save_interval_minutes,
        spinup_days=cfg.spinup_days,
    )

    # loaded once, up front: both branches' OWN recorded true_interval_minutes
    # (see run_model.py's run()) is the ground truth for what actually
    # happened during generation, more authoritative than re-deriving it from
    # cfg.save_interval_minutes (the request, not the achievement) - used
    # below for both the loss plot's x-axis and the comparison hour labels.
    neural_data = torch.load(neural_save_path, weights_only=False)
    ref_data = torch.load(ref_save_path, weights_only=False)
    true_interval_minutes = ref_data["metadata"]["true_interval_minutes"]
    print(f"    achieved true_interval_minutes: reference = {true_interval_minutes:.2f} min | "
          f"neural = {neural_data['metadata']['true_interval_minutes']:.2f} min")

    return SimpleNamespace(
        model_dir=model_dir,
        info_path=info_path,
        neural_save_path=neural_save_path,
        ref_save_path=ref_save_path,
        neural_data=neural_data,
        ref_data=ref_data,
        true_interval_minutes=true_interval_minutes,
    )


def _config_path():
    return sys.argv[1] if len(sys.argv) > 1 else "config.yml"


def main():
    cfg = load_config()

    if cfg.ic == "galewsky":
        # no dataset of times - galewsky is a SINGLETON evaluation set (one
        # canonical rollout pair per checkpoint, see run_model.py's
        # _data_path_from_checkpoint). run_batch_inference.py handles n=1
        # gracefully (degenerate/flat mean+-std bands), so it still runs here
        # rather than being skipped - a galewsky run should produce the same
        # per-step-loss/comparison diagnostics a real-world one does.
        cfg.ic_time = None
        run_rollout_pair(cfg)
        run_batch_inference(config_path=_config_path())
        return

    # ic == "real_world": the whole evaluation dataset IS the initial-condition
    # set - one rollout pair per valid_time, each independently cacheable/
    # resumable via run()'s own skip-if-already-there check.
    times = resolve_eval_times(cfg.dataset_name)
    print_in_box({
        "title": "Batch Inference Over Evaluation Dataset",
        "lines": [
            f"dataset_name = {cfg.dataset_name}",
            f"{len(times)} time point(s) found in reanalysis_data/{cfg.dataset_name}/data.nc",
        ],
    })
    for t in tqdm.tqdm(times, desc="Evaluation dataset"):
        cfg.ic_time = str(np.datetime_as_string(t, unit="s"))
        run_rollout_pair(cfg)

    run_batch_inference(config_path=_config_path())


def run_batch_inference(config_path):
    """Invoke task/batch_inference/run_batch_inference.py (the aggregate
    diagnostics - combined/per-channel L2 spectral loss, sample box/sphere
    comparisons) against `config_path` once every trajectory in the
    evaluation dataset has been computed above. Shelled out to (rather than
    imported) so src/entries/inference.py - core, reusable pipeline machinery
    - doesn't take a hard import dependency on task/, which holds one-off,
    per-experiment driver scripts (and isn't even version-controlled, see
    .gitignore); this mirrors exactly what
    task/batch_inference/run_batch_inference.sbatch already does by running
    the two scripts one after another.
    """
    script_path = SRC_DIR / "task" / "batch_inference" / "run_batch_inference.py"
    print_in_box({
        "title": "Running Batch Inference Diagnostics",
        "lines": [f"config = {config_path}", f"script = {script_path}"],
    })
    subprocess.run([sys.executable, str(script_path), config_path], check=True)


def plot_per_step_loss(neural_traj_path, ref_traj_path, info_path, lmax, grid,
                        save_interval_minutes, output_path):
    """Per-step relative spectral L2 loss of one SFNO trajectory against its
    reference. No longer called from main() (see module docstring - `make
    inference` only computes/saves trajectories now; task/batch_inference/
    run_batch_inference.py has the aggregate, whole-evaluation-set version of
    this computation) - kept here as a single-trajectory utility for
    src/entries/inference_era5_direct.py, which has no reference-numerical
    trajectory to aggregate over and so plots one trajectory's loss directly.
    """
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
    ax.plot(hours, losses, marker=".", markersize=3, linewidth=1)
    ax.set_ylim(0, 1.0)
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
