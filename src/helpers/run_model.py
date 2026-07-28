"""
Generate data from model checkpoints and provide the naming/organization
convention (see README.md) shared by the solver entry point and the trainer.

@author Haoyu Tang hytang2@illinois.edu
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

import glob
import json
import os
import time

import torch
import tqdm

from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator as SFNO
from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.helpers.print import print_in_box, finish_simulation_log


# --------------------------------------------------------------------------- #
# Naming / organization convention (README.md)
#
# Both `checkpoints/` and `model_output/` share the same tree, split first by
# model class (`numerical/` vs `neural_operator/`). The helpers below turn a
# configuration into the (directory, file_name) pair the convention prescribes,
# so run_solver.py can look up whether a checkpoint / dataset already exists.
# --------------------------------------------------------------------------- #
GRID_SHORTHAND = {
    "legendre-gauss": "lg",
    "equiangular": "eq",
    "lobatto": "lb",
}

IC_SHORTHAND = {
    "galewsky": "galewsky",
    "real_world": "rw",
}


def _fmt_duration(duration):
    """`20.0 -> '20'`, `2.5 -> '2.5'` so `duration_*` nodes stay clean."""
    duration = float(duration)
    return str(int(duration)) if duration.is_integer() else repr(duration)


def _tau_tag(tau):
    """`(30000, 30000, 30) -> '[30000,30000,30]'` (mirrors README)."""
    return "[" + ",".join(str(int(t)) for t in tau) + "]"


def _sanitize_time(ic_time):
    """`2000-07-01T00:00:00 -> 2000-07-01-000000` (filesystem-friendly)."""
    return str(ic_time).replace(":", "").replace("T", "-")


def numerical_checkpoint_path(lmax, tau, grid, semi_implicit):
    """(directory, file_name) of a numerical solver checkpoint.

    Tree: checkpoints/numerical/resol_*/tau_*/grid_*/method_*/
    File: <lmax>_<tau>_<grid>_<method>.pt
    """
    grid_short = GRID_SHORTHAND[grid]
    method = "implicit" if semi_implicit else "explicit"
    tau_tag = _tau_tag(tau)

    directory = os.path.join(
        "checkpoints", "numerical",
        f"resol_{lmax}", f"tau_{tau_tag}",
        f"grid_{grid_short}", f"method_{method}",
    )
    file_name = f"{lmax}_{tau_tag}_{grid_short}_{method}.pt"
    return directory, file_name


def numerical_data_path(lmax, tau, grid, semi_implicit, duration, ic,
                        pressure=None, ic_time=None):
    """(directory, file_name) of a numerical simulation output.

    Tree: <checkpoint tree>/duration_*/ic_*[/pressure_*/time_*]
    File name mirrors the path (README).
    """
    grid_short = GRID_SHORTHAND[grid]
    method = "implicit" if semi_implicit else "explicit"
    tau_tag = _tau_tag(tau)
    dur_tag = _fmt_duration(duration)
    ic_short = IC_SHORTHAND[ic]

    parts = [
        "model_output", "numerical",
        f"resol_{lmax}", f"tau_{tau_tag}",
        f"grid_{grid_short}", f"method_{method}",
        f"duration_{dur_tag}", f"ic_{ic_short}",
    ]
    name_parts = [str(lmax), tau_tag, grid_short, method, dur_tag, ic_short]

    if ic == "real_world":
        time_tag = _sanitize_time(ic_time)
        parts += [f"pressure_{pressure}", f"time_{time_tag}"]
        name_parts += [str(pressure), time_tag]

    directory = os.path.join(*parts)
    file_name = "_".join(name_parts) + ".pt"
    return directory, file_name

def _neural_config_nodes(resol, n_future, num_layers, embed_dim,
                         pos_embed, trainData_path, grid):
    """Shared tree nodes + file stem for the neural-operator convention.

    The tree layers (1-indexed, README.md):
      1 resol  2 nfuture  3 nlayer  4 embed  5 trainData  6 posEmbed  7 grid
    followed by a training-config index sub-directory (0/, 1/, ...).

    Returns (nodes, stem) where ``nodes`` is the ordered list of directory
    components and ``stem`` mirrors those values for file names.
    """
    if len(resol) != 2:
        raise ValueError("resol must be (Int, Int), like (128,256) specifying (nlat, nlon)")
    nlat, nlon = int(resol[0]), int(resol[1])
    grid_short = GRID_SHORTHAND.get(grid, grid)
    pe = str(pos_embed).replace(" ", "-")  # keep file-system friendly

    nodes = [
        f"resol_{nlat}|{nlon}",
        f"nfuture_{n_future}",
        f"nlayer_{num_layers}",
        f"embed_{embed_dim}",
        f"trainData_({trainData_path})",
        f"posEmbed_{pe}",
        f"grid_{grid_short}",
    ]
    stem = f"{nlat}|{nlon}_{n_future}_{num_layers}_{embed_dim}_{trainData_path}_{pe}_{grid_short}"
    return nodes, stem


def neural_model_path(resol,
                      n_future,
                      num_layers,
                      embed_dim,
                      pos_embed,
                      trainData_path,
                      grid,
                      index=0):
    """
    Get the path of a neural-operator model based on its configuration.

    Unlike the numerical checkpoint (a single .pt), each neural-operator run
    directory holds two files: ``model_info.json`` and the actual checkpoint,
    whose name mirrors the path with a ``_single.pt`` / ``_multi.pt`` suffix.

    args:
        resol: (nlat, nlon) resolution tuple.
        grid:  quadrature grid (7th tree layer).
        index: since the same architecture can be trained under different
               optimization configs, ``index`` selects the 0/, 1/, ... slot.

    Returns (directory, single_file_name, multi_file_name, info_file_name).
    """
    nodes, stem = _neural_config_nodes(
        resol, n_future, num_layers, embed_dim, pos_embed, trainData_path, grid)

    directory = os.path.join("checkpoints", "neural_operator", *nodes, str(index))

    single_file_name = f"{stem}_single.pt"
    multi_file_name = f"{stem}_multi.pt"
    info_file_name = "model_info.json"

    return directory, single_file_name, multi_file_name, info_file_name


def neural_data_path(resol,
                     n_future,
                     num_layers,
                     embed_dim,
                     pos_embed,
                     trainData_path,
                     grid,
                     duration,
                     ic,
                     pressure=None,
                     ic_time=None,
                     index=0):
    """(directory, file_name) of a neural-operator inference output.

    Parallels ``numerical_data_path``: the inference lives under
    ``model_output/neural_operator/`` inside the model's own config tree, then
    the shared ``duration_*/ic_*[/pressure_*/time_*]`` data sub-tree. The file
    name mirrors the path (same as the numerical data case).
    """
    nodes, stem = _neural_config_nodes(
        resol, n_future, num_layers, embed_dim, pos_embed, trainData_path, grid)

    dur_tag = _fmt_duration(duration)
    ic_short = IC_SHORTHAND[ic]

    data_nodes = [f"duration_{dur_tag}", f"ic_{ic_short}"]
    name_parts = [stem, dur_tag, ic_short]

    if ic == "real_world":
        time_tag = _sanitize_time(ic_time)
        data_nodes += [f"pressure_{pressure}", f"time_{time_tag}"]
        name_parts += [str(pressure), time_tag]

    directory = os.path.join(
        "model_output", "neural_operator", *nodes, str(index), *data_nodes)
    file_name = "_".join(name_parts) + ".pt"
    return directory, file_name




def run(model_checkpoint,
        initial_condition,
        output_dir,
        file_name,
        duration,
        save_interval_minutes=30,
        single_step=True):
    """
        A ubiquitous interface for both inferencing SFNO and psuedospectral simulation.
        Generates a .pt file of prediction data at ``output_dir/file_name``.

        params:
            model_checkpoint: for the neural operator a directory containing
                              model_info.json and the checkpoint; for the
                              numerical solver a .pt file holding a pickled
                              ShallowWaterSolver.
            initial_condition: state at t_0 (spectral (3, lmax, mmax) 
            file_name:        name of the trajectory .pt file (see convention).
            save_interval_minutes: interval to save in simulation time, minutes.
            single_step:      use single step model if True, else multistep.
    """
    start_time = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("Device is now CPU !")

    # number of frames in the simulation
    number_of_frames = max(1, int((duration * 1440) / save_interval_minutes))

    # ------------------------------------------------------------------ #
    # Neural operator inference: model_checkpoint is a directory holding
    # model_info.json alongside the checkpoint file.
    # ------------------------------------------------------------------ #
    if os.path.isdir(model_checkpoint):
        print("📖 🌍 Generating prognostic data using SFNO 📖 🌍".center(60))

        info_path = os.path.join(model_checkpoint, "model_info.json")
        with open(info_path, 'r') as f:
            model_info = json.load(f)

        # checkpoint file mirrors the path and ends with _single.pt / _multi.pt
        tag = "single" if single_step else "multi"
        matches = sorted(glob.glob(os.path.join(model_checkpoint, f"*_{tag}.pt")))
        if not matches:
            raise FileNotFoundError(
                f"No '*_{tag}.pt' checkpoint found in {model_checkpoint}")
        checkpoint_path = matches[0]

        n_future = model_info['n_future']
        nlat, nlon = model_info['nlat'], model_info['nlon']
        grid = model_info.get('grid', 'equiangular')

        if save_interval_minutes % (30 * n_future) != 0:
            raise RuntimeError(
                f"When running neural operators, save interval must be an integer "
                f"multiple of {30 * n_future} minutes")
        # number of autoregressive model calls between two saved frames
        model_steps_per_save = max(1, int(save_interval_minutes // (30 * n_future)))

        run_log_content = {
            "title": "Running Single-Step SFNO Inference",
            "lines": [
                f"Days : {duration} | save_interval : {save_interval_minutes}(minutes) | Total frames = {number_of_frames}"
                # f"output_dir = {output_dir} | file_name = {file_name}",
            ],
        }
        print_in_box(run_log_content)

        model = SFNO(
            img_size=(nlat, nlon), grid=grid,
            num_layers=model_info['num_layers'],
            scale_factor=model_info['scale_factor'],
            embed_dim=model_info['embed_dim'],
            residual_prediction=model_info.get('residual_prediction', True),
            pos_embed=model_info['pos_embed'], use_mlp=True,
            normalization_layer=model_info.get('normalization_layer', 'none'),
        ).to(device)

        print(" Loading weights ...")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Per-channel grid-space stats used to (de)normalize the state.
        mean = torch.tensor(
            [model_info['inp_mean_gp'], model_info['inp_mean_zeta'], model_info['inp_mean_delta']],
            device=device).reshape(3, 1, 1)
        std = torch.tensor(
            [model_info['inp_std_gp'], model_info['inp_std_zeta'], model_info['inp_std_delta']],
            device=device).reshape(3, 1, 1)

        # dtype needs to be complex64 to prevent complex->real casting 
        trajectory = torch.empty((number_of_frames, *(initial_condition.shape)), dtype=initial_condition.dtype)

        print("Preparing solver for grid->spec transformation and initial spin-up...")
        solver = ShallowWaterSolver(lmax=nlat//2, grid=grid, dealias=False)
        solver.to(device)
        init_end_time = time.perf_counter()
        print(f"⏰ Finished model initialization in {init_end_time - start_time:.3f} seconds")

        start_sim_time = time.perf_counter()
        # spin up the initial condition using numerical solver 
        state = initial_condition
        with torch.no_grad():
            for i in tqdm.trange(int(86400//solver.dt), desc="🐺->...->🦮  Spinning up the initial condition for 1 day"):
                state = solver.timestep(uspec=state, nsteps=1)
        
        state = ((solver.spec2grid(state.to(device)) - mean)/std).float() #cast to float to match with data type of weights
        with torch.no_grad():
            for i in tqdm.trange(number_of_frames, desc='Inference in Progress'):
                trajectory[i] = (solver.grid2spec(state * std + mean)).cpu()
                if i < number_of_frames - 1:
                    for _ in range(model_steps_per_save):
                        state = model(state.unsqueeze(0)).squeeze(0)
        end_sim_time = time.perf_counter()
        print(f"⏰ Finished model inference in {end_sim_time - start_sim_time:.3f} seconds")
        data = {
            'metadata': {
                'nlat': nlat,
                'nlon': nlon,
                'lmax': nlat // 2,
                'mmax': nlat // 2,
                'grid': grid,
                'n_future': n_future,
                'step_per_save': model_steps_per_save,
                'save_interval_minutes': save_interval_minutes,
            },
            'trajectory': trajectory,
        }
        print(f"metadata is {data['metadata']}")

    # ------------------------------------------------------------------ #
    # Numerical simulation: model_checkpoint is a pickled ShallowWaterSolver.
    # ------------------------------------------------------------------ #
    else:
        print("🧮 🌍 Generating prognostic data using Psuedospectral Solver 🧮 🌍".center(60))

        solver = torch.load(model_checkpoint, weights_only=False)
        solver.to(device)
        # restart the leapfrog scheme from the fresh initial condition
        solver.reset_time()

        uspec = initial_condition.to(device)

        # number of solver steps between two saved frames
        step_per_save = max(1, round(save_interval_minutes * 60.0 / solver.dt))

        run_log_content = {
            "title": "Running Psuedo-Spectral Solver",
            "lines": [
                f"Days : {duration} | save_interval : {save_interval_minutes}(minutes) | Total frames = {number_of_frames}",
                f"dt = {solver.dt:.2f}s | step_per_save = {step_per_save}"
                # f"output_dir = {output_dir} | file_name = {file_name}",
            ],
        }
        print_in_box(run_log_content)

        content = {
            'title': " Hyperdiffusion ",
            'lines': [
                f"model e-fold time (𝝉₂, 𝛕₄, 𝛕₈) = {tuple(solver.tau)} hours"
            ],
        }
        print_in_box(content)

        trajectory = torch.empty((number_of_frames, *uspec.shape), dtype=uspec.dtype)
        with torch.no_grad():
            for i in tqdm.trange(number_of_frames, desc='Simulation in Progress'):
                trajectory[i] = uspec.cpu()
                if i < number_of_frames - 1:  # no simulation for the last step
                    uspec = solver.timestep(uspec, step_per_save)

        data = {
            'metadata': {
                'dt': solver.dt,
                'tau': solver.tau,
                'nlat': solver.nlat,
                'nlon': solver.nlon,
                'lmax': solver.lmax,
                'mmax': solver.mmax,
                'grid': solver.grid,
                'step_per_save': step_per_save,
                'save_interval_minutes': save_interval_minutes,
            },
            'trajectory': trajectory,
        }

    # persist the trajectory following the naming convention
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_name)
    torch.save(data, save_path)
    finish_simulation_log(save_path, time=(time.perf_counter() - start_time))
    return save_path


