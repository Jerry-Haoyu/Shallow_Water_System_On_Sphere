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

import numpy as np
import torch
import tqdm

from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator as SFNO
from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.helpers.print import print_in_box, finish_simulation_log


# --------------------------------------------------------------------------- #
# Non-dimensionalization scales (see ShallowWaterSolver's non_dimensional
# docstring). Trajectories saved by `run()` are always in physical units
# regardless of whether the simulation/inference that produced them ran
# non-dimensionally internally - these helpers compute the (T, U) needed for
# that final conversion, and convert a spectral state with them.
# --------------------------------------------------------------------------- #
EARTH_RADIUS = 6.37122e6    # m, mirrors AbstractSWSolver's default
EARTH_GRAVITY = 9.80616     # m/s^2, mirrors AbstractSWSolver's default
DEFAULT_HAVG = 10.0e3       # m, reference height for runs with no dataset (e.g. galewsky)


def load_h_stats(dataset_name):
    """Dataset-wide (h_avg, h_amp) in meters, averaged over time, from the
    h_stats.npz that `make download_era5` / src.analyze.statistics computes
    for this dataset. arr_0/arr_1 are the per-time h_avg/h_amp arrays."""
    stats_path = Path("reanalysis_data") / dataset_name / "h_stats.npz"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"{stats_path} not found; run `make download_era5` for dataset "
            f"'{dataset_name}' first (it computes h_stats.npz via src/analyze/statistics.py)."
        )
    stats = np.load(stats_path)
    return float(stats["mean"][0]), float(stats["mean"][1])


def physical_scales(h_avg=None):
    """Velocity scale U=sqrt(g*h_avg) and time scale T=radius/U - the same
    scales ShallowWaterSolver's non_dimensional=True rescaling uses internally.
    h_avg is the dataset-wide reference height (m, from load_h_stats); pass
    None (e.g. galewsky, which has no dataset) to fall back to DEFAULT_HAVG.
    """
    h_avg = DEFAULT_HAVG if h_avg is None else float(h_avg)
    U = float(np.sqrt(EARTH_GRAVITY * h_avg))
    T = EARTH_RADIUS / U
    return T, U


def _nondim_to_physical(uspec, T, U):
    """Undo ShallowWaterSolver's non_dimensional=True rescaling: geopotential
    (channel 0) was scaled down by U**2 and vorticity/divergence (channels
    1:) by T (both carry units of 1/time). Works on a single (3, lmax, mmax)
    frame or a (N, 3, lmax, mmax) trajectory alike - the channel axis is
    always third-from-last.
    """
    out = uspec.clone()
    out[..., 0, :, :] = out[..., 0, :, :] * (U ** 2)
    out[..., 1:, :, :] = out[..., 1:, :, :] / T
    return out


def physical_to_nondim(x, T, U):
    """Inverse of _nondim_to_physical: rescale a physical-unit state into
    ShallowWaterSolver's non-dimensional units (geopotential / U**2,
    vorticity+divergence * T). This is the normalization the neural-operator
    pipeline trains/infers on (see dataset.py) in place of a z-score. The
    scaling is per-channel and linear, so it commutes with the (linear)
    spherical-harmonic transform - `x` may be spectral (..., 3, lmax, mmax)
    or grid-space (..., 3, nlat, nlon) alike, channel axis third-from-last.
    """
    out = x.clone()
    out[..., 0, :, :] = out[..., 0, :, :] / (U ** 2)
    out[..., 1:, :, :] = out[..., 1:, :, :] * T
    return out


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

NORM_SHORTHAND = {
    "none": "none",
    "layer_norm": "layer",
    "instance_norm": "instance",
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
    """tau_(30000,30000,30]'"""
    return "(" + ",".join(str(int(t)) for t in tau) + ")"


def _date_tag(ic_time):
    """`2000-07-01T00:00:00 -> '2000_07_01'` (date-only, training-data exception)."""
    date_part = str(ic_time).split("T")[0].split(" ")[0]
    return date_part.replace("-", "_")


def numerical_checkpoint_path(lmax, tau, grid, semi_implicit, rad=False, dataset_name=None):
    """Directory of a numerical solver checkpoint.

    Tree: checkpoints/numerical/resol_*/tau_*/grid_*/method_*/radiation_*/[dataset_*/]
    Holds: model_info.json + checkpoints.pt (see save_numerical_checkpoint /
    load_numerical_checkpoint) - like a neural-operator checkpoint, this is a
    directory, not a single file.

    rad: whether geopotential relaxes toward a radiative-equilibrium background
        (see ShallowWaterSolver's rad/tau_rad). Always present as either
        radiation_rad or radiation_no_rad (unlike dataset_name below, this node
        is never omitted) so rad/no_rad checkpoints never collide.

    dataset_name: real-world runs derive havg/hamp (and so dt, hyperdiffusion,
        Coriolis) from the dataset's h_stats.npz, so their checkpoint is only
        valid for that one dataset. Pass the dataset name to keep checkpoints
        from different datasets from colliding; omit (galewsky) to keep the
        tree as before.
    """
    grid_short = GRID_SHORTHAND[grid]
    method = "implicit" if semi_implicit else "explicit"
    tau_tag = _tau_tag(tau)

    parts = [
        "checkpoints", "numerical",
        f"resol_{lmax}", f"tau_{tau_tag}",
        f"grid_{grid_short}", f"method_{method}",
        "radiation_rad" if rad else "radiation_no_rad",
    ]
    if dataset_name:
        parts.append(f"dataset_{dataset_name}")

    return os.path.join(*parts)


def save_numerical_checkpoint(solver, ckpt_dir, dataset_name=None, pressure=None):
    """Persist a numerical solver checkpoint as model_info.json (reconstruction
    config + dataset provenance) + checkpoints.pt (its state_dict), replacing
    the old whole-object torch.save(solver, ...) pickle - see model_info.json's
    schema in README.md. Everything but dataset_name/pressure (not derivable
    from the solver itself) is read straight off `solver`.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    umax_phys = solver.umax * solver.U if solver.non_dimensional else solver.umax
    model_info = {
        "type": "numerical",
        "lmax": solver.lmax,
        "tau": list(solver.tau),
        "cfl": solver.cfl,
        "grid": solver.grid,
        "semi_implicit": solver.semi_implicit,
        "dealias": solver.dealias,
        "robert_coeff": solver.robert_coeff,
        "umax": umax_phys,
        "non_dimensional": solver.non_dimensional,
        "rad": solver.rad,
        "tau_rad": solver.tau_rad_days,  # days (see ShallowWaterSolver's rad/tau_rad)
        "dataset_name": dataset_name,
        "pressure": pressure,
        "h_avg": solver.havg_phys,
        "h_amp": solver.hamp_phys,
        "U": solver.U,
        "T": solver.T,
    }
    with open(os.path.join(ckpt_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)
    torch.save(solver.state_dict(), os.path.join(ckpt_dir, "checkpoints.pt"))


def load_numerical_checkpoint(ckpt_dir):
    """Inverse of save_numerical_checkpoint: rebuild the solver from
    ckpt_dir/model_info.json (reconstruction config) and, if present, restore
    ckpt_dir/checkpoints.pt's state_dict on top (a cache of the deterministic
    buffers/transforms the constructor already derives from that same config).

    Returns (solver, model_info).
    """
    model_info = load_model_info(ckpt_dir)
    solver = ShallowWaterSolver(
        lmax=model_info["lmax"], tau=tuple(model_info["tau"]), cfl=model_info["cfl"],
        grid=model_info["grid"], semi_implicit=model_info["semi_implicit"],
        dealias=model_info["dealias"], robert_coeff=model_info["robert_coeff"],
        umax=model_info["umax"], h_avg=model_info["h_avg"], h_amp=model_info["h_amp"],
        non_dimensional=model_info["non_dimensional"],
        rad=model_info.get("rad", False), tau_rad=model_info.get("tau_rad"),
    )
    state_path = os.path.join(ckpt_dir, "checkpoints.pt")
    if os.path.isfile(state_path):
        solver.load_state_dict(torch.load(state_path, weights_only=True))
    return solver, model_info


def load_model_info(model_checkpoint):
    """model_info.json of a checkpoint directory (numerical or neural alike -
    both are directories, see numerical_checkpoint_path / neural_model_path)."""
    with open(os.path.join(model_checkpoint, "model_info.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def parse_dataset_and_pressure(data_dir):
    """(dataset_name, pressure) read off a model_output/-convention directory's
    own path nodes (inverse of _data_path_from_checkpoint's dataset_*/pressure_*
    nodes) - (None, None) if absent (e.g. galewsky-only data, which has neither).
    Used to recover a neural-operator's training dataset identity from its
    training_data_dir.
    """
    dataset_name = None
    pressure = None
    for part in Path(data_dir).parts:
        if part.startswith("dataset_"):
            dataset_name = part[len("dataset_"):]
        elif part.startswith("pressure_"):
            pressure = part[len("pressure_"):]
    return dataset_name, pressure


def _neural_config_nodes(resol, n_future, num_layers, embed_dim,
                         pos_embed, trainData_path, grid,
                         normalization_layer, loss_type):
    """Tree nodes for the neural-operator convention.

    The tree layers (1-indexed, README.md):
      1 resol  2 nfuture  3 nlayer  4 embed  5 trainData  6 posEmbed  7 grid
      8 norm  9 loss
    followed by a training-config index sub-directory (0/, 1/, ...).
    """
    if len(resol) != 2:
        raise ValueError("resol must be (Int, Int), like (128,256) specifying (nlat, nlon)")
    nlat, nlon = int(resol[0]), int(resol[1])
    grid_short = GRID_SHORTHAND.get(grid, grid)
    pe = str(pos_embed).replace(" ", "-")  # keep file-system friendly
    norm_short = NORM_SHORTHAND.get(normalization_layer, normalization_layer)

    nodes = [
        f"resol_{nlat}|{nlon}",
        f"nfuture_{n_future}",
        f"nlayer_{num_layers}",
        f"embed_{embed_dim}",
        f"trainData_({trainData_path})",
        f"posEmbed_{pe}",
        f"grid_{grid_short}",
        f"norm_{norm_short}",
        f"loss_{loss_type}",
    ]
    return nodes


def neural_model_path(resol,
                      n_future,
                      num_layers,
                      embed_dim,
                      pos_embed,
                      trainData_path,
                      grid,
                      normalization_layer="none",
                      loss_type="spectral",
                      index=0):
    """
    Get the path of a neural-operator model based on its configuration.

    Unlike the numerical checkpoint (a single .pt), each neural-operator run
    directory holds two files: ``model_info.json`` and the actual checkpoint,
    named ``checkpoints_single.pt`` / ``checkpoints_multi.pt``.

    args:
        resol: (nlat, nlon) resolution tuple.
        grid:  quadrature grid (7th tree layer).
        normalization_layer: SFNO's own norm choice - none | layer_norm |
            instance_norm (8th tree layer).
        loss_type: which loss function trained this run - "grid" | "spectral"
            (9th tree layer, see src/neural_operator/loss.py's LOSS_FUNCTIONS).
        index: since the same architecture/loss can be trained under different
               optimization configs, ``index`` selects the 0/, 1/, ... slot.

    Returns (directory, single_file_name, multi_file_name, info_file_name).
    """
    nodes = _neural_config_nodes(
        resol, n_future, num_layers, embed_dim, pos_embed, trainData_path, grid,
        normalization_layer, loss_type)

    directory = os.path.join("checkpoints", "neural_operator", *nodes, str(index))

    single_file_name = "checkpoints_single.pt"
    multi_file_name = "checkpoints_multi.pt"
    info_file_name = "model_info.json"

    return directory, single_file_name, multi_file_name, info_file_name


def is_neural_checkpoint(model_checkpoint):
    """Whether `model_checkpoint` points at an SFNO run directory (True) or a
    numerical solver checkpoint directory (False) - both are directories
    holding model_info.json + their own checkpoint file(s), see
    save_numerical_checkpoint / neural_model_path.

    Decided from the ROOT NODE of its own checkpoint tree (README.md) -
    checkpoints/numerical/... vs checkpoints/neural_operator/... - rather
    than by inspecting the directory's contents, so it works purely off the
    path string, for a path that may not exist on disk yet too.
    """
    parts = Path(model_checkpoint).parts
    try:
        idx = parts.index("checkpoints")
        root = parts[idx + 1]
    except (ValueError, IndexError):
        raise ValueError(
            f"model_checkpoint must live under a 'checkpoints/<numerical|neural_operator>/' "
            f"root per the naming convention (README.md); got '{model_checkpoint}'."
        )
    if root == "numerical":
        return False
    if root == "neural_operator":
        return True
    raise ValueError(
        f"unrecognized checkpoint root node '{root}' in '{model_checkpoint}' "
        f"(expected 'numerical' or 'neural_operator')."
    )


def _data_path_from_checkpoint(model_checkpoint, duration, ic,
                               pressure=None, ic_time=None, dataset_name=None):
    """(directory, file_name) of a run's trajectory output, derived entirely
    from `model_checkpoint`'s own path plus the data-specific info a single
    checkpoint doesn't pin down (duration, ic, ...).

    The data tree is isomorphic to the checkpoint tree (README.md): the same
    config-identifying prefix, rooted at model_output/ instead of
    checkpoints/, with data-specific nodes (duration_*, ic_*, pressure_*)
    branching further off it before the final dataset_<name>/ leaf
    (real-world only) - so none of the checkpoint's own configuration
    (lmax/tau/grid/... or resol/nfuture/.../index) needs to be re-supplied
    here; it's read straight off the path string.
    """
    parts = list(Path(model_checkpoint).parts)
    try:
        idx = parts.index("checkpoints")
    except ValueError:
        raise ValueError(
            f"model_checkpoint must live under a 'checkpoints/' root per the "
            f"naming convention (README.md); got '{model_checkpoint}'."
        )
    parts[idx] = "model_output"

    # a numerical checkpoint for a real-world run carries its own dataset_<name>
    # leaf (see numerical_checkpoint_path - h_avg/h_amp come from that dataset,
    # so the checkpoint is only valid for it) - lift it off the prefix; the
    # data tree puts dataset_<name> at ITS OWN leaf instead, after
    # duration/ic/pressure, so many ic_times pool into the same directory.
    ckpt_dataset_name = None
    if parts[-1].startswith("dataset_"):
        ckpt_dataset_name = parts.pop()[len("dataset_"):]

    if ckpt_dataset_name is not None and dataset_name is not None and ckpt_dataset_name != dataset_name:
        raise ValueError(
            f"dataset_name='{dataset_name}' does not match the dataset "
            f"'{ckpt_dataset_name}' baked into checkpoint '{model_checkpoint}'."
        )
    dataset_name = dataset_name or ckpt_dataset_name

    dur_tag = _fmt_duration(duration)
    ic_short = IC_SHORTHAND[ic]
    parts += [f"duration_{dur_tag}", f"ic_{ic_short}"]

    if ic == "real_world":
        if not dataset_name:
            raise ValueError(
                "dataset_name is required when ic == 'real_world' - either baked "
                "into a numerical checkpoint's own path, or passed explicitly "
                "(always required for a neural-operator checkpoint, which isn't "
                "tied to any one real-world dataset)."
            )
        parts += [f"pressure_{pressure}", f"dataset_{dataset_name}"]
        file_name = f"{_date_tag(ic_time)}.pt"
    else:
        file_name = "model_output.pt"

    directory = os.path.join(*parts)
    return directory, file_name


def run(model_checkpoint,
        initial_condition,
        duration,
        ic,
        pressure=None,
        ic_time=None,
        dataset_name=None,
        save_interval_minutes=30,
        single_step=True,
        phi_eq_spec=None):
    """
        A ubiquitous interface for both inferencing SFNO and psuedospectral simulation.
        Generates a .pt file of prediction data, then returns its path.

        Args:
            model_checkpoint: path to the model checkpoint directory
            initial_condition: state at t_0 (spectral (3, lmax, mmax); units
                              dictated by model_checkpoint's own
                              model_info.json - see non_dimensional below.
            duration:         simulation/rollout length, days.
            ic:               "galewsky" or "real_world" (naming convention).
            pressure, ic_time: required when ic == "real_world" (naming
                              convention only; ic_time also selects the IC).
            dataset_name:     required when ic == "real_world", *except* a
                              numerical checkpoint already tied to one dataset
                              (see numerical_checkpoint_path) - passing a
                              different one there is an error, not an override.
                              Note this identifies `initial_condition`'s own
                              dataset (the rollout being run), which need not
                              be the dataset model_checkpoint was itself
                              built/trained against.
            save_interval_minutes: interval to save in simulation time, minutes.
            single_step:      use single step model if True, else multistep.
            phi_eq_spec:      required when model_checkpoint is a numerical checkpoint
                              with rad=True (see ShallowWaterSolver's rad/tau_rad) -
                              the zonally-symmetric equilibrium geopotential dudtspec's
                              rad term relaxes toward (see
                              initial_condition.radiative_equilibrium_geopotential).
                              This function reconstructs its own solver from
                              model_checkpoint internally (numerical branch below), so
                              a phi_eq_spec set via set_equilibrium_geopotential on some
                              other solver instance the caller may hold never reaches
                              the one actually used here - it must be passed in
                              directly. Ignored for neural checkpoints / rad=False.

        non_dimensional (whether `initial_condition` operates in
        ShallowWaterSolver's non-dimensional units) and T/U (the
        non-dimensionalization time/velocity scales, T=radius/U,
        U=sqrt(g*h_avg)) are no longer caller-supplied - both checkpoint types
        are self-describing (model_info.json), so they're read straight off
        model_checkpoint's own record instead: solver.T/solver.U (numerical,
        via load_numerical_checkpoint) or model_info["T"]/["U"] (neural,
        always non_dimensional=True - see train_singlestep.py). The saved
        trajectory is always converted back to physical units before being
        written to disk, so every .pt file on disk stays in physical units
        regardless of how the run itself was computed.
    """
    # dispatch (numerical vs neural) is decided purely by what model_checkpoint
    # points at, so it can be resolved before any heavy lifting - both to pick
    # the right branch, further down, and to interpret its path correctly.
    is_neural = is_neural_checkpoint(model_checkpoint)

    output_dir, file_name = _data_path_from_checkpoint(
        model_checkpoint, duration, ic,
        pressure=pressure, ic_time=ic_time, dataset_name=dataset_name,
    )

    # resolved before touching model_checkpoint's own model_info.json at all,
    # so a caller doesn't need model_checkpoint to even be fully written yet
    # just to find out the trajectory is already cached.
    save_path = os.path.join(output_dir, file_name)
    if os.path.isfile(save_path):
        print(f"✅ Trajectory already exists at {save_path}; skipping.")
        return save_path

    # T/U/non_dimensional are properties of model_checkpoint itself (see
    # docstring above), not something callers compute and pass in.
    model_info = load_model_info(model_checkpoint)
    T, U = model_info["T"], model_info["U"]
    non_dimensional = True if is_neural else model_info["non_dimensional"]

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
    if is_neural:
        print("📖 🌍 Generating prognostic data using SFNO 📖 🌍".center(60))

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

        # dtype needs to be complex64 to prevent complex->real casting
        trajectory = torch.empty((number_of_frames, *(initial_condition.shape)), dtype=initial_condition.dtype)

        solver = ShallowWaterSolver(lmax=nlat//2, grid=grid, dealias=False)
        solver.to(device)
        
        init_end_time = time.perf_counter()
        print(f"⏰ Finished model initialization in {init_end_time - start_time:.3f} seconds")

        start_sim_time = time.perf_counter()

        # the SFNO was trained directly on T/U-non-dimensionalized grid fields
        # (see dataset.py) - `state` is already in those units after the
        # non-dimensional spin-up above (is_neural implies non_dimensional
        # is always True here), so it feeds the model as-is; only the *saved*
        # trajectory needs converting back to physical units, each step.
        state = solver.spec2grid(initial_condition.to(device)).float()  # cast to float to match the weights' dtype
        with torch.no_grad():
            for i in tqdm.trange(number_of_frames, desc='Inference in Progress'):
                trajectory[i] = _nondim_to_physical(solver.grid2spec(state), T, U).cpu()
                if i < number_of_frames - 1:
                    for _ in range(model_steps_per_save):
                        state += model(state.unsqueeze(0)).squeeze(0)
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
    # Numerical simulation: model_checkpoint is a directory holding
    # model_info.json + checkpoints.pt (see load_numerical_checkpoint).
    # ------------------------------------------------------------------ #
    else:
        print("🧮 🌍 Generating prognostic data using Psuedospectral Solver 🧮 🌍".center(60))

        solver, _ = load_numerical_checkpoint(model_checkpoint)
        solver.to(device)
        # restart the leapfrog scheme from the fresh initial condition
        solver.reset_time()

        if solver.rad:
            if phi_eq_spec is None:
                raise ValueError(
                    f"model_checkpoint={model_checkpoint!r} has rad=True but run() "
                    f"was not given phi_eq_spec (see initial_condition."
                    f"radiative_equilibrium_geopotential)."
                )
            solver.set_equilibrium_geopotential(phi_eq_spec.to(device))

        uspec = initial_condition.to(device)

        # number of solver steps between two saved frames. solver.dt is in units of
        # solver.T (=1 second when the solver is dimensional), so convert the physical
        # save_interval into solver.dt's own units before dividing.
        solver_T = getattr(solver, 'T', 1.0)
        step_per_save = max(1, round(save_interval_minutes * 60.0 / (solver.dt * solver_T)))

        run_log_content = {
            "title": "Running Psuedo-Spectral Solver",
            "lines": [
                f"Days : {duration} | save_interval : {save_interval_minutes}(minutes) | Total frames = {number_of_frames}",
                f"dt = {solver.dt:.4g} ({'non-dim' if getattr(solver, 'non_dimensional', False) else 's'}) | step_per_save = {step_per_save}"
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

        # the solver may have integrated in non-dimensional units; the saved
        # trajectory is always converted back to physical units so every .pt
        # on disk stays consistent regardless of how it was produced.
        if non_dimensional:
            trajectory = _nondim_to_physical(trajectory, T, U)

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
                # provenance only - the trajectory above is always physical.
                'non_dimensional': non_dimensional,
                'T': T,
                'U': U,
            },
            'trajectory': trajectory,
        }

    # persist the trajectory following the naming convention
    os.makedirs(output_dir, exist_ok=True)
    torch.save(data, save_path)
    finish_simulation_log(save_path, time=(time.perf_counter() - start_time))
    return save_path


