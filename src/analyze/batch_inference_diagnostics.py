#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Generate diagnostics for batch inference with mean, mean ± std for three channels 

@author Haoyu Tang hytang2@illinois.edu and Claude
"""
import sys
import os
import glob
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.neural_operator.loss import LOSS_FUNCTIONS
from src.analyze.visualization import plot_sphere_comparison, plot_box_comparison, plot_trajectory_diagnostics
from src.helpers.run_model import load_model_info, physical_to_nondim, _data_path_from_checkpoint
from src.helpers.print import print_in_box
from src.entries.inference import load_config, resolve_checkpoint_dirs, dataset_tag

CHANNEL_NAMES = ["geopotential", "vorticity", "divergence"]
CHANNEL_COLORS = {"geopotential": "tab:red", "vorticity": "tab:blue", "divergence": "tab:green"}


def resolve_trajectory_dirs(cfg):
    """(neural_dir, ref_dir, galewsky_file_name): the two model_output/ leaf
    directories `make inference` wrote this evaluation dataset's trajectories
    into - the exact same resolution `run()` itself uses (see run_model.py's
    _data_path_from_checkpoint), so every .pt file it already wrote is found
    here without recomputing or re-running anything.

    galewsky_file_name is the exact filename THIS cfg's own trajectory got
    (None for ic == "real_world", where every file in the directory belongs to
    the evaluation set - see find_trajectory_pairs). galewsky is nominally a
    SINGLETON evaluation set, but cfg.spinup_days is tagged into the filename,
    not the directory (see _data_path_from_checkpoint's spinup_days docstring)
    - so re-running `make inference` with a different spinup_days leaves the
    PREVIOUS spinup_days's cached trajectory sitting in the same directory
    rather than overwriting it. Without filtering down to this exact filename,
    a directory-wide glob (as real_world correctly wants, since there every
    file is a distinct, currently-valid ic_time) would silently treat every
    stale spinup_days variant ever run as another "trajectory" in the
    aggregate mean/std bands and sample comparisons.
    """
    model_dir, ref_ckpt_dir, _ = resolve_checkpoint_dirs(cfg)
    # dataset_tag(cfg), not the raw cfg.dataset_name: for ic == "galewsky" the
    # reference checkpoint is tagged with cfg.trainData (see
    # inference.py's resolve_checkpoint_dirs), and _data_path_from_checkpoint
    # raises if the dataset_name passed in here doesn't match a checkpoint's
    # own baked-in tag - passing the stale/irrelevant cfg.dataset_name here
    # would false-positive that mismatch check.
    tag = dataset_tag(cfg)
    # only affects the returned file_name (galewsky) - harmless to pass for
    # real_world too, where spinup_days plays no role in that branch's own
    # ic_time-keyed file_name (see _data_path_from_checkpoint).
    spinup_days = getattr(cfg, "spinup_days", None)
    neural_dir, file_name = _data_path_from_checkpoint(
        model_dir, cfg.duration, cfg.ic, pressure=cfg.pressure,
        dataset_name=tag, single_step=cfg.single_step, spinup_days=spinup_days,
    )
    ref_dir, _ = _data_path_from_checkpoint(
        ref_ckpt_dir, cfg.duration, cfg.ic, pressure=cfg.pressure,
        dataset_name=tag, single_step=cfg.single_step, spinup_days=spinup_days,
    )
    galewsky_file_name = file_name if cfg.ic == "galewsky" else None
    return neural_dir, ref_dir, galewsky_file_name


def find_trajectory_pairs(neural_dir, ref_dir, only_file_name=None):
    """(name, neural_path, ref_path) for every trajectory `make inference`
    wrote for this dataset, matched by filename - both trees share the same
    date-tag naming (see run_model.py's _data_path_from_checkpoint).

    only_file_name restricts this to that one filename instead of globbing
    every "*.pt" in neural_dir - see resolve_trajectory_dirs' docstring for
    why galewsky (a nominally singleton evaluation set) needs this: its
    directory can hold more than one cached trajectory (one per spinup_days
    value ever run), and only the one matching the CURRENT cfg should count.
    """
    if only_file_name is not None:
        neural_paths = [os.path.join(neural_dir, only_file_name)]
        if not os.path.isfile(neural_paths[0]):
            neural_paths = []
    else:
        neural_paths = sorted(glob.glob(os.path.join(neural_dir, "*.pt")))

    pairs = []
    for neural_path in neural_paths:
        name = os.path.basename(neural_path)
        ref_path = os.path.join(ref_dir, name)
        if os.path.isfile(ref_path):
            pairs.append((name, neural_path, ref_path))
        else:
            print(f"⚠️  no matching reference trajectory for {name}; skipping.")
    return pairs


def per_step_losses(neural_path, ref_path, solver, loss_fn, T, U):
    """(combined, per_channel, true_interval_minutes) for one trajectory pair:
    combined is (n_steps,), per_channel is (n_steps, 3) - relative spectral L2
    loss of the SFNO rollout against its reference, per saved frame. Mirrors
    the old (single-trajectory) inference.py plot_per_step_loss's loss
    computation, just returning the raw arrays instead of plotting one figure.
    """
    neural = torch.load(neural_path, weights_only=False)
    ref = torch.load(ref_path, weights_only=False)
    neural_traj = neural["trajectory"].to(solver.device)
    ref_traj = ref["trajectory"].to(solver.device)
    true_interval_minutes = ref["metadata"]["true_interval_minutes"]

    n_steps = min(neural_traj.shape[0], ref_traj.shape[0])
    combined = np.empty(n_steps)
    per_channel = np.empty((n_steps, len(CHANNEL_NAMES)))
    with torch.no_grad():
        for t in range(n_steps):
            prd = solver.spec2grid(neural_traj[t].to(torch.complex64))
            tar = solver.spec2grid(ref_traj[t].to(torch.complex64))
            prd_n = physical_to_nondim(prd, T, U)
            tar_n = physical_to_nondim(tar, T, U)
            channel_loss = loss_fn(
                solver, prd_n.unsqueeze(0), tar_n.unsqueeze(0),
                relative=True, squared=False, reduce_channels=False,
            )
            per_channel[t] = channel_loss.detach().cpu().numpy()
            combined[t] = float(channel_loss.mean())
    return combined, per_channel, true_interval_minutes


def aggregate_losses(pairs, lmax, grid, loss_type, T, U):
    """(combined_arr, channel_arr, true_interval_minutes): combined_arr is
    (n_traj, n_steps), channel_arr is (n_traj, n_steps, 3) - every
    trajectory's per_step_losses stacked together, truncated to the shortest
    trajectory's length (all trajectories share the same duration/
    save_interval_minutes, so this is normally a no-op)."""
    solver = ShallowWaterSolver(lmax=lmax, grid=grid, dealias=False, non_dimensional=False)
    solver.to(solver.device)
    loss_fn = LOSS_FUNCTIONS[loss_type]

    combined_list, channel_list, true_interval_minutes = [], [], None
    for _, neural_path, ref_path in tqdm.tqdm(pairs, desc="Computing per-trajectory spectral L2 loss"):
        combined, channel, tim = per_step_losses(neural_path, ref_path, solver, loss_fn, T, U)
        combined_list.append(combined)
        channel_list.append(channel)
        true_interval_minutes = tim

    n_steps = min(len(c) for c in combined_list)
    if any(len(c) != n_steps for c in combined_list):
        print(f"⚠️  Trajectory lengths differ across the evaluation set; comparing the first {n_steps} steps.")
    combined_arr = np.stack([c[:n_steps] for c in combined_list])
    channel_arr = np.stack([c[:n_steps] for c in channel_list])
    return combined_arr, channel_arr, true_interval_minutes


def plot_combined_loss(combined_arr, true_interval_minutes, output_path):
    hours = np.arange(combined_arr.shape[1]) * true_interval_minutes / 60.0
    mean, std = combined_arr.mean(axis=0), combined_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, mean, color="tab:blue", linewidth=2, label="mean")
    ax.plot(hours, mean + std, color="tab:blue", linestyle="dashed", linewidth=1, label="mean ± std")
    ax.plot(hours, mean - std, color="tab:blue", linestyle="dashed", linewidth=1)
    ax.fill_between(hours, mean - std, mean + std, color="tab:blue", alpha=0.2)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("relative spectral L2 loss")
    ax.set_title(f"SFNO rollout vs. numerical reference  ({combined_arr.shape[0]} trajectories)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"📈 Saved combined L2 spectral loss plot -> {output_path}")


def plot_channel_loss(channel_arr, true_interval_minutes, output_path):
    hours = np.arange(channel_arr.shape[1]) * true_interval_minutes / 60.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, name in enumerate(CHANNEL_NAMES):
        color = CHANNEL_COLORS[name]
        series = channel_arr[:, :, c]
        mean, std = series.mean(axis=0), series.std(axis=0)
        ax.plot(hours, mean, color=color, linewidth=2, label=f"{name} (mean)")
        ax.plot(hours, mean + std, color=color, linestyle="dashed", linewidth=1)
        ax.plot(hours, mean - std, color=color, linestyle="dashed", linewidth=1)
        ax.fill_between(hours, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("relative spectral L2 loss")
    ax.set_title(f"Per-channel SFNO rollout vs. numerical reference  ({channel_arr.shape[0]} trajectories)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"📈 Saved per-channel L2 spectral loss plot -> {output_path}")


def plot_sample_comparisons(pairs, sample_number_to_plot, output_dir, comparison_hours, seed=42):
    k = min(sample_number_to_plot, len(pairs))
    samples = random.Random(seed).sample(pairs, k)
    print(f"🎲 Rendering box/sphere comparisons for {k} randomly-selected trajectories.")
    for name, neural_path, ref_path in samples:
        stem = Path(name).stem
        neural_data = torch.load(neural_path, weights_only=False)
        ref_data = torch.load(ref_path, weights_only=False)
        plot_sphere_comparison(
            ref_data=ref_data, inf_data=neural_data, var="pv", hours=comparison_hours,
            output_path=str(output_dir / f"{stem}_sphere_comparison.png"),
            ref_label="Numerical (ground truth)", inf_label="SFNO (inference)",
        )
        plot_box_comparison(
            ref_data=ref_data, inf_data=neural_data, var="pv", hours=comparison_hours,
            output_path=str(output_dir / f"{stem}_box_comparison.png"),
            ref_label="Numerical (ground truth)", inf_label="SFNO (inference)",
        )
        # companion reference-numerical trajectory's own mean+-amplitude
        # diagnostics (geopotential/vorticity/divergence vs. lead time) - see
        # run_solver.py's original use of this same plot for a standalone
        # numerical run; here it's the reference half of each sampled
        # SFNO-vs-reference pair, not the SFNO rollout itself. Saved next to
        # ref_path itself (the model_output/ tree `make inference` wrote it
        # into), not under output_dir (task/batch_inference/...) - it's a
        # per-trajectory companion artifact, not an aggregate diagnostic.
        plot_trajectory_diagnostics(ref_path, Path(ref_path).parent)


def main():
    cfg = load_config()
    # galewsky has no evaluation dataset of times - `run()` writes it a single
    # fixed "model_output.pt" per checkpoint (see run_model.py's
    # _data_path_from_checkpoint), so it's just a SINGLETON evaluation set
    # (one trajectory pair) rather than the usual many real-world ones. Every
    # step below already degrades gracefully to n=1 (aggregate_losses'
    # mean/std over one trajectory, random.sample of one pair, ...) - only the
    # output-directory name needs a non-dataset fallback, since galewsky has
    # no dataset_name.
    dataset_label = cfg.dataset_name if cfg.ic == "real_world" else cfg.ic

    neural_dir, ref_dir, galewsky_file_name = resolve_trajectory_dirs(cfg)
    pairs = find_trajectory_pairs(neural_dir, ref_dir, only_file_name=galewsky_file_name)
    if not pairs:
        raise FileNotFoundError(
            f"No trajectory pairs found under {neural_dir} / {ref_dir}"
            + (f" (expected '{galewsky_file_name}')" if galewsky_file_name else "")
            + "; run `make inference` first."
        )

    model_dir, _, _ = resolve_checkpoint_dirs(cfg)
    model_info = load_model_info(str(model_dir))
    T, U, loss_type = model_info["T"], model_info["U"], model_info.get("loss_type", "spectral")

    print_in_box({
        "title": "Batch Inference Diagnostics",
        "lines": [
            f"evaluation set = {dataset_label} | {len(pairs)} trajectory pair(s) found"
            + ("" if len(pairs) > 1 else "  (singleton - mean/std bands degenerate to a single line)"),
            f"neural_dir = {neural_dir}",
            f"ref_dir    = {ref_dir}",
            f"sample_number_to_plot = {cfg.sample_number_to_plot}",
            f"comparison_steps = {cfg.comparison_steps}",
        ],
    })

    combined_arr, channel_arr, true_interval_minutes = aggregate_losses(
        pairs, lmax=cfg.nlat // 2, grid=cfg.grid, loss_type=loss_type, T=T, U=U,
    )

    output_dir = Path("task") / "batch_inference" / dataset_label
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_combined_loss(combined_arr, true_interval_minutes, output_dir / "l2_spectral_loss_combined.png")
    plot_channel_loss(channel_arr, true_interval_minutes, output_dir / "l2_spectral_loss_per_channel.png")

    # inference.comparison_steps is in saved-frame units (0 = the IC itself,
    # 1 = one save_interval_minutes later, ...) - converted to hours here since
    # that's what plot_sphere_comparison/plot_box_comparison take (they round
    # back to a frame index internally via each trajectory's own sec_per_frame).
    # round(), not floor division: true_interval_minutes is the ACHIEVED
    # cadence (see compute_step_per_save), which only approximates the
    # requested save_interval_minutes - e.g. 358.36 min for a nominal 6h
    # (360 min) request. Flooring truncated every label down a full hour
    # (step=1 -> 5h instead of 6h, step=5 -> 29h instead of 30h) even though
    # round-tripping through round() (as the plotting functions already do
    # to get back a frame index) recovers the exact same frame either way -
    # only the printed label was wrong.
    comparison_hours = [round(step * true_interval_minutes / 60) for step in cfg.comparison_steps]
    plot_sample_comparisons(pairs, cfg.sample_number_to_plot, output_dir, comparison_hours)

    print(f"✅ Batch inference diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
