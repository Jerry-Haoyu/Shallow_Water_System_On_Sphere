"""
Inference for subtask 2 (debug_train_8_26/stage0.md): a model trained
directly on ERA5 (src/entries/build_era5_trajectory_dataset.py output), not on
PS-solver output. `src/entries/inference.py` can't be reused as-is for this:
its `numerical_model_info_from_neural_opeartor_model_info` recovers a
reference *numerical solver* config by regex-matching the PS-solver's own
`tau_(...)_method_..._radiation_...` path nodes out of the neural model's
recorded `training_data` path - nodes that don't exist for an ERA5-direct
model, and conceptually there's no "rerun the numerical solver" reference for
it anyway. The natural ground truth here is the real ERA5 trajectory itself.

Procedure: take one already-converted trajectory chunk (see
build_era5_trajectory_dataset.py) as ground truth, feed its first frame to the
trained model as the initial condition, roll the model forward autoregressively
for that chunk's duration, and compare against the same chunk's real frames
(subsampled to the model's own n_future*frame_interval cadence, since the
model can only step forward in n_future-frame increments). Reuses
inference.py's plot_per_step_loss / plot_sphere_comparison unmodified.

Configuration is read from the ``inference_era5_direct`` section of a YAML
config (pass the path as CLI arg 1).

@author Haoyu Tang hytang2@illinois.edu
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from types import SimpleNamespace

import torch
import yaml

from src.entries.inference import plot_per_step_loss, plot_sphere_comparison
from src.helpers.print import print_in_box
from src.helpers.run_model import load_model_info, neural_model_path, physical_to_nondim, run

DEFAULT_CONFIG = {
    "resol": [128, 256],
    "n_future": 6,
    "num_layers": 4,
    "embed_dim": 128,
    "pos_embed": "none",
    "trainData": "era5_direct_2023_06_500",
    "grid": "equiangular",
    "normalization_layer": "layer_norm",
    "loss_type": "grid",
    "index": 0,
    "dataset_name": "era5_direct_2023_06_500",
    "pressure": "500",
    # which already-converted chunk (build_era5_trajectory_dataset.py output)
    # to use as ground truth/IC - the file stem under
    # model_output/neural_direct/era5/dataset_<name>/pressure_<pressure>/
    "ref_chunk_stem": "2023_06_28_00_to_2023_06_30_23",
    "single_step": True,
}


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    raw = DEFAULT_CONFIG | config.get("inference_era5_direct", {})
    return SimpleNamespace(**raw)


def main():
    cfg = load_config()

    model_kwargs = dict(
        resol=tuple(cfg.resol), n_future=cfg.n_future, num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim, pos_embed=cfg.pos_embed, trainData_path=cfg.trainData,
        grid=cfg.grid, normalization_layer=cfg.normalization_layer,
        loss_type=cfg.loss_type, index=cfg.index,
    )
    model_dir, single_name, multi_name, info_name = neural_model_path(**model_kwargs)
    ckpt_name = single_name if cfg.single_step else multi_name
    if not (Path(model_dir) / ckpt_name).is_file():
        raise FileNotFoundError(
            f"No trained model found at {model_dir} (expected {ckpt_name}). "
            f"Train it first (debug_train_8_26/config_train_era5_direct.yml).")

    model_info = load_model_info(model_dir)
    T, U = model_info["T"], model_info["U"]
    train_interval_minutes = model_info["true_interval_minutes"]
    n_future = model_info["n_future"]
    step_minutes = train_interval_minutes * n_future  # finest legal rollout cadence

    ref_dir = (Path("model_output") / "neural_direct" / "era5"
               / f"dataset_{cfg.dataset_name}" / f"pressure_{cfg.pressure}")
    ref_path = ref_dir / f"{cfg.ref_chunk_stem}.pt"
    ref_data = torch.load(ref_path, weights_only=False)
    ref_trajectory = ref_data["trajectory"]  # (n_frames, 3, lmax, mmax), physical units, hourly
    ic_time = ref_data["metadata"]["start_time"]

    n_frames_ref = ref_trajectory.shape[0]
    n_steps = (n_frames_ref - 1) // n_future  # whole model-steps that fit in this chunk
    duration_days = (n_steps * step_minutes) / 1440.0

    print_in_box({
        "title": "ERA5-direct Inference (subtask 2)",
        "lines": [
            f"model = {model_dir}",
            f"ground truth chunk = {ref_path} ({n_frames_ref} hourly frames)",
            f"rollout cadence = {step_minutes} min ({n_future} model steps of "
            f"{train_interval_minutes} min each) -> {n_steps} steps, {duration_days:.2f} days",
        ],
    })

    initial_condition = physical_to_nondim(ref_trajectory[0].clone(), T, U)

    neural_save_path = run(
        model_checkpoint=str(model_dir),
        initial_condition=initial_condition,
        duration=duration_days,
        ic="real_world",
        pressure=cfg.pressure,
        ic_time=str(ic_time),
        dataset_name=cfg.dataset_name,
        save_interval_minutes=step_minutes,
        single_step=cfg.single_step,
    )

    data_dir = Path(neural_save_path).parent
    data_file = Path(neural_save_path).name

    # subsample the real trajectory to the model's own cadence so
    # plot_per_step_loss compares matching timestamps (frame i <-> real hour
    # i*n_future). Written under data_dir (the rollout's own output tree,
    # rooted at model_output/.../checkpoints-mirror/ - see
    # _data_path_from_checkpoint in run_model.py), NOT ref_dir: ref_dir is
    # SWEDataset's training data directory (dataset.py globs every *.pt file
    # there), so writing a derived file into it would get picked up as a bogus
    # "trajectory" on the next training run and crash (too few frames for the
    # configured n_future/warmup window).
    ref_subsampled = ref_trajectory[0: n_steps * n_future + 1: n_future]
    ref_sub_path = data_dir / f"{cfg.ref_chunk_stem}_subsampled_every{n_future}h.pt"
    torch.save({
        "metadata": {**ref_data["metadata"], "true_interval_minutes": step_minutes},
        "trajectory": ref_subsampled,
    }, ref_sub_path)

    plot_per_step_loss(
        neural_traj_path=Path(neural_save_path),
        ref_traj_path=ref_sub_path,
        info_path=Path(model_dir) / info_name,
        lmax=cfg.resol[0] // 2,
        grid=cfg.grid,
        save_interval_minutes=step_minutes,
        output_path=data_dir / f"{Path(data_file).stem}_l2_spectral_loss.png",
    )

    neural_data = torch.load(neural_save_path, weights_only=False)
    ref_sub_data = torch.load(ref_sub_path, weights_only=False)
    plot_sphere_comparison(
        ref_data=ref_sub_data,
        inf_data=neural_data,
        var="pv",
        hours=[0, step_minutes / 60.0, 2 * step_minutes / 60.0, min(24.0, duration_days * 24)],
        output_path=str(data_dir / f"{Path(data_file).stem}_sphere_comparison.png"),
        ref_label="ERA5 (ground truth)",
        inf_label="SFNO (inference)",
    )
    print(f"Wrote {neural_save_path}")


if __name__ == "__main__":
    main()
