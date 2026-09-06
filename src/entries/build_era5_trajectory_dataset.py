"""
Convert a CONTINUOUS ERA5 (u,v) netCDF (see download_era5.py, run with
time_option="all" over a contiguous date span - subtask 2 of
debug_train_8_26/stage0.md) into chunked spectral (Phi,zeta,delta) trajectory
.pt files matching SWEDataset's expected format. Each frame is converted with
the same per-frame (u,v) -> (Phi,zeta,delta) balance-equation diagnosis
src/numerical_solver/initial_condition.py's rw_initial_condition already does
for a single initial condition - just applied across every hourly frame of a
continuous real trajectory instead of one IC.

Output is written to <output_root>/dataset_<name>/pressure_<p>/, which
satisfies parse_dataset_and_pressure's dataset_*/pressure_* path convention
(see src/helpers/run_model.py), so SWEDataset can be pointed at that directory
directly - no other code changes needed to train on it.

Configuration is read from the ``build_era5_trajectory`` section of a YAML
config (pass the path as CLI arg 1); see
debug_train_8_26/config_era5_direct_download.yml for the matching download step.

@author Haoyu Tang hytang2@illinois.edu
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

import time
from types import SimpleNamespace

import torch
import tqdm
import xarray as xr
import yaml
from torch_harmonics import RealVectorSHT

from src.helpers.print import print_in_box
from src.helpers.run_model import load_h_stats
from src.numerical_solver.initial_condition import rw_initial_condition
from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver

DEFAULT_CONFIG = {
    "dataset_name": None,   # required: reanalysis_data/<dataset_name>/data.nc (from download_era5.py)
    "pressure": None,       # required, for naming (e.g. "500")
    "lmax": 64,
    "grid": "equiangular",
    "chunk_hours": 240,             # frames per output .pt file
    "frame_interval_minutes": 60,   # ERA5's native hourly cadence
    "output_root": "model_output/neural_direct/era5",
}


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    raw = DEFAULT_CONFIG | config.get("build_era5_trajectory", {})
    cfg = SimpleNamespace(**raw)
    if not cfg.dataset_name:
        raise ValueError("build_era5_trajectory.dataset_name is required.")
    if cfg.pressure is None:
        raise ValueError("build_era5_trajectory.pressure is required.")
    return cfg


def main():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("Device is now CPU !")

    nc_path = Path("reanalysis_data") / cfg.dataset_name / "data.nc"

    era5_dataset = xr.open_dataset(nc_path).load()
    times = era5_dataset['valid_time'].values
    n_frames_total = len(times)
    nlat_data, nlon_data = era5_dataset.sizes['latitude'], era5_dataset.sizes['longitude']

    n_chunks = n_frames_total // cfg.chunk_hours
    if n_chunks == 0:
        raise ValueError(
            f"Only {n_frames_total} frames available in {nc_path}, fewer than "
            f"chunk_hours={cfg.chunk_hours}; download more data or lower chunk_hours.")

    print_in_box({
        "title": "Build ERA5-direct Trajectory Dataset (subtask 2)",
        "lines": [
            f"source = {nc_path} ({n_frames_total} frames, {times[0]} .. {times[-1]})",
            f"native grid = ({nlat_data}, {nlon_data})",
            f"lmax = {cfg.lmax} | grid = {cfg.grid}",
            f"chunk_hours = {cfg.chunk_hours} -> {n_chunks} trajectory file(s) "
            f"({n_frames_total - n_chunks * cfg.chunk_hours} leftover frame(s) dropped)",
            f"frame_interval_minutes = {cfg.frame_interval_minutes}",
        ],
    })

    h_avg, h_amp = load_h_stats(cfg.dataset_name)
    solver = ShallowWaterSolver(
        lmax=cfg.lmax, grid=cfg.grid, dealias=False,
        h_avg=h_avg, h_amp=h_amp, non_dimensional=False,
    )
    solver.to(device)

    vSHT = RealVectorSHT(
        nlat=nlat_data, nlon=nlon_data, lmax=nlat_data // 2, mmax=nlat_data // 2,
        grid='equiangular', csphase=False,
    ).to(device)

    output_dir = Path(cfg.output_root) / f"dataset_{cfg.dataset_name}" / f"pressure_{cfg.pressure}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for c in range(n_chunks):
        chunk_start_time = time.perf_counter()
        frame_idxs = list(range(c * cfg.chunk_hours, (c + 1) * cfg.chunk_hours))

        frames = []
        with torch.no_grad():
            for i in tqdm.tqdm(frame_idxs, desc=f"chunk {c}/{n_chunks - 1}"):
                phivrtdiv_spec = rw_initial_condition(
                    model=solver, vSHT=vSHT, era5_dataset=era5_dataset,
                    ic_time=times[i], balanced=True,
                )
                frames.append(phivrtdiv_spec.cpu())
        trajectory = torch.stack(frames, dim=0)

        t0, t1 = times[frame_idxs[0]], times[frame_idxs[-1]]
        stem = f"{str(t0)[:13].replace('-', '_').replace('T', '_')}_to_{str(t1)[:13].replace('-', '_').replace('T', '_')}"
        save_path = output_dir / f"{stem}.pt"
        torch.save({
            'metadata': {
                'nlat': solver.nlat, 'nlon': solver.nlon,
                'lmax': solver.lmax, 'mmax': solver.mmax,
                'grid': solver.grid,
                # ERA5-direct frames are literally on the native hourly grid -
                # no substep quantization happens here (unlike run_model.py's
                # numerical branch), so the achieved cadence exactly equals the
                # requested one; still recorded under the same 'true_interval_minutes'
                # key every downstream consumer (SWEDataset, run_model.run()) reads.
                'true_interval_minutes': cfg.frame_interval_minutes,
                'source': 'era5_direct',
                'start_time': str(t0), 'end_time': str(t1),
            },
            'trajectory': trajectory,
        }, save_path)
        print(f"  wrote {save_path}  shape={tuple(trajectory.shape)}  "
              f"({time.perf_counter() - chunk_start_time:.1f}s)")

    print(f"Finished. {n_chunks} trajectory file(s) written to {output_dir}")


if __name__ == "__main__":
    main()
