"""
Anomaly Correlation Coefficient (ACC) diagnostics comparing model trajectories
against ERA5 ground truth.

@author Haoyu Tang hytang2@illinois.edu and Claude
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

import itertools

import numpy as np
import pandas as pd
import torch
import tqdm
import os
import xarray as xr
from torch_harmonics.sht import RealVectorSHT
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from src.helpers.run_model import (
    run, is_neural_checkpoint, load_model_info, load_numerical_checkpoint,
    numerical_checkpoint_path, save_numerical_checkpoint,
)
from src.numerical_solver.initial_condition import (
    rw_initial_condition, radiative_equilibrium_geopotential, day_of_year_climatology,
)
from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver

# Fixed for every ACC trajectory: daily sampling (matches sample_days'
# granularity), the single-step model variant, and the phi_eq spectral-smoothing
# fraction for rad=True checkpoints (see initial_condition.radiative_equilibrium_geopotential,
# matches config.yml's rad_smooth_fraction default). These used to be caller-
# supplied kwargs (with these exact defaults); compute_acc_over_a_set's
# signature is now strictly 5 params, so they're constants instead.
_ACC_SAVE_INTERVAL_MINUTES = 1440
_ACC_SINGLE_STEP = True
_RAD_SMOOTH_FRACTION = 0.5


def compute_acc_single(pred, actual, climatology, duration: float, weights=None, log=False):
    """
        Compute Anamoly Correlation Coefficient(latitude weighted)

        "Anomaly" is the deviation from `climatology` - the standard NWP
        definition - rather than each field's own spatial mean, now that
        compute_acc_over_a_set can build a real day-of-year climatology from
        a multi-year ERA5 dataset (see its groupby('valid_time.dayofyear')
        step). ACC then reduces to a latitude-weighted spatial pattern
        correlation between the forecast and truth anomalies.

        Args:
            pred, actual, climatology: (nlat, nlon) real-valued grid fields
                (e.g. height) on the same equiangular lat/lon grid - pred/
                actual at the same valid time, climatology the day-of-year
                reference both are compared against.
            duration: lead time (days) this pair corresponds to; only used
                for the optional log message, not the computation itself.
            weights: optional (nlat,) or (nlat, 1) latitude quadrature weight
                (e.g. a solver's `quad_weights`); defaults to a cos(latitude)
                area weight built from `pred`'s own shape (equiangular grid).
            log: print the resulting ACC value if True.

        Returns:
            float: the ACC value, in [-1, 1].
    """
    pred = torch.as_tensor(pred, dtype=torch.float64)
    actual = torch.as_tensor(actual, dtype=torch.float64)
    climatology = torch.as_tensor(climatology, dtype=torch.float64)
    if not (pred.shape == actual.shape == climatology.shape):
        raise ValueError(
            f"pred/actual/climatology shape mismatch: {tuple(pred.shape)}, "
            f"{tuple(actual.shape)}, {tuple(climatology.shape)}"
        )

    nlat = pred.shape[-2]
    if weights is None:
        lats = torch.linspace(torch.pi / 2, -torch.pi / 2, nlat + 2)[1:-1]
        w = torch.cos(lats)
    else:
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    w = (w / w.sum()).reshape(nlat, 1)

    pred_anom = pred - climatology
    actual_anom = actual - climatology

    numerator = (w * pred_anom * actual_anom).sum()
    denominator = torch.sqrt((w * pred_anom ** 2).sum() * (w * actual_anom ** 2).sum())
    acc = (numerator / denominator).item()

    if log:
        print(f"    ACC at duration={duration:g} day(s): {acc:.4f}")
    return acc


FIELD_NAMES = ("height", "pv", "divergence")


def _extract_fields(solver, uspec):
    """Physical (height, potential-vorticity, divergence) grid fields on
    `solver`'s grid, all derived from the same spectral state `uspec`
    (channels: phi, vrt, div - see ShallowWaterSolver.dudtspec). Trajectories
    produced by src.helpers.run_model.run() are always saved in physical
    units, so no rescaling is needed here regardless of whether the run that
    produced them integrated non-dimensionally internally.

    Returns {"height": ..., "pv": ..., "divergence": ...}, one (nlat, nlon)
    real-valued grid field per FIELD_NAMES entry.
    """
    uspec = uspec.to(solver.device)
    ugrid = solver.spec2grid(uspec)
    return {
        "height": ugrid[0] / solver.gravity,
        "pv": solver.potential_vorticity(uspec),
        "divergence": ugrid[2],
    }


def _resolve_solvers(model_checkpoint):
    """Build the solver used to generate each trajectory's initial condition
    (in `model_checkpoint`'s own units) and a physical (non_dimensional=False)
    twin at the same resolution/grid used for every grid-space extraction
    (climatology, truth, prediction), so they stay unit-consistent regardless
    of whether the model itself runs non-dimensionally.

    This dispatch (numerical vs. neural) is orthogonal to - and unrelated to -
    the one src.helpers.run_model.run() makes internally: run() decides where
    a trajectory lives and how to produce it; this only decides what solver
    ACC itself needs for balance-equation IC/climatology/truth computation,
    which is a different task with no way around needing the model's own
    solver either way. It reuses run_model.is_neural_checkpoint rather than
    a local check so the two stay in agreement.

    h_avg/h_amp/dataset_name/pressure all now live in model_checkpoint's own
    model_info.json (both numerical and neural checkpoints are self-describing
    directories - see run_model.save_numerical_checkpoint /
    train_singlestep.py) rather than being supplied by the caller. T/U aren't
    needed here at all anymore either - run_model.run() reads them back off
    each checkpoint's own model_info.json itself now.

    Returns (ic_solver, eval_solver, dataset_name, pressure).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_info = load_model_info(model_checkpoint)
    h_avg, h_amp = model_info["h_avg"], model_info["h_amp"]
    dataset_name, pressure = model_info.get("dataset_name"), model_info.get("pressure")

    if is_neural_checkpoint(model_checkpoint):
        ic_solver = ShallowWaterSolver(
            lmax=model_info["nlat"] // 2, grid=model_info.get("grid", "equiangular"),
            dealias=False, h_avg=h_avg, h_amp=h_amp, non_dimensional=True,
        )
    else:
        ic_solver, _ = load_numerical_checkpoint(model_checkpoint)
    ic_solver.to(device)

    eval_solver = ShallowWaterSolver(
        lmax=ic_solver.lmax, grid=ic_solver.grid, dealias=False, non_dimensional=False,
    )
    eval_solver.to(device)
    return ic_solver, eval_solver, dataset_name, pressure


def _climatology_fields(eval_solver, vSHT, clim_ds, doy):
    """Physical climatological (height, pv, divergence) fields for
    day-of-year `doy`, balanced from `clim_ds`'s (u, v) climatology
    (compute_acc_over_a_set's
    ``era5_dataset.groupby('valid_time.dayofyear').mean('valid_time')``).

    The single day-of-year slice is wrapped in a synthetic one-point
    'valid_time' axis so it can be fed through the same balance-equation code
    (rw_initial_condition) used for real ERA5 snapshots, rather than
    duplicating that logic here.
    """
    day_slice = clim_ds.sel(dayofyear=int(doy), method="nearest")
    fake_time = np.datetime64("2001-01-01") + np.timedelta64(int(doy) - 1, "D")
    day_ds = day_slice.expand_dims(valid_time=[fake_time])
    with torch.no_grad():
        uspec = rw_initial_condition(eval_solver, vSHT, day_ds, str(fake_time), balanced=True)
        return _extract_fields(eval_solver, uspec)


def compute_acc_over_a_set(model, era5_dataset: xr.Dataset, trajectory_set, start_day: int,
                            sample_days: list[int]
                            ) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    """
        Compute Anamoly Correlation Coefficient(latitude weighted) over a set of trajectories,
        simulated/inferred on the fly with `model` via the generic
        src.helpers.run_model.run() interface.

        Each trajectory is generated (or reused, if already there) at its
        usual place in the shared model_output/ cache (README.md's naming
        convention), exactly like run_solver.py/inference.py - there is no
        separate/ephemeral location for ACC trajectories, so re-running this
        over an overlapping trajectory_set or sample_days reuses whatever was
        already computed instead of redoing the simulation, and results stay
        around afterward for anyone else (e.g. training data collection) to
        reuse too. That reuse is keyed by `model`'s own training dataset (its
        `dataset_name`/`pressure`, read from model_info.json), NOT by
        `era5_dataset` - see that arg's own note below.

        Args:
            model: either a numerical model or machine learning model checkpoint
                directory (see run_model.numerical_checkpoint_path /
                neural_model_path) - self-describing via its own
                model_info.json, which supplies everything else this function
                used to need as separate arguments: h_avg/h_amp (dataset-wide
                reference height/amplitude, m), and dataset_name/pressure for
                naming the trajectories this generates in the shared
                model_output/ tree (see run_model.save_numerical_checkpoint,
                train_singlestep.py).

            era5_dataset: a loaded dataset (via xr.open_dataset().load()); this is a multi-year
                dataset that allows us to compute climatology. It serves 4 purposes:
                a) To compute the climatology via .groupby('valid_time.dayofyear').mean('valid_time')
                b) As initial condition to produce prediction using the model
                c) As ground truth, sampled at sample_days
                d) If `model` is a numerical checkpoint with rad=True, as the source of
                   each trajectory's phi_eq_spec (radiative_equilibrium_geopotential,
                   cached per day-of-year) - required since run() reconstructs its own
                   solver from `model` and can't otherwise recover the equilibrium
                   background it needs to relax toward.
                It need not be the same dataset `model` was built/trained against (e.g. a
                different period), so it is never the source of `dataset_name`/`pressure`
                for naming this function's own output trajectories - only `model`'s own
                model_info.json is.

            trajectory_set: The set of trajectories we are going to average, defined by the
                years and months, e.g. ([2020, 2021, 2022, 2023, 2024, 2025], [1, 7, 11]) - the
                actual set of trajectories is the cartesian product, the above example would have
                5*3=15 trajectories.

            start_day: The day of month to start simulation/inference, e.g. 1

            sample_days: A list of day (of month) to compute acc on, e.g. [1, 5, 10, 15] would
                compute acc at these days. Also this gives the duration required, through
                sample_days[-1]-start_day+1, e.g. the above example would have a duration of
                15-1+1=15 days.

        Returns:
            (height_mean, height_std, pv_mean, pv_std, divergence_mean, divergence_std):
                six list[float], one entry per sample_days element - the mean
                and (sample, ddof=0) standard deviation across trajectories of
                the ACC at each sample day, for each of FIELD_NAMES
                ("height", "pv", "divergence") in turn. All three are read off
                the same predicted/truth/climatology spectral state per
                sample day (see _extract_fields), so they cost nothing extra
                in terms of trajectories generated - only the ACC reduction
                itself is repeated per field.
    """
    years, months = trajectory_set
    nlat_data, nlon_data = era5_dataset.sizes["latitude"], era5_dataset.sizes["longitude"]

    ic_solver, eval_solver, dataset_name, pressure = _resolve_solvers(model)
    vSHT = RealVectorSHT(nlat=nlat_data, nlon=nlon_data, lmax=nlat_data // 2, mmax=nlat_data // 2,
                          grid="equiangular", csphase=False).to(eval_solver.device)
    weights = eval_solver.quad_weights.cpu()

    # (a) day-of-year climatology, computed once from the whole multi-year dataset,
    # then balanced into (height, pv, divergence) fields lazily (and cached)
    # per (month, day).
    clim_ds = day_of_year_climatology(era5_dataset)
    climatology_cache = {}

    def climatology_for(month, day):
        key = (month, day)
        if key not in climatology_cache:
            doy = pd.Timestamp(2001, month, day).dayofyear  # arbitrary non-leap reference year
            climatology_cache[key] = _climatology_fields(eval_solver, vSHT, clim_ds, doy)
        return climatology_cache[key]

    # (b) if model's own checkpoint has rad=True, run() needs a phi_eq_spec for
    # every trajectory it generates (it reconstructs its own solver from `model`
    # internally and can't otherwise recover the equilibrium background - see
    # run_model.run()'s phi_eq_spec docstring). Reuses the same clim_ds as (a)
    # rather than repeating the expensive groupby per trajectory; cached by
    # day-of-year since every trajectory in a given month shares one ic_time day.
    phi_eq_cache = {}

    def phi_eq_for(ic_time):
        doy = pd.Timestamp(ic_time).dayofyear
        if doy not in phi_eq_cache:
            phi_eq_cache[doy] = radiative_equilibrium_geopotential(
                ic_solver, vSHT, clim_ds, ic_time,
                smooth_fraction=_RAD_SMOOTH_FRACTION, log=False,
            )
        return phi_eq_cache[doy]

    run_duration = sample_days[-1] - start_day + 1
    trajectories = list(itertools.product(years, months))
    acc_values = np.empty((len(trajectories), len(sample_days), len(FIELD_NAMES)), dtype=np.float64)

    for traj_idx, (year, month) in enumerate(tqdm.tqdm(trajectories, desc="Computing ACC over trajectories")):
        ic_time = f"{year:04d}-{month:02d}-{start_day:02d}T00:00:00"
        with torch.no_grad():
            phivrtdivspec_0 = rw_initial_condition(ic_solver, vSHT, era5_dataset, ic_time, balanced=True)

        save_path = run(
            model_checkpoint=model,
            initial_condition=phivrtdivspec_0.clone(),
            duration=run_duration,
            ic="real_world",
            pressure=pressure,
            ic_time=ic_time,
            dataset_name=dataset_name,
            save_interval_minutes=_ACC_SAVE_INTERVAL_MINUTES,
            single_step=_ACC_SINGLE_STEP,
            phi_eq_spec=phi_eq_for(ic_time) if ic_solver.rad else None,
        )
        saved = torch.load(save_path, map_location="cpu", weights_only=False)
        trajectory = saved["trajectory"]
        n_frames = trajectory.shape[0]

        with torch.no_grad():
            for i, day in enumerate(sample_days):
                frame_idx = round((day - start_day) * 1440.0 / _ACC_SAVE_INTERVAL_MINUTES)
                if not (0 <= frame_idx < n_frames):
                    raise ValueError(
                        f"sample_day={day} (frame {frame_idx}) is out of range for "
                        f"trajectory '{save_path}' with {n_frames} frame(s)."
                    )
                pred_fields = _extract_fields(eval_solver, trajectory[frame_idx])

                verification_time = f"{year:04d}-{month:02d}-{day:02d}T00:00:00"
                truth_uspec = rw_initial_condition(eval_solver, vSHT, era5_dataset, verification_time, balanced=True)
                truth_fields = _extract_fields(eval_solver, truth_uspec)

                clim_fields = climatology_for(month, day)

                for f_idx, field in enumerate(FIELD_NAMES):
                    acc_values[traj_idx, i, f_idx] = compute_acc_single(
                        pred_fields[field].cpu().numpy(), truth_fields[field].cpu().numpy(),
                        clim_fields[field].cpu().numpy(),
                        duration=day - start_day, weights=weights,
                    )

    acc_mean = acc_values.mean(axis=0)  # (n_sample_days, n_fields)
    acc_std = acc_values.std(axis=0)
    height_mean, pv_mean, divergence_mean = acc_mean.T.tolist()
    height_std, pv_std, divergence_std = acc_std.T.tolist()
    return height_mean, height_std, pv_mean, pv_std, divergence_mean, divergence_std

def compute_accs(tau4s,
                tau8s,
                tau_rads,
                resolution,
                era5_dataset_name,
                trajectory_set,
                start_day,
                sample_days,
                task_dir="acc"
                ):
    """
    Grid-search either (tau4, tau8) hyperdiffusion pairs (rad=False throughout, when
    tau_rads is None/empty) or tau_rad relaxation values (tau4/tau8 held fixed at
    their first entry, when tau_rads is given - entries may be None to include a
    rad=False baseline alongside the numeric tau_rad values), computing ACC over
    trajectory_set for each config.
    """
    era5_dataset = xr.open_dataset(f"reanalysis_data/{era5_dataset_name}/data.nc").load()
    h_avg, h_amp = np.load(f"reanalysis_data/{era5_dataset_name}/h_stats.npz")['mean']

    def _build_and_run(tau4, tau8, tau_rad):
        """Build (or reuse) the checkpoint for this (tau4, tau8, tau_rad) config and
        compute ACC over trajectory_set with it. tau_rad=None means rad=False.

        The checkpoint-path dispatcher (numerical_checkpoint_path) is keyed on `rad`
        (its radiation_rad/radiation_no_rad tree node), so rad=True and rad=False
        configs always land under different checkpoint/model_output directories and
        never collide/overwrite each other, even when swept alongside the same
        (tau4, tau8).
        """
        print(f" Considering config (tau_4 = {tau4}, tau_8 = {tau8}, tau_rad = {tau_rad})")
        rad = tau_rad is not None
        solver = ShallowWaterSolver(lmax=resolution,
                        tau=(30000, tau4, tau8),
                        h_avg=h_avg,
                        h_amp=h_amp,
                        rad=rad,
                        tau_rad=tau_rad)
        solver.to(solver.device)

        ckpt_dir = numerical_checkpoint_path(
            lmax=resolution, tau=(30000, tau4, tau8), grid=solver.grid,
            semi_implicit=solver.semi_implicit, rad=rad, dataset_name=era5_dataset_name,
        )
        save_numerical_checkpoint(
            solver, ckpt_dir, dataset_name=era5_dataset_name,
            pressure=500  # hard coded to 500hPa
        )
        return compute_acc_over_a_set(ckpt_dir, era5_dataset, trajectory_set, start_day, sample_days)

    if tau_rads is None or len(tau_rads) == 0:
        print(" Grid searching damping coefficients tau4, tau8")
        results = np.empty(shape=(len(tau4s), len(tau8s), 6, len(sample_days)))  # X,Y,6,T
        for i, tau4 in enumerate(tau4s):
            for j, tau8 in enumerate(tau8s):
                results[i][j] = _build_and_run(tau4, tau8, None)
    else:
        print(" Grid searching radiation coefficients tau_rad")
        # sweeps tau_rad alone, holding tau4/tau8 fixed at their first entry; kept 4D
        # (singleton 2nd axis) so plot_acc_time_series's fixed-index transpose stays valid.
        tau4, tau8 = tau4s[0], tau8s[0]
        results = np.empty(shape=(len(tau_rads), 1, 6, len(sample_days)))
        for k, tau_rad in enumerate(tau_rads):
            results[k, 0] = _build_and_run(tau4, tau8, tau_rad)

    os.makedirs(name=task_dir, exist_ok=True)
    np.save(file=task_dir+"/accs.npy", arr=results)
    if tau_rads is None or len(tau_rads) == 0:
        plot_acc_for_different_taus(task_dir=task_dir, sample_days=sample_days, vars=['pv', 'h'])
    plot_acc_time_series(task_dir=task_dir, sample_days=sample_days)

def plot_acc_for_different_taus(task_dir, sample_days, vars:list[str]):
    results = np.load(file=f"{task_dir}/accs.npy")
    
    results = np.transpose(results, [3,2,0,1])
    
    ncols = 3
    nrows = int(np.ceil(len(sample_days)/ncols))
    for var in vars:
        if var == 'h':
            var_idx = 0
        elif var == 'pv':
            var_idx = 2
        else:
            raise ValueError(" no such variable")

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw={'projection':'3d'}, figsize=(20,20))
        fig.suptitle(rf"ACC of {var} at different $(\tau_4, \tau_8)$", fontsize=40)
        X, Y = np.meshgrid(np.arange(len(tau4s)),np.arange(len(tau8s)), indexing='ij')
        fig.subplots_adjust(left=0.00, right=0.99, top=0.90, bottom=0.02,
                        wspace=-0.25, hspace=0.1)
        
        tau4_labels = [f"{str(int(t))}" if t < 10000 else rf"$\infty$" for t in tau4s]
        tau8_labels = [f"{str(int(t))}" if t < 10000 else rf"$\infty$" for t in tau8s]
        for t in range(len(sample_days)):
            row = t//ncols
            col = t-row*ncols

            idx = col if nrows<= 1 else (row,col)
            ax[idx].set_zlim(0.0, 1.0)
            ax[idx].set_box_aspect((1, 1, 2)) 
            ax[idx].view_init(elev=35,azim=70)
            ax[idx].plot_surface(X, Y, results[t][var_idx], cmap='coolwarm')
            ax[idx].set_xlabel(r"$\tau_4$ (days)", fontsize=15)
            ax[idx].set_ylabel(r"$\tau_8$ (days)", fontsize=15)
            ax[idx].set_zlabel(r"ACC")
            ax[idx].set_title(f"Day {sample_days[t]}", fontsize=22)
            ax[idx].set_xticks(ticks=np.arange(len(tau4s)), labels=tau4_labels, fontsize=12)
            ax[idx].set_yticks(ticks=np.arange(len(tau8s)), labels=tau8_labels, fontsize=12)

            fig.savefig(task_dir+f"/acc_plot_{var}.png")
            

def plot_acc_time_series_comparison(baseline_dir, task_dir, sample_days):
    """Same time-series ACC plot as plot_acc_time_series, but overlaid with a
    baseline run's curves (e.g. a rad=False sweep) as dashed lines for direct
    comparison against task_dir's own (solid) curves - useful for judging whether
    a tau_rad sweep in task_dir actually improves on the rad=False baseline.
    Saved to task_dir/acc_timeseries_comp.png (baseline_dir is read-only).
    """
    def _load_and_collapse(directory):
        results = np.load(file=f"{directory}/accs.npy")
        results = np.transpose(results, [3, 2, 0, 1])  # (T, 6, dim0, dim1)
        return np.max(results, axis=(2, 3))  # (T, 6)

    baseline = _load_and_collapse(baseline_dir)
    results = _load_and_collapse(task_dir)

    fig, ax = plt.subplots()
    ax.set_title(" ACC timeseries (vs. baseline)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Anamoly Correlation Coefficient")
    ax.plot(np.linspace(1, sample_days[-1], 100), np.ones(100) * 0.6, linestyle='dashed', color='grey')

    # (mean_idx, std_idx, color, label) per field, matching compute_acc_over_a_set's
    # (height_mean, height_std, pv_mean, pv_std, divergence_mean, divergence_std) order.
    field_specs = [
        (0, 1, 'b', r"$h$(height)"),
        (2, 3, 'r', r"$q$(pv)"),
        (4, 5, 'g', r"$\delta$(div)"),
    ]
    for mean_idx, std_idx, color, label in field_specs:
        # current run: solid line + std band
        ax.plot(sample_days, results[:, mean_idx], color=color, label=label)
        ax.fill_between(sample_days, results[:, mean_idx] - results[:, std_idx],
                         results[:, mean_idx] + results[:, std_idx], alpha=0.3, color=color)
        # baseline: dashed line only (no band, to keep the overlay readable)
        ax.plot(sample_days, baseline[:, mean_idx], color=color, linestyle='dashed',
                 label=f"{label} (baseline)")

    ax.legend(loc='upper right')

    fig.savefig(task_dir + "/acc_timeseries_comp.png")

def plot_acc_time_series(task_dir, sample_days):
    results = np.load(file=f"{task_dir}/accs.npy")
    
    results = np.transpose(results, [3,2,0,1])
    if len(results.shape) > 2:
        if len(results.shape) > 3:
            results = np.max(results, axis=(2,3))
        else:
            results = np.max(results, axis=(2))
            
    fig, ax = plt.subplots()
    ax.set_title(" ACC timeseires")
    ax.set_xlabel("Days")
    ax.set_ylabel("Anamoly Correlation Coefficient")
    ax.plot(np.linspace(1, sample_days[-1], 100), np.ones(100) * 0.6, linestyle='dashed', color='grey')
    # plot pv
    ax.plot(sample_days, results[:, 0], color='b', label=r"$h$(height)")
    ax.fill_between(sample_days, results[:, 0]-results[:, 1], results[:, 0]+results[:, 1], alpha=0.3)
    # plot h 
    ax.plot(sample_days, results[:, 2], color='r',label=r"$q$(pv)")
    ax.fill_between(sample_days, results[:, 2]-results[:, 3], results[:, 2]+results[:, 3], alpha=0.3)
    # plot div
    ax.plot(sample_days, results[:, 4], color='g',label=r"$\delta$(div)")
    ax.fill_between(sample_days, results[:, 4]-results[:, 5], results[:, 4]+results[:, 5], alpha=0.3)
    ax.legend(loc='upper right')
    
    fig.savefig(task_dir+f"/acc_time_series.png")
    
if __name__ == "__main__":
    # model = "checkpoints/numerical/resol_64/tau_[30000,30000,20]/grid_eq/method_implicit/dataset_1980_2025_odd_month_500"
    # erat_dath_path = "reanalysis_data/1995_2025_(1,7,11)_(1-7)_500/data.nc"
    # era5_dataset = xr.open_dataset("reanalysis_data/1995_2025_(1,7,11)_(1-7)_500/data.nc").load()
    # height_mean, height_std, pv_mean, pv_std, divergence_mean, divergence_std = compute_acc_over_a_set(
    #     model=model,
    #     era5_dataset=era5_dataset,
    #     trajectory_set=([2020, 2021], [1]),
    #     start_day=1,
    #     sample_days=[1, 2, 3, 4, 5, 6, 7],
    # )
    # print(f"height_mean = {height_mean}")
    # print(f"height_std = {height_std}")
    # print(f"pv_mean = {pv_mean}")
    # print(f"pv_std = {pv_std}")
    # print(f"divergence_mean = {divergence_mean}")
    # print(f"divergence_std = {divergence_std}")
    
    tau4s=[2]
    tau8s=[2]
    era5_dataset_name="1995_2025_(1,7,11)_(1-7)_500"
    tau_rads=[2, 5, 15, 30, 40]
    # compute_accs(tau4s, 
    #             tau8s, 
    #             tau_rads,
    #             resolution=64, 
    #             era5_dataset_name=era5_dataset_name, 
    #             trajectory_set=([2010, 2015, 2020, 2025], [1, 7, 11]),
    #             start_day=1,
    #             sample_days=[1, 2, 3, 4, 5, 6],
    #             task_dir="task_output/acc_rad_"
    #             )
    # plot_acc_for_different_taus("task_output/acc", [1,2,3,4,5,6], vars=['h'])
    # plot_acc_time_series("task_output/acc", [1,2,3,4,5,6])
    plot_acc_time_series_comparison(baseline_dir="task_output/acc", task_dir="task_output/acc_rad_",
                                    sample_days=[1, 2, 3, 4, 5, 6])
