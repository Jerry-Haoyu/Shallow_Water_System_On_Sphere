import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))


import glob
import os
import random

import torch

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.helpers.run_model import (
    parse_dataset_and_pressure,
    load_h_stats,
    physical_scales,
    physical_to_nondim,
)


class SWEDataset(torch.utils.data.Dataset):
    """Custom Dataset class for PDE training data.

    A single directory of solver outputs is treated as one dataset. Every ``.pt``
    file in ``simulation_data_dir`` is a solver trajectory saved as
    ``{'metadata': {...}, 'trajectory': tensor}`` where ``trajectory`` has shape
    ``(frames, 3, lmax, mmax)`` of spectral coefficients.

    All files are assumed to share the same solver configuration
    """

    def __init__(self, simulation_data_dir, n_future, mode='absolute', seed=None,
                 samples_per_file=1, cache_in_memory=True):
        self.simulation_data_dir = simulation_data_dir
        self.nfuture = n_future
        self.mode= mode
        if seed is not None:
            print(f"Seeding random to {seed}")
        self._rng = random.Random(seed)

        # Pick starting opint randomly
        self.random_flag = True
        # Store trajectories type: either 'spec' or 'physical'
        self.input_type = "spec"

        # How many independent random windows to draw from each trajectory file
        # per epoch (default 1, matching the historical one-window-per-file
        # behavior). Raising this multiplies __len__ without touching the file
        # list itself - see cache_in_memory below, which is what makes this
        # cheap: without it, samples_per_file > 1 would multiply the full-file
        # disk read (see __getitem__) by the same factor every epoch.
        self.samples_per_file = samples_per_file

        # Cache each file's full trajectory in host RAM after its first load,
        # keyed by path, instead of re-reading it from disk on every
        # __getitem__ call (every epoch). Safe because train/val file
        # membership is fixed once at dataset construction (random_split
        # operates on indices, not on file contents) and files are never
        # modified during training. Sized against the *whole* directory
        # (~52GB for subtask 1's 276 files as of debug_train_8_26/stage0.md) -
        # the caller's job must request enough host memory to hold it.
        self.cache_in_memory = cache_in_memory
        self._cache = {} if cache_in_memory else None

        # every trajectory file in the directory forms the sample pool
        self.file_list = sorted(glob.glob(os.path.join(simulation_data_dir, "*.pt")))
        if not self.file_list:
            raise FileNotFoundError(f"No .pt trajectory files found in {simulation_data_dir}")

        # ------------------------------------------------------------------ #
        # Rebuild the solver once from the stored metadata (all files in the
        # directory share the configuration). Only spec2grid is exercised here,
        # which depends solely on the spectral truncation and quadrature grid;
        # the remaining solver knobs (tau, cfl, semi_implicit, dealias) do not
        # affect the transform, so their defaults are fine and dealias is turned
        # off to skip building the unused padded transforms.
        # ------------------------------------------------------------------ #
        first = torch.load(self.file_list[0], map_location="cpu", weights_only=False)
        metadata = first["metadata"]
        self.solver = ShallowWaterSolver(lmax=metadata["lmax"], grid=metadata["grid"], dealias=False, non_dimensional=False)
        self.solver.to(self.solver.device)
        self.device = self.solver.device
        if self.cache_in_memory:
            # already loaded above (on CPU) for metadata - reuse it instead of
            # discarding and re-reading it in __getitem__.
            self._cache[self.file_list[0]] = first["trajectory"]

        # REAL elapsed minutes between consecutive saved frames in these
        # trajectory files - the achieved cadence run()'s numerical branch (or
        # build_era5_trajectory_dataset.py, exactly, for ERA5-direct data)
        # actually realized, not a nominal/requested one (see stage0.md
        # Findings: frame-cadence rounding drift). Recorded into
        # model_info.json so inference-time rollout (run_model.py's run())
        # knows how many real minutes n_future actually spans.
        self.true_interval_minutes = metadata["true_interval_minutes"]

        # non-dimensionalization scales (T, U), keyed by this training data's
        # own dataset-wide h_avg (README.md convention: recovered from
        # simulation_data_dir's own dataset_*/pressure_* path nodes) - (None,
        # None)/DEFAULT_HAVG fallback for galewsky-only data, which has no
        # real-world dataset. Used in place of a z-score to scale samples
        # in __getitem__ below.
        self.dataset_name, self.pressure = parse_dataset_and_pressure(self.simulation_data_dir)
        self.h_avg, self.h_amp = (
            load_h_stats(self.dataset_name) if self.dataset_name else (None, None)
        )
        self.T, self.U = physical_scales(self.h_avg)

        # step_window spans the whole trajectory: any start step whose n_future
        # target still lands inside the trajectory. Inferred from its length.
        # Only sample after a 2-day warmup (skips the file's own spin-up/edge
        # effects) - in units of this file's own frame cadence, so it's still
        # "2 days" regardless of whether frames are 30 min apart (PS-solver
        # output, warmup=96) or 60 min apart (ERA5-direct, warmup=48).
        n_frames = first["trajectory"].shape[0]
        print(f"💿 SWE Dataset: Each trajectory has n_frames={n_frames} ")
        warmup_steps = max(1, round(2 * 24 * 60 / self.true_interval_minutes))
        self.step_window = (warmup_steps, n_frames - 1 - self.nfuture)
        if self.step_window[1] < self.step_window[0]:
            raise ValueError(
                f"Trajectory of length {n_frames} is too short for n_future={self.nfuture}"
            )

    def __len__(self):
        # samples_per_file independent random windows are drawn from each
        # file per epoch (default 1) - see __init__ and __getitem__.
        return len(self.file_list) * self.samples_per_file

    def _spec_to_grid(self, uspec_single):
        """Convert spectral coefficients to grid space based on input_type."""
        if self.input_type == "uvh":
            return self.solver.gethuv(uspec_single)
        else:
            return self.solver.spec2grid(uspec_single)

    def _load_trajectory(self, file_idx):
        """Return the (file_idx's) file's full spectral trajectory, on
        self.device if uncached or on CPU if cached (see cache_in_memory).
        Falls back to the next file in the list if loading fails."""
        file = self.file_list[file_idx]
        map_location = "cpu" if self.cache_in_memory else self.device
        cache = self._cache if self.cache_in_memory else None

        if cache is not None and file in cache:
            return cache[file]
        try:
            uspec = torch.load(file, map_location=map_location, weights_only=False)["trajectory"]
        except Exception as e:
            print(f"Warning: failed to load {file}: {e}. Falling back to next file.")
            # fall back to the next valid file in the list
            fallback_idx = (file_idx + 1) % len(self.file_list)
            return self._load_trajectory(fallback_idx)
        if cache is not None:
            cache[file] = uspec
        return uspec

    def __getitem__(self, index):
        """
            Each sample is a (phivortdivphi_t, (phivortdiv_{t+1}-uvphi_t)), i.e., the input is the
            state in grid space at timestep t and the output is the difference of the same variables between step t and t+1
        """
        # samples_per_file windows share the same file, cycling through
        # file_list; index // len(file_list) only picks which of the
        # samples_per_file draws this is (the actual window is still random).
        file_idx = index % len(self.file_list)
        uspec = self._load_trajectory(file_idx)

        # pick a random starting step within the whole-trajectory window
        step_start = self._rng.randint(self.step_window[0], self.step_window[1])
        step_end = step_start + self.nfuture
        # print(f"Picking trajectory number {index} ")
        # print(f"step_start = {step_start}, step_end = {step_end}")

        uspec_target = uspec[step_start: step_end + 1]

        # first and last steps - convert based on input_type. .to(self.device)
        # is a no-op in the uncached path (uspec already lives there) and the
        # real CPU->GPU transfer in the cached path - crucially only the 2
        # selected frames cross PCIe, not the whole cached file.
        u_curr = self._spec_to_grid(uspec_target[0].to(self.device)).float()
        u_next = self._spec_to_grid(uspec_target[-1].to(self.device)).float()

        u_curr = physical_to_nondim(u_curr, self.T, self.U)
        u_next = physical_to_nondim(u_next, self.T, self.U)
        
        if self.mode == 'residual':
            return u_curr.clone(), (u_next-u_curr).clone(), (index, step_start)
        elif self.mode == 'absolute':
            return u_curr.clone(), u_next.clone(), (index, step_start)


class SWEMultiStepDataset(torch.utils.data.Dataset):
    """Windowed dataset for multi-step curriculum training (see train_multistep.py).

    Each item is a window of ``max_subsequent_steps + 1`` grid-space frames,
    spaced ``n_future`` trajectory frames apart, starting at a random offset:
    ``window[j]`` is the frame at trajectory step ``start + j * n_future`` for
    ``j = 0 .. max_subsequent_steps``. In the curriculum pseudocode, ``window[0]``
    plays "obs1 at t(1)" (what the frozen teacher rolls out from) and
    ``window[j]`` for ``j >= 1`` plays "obs2"/"target at t(j+1)" depending on
    which curriculum stage's ``teacher_step`` is indexing into it.
    """

    def __init__(self, simulation_data_dir, n_future, max_subsequent_steps, T, U,
                 samples_per_file=1, cache_in_memory=True):
        self.simulation_data_dir = simulation_data_dir
        self.n_future = n_future
        self.max_subsequent_steps = max_subsequent_steps
        self.T = T
        self.U = U

        # see SWEDataset's identical knobs for the full rationale: draw this
        # many independent random windows from each file per epoch (default
        # 1, matching the old one-window-per-file behavior), and cache each
        # file's full trajectory in host RAM after its first load instead of
        # re-reading it from disk every epoch - only cheap to raise
        # samples_per_file above 1 when cache_in_memory is also True.
        self.samples_per_file = samples_per_file
        self.cache_in_memory = cache_in_memory
        self._cache = {} if cache_in_memory else None

        self.file_list = sorted(glob.glob(os.path.join(simulation_data_dir, "*.pt")))
        if not self.file_list:
            raise FileNotFoundError(f"No .pt trajectory files found in {simulation_data_dir}")

        # solver rebuilt from the shared metadata, same reasoning as SWEDataset.
        first = torch.load(self.file_list[0], map_location="cpu", weights_only=False)
        metadata = first["metadata"]
        self.solver = ShallowWaterSolver(lmax=metadata["lmax"], grid=metadata["grid"], dealias=False, non_dimensional=False)
        self.solver.to(self.solver.device)
        self.device = self.solver.device
        if self.cache_in_memory:
            # already loaded above (on CPU) for metadata - reuse it instead of
            # discarding and re-reading it in __getitem__.
            self._cache[self.file_list[0]] = first["trajectory"]

        # real elapsed minutes between consecutive saved frames - same field
        # SWEDataset reads, used the same way below (2-day warmup in units of
        # this file's own frame cadence).
        self.true_interval_minutes = metadata["true_interval_minutes"]

        self.window_frames = max_subsequent_steps * n_future + 1
        n_frames = first["trajectory"].shape[0]
        print(f"💿 SWE Multi-Step Dataset: n_frames={n_frames}, window_frames={self.window_frames} "
              f"(max_subsequent_steps={max_subsequent_steps}, n_future={n_future})")
        # Only sample after a 2-day warmup, skipping the file's own
        # spin-up/edge effects - same reasoning and formula as SWEDataset's
        # step_window (dataset.py), previously missing here entirely (this
        # class started sampling from step 0).
        warmup_steps = max(1, round(2 * 24 * 60 / self.true_interval_minutes))
        self.step_window = (warmup_steps, n_frames - self.window_frames)
        if self.step_window[1] < self.step_window[0]:
            raise ValueError(
                f"Trajectory of length {n_frames} is too short for max_subsequent_steps="
                f"{max_subsequent_steps} at n_future={n_future} with a {warmup_steps}-step "
                f"warmup (needs >= {warmup_steps + self.window_frames} frames)."
            )

    def __len__(self):
        # samples_per_file independent random windows are drawn from each
        # file per epoch (default 1) - see __init__ and __getitem__.
        return len(self.file_list) * self.samples_per_file

    def _load_trajectory(self, file_idx):
        """Return the (file_idx's) file's full spectral trajectory, on
        self.device if uncached or on CPU if cached (see cache_in_memory).
        Falls back to the next file in the list if loading fails."""
        file = self.file_list[file_idx]
        map_location = "cpu" if self.cache_in_memory else self.device
        cache = self._cache if self.cache_in_memory else None

        if cache is not None and file in cache:
            return cache[file]
        try:
            uspec = torch.load(file, map_location=map_location, weights_only=False)["trajectory"]
        except Exception as e:
            print(f"Warning: failed to load {file}: {e}. Falling back to next file.")
            fallback_idx = (file_idx + 1) % len(self.file_list)
            return self._load_trajectory(fallback_idx)
        if cache is not None:
            cache[file] = uspec
        return uspec

    def __getitem__(self, index):
        file_idx = index % len(self.file_list)
        uspec = self._load_trajectory(file_idx)

        # pick a random starting step within the whole-trajectory window
        start = random.randint(self.step_window[0], self.step_window[1])
        idxs = [start + j * self.n_future for j in range(self.max_subsequent_steps + 1)]
        uspec_window = uspec[idxs].to(self.device)

        window = self.solver.spec2grid(uspec_window).float()
        window = physical_to_nondim(window, self.T, self.U)

        return window.clone(), (index, start)
