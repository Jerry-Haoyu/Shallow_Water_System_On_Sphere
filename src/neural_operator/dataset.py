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

    def __init__(self, simulation_data_dir, n_future, mode='residual', seed=None):
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
        n_frames = first["trajectory"].shape[0]
        print(f"💿 SWE Dataset: Each trajectory has n_frames={n_frames} ")
        self.step_window = (96, n_frames - 1 - self.nfuture) # only sample after 2 days
        if self.step_window[1] < self.step_window[0]:
            raise ValueError(
                f"Trajectory of length {n_frames} is too short for n_future={self.nfuture}"
            )

    def __len__(self):
        return len(self.file_list)

    def _spec_to_grid(self, uspec_single):
        """Convert spectral coefficients to grid space based on input_type."""
        if self.input_type == "uvh":
            return self.solver.gethuv(uspec_single)
        else:
            return self.solver.spec2grid(uspec_single)

    def __getitem__(self, index):
        """ 
            Each sample is a (phivortdivphi_t, (phivortdiv_{t+1}-uvphi_t)), i.e., the input is the 
            state in grid space at timestep t and the output is the difference of the same variables between step t and t+1
        """
        file = self.file_list[index]
        try:
            uspec = torch.load(file, map_location=self.device, weights_only=False)["trajectory"]
        except Exception as e:
            print(f"Warning: failed to load {file}: {e}. Falling back to next file.")
            # fall back to the next valid file in the list
            fallback_index = (index + 1) % len(self.file_list)
            file = self.file_list[fallback_index]
            uspec = torch.load(file, map_location=self.device, weights_only=False)["trajectory"]

        # pick a random starting step within the whole-trajectory window
        step_start = self._rng.randint(self.step_window[0], self.step_window[1])
        step_end = step_start + self.nfuture
        # print(f"Picking trajectory number {index} ")
        # print(f"step_start = {step_start}, step_end = {step_end}")

        uspec_target = uspec[step_start: step_end + 1]

        # first and last steps - convert based on input_type
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

    def __init__(self, simulation_data_dir, n_future, max_subsequent_steps, T, U):
        self.simulation_data_dir = simulation_data_dir
        self.n_future = n_future
        self.max_subsequent_steps = max_subsequent_steps
        self.T = T
        self.U = U

        self.file_list = sorted(glob.glob(os.path.join(simulation_data_dir, "*.pt")))
        if not self.file_list:
            raise FileNotFoundError(f"No .pt trajectory files found in {simulation_data_dir}")

        # solver rebuilt from the shared metadata, same reasoning as SWEDataset.
        first = torch.load(self.file_list[0], map_location="cpu", weights_only=False)
        metadata = first["metadata"]
        self.solver = ShallowWaterSolver(lmax=metadata["lmax"], grid=metadata["grid"], dealias=False, non_dimensional=False)
        self.solver.to(self.solver.device)
        self.device = self.solver.device

        self.window_frames = max_subsequent_steps * n_future + 1
        n_frames = first["trajectory"].shape[0]
        print(f"💿 SWE Multi-Step Dataset: n_frames={n_frames}, window_frames={self.window_frames} "
              f"(max_subsequent_steps={max_subsequent_steps}, n_future={n_future})")
        self.step_window = (0, n_frames - self.window_frames)
        if self.step_window[1] < self.step_window[0]:
            raise ValueError(
                f"Trajectory of length {n_frames} is too short for max_subsequent_steps="
                f"{max_subsequent_steps} at n_future={n_future} (needs >= {self.window_frames} frames)."
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        try:
            uspec = torch.load(file, map_location=self.device, weights_only=False)["trajectory"]
        except Exception as e:
            print(f"Warning: failed to load {file}: {e}. Falling back to next file.")
            fallback_index = (index + 1) % len(self.file_list)
            file = self.file_list[fallback_index]
            uspec = torch.load(file, map_location=self.device, weights_only=False)["trajectory"]

        # pick a random starting step within the whole-trajectory window
        start = random.randint(self.step_window[0], self.step_window[1])
        idxs = [start + j * self.n_future for j in range(self.max_subsequent_steps + 1)]
        uspec_window = uspec[idxs].to(self.device)

        window = self.solver.spec2grid(uspec_window).float()
        window = physical_to_nondim(window, self.T, self.U)

        return window.clone(), (index, start)
