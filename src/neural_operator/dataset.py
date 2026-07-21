import glob
import os
import random

import torch

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver


class SWEDataset(torch.utils.data.Dataset):
    """Custom Dataset class for PDE training data.

    A single directory of solver outputs is treated as one dataset. Every ``.pt``
    file in ``simulation_data_dir`` is a solver trajectory saved as
    ``{'metadata': {...}, 'trajectory': tensor}`` where ``trajectory`` has shape
    ``(frames, 3, lmax, mmax)`` of spectral coefficients.

    All files are assumed to share the same solver configuration, so the solver
    used to map spectral coefficients back to the grid is rebuilt exactly once in
    ``__init__`` from the stored metadata. The behaviour is fixed to the common
    training setup: normalization is always on, the sampling window covers the
    whole trajectory, a random start step is drawn per sample, and inputs are the
    spectral-to-grid fields (``spec2grid``). The normalization statistics are
    computed in place from the data during construction.
    """

    def __init__(self, simulation_data_dir, n_future):
        self.simulation_data_dir = simulation_data_dir
        self.nfuture = n_future

        # fixed behaviour (previously configurable)
        self.normalize = True
        self.random_flag = True
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
        self.solver = ShallowWaterSolver(lmax=metadata["lmax"], grid=metadata["grid"], dealias=False)
        self.solver.to(self.solver.device)
        self.device = self.solver.device

        # step_window spans the whole trajectory: any start step whose n_future
        # target still lands inside the trajectory. Inferred from its length.
        n_frames = first["trajectory"].shape[0]
        self.step_window = (0, n_frames - 1 - self.nfuture)
        if self.step_window[1] < self.step_window[0]:
            raise ValueError(
                f"Trajectory of length {n_frames} is too short for n_future={self.nfuture}"
            )

        # compute normalization statistics (per-channel mean/std over grid space,
        # accumulated across every frame of every trajectory) in place.
        self._compute_stats()

    def _compute_stats(self):
        """Per-channel grid-space mean/std over all frames of all files."""
        count = 0
        sum_c = None
        sumsq_c = None
        with torch.no_grad():
            for file in self.file_list:
                trajectory = torch.load(file, map_location=self.device, weights_only=False)["trajectory"]
                for t in range(trajectory.shape[0]):
                    grid = self._spec_to_grid(trajectory[t].to(self.device))
                    flat = grid.reshape(grid.shape[0], -1).double()
                    if sum_c is None:
                        channels = flat.shape[0]
                        sum_c = torch.zeros(channels, dtype=torch.float64, device=self.device)
                        sumsq_c = torch.zeros(channels, dtype=torch.float64, device=self.device)
                    sum_c += flat.sum(dim=1)
                    sumsq_c += (flat ** 2).sum(dim=1)
                    count += flat.shape[1]

        mean = sum_c / count
        std = torch.sqrt(torch.clamp(sumsq_c / count - mean ** 2, min=0.0))
        self.inp_mean = mean.reshape(-1, 1, 1).float().to(self.device)
        self.inp_std = std.reshape(-1, 1, 1).float().to(self.device)

    def __len__(self):
        return len(self.file_list)

    def _spec_to_grid(self, uspec_single):
        """Convert spectral coefficients to grid space based on input_type."""
        if self.input_type == "uvh":
            return self.solver.gethuv(uspec_single)
        else:
            return self.solver.spec2grid(uspec_single)

    def __getitem__(self, index):

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
        self.step_srt = random.randint(self.step_window[0], self.step_window[1])
        self.step_end = self.step_srt + self.nfuture

        uspec_target = uspec[self.step_srt: self.step_end + 1]

        # first and last steps - convert based on input_type
        inp = self._spec_to_grid(uspec_target[0].to(self.device)).float()
        tar = self._spec_to_grid(uspec_target[-1].to(self.device)).float()

        if self.normalize:
            inp = (inp - self.inp_mean) / self.inp_std
            tar = (tar - self.inp_mean) / self.inp_std

        return inp.clone(), tar.clone(), (index, self.step_srt)
