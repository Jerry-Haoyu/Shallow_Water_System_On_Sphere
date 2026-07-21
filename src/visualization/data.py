import os

import numpy as np
import torch
import tqdm

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver

# variable name -> (human-readable label, default colormap).
# Aliases map onto the same physical field.
_VAR_INFO = {
    "phi":                 ("geopotential ɸ",      "coolwarm"),
    "geopotential":        ("geopotential ɸ",      "coolwarm"),
    "height":              ("height h (m)",        "coolwarm"),
    "h":                   ("height h (m)",        "coolwarm"),
    "vorticity":           ("vorticity 𝛇",         "RdBu_r"),
    "vort":                ("vorticity 𝛇",         "RdBu_r"),
    "divergence":          ("divergence δ",        "PuOr"),
    "div":                 ("divergence δ",        "PuOr"),
    "u":                   ("zonal wind u (m/s)",  "coolwarm"),
    "v":                   ("meridional wind v (m/s)", "coolwarm"),
    "speed":               ("wind speed |u| (m/s)", "viridis"),
    "pv":                  ("potential vorticity", "coolwarm"),
    "potential_vorticity": ("potential vorticity", "coolwarm"),
}


def _solver_from_metadata(metadata):
    """Rebuild a solver from saved metadata so we can map spectral -> grid."""
    # Resolution is fully determined by lmax (mmax=lmax, nlat=2*lmax, nlon=2*nlat) and dt
    # is derived from CFL — neither is needed from metadata for reconstruction.
    solver = ShallowWaterSolver(
        lmax=metadata["lmax"],
        grid=metadata.get("grid", "equiangular"),
    )
    solver.to(solver.device)
    return solver


def _field_from_uspec(solver, uspec, var, _cache):
    """Compute a single (nlat, nlon) grid field for `var` from one spectral frame.

    `_cache` is reused within a frame so shared transforms (spec2grid, getuv)
    are only computed once per frame across the requested variables.
    """
    if "grid" not in _cache:
        _cache["grid"] = solver.spec2grid(uspec)        # (3, nlat, nlon): ɸ, 𝛇, δ
    grid = _cache["grid"]

    if var in ("phi", "geopotential"):
        return grid[0]
    if var in ("height", "h"):
        return grid[0] / solver.gravity
    if var in ("vorticity", "vort"):
        return grid[1]
    if var in ("divergence", "div"):
        return grid[2]
    if var in ("u", "v", "speed"):
        if "uv" not in _cache:
            _cache["uv"] = solver.getuv(uspec[1:])        # (2, nlat, nlon): u, v
        u, v = _cache["uv"]
        if var == "u":
            return u
        if var == "v":
            return v
        return torch.sqrt(u**2 + v**2)
    if var in ("pv", "potential_vorticity"):
        return solver.potential_vorticity(uspec)

    raise ValueError(
        f"unknown variable '{var}'. supported: {sorted(set(_VAR_INFO))}"
    )


def _coarsen_view(solver, trajectory, factor):
    """Build a lower-resolution solver and spectrally truncate the trajectory so
    fields can be reconstructed on a coarse grid — the inverse SHT then runs over
    far fewer grid points and modes. Returns (coarse_solver, truncated_trajectory).
    """
    # Truncating the spectral resolution by `factor` automatically coarsens the grid
    # too (the solver derives nlat = 2*lmax, nlon = 2*nlat), so the inverse SHT runs
    # over far fewer modes and grid points.
    lmax_c = max(4, solver.lmax // factor)
    coarse = ShallowWaterSolver(lmax=lmax_c, grid=solver.grid)
    coarse = coarse.to(trajectory.device)
    traj_c = trajectory[:, :, :lmax_c, :lmax_c].contiguous()
    return coarse, traj_c


def _build_var_video(solver, trajectory, var, spatial_coarsen=1, temporal_stride=1):
    """Convert the spectral trajectory into a list of grid frames for `var`.

    spatial_coarsen : reconstruct on a grid coarsened by this integer factor (>1
                      is much faster; fields are spectrally truncated to match).
    temporal_stride : reconstruct only every N-th frame (skips work for frames
                      that won't be displayed).
    """
    if temporal_stride and temporal_stride > 1:
        trajectory = trajectory[::temporal_stride]
    if spatial_coarsen and spatial_coarsen > 1:
        solver, trajectory = _coarsen_view(solver, trajectory, spatial_coarsen)
    # Reconstruct on whatever device the solver lives on. The trajectory can stay on
    # the CPU (e.g. an mmap'd multi-GB file); only the small per-frame spectral slice
    # (~few MB) is moved to the solver's device, so a GPU solver accelerates the inverse
    # SHT without ever holding the whole trajectory in GPU memory.
    dev = solver.lap.device
    video = []
    with torch.no_grad():
        for t in tqdm.trange(trajectory.shape[0], desc=f"Reconstructing {var}"):
            field = _field_from_uspec(solver, trajectory[t].to(dev), var, _cache={})
            video.append(field.detach().cpu().numpy())
    return video

def get_var_field(data_path, var, device="cpu"):
    """
    Given the path of the data, return the grid and the the a list of gridded frames for 'var'
        (X, Y, [Z_t]t)
    `device` controls where the inverse-SHT reconstruction runs ("cuda" is much
    faster for large lmax); frames are returned as CPU numpy arrays regardless.
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    metadata = data['metadata']
    solver = _solver_from_metadata(metadata).to(device)
    trajectory = data['trajectory'].to(device)
    video = _build_var_video(solver, trajectory, var)
    print(f"Returning a tuple  (X, Y, [Z_i]_i) where X=lons, Y=lats and Z is {var} \n  i ⊂ [1, {len(video)} | dt = {metadata['dt']}] | Shape = {video[0].shape} ")
    return solver.lons.detach().cpu(), solver.lats.detach().cpu(), video