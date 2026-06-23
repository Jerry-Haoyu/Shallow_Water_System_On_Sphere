import os

import numpy as np
import torch
import tqdm

from psuedo_spectral_solver_naive import ShallowWaterSolver

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
    return ShallowWaterSolver(
        dt=metadata["dt"],
        nlat=metadata["nlat"],
        nlon=metadata["nlon"],
        lmax=metadata["lmax"],
        mmax=metadata["mmax"],
        grid=metadata.get("grid", "equiangular"),
    )


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


def _build_var_video(solver, trajectory, var):
    """Convert the whole spectral trajectory into a list of grid frames for `var`."""
    video = []
    with torch.no_grad():
        for t in range(trajectory.shape[0]):
            field = _field_from_uspec(solver, trajectory[t], var, _cache={})
            video.append(field.detach().cpu().numpy())
    return video

def get_var_field(data_path, var):
    """
    Given the path of the data, return the grid and the the a list of gridded frames for 'var' 
        (X, Y, [Z_t]t)
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    metadata = data['metadata']
    solver = _solver_from_metadata(metadata)
    video = _build_var_video(solver, data['trajectory'], var)
    print(f"Returning a tuple  (X, Y, [Z_i]_i) where X=lons, Y=lats and Z is {var} \n  i ⊂ [1, {len(video)} | dt = {metadata['dt']}] | Shape = {video[0].shape} ")
    return solver.lons, solver.lats, video