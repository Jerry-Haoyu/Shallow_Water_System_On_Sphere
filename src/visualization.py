import os

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import torch
import tqdm

from data import _solver_from_metadata, _field_from_uspec, _build_var_video

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


def _common_setup(data):
    """Shared front matter: rebuild solver, derive grid, normalise inputs."""
    metadata = data["metadata"]
    solver = _solver_from_metadata(metadata)

    trajectory = data["trajectory"]
    if not torch.is_tensor(trajectory):
        trajectory = torch.as_tensor(trajectory)

    lats = solver.lats.detach().cpu().numpy()
    lons = solver.lons.detach().cpu().numpy()

    # physical seconds advanced per saved frame (step_per_save may be absent on
    # older save files, in which case titles fall back to the frame index).
    step_per_save = metadata.get("step_per_save")
    sec_per_frame = metadata["dt"] * step_per_save if step_per_save else None

    return solver, trajectory, lats, lons, sec_per_frame


def _frame_title(label, frame_idx, coarsen_factor, sec_per_frame):
    t = frame_idx * coarsen_factor
    if sec_per_frame is None:
        return f"{label}  (frame {t})"
    days = t * sec_per_frame / 86400.0
    return f"{label}  t={days:.2f} (days)"


def _color_limits(video):
    sample = video[:: max(1, len(video) // 20)]
    all_vals = np.concatenate([v.ravel() for v in sample])
    return float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))


def animate_swe_on_sphere(data, variables, output_dir, fps=15, coarsen_factor=1):
    """
    Animate saved SWE simulation variables on a sphere — one .mp4 per variable.

    Parameters
    ----------
    data           : dict {'metadata', 'trajectory'} as written by the solver's
                     ``run``. ``trajectory`` holds spectral coefficients
                     (nframes, 3, lmax, mmax); grid fields are reconstructed here.
    variables      : list of variable names to visualise, e.g.
                     ['phi', 'vorticity', 'divergence', 'u', 'v', 'speed', 'pv'].
    output_dir     : directory for the output videos; each var is saved as
                     ``<output_dir>/<var>.mp4``.
    fps            : frames per second of the output video.
    coarsen_factor : use every N-th frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    solver, trajectory, lats, lons, sec_per_frame = _common_setup(data)

    Lons, Lats = np.meshgrid(lons, lats)
    x = np.cos(Lats) * np.cos(Lons)
    y = np.cos(Lats) * np.sin(Lons)
    z = (296 / 297) * np.sin(Lats)

    print(f" Physical Grid Shape : {lats.shape[0]} ✖️ {lons.shape[0]}")

    for var in variables:
        label, cmap_name = _VAR_INFO.get(var, (var, "coolwarm"))
        output_path = os.path.join(output_dir, f"{var}.mp4")
        print(f" Animating {var} on sphere -> {output_path}")

        video = _build_var_video(solver, trajectory, var)
        total_frames = int(len(video) / coarsen_factor)

        vmin, vmax = _color_limits(video)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(elev=15, azim=35)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label(label)

        surface = ax.plot_surface(
            x, y, z, zorder=1,
            facecolors=cmap(norm(video[0])),
            rcount=x.shape[0], ccount=x.shape[1],
            shade=False, antialiased=False, linewidth=0,
        )
        ax.set_title(_frame_title(label, 0, coarsen_factor, sec_per_frame))

        pbar = tqdm.tqdm(total=total_frames)

        def update(frame_idx, _surface=[surface]):
            t = frame_idx * coarsen_factor
            _surface[0].remove()
            _surface[0] = ax.plot_surface(
                x, y, z, zorder=1,
                facecolors=cmap(norm(video[t])),
                rcount=x.shape[0], ccount=x.shape[1],
                shade=False, antialiased=False, linewidth=0,
            )
            ax.set_title(_frame_title(label, frame_idx, coarsen_factor, sec_per_frame))
            pbar.update(1)

        result = anim.FuncAnimation(fig, update, frames=total_frames)
        result.save(filename=output_path, fps=fps)
        pbar.close()
        plt.close(fig)
        print(f"Saved animation -> {output_path}")


def animate_swe_on_box(data, variables, output_dir, fps=15, coarsen_factor=1):
    """
    Animate saved SWE simulation variables on a flat lon/lat box — one .mp4 per variable.

    A fast 2-D counterpart to :func:`animate_swe_on_sphere`: each frame is a single
    ``imshow`` image (one blitted image vs. a re-meshed surface per frame).

    Parameters
    ----------
    data           : dict {'metadata', 'trajectory'} as written by the solver's
                     ``run``. ``trajectory`` holds spectral coefficients
                     (nframes, 3, lmax, mmax); grid fields are reconstructed here.
    variables      : list of variable names to visualise (see _VAR_INFO).
    output_dir     : directory for the output videos; each var is saved as
                     ``<output_dir>/<var>.mp4``.
    fps            : frames per second of the output video.
    coarsen_factor : use every N-th frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    solver, trajectory, lats, lons, sec_per_frame = _common_setup(data)

    # Plot in degrees for readable axes.
    extent = [np.degrees(lons[0]), np.degrees(lons[-1]),
              np.degrees(lats[0]), np.degrees(lats[-1])]

    print(f" Physical Grid Shape : {lats.shape[0]} ✖️ {lons.shape[0]}")

    for var in variables:
        label, cmap_name = _VAR_INFO.get(var, (var, "coolwarm"))
        output_path = os.path.join(output_dir, f"{var}.mp4")
        print(f" Animating {var} on box -> {output_path}")

        video = _build_var_video(solver, trajectory, var)
        total_frames = int(len(video) / coarsen_factor)

        vmin, vmax = _color_limits(video)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)

        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)")

        # Origin "lower" so increasing latitude points up.
        im = ax.imshow(video[0], origin="lower", extent=extent,
                       aspect="auto", norm=norm, cmap=cmap,
                       interpolation="bilinear")
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label(label)
        ax.set_title(_frame_title(label, 0, coarsen_factor, sec_per_frame))

        pbar = tqdm.tqdm(total=total_frames)

        def update(frame_idx):
            t = frame_idx * coarsen_factor
            im.set_data(video[t])
            ax.set_title(_frame_title(label, frame_idx, coarsen_factor, sec_per_frame))
            pbar.update(1)
            return (im,)

        result = anim.FuncAnimation(fig, update, frames=total_frames, blit=False)
        result.save(filename=output_path, fps=fps)
        pbar.close()
        plt.close(fig)
        print(f"Saved animation -> {output_path}")
