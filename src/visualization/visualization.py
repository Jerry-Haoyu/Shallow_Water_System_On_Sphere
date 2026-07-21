import os

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import torch
import tqdm

import matplotlib.animation as animation

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.visualization.data import _solver_from_metadata, _field_from_uspec, _build_var_video
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


def _common_setup(data, device="cpu"):
    """Shared front matter: rebuild solver, derive grid, normalise inputs.

    `device` is where the inverse-SHT reconstruction runs ("cuda" is much faster for
    large lmax); the trajectory is left where it is and moved frame-by-frame, and grid
    fields come back as CPU numpy arrays for matplotlib regardless.
    """
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


def _center_longitudes(lons):
    """Recenter longitudes from ``[0, 2π)`` onto ``[-π, π)``.

    torch_harmonics places longitudes in ``[0, 360)``; shifting to ``[-180, 180)``
    puts the prime meridian in the middle of the plot. Returns the recentered,
    ascending longitudes (radians) and an ``order`` index that reorders the
    longitude axis (last axis) of any field so its columns line up with them.
    """
    lons_centered = (lons + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(lons_centered)
    return lons_centered[order], order


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

    # Recenter longitude onto [-180, 180); `lon_order` reorders each field's
    # longitude axis to match the recentered coordinates.
    lons, lon_order = _center_longitudes(lons)

    Lons, Lats = np.meshgrid(lons, lats)
    x = np.cos(Lats) * np.cos(Lons)
    y = np.cos(Lats) * np.sin(Lons)
    z = (296 / 297) * np.sin(Lats)

    print(f" Physical Grid Shape : {lats.shape[0]} ✖️ {lons.shape[0]}")

    for var in variables:
        label, cmap_name = _VAR_INFO.get(var, (var, "coolwarm"))
        output_path = os.path.join(output_dir, f"{var}.mp4")
        print(f" Animating {var} on sphere -> {output_path}")

        video = [frame[:, lon_order]
                 for frame in _build_var_video(solver, trajectory, var)]
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


def animate_swe_on_box(data, variables, output_dir, fps=15, coarsen_factor=1,
                       coastline=False):
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
    coastline      : if True, render onto a cartopy PlateCarree map with
                     coastlines overlaid (as in :func:`animate_uv_quiver_on_box`);
                     otherwise use a plain ``imshow`` lon/lat box.
    """
    os.makedirs(output_dir, exist_ok=True)
    solver, trajectory, lats, lons, sec_per_frame = _common_setup(data)

    # Recenter longitude onto [-180, 180); `lon_order` reorders each field's
    # longitude axis to match the recentered coordinates.
    lons, lon_order = _center_longitudes(lons)

    # Plot in degrees for readable axes.
    extent = [np.degrees(lons[0]), np.degrees(lons[-1]),
              np.degrees(lats[-1]), np.degrees(lats[0])]

    print(f" Physical Grid Shape : {lats.shape[0]} ✖️ {lons.shape[0]}")

    for var in variables:
        label, cmap_name = _VAR_INFO.get(var, (var, "coolwarm"))
        output_path = os.path.join(output_dir, f"{var}.mp4")
        print(f" Animating {var} on box -> {output_path}")

        # Coarsen temporally during reconstruction, not just at display time:
        # `temporal_stride` skips frames so they're never reconstructed or held in the
        # video list. This matches the previous displayed animation (every Nth frame at
        # full resolution) but keeps the list ~coarsen_factor times smaller, which is
        # what avoids the OOM on large-nframes runs.
        video = [frame[:, lon_order]
                 for frame in _build_var_video(solver, trajectory, var,
                                               temporal_stride=coarsen_factor)]
        total_frames = len(video)

        vmin, vmax = _color_limits(video)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)

        if coastline:
            # Render onto a cartopy map so real-world coastlines line up with the
            # [-180, 180) longitude extent (same projection as the uv quiver).
            fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
            ax.set_global()
            ax.coastlines(resolution="110m", color="red", linewidth=1)
            # Origin "lower" so increasing latitude points up; `transform` tells
            # cartopy the data itself is in PlateCarree lon/lat.
            im = ax.imshow(video[0], origin="lower", extent=extent,
                           norm=norm, cmap=cmap, interpolation="bilinear",
                           transform=ccrs.PlateCarree())
        else:
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
            # video[frame_idx] is already the temporally-strided frame; multiply by
            # coarsen_factor only to recover the original frame index for the title.
            im.set_data(video[frame_idx])
            ax.set_title(_frame_title(label, frame_idx, coarsen_factor, sec_per_frame))
            pbar.update(1)
            return (im,)

        result = anim.FuncAnimation(fig, update, frames=total_frames, blit=False)
        result.save(filename=output_path, fps=fps)
        pbar.close()
        plt.close(fig)
        print(f"Saved animation -> {output_path}")


def visualize_initial_condition_on_box(
    model,
    initial_condition,
    plot_title=None,
    initial_condition_params=None):
    """
        Genearte a plot of the initial conndition
        
        Params:
            initial_condition_params: a dict of params to be passed in the callalbe initial_condition 
            excluding 'model'
    """
    if initial_condition_params is not None:
        pvdspec = initial_condition(model=model, **initial_condition_params)
    else :
        pvdspec = initial_condition(model=model)
    
    pvd = model.spec2grid(pvdspec)
    
    pvd = pvd.to('cpu')
    
    layout = [
        ["A", "A", "B", "B"],
        [".", "C", "C", "."]
    ]
    
    fig, axd = plt.subplot_mosaic(layout, 
                                  figsize=(8, 5),
                                  gridspec_kw={'width_ratios':[1, 1, 1, 1]})   
    
    if plot_title is None:
        title = "Initial Condition Visualization"
    else: 
       title = plot_title
    
    fig.suptitle(title)
    
    # state layout is (ɸ, 𝛇, δ): geopotential, vorticity, divergence
    axd['A'].imshow(pvd[1])
    axd['A'].set_title(rf"Vorticity $\zeta$ ")
    axd['B'].imshow(pvd[2])
    axd['B'].set_title(rf"Divergence $\delta$")
    axd['C'].imshow(pvd[0])
    axd['C'].set_title(rf"Geopotential $\Phi$")

    plt.tight_layout()
    fig.savefig(f"visualization_output/{(title.replace(' ', '_')).lower()}")


def visualize_initial_condition_on_sphere(
    model,
    initial_condition,
    plot_title=None,
    initial_condition_params=None,
    output_dir="visualization_output"):
    """
        Generate a plot of the initial condition rendered on three spheres, one per
        prognostic field (geopotential, vorticity, divergence). This is the spherical
        counterpart to :func:`visualize_initial_condition_on_box`.

        Params:
            model                    : the SWE solver (provides spec2grid, lats, lons).
            initial_condition        : callable returning spectral coeffs (3, lmax, mmax)
                                       of (ɸ, 𝛇, δ); called as ``initial_condition(model=model, **params)``.
            plot_title               : figure title (also used to name the saved file).
            initial_condition_params : dict of extra kwargs for ``initial_condition``
                                       (excluding ``model``).
            output_dir               : directory for the saved figure.
    """
    if initial_condition_params is not None:
        pvdspec = initial_condition(model=model, **initial_condition_params)
    else:
        pvdspec = initial_condition(model=model)

    # state layout is (ɸ, 𝛇, δ): geopotential, vorticity, divergence
    pvd = model.spec2grid(pvdspec).detach().cpu().numpy()

    lats = model.lats.detach().cpu().numpy()
    lons = model.lons.detach().cpu().numpy()

    # Cartesian coordinates of the sphere (mirrors animate_swe_on_sphere).
    Lons, Lats = np.meshgrid(lons, lats)
    x = np.cos(Lats) * np.cos(Lons)
    y = np.cos(Lats) * np.sin(Lons)
    z = (296 / 297) * np.sin(Lats)

    fields = [
        (pvd[0], r"Geopotential $\Phi$", _VAR_INFO["phi"][1]),
        (pvd[1], r"Vorticity $\zeta$",   _VAR_INFO["vorticity"][1]),
        (pvd[2], r"Divergence $\delta$", _VAR_INFO["divergence"][1]),
    ]

    title = "Initial Condition Visualization" if plot_title is None else plot_title

    fig, axs = plt.subplots(1, 3, figsize=(15, 5),
                            subplot_kw={"projection": "3d"})
    fig.suptitle(title)

    for ax, (field, label, cmap_name) in zip(axs, fields):
        vmin = float(np.percentile(field, 2))
        vmax = float(np.percentile(field, 98))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)

        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(elev=15, azim=35)

        ax.plot_surface(
            x, y, z, zorder=1,
            facecolors=cmap(norm(field)),
            rcount=x.shape[0], ccount=x.shape[1],
            shade=False, antialiased=False, linewidth=0,
        )
        ax.set_title(label)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_sphere")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved initial-condition sphere plot -> {output_path}")


def animate_uv_quiver_on_box(data, 
                             output_dir, 
                             file_name="uv_quiver", 
                             time_coarsen_factor=4):
    os.makedirs(output_dir, exist_ok=True)

    solver, trajectory, lats, lons, sec_per_frame = _common_setup(data)
    u =  _build_var_video(solver, trajectory, 'u', temporal_stride=time_coarsen_factor)
    v =  _build_var_video(solver, trajectory, 'v', temporal_stride=time_coarsen_factor)
    h =  _build_var_video(solver, trajectory, 'h', temporal_stride=time_coarsen_factor)

    fig, ax = plt.subplots(figsize=(8, 6), 
                        dpi=150, 
                        subplot_kw = {'projection' :ccrs.PlateCarree()}) 
    
    ax.set_title(_frame_title("uv wind field", 0, time_coarsen_factor, sec_per_frame))

    # sampling to plot the tangent bundle 
    sp_coarsen_factor= int(3 * (data['metadata']['lmax']//64))

    progress = tqdm.tqdm(total=len(u), desc="Plotting uv wind field")

    ax.set_global()
    ax.coastlines(resolution='110m', color='black', linewidth=1)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)

    lons_deg, lats_deg = np.meshgrid(np.degrees(lons),
                            np.degrees(lats),
                            indexing='xy'
                            ) 

    print(np.swapaxes(u[2][::sp_coarsen_factor, ::sp_coarsen_factor], 0, 1).shape)
    
    label, cmap_name = _VAR_INFO.get('h', ('height', "coolwarm"))
    vmin, vmax = _color_limits(pv)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    
    scalar = ax.pcolormesh(
        lons_deg,
        lats_deg,
        h[0],
        cmap=cmap,
        norm=norm,
        shading='auto',
        transform=ccrs.PlateCarree()
    )
    
    quiver = ax.quiver(
        lons_deg[::sp_coarsen_factor, ::sp_coarsen_factor], 
        lats_deg[::sp_coarsen_factor, ::sp_coarsen_factor],
        u[2][::sp_coarsen_factor, ::sp_coarsen_factor], 
        v[2][::sp_coarsen_factor, ::sp_coarsen_factor],
        width=0.001,
        headwidth=5,
        headaxislength=2.3,
        headlength=3.5,
        transform=ccrs.PlateCarree()  
    )

    def animate(i):
        progress.update(1)
        U=u[i][::sp_coarsen_factor,::sp_coarsen_factor]
        V=v[i][::sp_coarsen_factor,::sp_coarsen_factor]
        scalar.set_array(pv[i].flatten())
        quiver.set_UVC(U=U, V=V)
        ax.set_title(_frame_title("uv wind field", i, time_coarsen_factor, sec_per_frame))

    ani = animation.FuncAnimation(fig, animate, len(u))
    path = output_dir + "/" + file_name + ".mp4"
    ani.save(path)
    print(f"✅ Finished Visualization of uv Field ✅  \n Saved to -----> {path}")

    
def track_tropical_cyclone(data, 
                        output_dir, 
                        initial_loc,
                        file_name="tropical_cyclone_tracking", 
                        time_coarsen_factor=4):
    os.makedirs(output_dir, exist_ok=True)

    solver, trajectory, lats, lons, sec_per_frame = _common_setup(data)
    u =  _build_var_video(solver, trajectory, 'u', temporal_stride=time_coarsen_factor)
    v =  _build_var_video(solver, trajectory, 'v', temporal_stride=time_coarsen_factor)
    h =  _build_var_video(solver, trajectory, 'h', temporal_stride=time_coarsen_factor)
    pv =  _build_var_video(solver, trajectory, 'pv', temporal_stride=time_coarsen_factor)
    
    fig, ax = plt.subplots(figsize=(8, 6), 
                        dpi=150, 
                        subplot_kw = {'projection' :ccrs.PlateCarree()}) 
    
    ax.set_title(_frame_title("uv wind field", 0, time_coarsen_factor, sec_per_frame))

    # sampling to plot the tangent bundle 
    sp_coarsen_factor= int(3 * (data['metadata']['lmax']//64))

    progress = tqdm.tqdm(total=len(u), desc="Plotting uv wind field")

    ax.set_global()
    ax.coastlines(resolution='110m', color='black', linewidth=1)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)

    lons_deg, lats_deg = np.meshgrid(np.degrees(lons),
                            np.degrees(lats),
                            indexing='xy'
                            ) 

    print(np.swapaxes(u[2][::sp_coarsen_factor, ::sp_coarsen_factor], 0, 1).shape)
    
    label, cmap_name = _VAR_INFO.get('h', ('height', "coolwarm"))
    vmin, vmax = _color_limits(h)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    
    scalar = ax.pcolormesh(
        lons_deg,
        lats_deg,
        h[0],
        cmap=cmap,
        norm=norm,
        shading='auto',
        transform=ccrs.PlateCarree()
    )
    
    quiver = ax.quiver(
        lons_deg[::sp_coarsen_factor, ::sp_coarsen_factor], 
        lats_deg[::sp_coarsen_factor, ::sp_coarsen_factor],
        u[2][::sp_coarsen_factor, ::sp_coarsen_factor], 
        v[2][::sp_coarsen_factor, ::sp_coarsen_factor],
        width=0.001,
        headwidth=5,
        headaxislength=2.3,
        headlength=3.5,
        transform=ccrs.PlateCarree()  
    )
    
    y, x = initial_loc 
    print(f"Initial Location is lats,lon = {y, x}")
    j, i = int(np.argmin(np.abs(np.degrees(lats)- y))), int(np.argmin(np.abs(np.degrees(lons)- x)))
    # print(f"Initial location has array index:  j, i = {j,i}")
    
    def update_cyclone_center(j, i, frame):
        pv_max = 0.0
        resol = 360 / len(lons) # how many degree does one grid cover
        width = int(10.0 // resol) # the width of the bounding box 
        height =  int(10.0 // resol) # the height of the bounding box
        new_i_, new_j_ = i, j
        for new_i in range(i-5, i+5, 1):
            for new_j in range(j-5, j+5, 1):
                new_i, new_j = int(new_i), int(new_j)
                left, right = max(new_i-width//2, 0), min(new_i+width//2, len(lons))
                down, top = max(new_j-height//2, 0), min(new_j+height//2, len(lats))
                pv_mean = np.mean(np.abs(pv[frame][down:top, left:right])) 
                if pv_mean > pv_max:
                    pv_max = pv_mean
                    new_i_, new_j_ = new_i, new_j
        return new_j_, new_i_, pv_max

    # a list that maintain the location of the center of the tropical cyclone 
    # [[j, i, pv]] 
    center_type= [('j', 'i4'), ('i', 'i4'), ('pv', 'f4')] # inhomogenous array
    center_trajectory = np.zeros(shape=(len(u), ), dtype=center_type)
    center_trajectory[0] = update_cyclone_center(j, i, 0)
    
    lats_deg_1d, lons_deg_1d = np.degrees(lats), np.degrees(lons)
    
    scatter = ax.scatter(lons_deg_1d[center_trajectory[0]['i']], \
                         lats_deg_1d[center_trajectory[0]['j']],  \
                         marker='*', color='green', s=30, \
                         transform=ccrs.PlateCarree())
   
    center_trajectory_plot, = ax.plot(lons_deg_1d[center_trajectory[0]['i']], \
                                      lats_deg_1d[center_trajectory[0]['j']], \
                                      color='green', \
                                      transform=ccrs.PlateCarree())
    
    lost_track = False
    lost_track_step_number = 0
    disappear_threshold = 0.005 * time_coarsen_factor 
    def animate(i):
        nonlocal lost_track, lost_track_step_number, disappear_threshold
        progress.update(time_coarsen_factor)
        U=u[i][::sp_coarsen_factor,::sp_coarsen_factor]
        V=v[i][::sp_coarsen_factor,::sp_coarsen_factor]
        scalar.set_array(h[i].flatten())
        quiver.set_UVC(U=U, V=V)
        if i >= 1:
            center_trajectory[i] = update_cyclone_center(center_trajectory[i-1]['j'], center_trajectory[i-1]['i'], i)
            # print(f"at time step {i}, the last center has state {center_trajectory[i-1]}, \n the next has state {center_trajectory[i]} \n \n")
            if lost_track is False:
                if ((center_trajectory[i]['pv'] - center_trajectory[i-1]['pv'])/center_trajectory[i-1]['pv']) < disappear_threshold: # changes by less than threshold% percent
                    current_lon = lons_deg_1d[center_trajectory[i]['i']]
                    current_lat = lats_deg_1d[center_trajectory[i]['j']]
                    scatter.set_offsets(np.c_[current_lon, current_lat])
                    center_trajectory_plot.set_ydata(lats_deg_1d[center_trajectory[:i+1]['j']])
                    center_trajectory_plot.set_xdata(lons_deg_1d[center_trajectory[:i+1]['i']])
                else: 
                    lost_track = True
                    lost_track_step_number = i
                

        ax.set_title(_frame_title("Tracking Tropical Cyclone", i, time_coarsen_factor, sec_per_frame))
                    
    
    ani = animation.FuncAnimation(fig, animate, len(u))
    path = output_dir + "/" + file_name + ".mp4"
    ani.save(path)
    print(f"✅ Finished TC tracking Visualization ✅  \n\n Saved to -----> {path}")
    if lost_track is True:
        print(f"\n 🫥 🫥 At step {lost_track_step_number} , The 🌀 disappeared  or we lost track of it 🫥 🫥 \n")

def animate_spectrum(data, 
                     output_dir,
                     time_coarsen_factor=2):
    os.makedirs(output_dir, exist_ok=True)

    solver, trajectory, lats, lons, sec_per_frame = _common_setup(data)

    pbar = tqdm.tqdm(total=len(trajectory)//time_coarsen_factor, desc="Visualizating Energy Spectrum")
    title = "Energy Spectrum"
    fig, ax = plt.subplots()
    ax.set_xlabel(r"Total Spatial Frequency $l$ ")
    ax.set_ylabel(r"Enstropy $\int |\zeta|^2$ ")
    ax.set_title(_frame_title("Energy Spectrum", 0, time_coarsen_factor, sec_per_frame))
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')
    
    def compute_enstropy(t):
        vort_spec = trajectory[t][1]
        # apply parsevel's theorem to compute total energy
        enstropy = 2 * np.sum((vort_spec[:, 1:].abs().numpy())**2, axis=1)  + (vort_spec[:, 0].abs().numpy())**2
        return enstropy
    line, = ax.loglog(np.logspace(1, np.log10(solver.lmax), int((solver.lmax))), compute_enstropy(0))

    def animate(i):
        pbar.update(1)
        line.set_ydata(compute_enstropy(i * time_coarsen_factor))
        ax.set_title(_frame_title("Energy Spectrum", i, time_coarsen_factor, sec_per_frame))
        
    ani = animation.FuncAnimation(fig, animate, len(trajectory)//time_coarsen_factor, interval=50)
    ani.save(f"{output_dir}/energy_spectrum.mp4")