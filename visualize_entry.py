from src.visualization.visualization import (
    animate_swe_on_box,
    animate_swe_on_sphere,
    visualize_initial_condition_on_sphere,
    animate_uv_quiver_on_box,
    _common_setup,
    track_tropical_cyclone,
    animate_spectrum
)
import argparse
import torch

parser = argparse.ArgumentParser(
                    prog='PlotSWE',
                    description='Plot the solution of SWE on sphere simulation')

parser.add_argument("--mode",
                    type=str,
                    help='what to visulize; either ic(picture), trajectory(animation)), vorticity_spectrum(animation) or cyclone_tracking(animation)'
                    )

parser.add_argument("-p",
                    "--projection", 
                    type=str,
                    help='projection used to visualize: box, sphere '
                         '(render the initial condition, i.e. the first saved frame, on three spheres)')

parser.add_argument("--path",
                    type=str,
                    help='path to the data'
                    )

parser.add_argument("--pv",
                    action='store_true',
                    help='visualize potential vorticity if this flag is provided'
                    )

parser.add_argument("--h",
                    action='store_true',
                    help='visualize height if this flag is provided'
                    )

parser.add_argument("--uv",
                    action='store_true',
                    help='visualize uv windfield(quiver animation) if this flag is provided'
                    )

parser.add_argument("--energy_spectrum",
                    action='store_true',
                    help='visualize the enstropy-order distribution if this flag is provided'
                    )

parser.add_argument("--output_dir",
                    type=str,
                    help="directory where the visualization output will be stored"
                    )

parser.add_argument("--coarsen_factor",
                    default=1,
                    type=int,
                    help='Coarses sampling rate during plot to speed up'
                    )

parser.add_argument("--coastline",
                    action='store_true',
                    help='Overlay coastlines using a cartopy PlateCarree map for box '
                         'animations (otherwise a plain imshow lon/lat box is used)'
                    )

parser.add_argument("--initial_location",
                    type=float,
                    nargs=2,
                    help='If mode is cyclone_tracking, this provides the approximate initial location of th cyclnoe \
                    syntax : (lat, lon)'
                    )

args = parser.parse_args()

mode, projection, path, coarsen_factor, output_dir = args.mode, args.projection, args.path, args.coarsen_factor, args.output_dir

pv, h, uv, energy_spectrum = args.pv, args.h, args.uv, args.energy_spectrum

coastline = args.coastline

initial_location = args.initial_location

try:
    data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
except (RuntimeError, ValueError):
    data = torch.load(path, map_location='cpu', weights_only=False)

if mode == 'trajectory':
    # mmap=True memory-maps the (multi-GB) trajectory instead of reading it all into
    # RAM up front; with temporal striding only the frames actually reconstructed get
    # paged in. Falls back to a normal load for older archives that can't be mmap'd.
    vars = []
    if pv is True:
        vars.append('pv')
    if h is True:
        vars.append('h')
    if projection == 'box':
        animate_swe_on_box(
            data=data,
            variables=vars,
            output_dir=output_dir,
            coarsen_factor=coarsen_factor,
            coastline=coastline
        )
    if uv is True:
        if projection == 'box':
            animate_uv_quiver_on_box(data, 
                                    output_dir=output_dir,
                                    time_coarsen_factor=coarsen_factor)
    if energy_spectrum is True:
        animate_spectrum(
            data=data,
            output_dir=output_dir,
            time_coarsen_factor=coarsen_factor
        )
    elif projection == 'sphere':
        animate_swe_on_sphere(
            data=data,
            variables=vars,
            output_dir=output_dir,
            coarsen_factor=coarsen_factor
        )
elif mode == 'ic':
    if projection == 'sphere_ic':
        solver, trajectory, _, _, _ = _common_setup(data)

        visualize_initial_condition_on_sphere(
            model=solver,
            # the "initial condition" is the trajectory's first spectral frame
            initial_condition=lambda model: trajectory[0],
            plot_title="Initial Condition",
            output_dir=output_dir,
        )
elif mode == 'cyclone_tracking':
    if initial_location is None:
        parser.error("if mode is cyclone_tracking, must provide an initial location of the cyclone")

    track_tropical_cyclone(
        data,
        initial_loc=initial_location,
        time_coarsen_factor=coarsen_factor,
        output_dir=output_dir
    )



        
        
