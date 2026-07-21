#!/data/gzhang13/a/hytang2/envs/swe/bin/python
from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from src.numerical_solver.initial_condition import *
import argparse 
import torch

parser = argparse.ArgumentParser(
                    prog='solve',
                    description='Solve the SWE on the sphere using psuedospectral method')

parser.add_argument("--saved_model_path",
                    type=str,
                    help=(" If this flag is provided, run the model using saved .pt file and don't need configuration parameters")
                    )

parser.add_argument("--lmax", 
                    type=int,
                    help='\n The resolution knob, specifically the largest degree allowed in the spectral representation \n')

parser.add_argument("--tau", 
                    type=int,
                    nargs=3,
                    help='The characteristic time(e-fold time) for viscosity damping, the tuple (𝛕2, 𝛕4, 𝛕₈) is for n=1 and n=2 and n=4 respectively \n')

parser.add_argument("-g", 
                    "--grid", 
                    type=str,
                    default='legendre-gauss',
                    help="One of equiangular, legendre-gauss, or lobatto \n")

parser.add_argument("--cfl", 
                    type=float,
                    default=0.25,
                    help="The percentage of grid that can be traversed by gravity wave c=√hg per dt, smaller for safer numerical margin \n")

parser.add_argument("--semi_implicit", 
                    action='store_true',
                    help=" If passed, use semi-implicit method, i.e. gravity wave terms is approximated by average at t-1 adn t+1 \n")

parser.add_argument("--dealias", 
                    action='store_true',
                    help=" If passed, use a padded grid(truncation is extended to 1.5l_max) for quadratic product(non-linear terms) \n")

parser.add_argument("-d",
                    "--duration",
                    type=float,
                    default=5,
                    help=' How many days to simulate \n')

parser.add_argument("--save_interval_minutes",
                    type=float,
                    default=30,
                    help=' Simulated-time interval (minutes) between saved trajectory frames \n')

parser.add_argument("-o", 
                    "--output_dir", 
                    type=str,
                    default='output',
                    help=" The directory to save the output \n")

parser.add_argument("--ic",
                    type=str,
                    default='galewsky',
                    help=" The initial condition type, either galewsky or real_world \n"
                    )

parser.add_argument("--netcdf_path",
                    type=str,
                    default=None,
                    help=" The path to ERA5 netcdf data \n"
                    )

parser.add_argument("--ic_time",
                    type=str,
                    default=None,
                    help=" In the case of real world IC, the time we chose to generate ic \n"
                    )

args = parser.parse_args()

if args.ic == 'real_world' :
    if not args.ic_time:
        parser.error("--ic_time which specifies the year-month-day-hour used to generate initial condition is required when --ic is real-world.")
    if not args.netcdf_path:
        parser.error("--netcdf_path which specifies the path to the netcdf file for generating initial condition is required when --ic is real-world.")

# From model.pt or rebuild new model from configuration?
saved_model_path = args.saved_model_path

# parse initial condition
ic, ic_time, netcdf_path = args.ic, args.ic_time, args.netcdf_path

# parse solver configuration
lmax, tau, cfl, grid, semi_implicit, dealias, duration, output_dir = \
    args.lmax, args.tau, args.cfl, args.grid, args.semi_implicit, \
        args.dealias, args.duration, args.output_dir


###### Automatically name the file ###########

# naming rule = (l_max)_(tau)_(duration)_{grid_type}_{implicity}_{apply_dealis_or_not}_{initial_condition_type}
grid_shorthand = {
    'legendre-gauss' : 'lg',
    'equiangular' : 'eq',
    'lobatto' : 'lb'
}

solver_implicity = 'implicit' if semi_implicit is True else 'explicit'

# ic_time may be a full ISO timestamp (e.g. 2000-01-01T00:00:00); strip the ':' and
# 'T' so the auto-generated output filename stays filesystem-friendly.
ic_time_tag = ic_time.replace(':', '').replace('T', '-') if ic_time else ic_time

initial_conidtion_shorthand = {
    'galewsky' : 'glsky',
    'real_world' : f'rw_{ic_time_tag}'
}

if dealias is True:
    file_name = f'{lmax}_{tau}_{duration}_{grid_shorthand[grid]}_{solver_implicity}_dealis_{initial_conidtion_shorthand[ic]}.pt'
else:
    file_name = f'{lmax}_{tau}_{duration}_{grid_shorthand[grid]}_{solver_implicity}_{initial_conidtion_shorthand[ic]}.pt'
###############################################

# From model.pt 
if saved_model_path is not None:
    solver = torch.load(saved_model_path, weights_only=False)
# rebuild new model from configuration?
else:
    solver = ShallowWaterSolver(lmax, tau, cfl, grid=grid, dealias=dealias, semi_implicit=semi_implicit)

solver.to(solver.device) 

if ic == 'galewsky':
    phivrtdivspec_0 = galewsky_initial_condition(model=solver)
elif ic == 'real_world':
    phivrtdivspec_0 = rw_initial_condition(model=solver, netcdf_path=netcdf_path, ic_time=ic_time)

    
solver.run(initial_condition=phivrtdivspec_0, days=duration, output_dir=output_dir, file_name=file_name,
           save_interval_minutes=args.save_interval_minutes)

