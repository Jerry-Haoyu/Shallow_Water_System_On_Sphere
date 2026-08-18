# Shallow Water System On the Sphere
## What is this about?
This is a repository to solve the shallow water equation both numerically and in the data-driven way. Specifically, the numerical side is consisted of the *psuedospectral solver* while the ML side we have the *Spherical Fourier Neural Operator*. SFNO based architecture was proved powerful as a prognostic model for real world data. This repo provides some more means of analysis and benchmarks on the idealistic models side. 

## Program Entries 
This repo is wired so that it is like a crude piece of software that runs on clusters with GPUs. Different tasks are organized into a $\textcolor{Orange}{\textbf{Makefile}}$. For instance, to train SFNO single-step(without multi-step fine-tuning), just type `make train_single`. To specify configurations like training dataset path, learning rate, where to log etc., modify attributes of corresponding task in the $\textcolor{Purple}{\text{config.yml}}$  file.In the `train_single` case, this looks like:

```
train:
  training_data_dir: //path of the directory that contains all traing data
  batch_size: 128
  lr: 1e-3
  n_future: 1  
  no_compile: True
  epochs: 300
  weight_decay: 0.01
  validation_cadence: 25
  warmup_epochs: 5
  warmup_start_factor: 0.001 
  restart: True
  restart_period: 30
  restart_mult: 2
  compile: False

```

Once a single-step model is trained, `make train_multi` fine-tunes it with teacher/student multi-step curriculum
training (see `multi_step_curriculum_training.png`): a frozen teacher (a snapshot of the student at the start of
each curriculum stage) is rolled out autoregressively to provide the student's input at increasingly long horizons,
while a single-step loss term keeps the student anchored to the ground truth. Its `train_multi` config section
identifies the pretrained run to build on the same way `inference` does (`nlat`, `n_future`, `num_layers`, ...,
`index`), and writes `checkpoints_multi.pt` alongside that run's `checkpoints_single.pt` (see Neural Operator Model
below) rather than starting a new run directory.

The $\textcolor{Orange}{\textbf{Makefile}}$ would contain a variety [not implemented yet] of different tasks spanning across benchmarking models, visualization. Please see the $\textcolor{Orange}{\textbf{Makefile}}$ and $\textcolor{Purple}{\text{config.yml}}$ for more detail.

## Repo Structure : Model and Data Classification
> To experiment with all kinds of different set ups, it is convenient to come up with a naming convention.


The models are stored in `checkpoints\` and the model output is stored in `model_output\`. They share the same tree structure:
### Numerical Models 
Numerical models are classified by their configurations. From root to leaves:

1. The first layer classifies resolution, e.g. `resol_64`, `resol_128` ...
2. The second layer classifies viscosity, e.g. `tau_[30000,30000,30]`. Since the damping term is:
$$
\sum_{n=2,4,8}\frac{1}{\tau_n}(\nabla ^2 \mathbf{u})^n
$$ 
`tau_[30000,30000,30]` correspond to $\tau_2=30000$(no damping at this scale), $\tau_4=30000$ and $\tau_8=30$.

3. The third layer classifies the grid, e.g. `grid_eq` means we use the equiangular grid. Other options include `legendre-gauss`
4. The fourth layer classifies the numerical engine used, e.g. `method_implicit` means we use the semi-implicit method(leap-frog for gravity waves)
5. The fifth layer classifies radiative relaxation of geopotential (see Radiative Relaxation below): `radiation_rad`
   or `radiation_no_rad`. Unlike the `dataset_<name>` layer below, this node is always present (never omitted) so
   `rad` and `no_rad` checkpoints/data never collide.
6. **Real-world runs only:** a sixth layer `dataset_<name>`, e.g. `dataset_2026_07_500`. Real-world initial conditions
   pull the dataset-wide reference height/amplitude (`h_avg`, `h_amp`, from `reanalysis_data/<name>/h_stats.npz`)
   into the solver's own constants (see Non-dimensionalization below), so the checkpoint is only valid for that one
   dataset - this node keeps checkpoints from different datasets from colliding. Galewsky runs have no dataset, so
   this node is omitted for them.

#### File name
Since the directory tree already encodes the full configuration, the file name is fixed:
`checkpoints.pt`. For example, the model checkpoint in `resol_64/tau_[30000,30000,30]/grid_eq/method_implicit/radiation_no_rad` is
`resol_64/tau_[30000,30000,30]/grid_eq/method_implicit/radiation_no_rad/checkpoints.pt`

### Non-dimensionalization
`ShallowWaterSolver` runs non-dimensional by default (`non_dimensional=True`). Given a length scale
`L = radius`, gravity-wave speed `U = sqrt(g*h_avg)` and time scale `T = L/U`, the solver's own constants
(`radius`, `gravity`, `havg`, `hamp`, `omega`, `umax`) are rescaled once at construction time; since the SWE
tendency equations are scale-covariant under this rescaling, the rest of the solver (CFL/dt, hyperdiffusion,
Coriolis, `dudtspec`, the leapfrog stepper) runs unchanged on the resulting non-dimensional state. For real-world
runs, `h_avg`/`h_amp` come from the dataset's `h_stats.npz` (see below); for galewsky they fall back to the
built-in 10 km / 120 m reference. Pass `non_dimensional=False` to run fully dimensional (physical units).

### Radiative Relaxation
`ShallowWaterSolver` can optionally relax geopotential toward a zonally-symmetric equilibrium background
`Phi_eq(lat)`, mimicking radiative heating/cooling, via a Newtonian relaxation term added to the mass-continuity
equation:
$$
\frac{\partial \Phi}{\partial t} \mathrel{+}= -\frac{\Phi - \Phi_{eq}}{\tau_{rad}}
$$
Enabled with `rad=True, tau_rad=<days>` (config.yml: `rad`/`tau_rad` under `run_solver`/`batch_simulation`), and
only supported for `ic == real_world` (it needs an ERA5 dataset to derive `Phi_eq` from). `Phi_eq` itself is built
by `src.numerical_solver.initial_condition.radiative_equilibrium_geopotential`:

1. the day-of-year climatological (mean across years) zonal-mean `u` field from the full multi-year ERA5 dataset,
   at the run's `ic_time` calendar day - the run duration (15-20 days) is short enough that this single
   climatological profile stands in for "equilibrium" over the whole run;
2. a stronger-than-model spectral truncation of that meridional profile (`rad_smooth_fraction` of the model's own
   `lmax`, default `0.5`), for a smooth background;
3. the same nonlinear balance equation used elsewhere in this file (`initial_condition._solve_balance_geopotential`,
   $\nabla^2 \Phi = \nabla \cdot [\mathbf{u}(\zeta+f)] - \nabla^2 K$) applied to the smoothed, purely-zonal `u_eq`
   (`v_eq=0`).

Because `Phi_eq` is IC/dataset/day-of-year specific rather than solver-architecture state, it is not a constructor
argument and is not part of the persisted checkpoint - it's set on the solver (via
`solver.set_equilibrium_geopotential(...)`) fresh at the start of every run, right after the initial condition
itself is generated. Whether a checkpoint is `rad` or `no_rad` (and, if `rad`, its `tau_rad`) *is* part of the
persisted config, and is what the `radiation_rad`/`radiation_no_rad` tree node (below) distinguishes.

### Numerical Data
Numerical data follows the same tree strucutre as above, under the last node, we classfies the output data by 

1. Duration in days, e.g. `duration_20/` would mean a simulation of 20 days 
2. Initial condition type, e.g. `ic_rw/` would mean real-world data
    - Under real world data, we classify by pressure level and then by dataset:
        - `pressure_500/` would mean 500hPa, `pressure_(100,1000)` would mean a vertically integration from 1000hPa to 100hPa. 
        - `dataset_<name>/` is the **leaf**: every `ic_time` drawn from that ERA5 dataset (e.g. `1980_2025_odd_month`)
          lands in this same directory, one file per ic_time (see File names below) - so the directory alone is a
          flat, poolable set of trajectories for that dataset.

Unlike the checkpoint tree (where `dataset_<name>/` sits right after `radiation_*/`, see Non-dimensionalization
above), here it sits at the very end instead of mid-tree, e.g. the full tree reads
`.../method_implicit/radiation_no_rad/duration_20/ic_rw/pressure_500/dataset_<name>/`. This is deliberate: pointing
`train_single`'s `training_data_dir` straight at a `dataset_<name>/` leaf is how SFNO training data is collected
(see [Training data](#trainig-data) below) - no separate flat-copy step needed. `run_solver.py` delegates deducing
this path, and reusing an already-generated trajectory, to `run()` (src/helpers/run_model.py) - see Program Entries
above.

#### File names
Galewsky runs: the directory tree already pins down the full configuration, so the file name is fixed:
`model_output.pt`. ex. the data file in `resol_64/tau_[30000,30000,30]/grid_eq/method_implicit/radiation_no_rad/duration_20/ic_galewsky` is
``resol_64/tau_[30000,30000,30]/grid_eq/method_implicit/radiation_no_rad/duration_20/ic_galewsky/model_output.pt``

Real-world (`ic_rw`) runs: since many ic_times share one `dataset_<name>/` leaf directory, the fixed
`model_output.pt` name would collide, so the file name instead encodes the date (day resolution) of that run's
`ic_time`, e.g. the data file for `ic_time = 2026-07-05T00:00:00` in
`.../radiation_no_rad/duration_20/ic_rw/pressure_500/dataset_2026_07_500/` is
``resol_64/tau_[30000,30000,30]/grid_eq/method_implicit/radiation_no_rad/duration_20/ic_rw/pressure_500/dataset_2026_07_500/2026_07_05.pt``

Alongside each `.pt` file, `run_solver.py` also writes `<stem>_diagnostics.png` in the same directory: a line
plot of the spatial-average (mean +/- amplitude band) of geopotential, vorticity and divergence over the
trajectory, computed straight from each saved frame's `l=0,m=0` spherical-harmonic coefficient (the domain mean)
and Parseval's theorem (the RMS deviation) - not by reconstructing the grid and integrating.

### Neural Operator Model
The neural operator model and data follows similar organization. However, the exact nodes are different. 
1. Classifes resolution

2. Prediction interval. For instance`nfuture_1/` means the model predicts the frame in next $1\cdot \Delta t$. 
3. Classifies number of layers, i.e., `nlayer_4/` would contain models with 4 SFNO layers 
4. Classifies embedded dimension, `ebd_16/` would contain all models with 16 channels. 
5. Classifies the training data used, the training data(see the next section) is named after the ERA5 data that is used as initial conditions. For example, `trainData_(1980_2025_odd_month_500)` would mean the model is trained using simulation data generated from `1980_2025_odd_month_500.nc
6. Classifes positional embedding.  
7. Classifies grid
8. Classifies the SFNO's own normalization layer, `norm_none` | `norm_layer` | `norm_instance` (none | layer_norm | instance_norm).
9. Classifies which loss function trained the run, `loss_grid` | `loss_spectral` (see `src/neural_operator/loss.py`'s `LOSS_FUNCTIONS`).

Unlike the numerical model checkpoints, under each directory, we have two files:
1. `model_info.json`
2. The actual checkpoint

The json file contains the model configuration(architecture) and training configuration. 

Also, since we might be training the same model for training config like different learning rate scheudling etc., subdivide the directory into `0/`, `1/`, `2/` for different training configurations. 

#### File name 
As with the numerical checkpoint, the file name is fixed - `checkpoints_single.pt` or `checkpoints_multi.pt` since both files live side by side in the same directory. For example, 
```
nfuture_1/nlayer_4/embd_16/trainData_(1980_2025_odd_month_500hPa)/norm_none/loss_spectral/0/checkpoints_single.pt
```

### Neural Operator Data 
The inference generated by neural operator model is stored under `model_output/neural_operator/` and follows the same organization as numerical data: `duration_*/ic_*/pressure_*/dataset_*`, with `dataset_<name>/` as the leaf for real-world initial conditions - same reasoning as the numerical case above.

#### File name 
Galewsky: fixed at `model_output.pt`, same as the numerical case. Real-world: date-only name (e.g. `2026_07_05.pt`), same exception as the numerical case - this keeps the file unique within the shared `dataset_<name>/` leaf, so re-running inference across many ic_times from the same dataset pools into one directory (and lets `inference.py` skip an ic_time it already has, via `run()`'s reuse) instead of scattering one file per timestamp.


## Trainig data
SFNO training data is just numerical `model_output` for real-world initial conditions - no separate location or
flattening step. Point `train_single`'s `training_data_dir` config directly at the `dataset_<name>/` leaf directory
(see Numerical Data above) for whichever (resol, tau, grid, method, rad, duration, ic_rw, pressure) configuration and
ERA5 dataset you want to train on, e.g.
`model_output/numerical/resol_64/tau_[30000,30000,30]/grid_eq/method_implicit/radiation_no_rad/duration_20/ic_rw/pressure_500/dataset_1980_2025_odd_month/`.
Generating that data is just running `run_solver.py` (directly, or via `batch_simulation.py` across many ic_times)
- since `run_solver.py` delegates trajectory-path deduction and reuse to `run()` (src/helpers/run_model.py),
re-running/resubmitting jobs for ic_times that already have a trajectory there is a cheap no-op rather than redoing
the simulation.

## Reanalysis Data
ERA5 downloads (`make download_era5`) are identified downstream by a **dataset name** (e.g. `1980_2025_odd_month`)
rather than a raw `.nc` path - every task that consumes real-world data (`run_solver`, `batch_simulation`,
`inference`) takes this name via a `dataset_name` config key and resolves it itself. Each dataset gets its own
directory under `reanalysis_data/`:

```
reanalysis_data/<dataset_name>/
    data.nc            the ERA5 netCDF (u, v on pressure levels)
    h_stats.npz         per-time h_avg / h_amp (arr_0 / arr_1), computed by src/analyze/statistics.py
    h_stats_plot.png     scatter of h_amp vs h_avg over time, same script
```

`h_stats.npz` is written by `download_era5.py` right after the download (via
`src.analyze.statistics.plot_h_stats_ic`). Its time-averaged `h_avg`/`h_amp` are what real-world `run_solver.py`
runs load to non-dimensionalize the solver (see Non-dimensionalization above).



