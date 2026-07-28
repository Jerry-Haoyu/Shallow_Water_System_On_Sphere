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

#### File name
The file name mirrors the path. For example, the model checkpoint in `resol_64/tau_[30000,30000,30]/grid_eq/method_implicit` would be
`64_[30000,30000,30]_eq_implicit.pt`

### Numerical Data
Numerical data follows the same tree strucutre as above, under the last node, we classfies the output data by 

1. Duration in days, e.g. `duration_20/` would mean a simulation of 20 days 
2. Initial condition type, e.g. `ic_rw/` would mean real-world data
    - Under real world data, we classify by pressure level and then by time :
        - `pressure_500/` would mean 500hPa, `pressure_(100,1000)` would mean a vertically integration from 1000hPa to 100hPa. 
        - `time_2000-07-01-000000/` would mean the initial condition is the snapshot taken on 2000-07-01 at 00:00:00 
#### File names
The data files have names that mirror their path,
ex. the data file in `resol_64/tau_[30000,30000,30]/grid_eq/method_implicit/duration_20/ic_rw/pressure_500/time_2000-07-01-00000` would be
``64_[30000,30000,30]_eq_implicit_20_rw_500_2000-07-01-000000.pt``

### Neural Operator Model
The neural operator model and data follows similar organization. However, the exact nodes are different. 
1. Classifes resolution

2. Prediction interval. For instance`nfuture_1/` means the model predicts the frame in next $1\cdot \Delta t$. 
3. Classifies number of layers, i.e., `nlayer_4/` would contain models with 4 SFNO layers 
4. Classifies embedded dimension, `ebd_16/` would contain all models with 16 channels. 
5. Classifies the training data used, the training data(see the next section) is named after the ERA5 data that is used as initial conditions. For example, `trainData_(1980_2025_odd_month_500)` would mean the model is trained using simulation data generated from `1980_2025_odd_month_500.nc
6. Classifes positional embedding.  
7. Classifies grid

Unlike the numerical model checkpoints, under each directory, we have two files:
1. `model_info.json`
2. The actual checkpoint

The json file contains the model configuration(architecture) and training configuration. 

Also, since we might be training the same model for training config like different learning rate scheudling etc., subdivide the directory into `0/`, `1/`, `2/` for different training configurations. 

#### File name 
The file name mirrors the path like with numerical model, but adds `_single.pt` or `_multi.pt` at the end. For example, 
```
nfuture_1/nlayer_4/embd_16/trainData_(1980_2025_odd_month_500hPa)/0
```

### Neural Operator Data 
The inference generated by neural operator model is stored under `model_output/neural_operator/` and follows the same organization as numerical data.That is, `duration_*/ic_*/pressure_*/time_*`. 

#### File name 
Same as the numerical data case


## Trainig data
An exception exists for the training data for SFNO, they exists in `model-output/numerical/training_data` and has a flat strucutre. They follow the same naming convention 



