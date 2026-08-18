"""
Diagnostics computed from ERA5-derived real-world initial conditions.

@author Haoyu Tang hytang2@illinois.edu and Claude
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import tqdm
import xarray as xr
from pathlib import Path
import time
import matplotlib.pyplot as plt

from src.numerical_solver.psuedo_spectral_solver_naive import ShallowWaterSolver
from torch_harmonics.sht import RealVectorSHT
from src.numerical_solver.initial_condition import rw_initial_condition

def compute_h_stats_ic(netcdf_path, 
                       nlat_data=721, 
                       nlon_data=1440, 
                       lmax_model=64, 
                       grid="equiangular", 
                       dealias=False):
    """
    For every time point in `netcdf_path`, reconstruct the balanced geopotential
    spectrum via rw_initial_condition (src.numerical_solver.initial_condition,
    which solves the balance equation for the geopotential from u, v) and reduce
    it to two domain-wide diagnostics of the height field h = ɸ/g:
        h_avg : area-weighted mean height
        h_amp : area-weighted RMS deviation of height around h_avg

    Both are read directly off the spectral coefficients (no inverse transform):
    with orthonormal real spherical harmonics, the l=0,m=0 coefficient is the
    domain mean up to a factor of sqrt(4*pi), and Parseval's theorem gives the
    mean-square directly from the coefficient magnitudes (m=0 counted once,
    m>0 counted twice for its +-m conjugate pair) -- so h_amp is exact, not a
    quadrature approximation, and is much cheaper than reconstructing the grid.

    Args:
        netcdf_path : path to an ERA5-style netcdf file with a 'valid_time'
            coordinate and u/v data variables (as consumed by rw_initial_condition)
        nlat_data, nlon_data : the shape of the ERA5 dataset
        lmax_model, grid, dealias : solver resolution/config used to reconstruct the
            balanced fields (see ShallowWaterSolver)

    Returns:
        h_avg, h_amp, year : 1-D numpy arrays, one entry per time point in
            `netcdf_path`'s 'valid_time' coordinate, in file order.
    """
    start_time = time.perf_counter()
    # this function derives h_avg/h_amp themselves, so the solver used to
    # reconstruct the balanced geopotential must stay in physical units.
    model = ShallowWaterSolver(lmax=lmax_model, grid=grid, dealias=dealias, non_dimensional=False)
    model.to(model.device)
    
    vSHT = RealVectorSHT(nlat=nlat_data, nlon=nlon_data, lmax=nlat_data// 2, mmax=nlat_data // 2,
                        grid='equiangular', csphase=False).to(model.device)
    
    start_load_time=time.perf_counter()
    era5_dataset = xr.open_dataset(netcdf_path).load()
    finish_load_time=time.perf_counter()
    print(f"Finished loading data in {(finish_load_time - start_load_time):2f} seconds")

    times = np.atleast_1d(era5_dataset["valid_time"].values)

    sqrt_4pi = float(4.0 * np.pi) ** 0.5

    h_avg = np.empty(len(times), dtype=np.float64)
    h_amp = np.empty(len(times), dtype=np.float64)
    year = np.empty(len(times), dtype=np.int32)
    

    with torch.no_grad():
        for i, t in enumerate(tqdm.tqdm(times, desc="Computing h_avg/h_amp")):
            if i % 10 == 0:
                curr_time=time.perf_counter()
                if (curr_time-start_time) > 300:
                    print("Taking too long(> 300 s), killing the job")
                    break 
            ic_time = str(np.datetime_as_string(t, unit="s"))
            uspec = rw_initial_condition(model, vSHT, era5_dataset, ic_time, balanced=True)
            phispec = uspec[0]  # (lmax, mmax) spectral geopotential

            phi_mean = phispec[0, 0].real / sqrt_4pi
            phi_meansq = (phispec[:, 0].abs() ** 2).sum() + 2.0 * (phispec[:, 1:].abs() ** 2).sum()
            phi_meansq = phi_meansq / (4.0 * np.pi)
            phi_var = phi_meansq - phi_mean ** 2

            h_avg[i] = (phi_mean / model.gravity).item()
            h_amp[i] = (torch.sqrt(phi_var) / model.gravity).item()
            year[i] = t.astype("datetime64[Y]").astype(int) + 1970

    return h_avg, h_amp, year



def plot_h_stats_ic(data_name):
    netcdf_path = f"reanalysis_data/{data_name}/data.nc"
    stats_path = f"reanalysis_data/{data_name}/h_stats.npz"
    fig_path = f"reanalysis_data/{data_name}/h_stats_plot.png"
    h_avgs, h_amps, date = compute_h_stats_ic(netcdf_path)
    
    time_averaged_stat = np.array([h_avgs.mean(), h_amps.mean()])
    np.savez(stats_path, h_avgs=h_avgs,h_amps=h_amps, dates=date, mean=time_averaged_stat)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\mu_h$ (m)")
    ax.set_ylabel(r"$\sigma_h$ (m)")
    ax.set_title(f" Nondimesionalization Stats For {data_name}")
    
    ax.scatter(h_avgs, h_amps, c=date, cmap='viridis')
    ax.scatter(time_averaged_stat[0], time_averaged_stat[1], c='r', marker='*')
    fig.savefig(fig_path)


if __name__ == "__main__":
    # data_name = "1980_2025_odd_month_500"
    data_name = "1995_2025_(1,7,11)_(1-7)_500"
    plot_h_stats_ic(data_name)