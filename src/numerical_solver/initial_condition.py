import torch
from torch_harmonics.sht import *
import xarray as xr
import pandas as pd
import time


def _solve_balance_geopotential(model, uv_grid, height_field):
    """
    Solve the (nonlinear) balance equation for geopotential given a wind field
    `uv_grid` (2, nlat, nlon) already on `model`'s own physical grid:
        $$\\nabla^2 \\Phi = \\nabla \\cdot \\left[ \\mathbf{u}(\\zeta + f) \\right] - \\nabla^2 K$$
    `height_field` is the background height (m, or model units) added as
    `g*height_field`; e.g. `model.havg + hbump` (galewsky) or a plain
    `model.havg.expand(model.nlat, model.nlon)` (real-world / equilibrium).

    Shared by galewsky_initial_condition, rw_initial_condition and
    radiative_equilibrium_geopotential, which all solve this same equation for
    different wind fields.
    """
    vrtdivspec = model.vrtdivspec(uv_grid)
    vrtdivgrid = model.spec2grid(vrtdivspec)
    A_spec = model.vrtdivspec(uv_grid * (vrtdivgrid[0] + model.coriolis))
    kinetic_energy = model.grid2spec(0.5 * torch.sum(uv_grid ** 2, dim=0))
    phispec = model.invlap * A_spec[0] - kinetic_energy + model.grid2spec(model.gravity * height_field)
    return phispec


# the function has been modified to accept more parameters for generating diverse initial conditions
def galewsky_initial_condition(model,
                                umax = 80., 
                               usouth = 1/7, 
                               unorth = 5/14, 
                               perturb_loc=0.25, 
                               perturb_amp=1., 
                               noise_level=1):
    """
    Initializes non-linear barotropically unstable shallow water test case of Galewsky et al. (2004, Tellus, 56A, 429-440).

    [1] Galewsky; An initial-value problem for testing numerical models of the global shallow-water equations;
        DOI: 10.1111/j.1600-0870.2004.00071.x; http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
    """
    device = model.lap.device

    # umax/noise_level are physical constants (m/s, m); rescale into the model's
    # own units so a non-dimensional model gets a non-dimensional IC.
    if getattr(model, 'non_dimensional', False):
        umax = umax / model.U
        noise_level = noise_level / model.havg_phys

    phi0 = torch.asarray(torch.pi * usouth, device=device)
    phi1 = torch.asarray(torch.pi * unorth, device=device)
    phi2 = perturb_loc * torch.pi
    en = torch.exp(torch.asarray(-4.0 / (phi1 - phi0)**2, device=device))
    alpha = 1. / 3.
    beta = 1. / 15.

    lats, lons = torch.meshgrid(model.lats, model.lons)

    u1 = (umax/en)*torch.exp(1./((lats-phi0)*(lats-phi1)))
    ugrid = torch.where(torch.logical_and(lats < phi1, lats > phi0), u1, torch.zeros(model.nlat, model.nlon, device=device))
    vgrid = torch.zeros((model.nlat, model.nlon), device=device)
    noise = noise_level * torch.randn(model.nlat, model.nlon, device=device)
    hbump = noise + model.hamp * perturb_amp * torch.cos(lats) * torch.exp(-((lons-torch.pi)/alpha)**2) * torch.exp(-(phi2-lats)**2/beta)

    # intial velocity field
    ugrid = torch.stack((ugrid, vgrid))
    # intial vorticity/divergence field
    vrtdivspec = model.vrtdivspec(ugrid)

    # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
    phispec = _solve_balance_geopotential(model, ugrid, model.havg + hbump)

    # assemble solution
    uspec = torch.zeros(3, model.lmax, model.mmax, dtype=vrtdivspec.dtype, device=device)
    uspec[0] = phispec
    uspec[1:] = vrtdivspec

    return torch.tril(uspec)

def random_initial_condition(model, mach=0.1, scaler=1) -> torch.Tensor:
    """
    random initial condition on the sphere
    """
    device = model.lap.device
    ctype = torch.complex128 if model.lap.dtype == torch.float64 else torch.complex64

    # mach number relative to wave speed
    llimit = mlimit = 120

    # initial geopotential
    uspec = torch.zeros(3, model.lmax, model.mmax, dtype=ctype, device=model.lap.device)
    uspec[:, :llimit, :mlimit] = scaler * torch.sqrt(torch.tensor(4 * torch.pi / llimit / (llimit+1), device=device, dtype=ctype)) * torch.randn_like(uspec[:, :llimit, :mlimit])

    uspec[0] = model.gravity * model.hamp * uspec[0]
    uspec[0, 0, 0] += torch.sqrt(torch.tensor(4 * torch.pi, device=device, dtype=ctype)) * model.havg * model.gravity
    uspec[1:] = mach * uspec[1:] * torch.sqrt(model.gravity * model.havg) / model.radius
    
    return torch.tril(uspec)



def rw_initial_condition(model, vSHT, era5_dataset, ic_time, balanced=True, log=False):
    """
        Generate a real-world based initial using the dataset:
        "ERA5 hourly data on pressure levels from 1940 to present"
        https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
        
        Args:
            model: swe solver model
            vSHT: the RealVectorSHT object with geometry compatbile to ERA5 datset
            grib_path: the path to the grib file
            ic_time: the year-month-day-hour used to generate ic, ex."2026-07-17"
            balanced: if set to True, only extract [u,v] and compute geopotential 
            by balance equation. Else, use realworld geopotential data
    """
    if log:
        print("Computing Initial Condition....")
    start_time = time.perf_counter()
    device = model.device
    if log:
        print(f"    device is {device}")
        print(f"    Preparing data ....")
    
    # select the date. squeeze() drops any leftover length-1 dims (e.g. a
    # single-pressure-level 
    ds = era5_dataset.sel(valid_time=ic_time, method='nearest').squeeze()
    
    # match the solver's spectral precision (buffers are float64); ERA5 is float32
    dtype = model.lap.dtype
    u_data = torch.tensor(ds['u'].to_numpy(), device=device, dtype=dtype)
    v_data = torch.tensor(ds['v'].to_numpy(), device=device, dtype=dtype)

    uv_data = torch.stack([u_data,v_data], dim=0)
    # ERA5 winds are physical (m/s); rescale into the model's own units before
    # any spectral transform so a non-dimensional model gets a non-dimensional IC.
    if getattr(model, 'non_dimensional', False):
        uv_data = uv_data / model.U
    
    data_finish_time = time.perf_counter()
    if log:
        print(f"    Finished preparing data in {(data_finish_time - start_time):.2f} seconds")
    
    if model.solver_type == 'psuedo_spectral_naive':
        spec_start = time.perf_counter()
        
        # compute spectral representation
        nlat, nlon = uv_data.shape[-2], uv_data.shape[-1]
        if nlat != vSHT.nlat or nlon != vSHT.nlon:
            raise ValueError(f"❌ vSHT.nlat and vSHT.nlon imcompatible with the data shape, they should be the same." + \
                             f"netCDF file has data of shape {nlat, nlon} while vSHT has {vSHT.nlat, vSHT.nlon}")

        # truncate spectral representation
        uv_model_spec = vSHT(uv_data)
        uv_model_spec_truncated = uv_model_spec[:, :model.lmax, :model.mmax]

        # map to vrtdiv
        vrtdiv_spec = model.lap * model.radius * uv_model_spec_truncated

        # wind on the model grid: needs the *inverse vector* SHT, not the scalar
        # spec2grid (the truncated coeffs are spheroidal/toroidal potentials, not u,v).
        uv_model = model.getuv(vrtdiv_spec)
        spec_end = time.perf_counter()
        if log:
            print(f"    Finished spectral truncation in {(spec_end - spec_start):2f} seconds")
        
        # Solve the balance Equation (see _solve_balance_geopotential)
        if balanced is True:
            balance_start = time.perf_counter()
            phispec = _solve_balance_geopotential(model, uv_model, model.havg.expand(model.nlat, model.nlon))
            balance_end = time.perf_counter()
            if log:
                print(f"    Computed balanced geopotential in {(balance_end - balance_start):2f} seconds")
        
        write_start = time.perf_counter()
        phivrtdiv_spec = torch.zeros(3, 
                                     model.lmax, 
                                     model.mmax, 
                                     dtype=vrtdiv_spec.dtype, 
                                     device=device)
        
        # print(f"shape of phispec is {phispec.shape}")
        # print(f"shape of vrtdiv_spec is {vrtdiv_spec.shape}")
        phivrtdiv_spec[0] = phispec 
        phivrtdiv_spec[1:] = vrtdiv_spec
        all_end_time = time.perf_counter()
        if log:
            print(f"    finished writing data in {(all_end_time - write_start):2f}")
            print(f"finished computing initial condition in {(all_end_time-start_time):.2f} seconds")
        return torch.tril(phivrtdiv_spec)


def day_of_year_climatology(era5_dataset):
    """Day-of-year climatology (mean across years) of `era5_dataset` - the expensive
    step behind radiative_equilibrium_geopotential's phi_eq. Factored out so callers
    that need many day-of-year slices (e.g. acc.py's per-trajectory ACC loop, which
    also uses the same climatology for its baseline fields) compute this once and
    reuse it, rather than every radiative_equilibrium_geopotential call repeating the
    groupby over the full multi-year dataset.
    """
    return era5_dataset.groupby('valid_time.dayofyear').mean('valid_time')


def radiative_equilibrium_geopotential(model, vSHT, clim_ds, ic_time, smooth_fraction=0.5, log=False):
    """
    Build the zonally-symmetric radiative-equilibrium geopotential phi_eq that
    ShallowWaterSolver.dudtspec's rad term relaxes the mass field toward (see
    ShallowWaterSolver.set_equilibrium_geopotential), from three steps:

      1. Climatological (day-of-year mean across years) zonal-mean u field at
         ic_time's calendar day -- the simulation duration (15-20 days) is short
         enough that this single climatological profile stands in for
         "equilibrium" over the whole run.
      2. A stronger-than-model spectral truncation of that meridional profile:
         keep only spherical-harmonic degrees l <= smooth_fraction * model.lmax
         (the field is already longitude-independent, so only m=0 is nonzero --
         this just low-pass filters it further in latitude).
      3. The same nonlinear balance equation used elsewhere in this file
         (_solve_balance_geopotential), applied to the smoothed, purely-zonal
         u_eq with v_eq=0.

    Args:
        model: swe solver model
        vSHT: RealVectorSHT with geometry compatible with clim_ds (same one used
            for rw_initial_condition)
        clim_ds: day-of-year climatology of the FULL multi-year ERA5 dataset, i.e.
            day_of_year_climatology(era5_dataset) (not a single-time-point slice --
            the climatology needs multiple years of data averaged together)
        ic_time: the run's initial-condition time; only its calendar day is used
        smooth_fraction: fraction (0, 1] of model.lmax kept in u_eq's spectrum
        log: print progress/timing if True

    Returns:
        phi_eq_spec: (model.lmax, model.mmax) complex tensor, triangularly truncated
    """
    if log:
        print("Computing radiative-equilibrium geopotential....")
    start_time = time.perf_counter()
    device = model.device

    doy = pd.Timestamp(ic_time).dayofyear
    day_slice = clim_ds.sel(dayofyear=doy, method='nearest').squeeze()

    dtype = model.lap.dtype
    u_clim = torch.tensor(day_slice['u'].to_numpy(), device=device, dtype=dtype)  # (nlat_data, nlon_data)
    u_zonal = u_clim.mean(dim=-1)  # zonal (longitude) mean -> (nlat_data,)

    nlat_data, nlon_data = u_clim.shape
    if nlat_data != vSHT.nlat or nlon_data != vSHT.nlon:
        raise ValueError(f"❌ vSHT.nlat and vSHT.nlon imcompatible with the data shape, they should be the same." +
                         f"climatology has shape {nlat_data, nlon_data} while vSHT has {vSHT.nlat, vSHT.nlon}")

    # background equilibrium wind: purely zonal (v_eq=0), broadcast across longitude.
    uv_data_eq = torch.zeros(2, nlat_data, nlon_data, device=device, dtype=dtype)
    uv_data_eq[0] = u_zonal.unsqueeze(-1).expand(nlat_data, nlon_data)

    # ERA5 winds are physical (m/s); rescale into the model's own units before any
    # spectral transform so a non-dimensional model gets a non-dimensional u_eq.
    if getattr(model, 'non_dimensional', False):
        uv_data_eq = uv_data_eq / model.U

    uv_model_spec = vSHT(uv_data_eq)
    uv_model_spec_truncated = uv_model_spec[:, :model.lmax, :model.mmax].clone()

    # additional, stronger truncation: smooth the meridional profile by zeroing
    # spherical-harmonic degrees above smooth_fraction * model.lmax.
    l_smooth = max(1, int(smooth_fraction * model.lmax))
    uv_model_spec_truncated[:, l_smooth:, :] = 0

    vrtdiv_spec = model.lap * model.radius * uv_model_spec_truncated
    uv_model = model.getuv(vrtdiv_spec)

    phispec = _solve_balance_geopotential(model, uv_model, model.havg.expand(model.nlat, model.nlon))

    if log:
        print(f"    finished computing radiative-equilibrium geopotential in {(time.perf_counter()-start_time):.2f} seconds")
    return torch.tril(phispec)
