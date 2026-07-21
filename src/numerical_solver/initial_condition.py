import torch
from torch_harmonics.sht import *
import xarray as xr
import time

   
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
    vrtdivgrid = model.spec2grid(vrtdivspec)

    # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
    tmp = ugrid * (vrtdivgrid[:1] + model.coriolis)
    tmpspec = model.vrtdivspec(tmp) # vrtdivspec may (u,v) to vrtdiv_spec
    tmpspec[1] = model.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
    phispec = model.invlap * tmpspec[0] - tmpspec[1] + model.grid2spec(model.gravity*(model.havg + hbump))
    
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


def rw_initial_condition(model, netcdf_path, ic_time, balanced=True):
    """
        Generate a real-world based initial using the dataset:
        "ERA5 hourly data on pressure levels from 1940 to present"
        https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
        
        Args:
            model: swe solver model
            grib_path: the path to the grib file
            ic_time: the year-month-day-hour used to generate ic, ex."2026-07-17"
            balanced: if set to True, only extract [u,v] and compute geopotential 
            by balance equation. Else, use realworld geopotential data
    """
    print("Computing Initial Computation....")
    start_time = time.perf_counter()
    device = model.device
    
    print("     Preparing data ....")
    # open the data set and average along the pressure level axis
    ds = (xr.open_dataset(netcdf_path))
    
    # select the date
    ds = ds.sel(time=ic_time, method='nearest')
    data_finish_time = time.perf_counter()
    print(f"    Finished preparing data in {data_finish_time - start_time:.6f} seconds")
    
    # match the solver's spectral precision (buffers are float64); ERA5 is float32
    dtype = model.lap.dtype
    u_data = torch.tensor(ds['u'].to_numpy(), device=device, dtype=dtype)
    v_data = torch.tensor(ds['v'].to_numpy(), device=device, dtype=dtype)
    
    uv_data = torch.stack([u_data,v_data], dim=0)
    
    if model.solver_type == 'psuedo_spectral_naive':
        # match the solver's transform convention: csphase=False and the same
        # (equiangular) ERA5 grid, otherwise odd-m coefficients carry the wrong sign.
        nlat, nlon = uv_data.shape[-2], uv_data.shape[-1]
        vSHT = RealVectorSHT(nlat=nlat, nlon=nlon, lmax=nlat // 2, mmax=nlat // 2,
                             grid='equiangular', csphase=False).to(device)

        # truncate spectral representation
        uv_model_spec = vSHT(uv_data)
        uv_model_spec_truncated = uv_model_spec[:, :model.lmax, :model.mmax]

        # map to vrtdiv
        vrtdiv_spec = model.lap * model.radius * uv_model_spec_truncated
        # map back to physcal space with vrtdiv
        vrtdiv_model = model.spec2grid(vrtdiv_spec)
        # wind on the model grid: needs the *inverse vector* SHT, not the scalar
        # spec2grid (the truncated coeffs are spheroidal/toroidal potentials, not u,v).
        uv_model = model.getuv(vrtdiv_spec)
        
        # Solve the balance Equation
        # $$\nabla^2 \Phi = \nabla \cdot \left[ \mathbf{u}(\zeta + f) \right] - \nabla^2 K$$
        if balanced is True:
            # A = u(𝛇 + f)
            A_spec = model.vrtdivspec(uv_model * (vrtdiv_model[0] + model.coriolis)) # vrtdivspec may (u,v) to vrtdiv_spec
            kinetic_energy = model.grid2spec(0.5 * torch.sum(uv_model ** 2, dim=0))
            phispec = model.invlap * A_spec[0] - kinetic_energy + model.grid2spec(model.gravity * ((model.havg).expand(model.nlat, model.nlon)))
        
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
        print(f"finished computing initial condition in {all_end_time-start_time:.6f} seconds")
        return torch.tril(phivrtdiv_spec)
