import math

import torch
from torch_harmonics.sht import *
import numpy as np
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



def _exponential_spectral_filter(lmax, a, p, dtype, device):
    """Smooth exponential spectral filter sigma(l) = exp(-a*(l/l_max))^p (l_max=lmax-1),
    used in place of a hard truncation cutoff when bringing real-world data (at its own
    native resolution) down to the model's spectral truncation. A hard slice (`spec[:lmax]`)
    leaves an abrupt edge at l=lmax-1 that rings in physical space (Gibbs phenomenon);
    this instead damps the coefficients smoothly and monotonically from l=0 (sigma=1)
    down to l=l_max (sigma=exp(-a)^p), with a*p controlling how aggressively the whole
    retained band is damped. Returns a (lmax, 1) real tensor (matching `dtype`, the
    model's own real precision, so multiplying it into a spectral tensor doesn't silently
    upcast the complex dtype) that broadcasts over the trailing `m` axis (and any leading
    channel axis) of a (..., lmax, mmax) spectral tensor.
    """
    l = torch.arange(lmax, dtype=dtype, device=device)
    l_max = max(lmax - 1, 1)
    return (torch.exp(-a * (l / l_max))**p).unsqueeze(-1)


def _infer_triangular_truncation(n_complex):
    """Native ECMWF triangular truncation T implied by a GRIB spectral field's complex
    coefficient count: a T-truncated triangular field stores N = (T+1)(T+2)/2 complex
    coefficients (m=0..T, n=m..T)."""
    T = int(round((-3 + math.sqrt(8 * n_complex + 1)) / 2))
    if (T + 1) * (T + 2) // 2 != n_complex:
        raise ValueError(f"❌ {n_complex} complex spectral coefficients don't correspond to any triangular truncation T.")
    return T


def _ecmwf_spectral_to_tensor(raw_values, native_trunc, lmax, mmax, ctype, device):
    """Convert one ERA5-complete GRIB spherical-harmonic field's raw coefficient array
    into a (lmax, mmax) complex tensor in torch_harmonics' orthonormal ("ortho") real-SHT
    convention (l=degree first axis, m=order second axis, m<=l), truncating or
    zero-padding to (lmax, mmax) as needed.

    ECMWF stores `raw_values` as (real, imag) pairs ordered by increasing zonal wavenumber
    m (0..native_trunc), and for each m by increasing total wavenumber n (m..native_trunc)
    -- the standard WMO GRIB triangular spectral layout (verified against this file's own
    `pentagonalResolutionParameterJ/K/M`). ECMWF's Legendre normalization is the "4*pi"
    convention (Y_0^0 = 1, i.e. the raw (n=0, m=0) coefficient IS the field's domain mean --
    verified against 'z''s ~55000 m^2/s^2 January-mean 500 hPa geopotential), whereas
    torch_harmonics' "ortho" convention has Y_0^0 = 1/sqrt(4*pi), so every coefficient is
    rescaled by sqrt(4*pi). Both conventions omit the Condon-Shortley phase, matching this
    codebase's `csphase=False` transforms.

    Cross-validated by comparing the vorticity/divergence spectra this produces against
    this repo's own forward RealVectorSHT of ERA5 GRIDDED (u, v) at a shared time point
    (1980-01-01T00, present in both reanalysis_data/1980_2025_odd_month_500 and
    1970_2025_sparse_500_vo_d_z_t128): the two agree to ~1% relative L2 error, attributable
    to the different native resolutions (T128 spectral vs 0.25 deg grid), confirming both
    the coefficient ordering and the sqrt(4*pi) normalization above.
    """
    n_complex = raw_values.size // 2
    if n_complex != (native_trunc + 1) * (native_trunc + 2) // 2:
        raise ValueError(f"❌ raw spectral array has {n_complex} complex coefficients, "
                         f"inconsistent with triangular truncation T{native_trunc}.")

    # GRIB storage order (m outer 0..T, n inner m..T) is exactly the row-major order
    # np.triu_indices returns for an upper triangle -> pairs[i] = (m_idx[i], n_idx[i]).
    m_idx, n_idx = np.triu_indices(native_trunc + 1)
    complex_vals = raw_values[0::2].astype(np.float64) + 1j * raw_values[1::2].astype(np.float64)
    native = np.zeros((native_trunc + 1, native_trunc + 1), dtype=np.complex128)
    native[n_idx, m_idx] = complex_vals  # (l=n, m) layout, lower-triangular (m<=l)
    native *= np.sqrt(4 * np.pi)

    out = torch.zeros(lmax, mmax, dtype=ctype, device=device)
    common_l = min(native_trunc + 1, lmax)
    common_m = min(native_trunc + 1, mmax)
    out[:common_l, :common_m] = torch.tensor(native[:common_l, :common_m], dtype=ctype, device=device)
    return out


def rw_initial_condition(model, vSHT, era5_dataset, ic_time, balanced=False, log=False, a=2, p=16):
    """
        Generate a real-world based initial condition from an ERA5 dataset, either
        GRIDDED (u, v on a lat/lon grid, e.g. from download_era5.py's
        "ERA5 hourly data on pressure levels" -
        https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download)
        or SPECTRAL (native spherical-harmonic vo/d/z coefficients, no lat/lon dims, from
        download_era5_spectral.py's "ERA5 complete" MARS retrieval) -- which of the two
        `era5_dataset` is auto-detected from its dimensions (a 'values' dim with no
        'latitude'/'longitude' means spectral).

        Args:
            model: swe solver model
            vSHT: RealVectorSHT with geometry compatible to a GRIDDED era5_dataset; ignored
                (may be None) for a spectral era5_dataset, which needs no vector transform.
            era5_dataset: opened xarray Dataset, gridded or spectral (see above)
            ic_time: the year-month-day-hour used to generate ic, ex."2026-07-17"
            balanced: if True, extract [u,v] (or, for spectral input, vorticity/divergence
                directly) and solve the balance equation for geopotential. If False
                (default), use the dataset's own real-world geopotential ('z') instead.
            a, p: exponential spectral filter parameters (see _exponential_spectral_filter)
                applied in place of a hard truncation when bringing era5_dataset's native
                resolution down to the model's (lmax, mmax).
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
    ctype = torch.complex128 if dtype == torch.float64 else torch.complex64

    # a spectral (native spherical-harmonic) dataset has a flat 'values' dim and no
    # lat/lon; a gridded one has 'latitude'/'longitude' (see rw_initial_condition's docstring).
    is_spectral = 'values' in ds.dims

    if model.solver_type != 'psuedo_spectral_naive':
        return

    sigma = _exponential_spectral_filter(model.lmax, a, p, dtype, device)

    if is_spectral:
        if log:
            print(f"    Detected spectral ERA5 dataset (native spherical-harmonic coefficients)")
        spec_start = time.perf_counter()

        vo_raw, d_raw = ds['vo'].to_numpy(), ds['d'].to_numpy()
        native_trunc = _infer_triangular_truncation(vo_raw.size // 2)

        vrt_spec = _ecmwf_spectral_to_tensor(vo_raw, native_trunc, model.lmax, model.mmax, ctype, device)
        div_spec = _ecmwf_spectral_to_tensor(d_raw, native_trunc, model.lmax, model.mmax, ctype, device)
        # vorticity/divergence are physical rates (1/s); model.T (=1 when
        # non_dimensional=False) rescales them into the model's own time units, the same
        # way __init__ rescales omega_phys -> self.omega.
        vrtdiv_spec = torch.stack([vrt_spec, div_spec], dim=0) * model.T
        vrtdiv_spec = vrtdiv_spec * sigma

        spec_end = time.perf_counter()
        if log:
            print(f"    Finished spectral remapping in {(spec_end - spec_start):2f} seconds")

        if balanced:
            uv_model = model.getuv(vrtdiv_spec)
        else:
            z_raw = ds['z'].to_numpy()
            phispec = _ecmwf_spectral_to_tensor(z_raw, native_trunc, model.lmax, model.mmax, ctype, device)
            if getattr(model, 'non_dimensional', False):
                phispec = phispec / (model.U ** 2)
            phispec = phispec * sigma
    else:
        u_data = torch.tensor(ds['u'].to_numpy(), device=device, dtype=dtype)
        v_data = torch.tensor(ds['v'].to_numpy(), device=device, dtype=dtype)

        uv_data = torch.stack([u_data, v_data], dim=0)
        # ERA5 winds are physical (m/s); rescale into the model's own units before
        # any spectral transform so a non-dimensional model gets a non-dimensional IC.
        if getattr(model, 'non_dimensional', False):
            uv_data = uv_data / model.U

        data_finish_time = time.perf_counter()
        if log:
            print(f"    Finished preparing data in {(data_finish_time - start_time):.2f} seconds")

        spec_start = time.perf_counter()

        # compute spectral representation
        nlat, nlon = uv_data.shape[-2], uv_data.shape[-1]
        if nlat != vSHT.nlat or nlon != vSHT.nlon:
            raise ValueError(f"❌ vSHT.nlat and vSHT.nlon imcompatible with the data shape, they should be the same." + \
                             f"netCDF file has data of shape {nlat, nlon} while vSHT has {vSHT.nlat, vSHT.nlon}")

        # filter (in place of a hard truncation) down to the model's spectral resolution
        uv_model_spec = vSHT(uv_data)
        uv_model_spec_truncated = uv_model_spec[:, :model.lmax, :model.mmax] * sigma

        # map to vrtdiv
        vrtdiv_spec = model.lap * model.radius * uv_model_spec_truncated

        # wind on the model grid: needs the *inverse vector* SHT, not the scalar
        # spec2grid (the truncated coeffs are spheroidal/toroidal potentials, not u,v).
        uv_model = model.getuv(vrtdiv_spec)
        spec_end = time.perf_counter()
        if log:
            print(f"    Finished spectral truncation in {(spec_end - spec_start):2f} seconds")

        if balanced:
            pass  # phispec computed below, from the balance equation
        else:
            z_data = torch.tensor(ds['z'].to_numpy(), device=device, dtype=dtype)
            z_sht = RealSHT(nlat=vSHT.nlat, nlon=vSHT.nlon, lmax=vSHT.lmax, mmax=vSHT.mmax,
                             grid=vSHT.grid, csphase=vSHT.csphase).to(device)
            phispec = z_sht(z_data)[:model.lmax, :model.mmax] * sigma
            if getattr(model, 'non_dimensional', False):
                phispec = phispec / (model.U ** 2)

    # Solve the balance Equation (see _solve_balance_geopotential) -- shared by both
    # branches above, since it only needs `uv_model` (wind on the model grid).
    if balanced:
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

    Reduces over whichever dimension the 'valid_time' coordinate is actually indexed
    by, rather than assuming it's named 'valid_time' itself: a gridded ERA5 dataset's
    time dimension IS named 'valid_time', but a spectral (cfgrib-opened) dataset names
    it 'time', with 'valid_time' only a coordinate on it - see rw_initial_condition's
    spectral/grid auto-detection for the same grid-vs-spectral distinction.
    """
    time_dim = era5_dataset['valid_time'].dims[0]
    return era5_dataset.groupby('valid_time.dayofyear').mean(time_dim)


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
