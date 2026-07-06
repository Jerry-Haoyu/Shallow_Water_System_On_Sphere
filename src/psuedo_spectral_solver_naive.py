import torch
import torch.nn as nn
import numpy as np
import math
from torch_harmonics.sht import *
import torch_harmonics as harmonics
from torch_harmonics.quadrature import *
from torch_harmonics.quadrature import _precompute_longitudes, _precompute_latitudes

import pickle
import os
import tqdm

from solver import AbstractSWSolver

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ShallowWaterSolver(AbstractSWSolver):
    """
    SWE solver class. Interface inspired bu pyspharm and SHTns
    Note: 
    1. uspec is frequency of (ɸ,𝛇,δ)
    """

    def __init__(self, dt, nlat=120, nlon=240, tau=2, lmax=None, mmax=None, grid="equiangular"):
        super().__init__(dt)

        # time stepping param
        self.dt = dt

        # grid parameters
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid

        # SHT
        self.sht = harmonics.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.isht = harmonics.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.vsht = harmonics.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.ivsht = harmonics.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)

        self.lmax = lmax or self.sht.lmax
        self.mmax = lmax or self.sht.mmax

        # compute gridpoints
        if self.grid == "legendre-gauss":
            cost, quad_weights = harmonics.quadrature.legendre_gauss_weights(self.nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, quad_weights = harmonics.quadrature.lobatto_weights(self.nlat, -1, 1)
        elif self.grid == "equiangular":
            cost, quad_weights = harmonics.quadrature.clenshaw_curtiss_weights(self.nlat, -1, 1)

        quad_weights = quad_weights.reshape(-1, 1)

        # apply cosine transform and flip them
        lats = -torch.arcsin(cost)
        lons = _precompute_longitudes(self.nlon)

        # compute the laplace and inverse laplace operators
        l = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l = l.expand(self.lmax, self.mmax)
        # the laplace operator acting on the coefficients is given by - l (l + 1)
        lap = - l * (l + 1) / self.radius**2
        invlap = - self.radius**2 / l / (l + 1)
        invlap[0] = 0.

        # compute coriolis force
        coriolis = 2 * self.omega * torch.sin(lats).reshape(self.nlat, 1)

        # hyperdiffusion
        self.tau = tau
        hyperdiff = torch.exp(torch.asarray((-self.dt / self.tau/ 3600.)*(lap / lap[-1, 0])**4))

        # register all
        self.register_buffer('lats', lats)
        self.register_buffer('lons', lons)
        self.register_buffer('l', l)
        self.register_buffer('lap', lap)
        self.register_buffer('invlap', invlap)
        self.register_buffer('coriolis', coriolis)
        self.register_buffer('hyperdiff', hyperdiff)
        self.register_buffer('quad_weights', quad_weights)

    def grid2spec(self, ugrid):
        """
        spectral coefficients from spatial data
        """
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        """
        spatial data from spectral coefficients
        """
        return self.isht(uspec)

    def vrtdivspec(self, ugrid):
        """spatial data from spectral coefficients"""
        vrtdivspec = self.lap * self.radius * self.vsht(ugrid)
        return vrtdivspec

    def getuv(self, vrtdivspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        return self.ivsht( self.invlap * vrtdivspec / self.radius)

    def gethuv(self, uspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        hgrid = self.spec2grid(uspec[:1])
        uvgrid = self.getuv(uspec[1:])
        return torch.cat((hgrid, uvgrid), dim=-3)

    def potential_vorticity(self, uspec):
        """
        Compute potential vorticity
        """
        ugrid = self.spec2grid(uspec)
        pvrt = (0.5 * self.havg * self.gravity / self.omega) * (ugrid[1] + self.coriolis) / ugrid[0]
        return pvrt

    def dimensionless(self, uspec):
        """
        Remove dimensions from variables
        """
        uspec[0] = (uspec[0] - self.havg * self.gravity) / self.hamp / self.gravity
        # vorticity is measured in 1/s so we normalize using sqrt(g h) / r
        uspec[1:] = uspec[1:] * self.radius / torch.sqrt(self.gravity * self.havg)
        return uspec

    def dudtspec(self, uspec):
        """
        Compute time derivatives from solution represented in spectral coefficients
        """

        dudtspec = torch.zeros_like(uspec)

        # compute the derivatives - this should be incorporated into the solver:
        ugrid = self.spec2grid(uspec)
        uvgrid = self.getuv(uspec[1:])

        # phi = ugrid[0]
        # vrtdiv = ugrid[1:]

        tmp = uvgrid * (ugrid[1] + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        # divergence
        dudtspec[2] = tmpspec[0]  # R.H.S vort_spec
        
        # vorticity
        dudtspec[1] = -1 * tmpspec[1] # R.H.S - div_spec

        tmp = uvgrid * ugrid[0]
        tmp = self.vrtdivspec(tmp)
        
        # geopotential
        dudtspec[0] = -1 * tmp[1] # R.H.S div_spec

        tmpspec = self.grid2spec(ugrid[0] + 0.5 * (uvgrid[0]**2 + uvgrid[1]**2))
        dudtspec[2] = dudtspec[2] - self.lap * tmpspec

        return dudtspec
    
    # the function has been modified to accept more parameters for generating diverse initial conditions
    def galewsky_initial_condition(self, umax = 80., usouth = 1/7, unorth = 5/14, perturb_loc=0.25, perturb_amp=1., noise_level=1):
        """
        Initializes non-linear barotropically unstable shallow water test case of Galewsky et al. (2004, Tellus, 56A, 429-440).

        [1] Galewsky; An initial-value problem for testing numerical models of the global shallow-water equations;
            DOI: 10.1111/j.1600-0870.2004.00071.x; http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
        """
        device = self.lap.device
        
        phi0 = torch.asarray(torch.pi * usouth, device=device)
        phi1 = torch.asarray(torch.pi * unorth, device=device)
        phi2 = perturb_loc * torch.pi
        en = torch.exp(torch.asarray(-4.0 / (phi1 - phi0)**2, device=device))
        alpha = 1. / 3.
        beta = 1. / 15.

        lats, lons = torch.meshgrid(self.lats, self.lons)

        u1 = (umax/en)*torch.exp(1./((lats-phi0)*(lats-phi1)))
        ugrid = torch.where(torch.logical_and(lats < phi1, lats > phi0), u1, torch.zeros(self.nlat, self.nlon, device=device))
        vgrid = torch.zeros((self.nlat, self.nlon), device=device)
        noise = noise_level * torch.randn(self.nlat, self.nlon, device=device)
        hbump = noise + self.hamp * perturb_amp * torch.cos(lats) * torch.exp(-((lons-torch.pi)/alpha)**2) * torch.exp(-(phi2-lats)**2/beta)

        # intial velocity field
        ugrid = torch.stack((ugrid, vgrid))
        # intial vorticity/divergence field
        vrtdivspec = self.vrtdivspec(ugrid)
        vrtdivgrid = self.spec2grid(vrtdivspec)

        # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
        tmp = ugrid * (vrtdivgrid[:1] + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
        phispec = self.invlap*tmpspec[0] - tmpspec[1] + self.grid2spec(self.gravity*(self.havg + hbump))

        # assemble solution
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=vrtdivspec.dtype, device=device)
        uspec[0] = phispec
        uspec[1:] = vrtdivspec

        return torch.tril(uspec)

    def random_initial_condition(self, mach=0.1, scaler=1) -> torch.Tensor:
        """
        random initial condition on the sphere
        """
        device = self.lap.device
        ctype = torch.complex128 if self.lap.dtype == torch.float64 else torch.complex64

        # mach number relative to wave speed
        llimit = mlimit = 120

        # initial geopotential
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=ctype, device=self.lap.device)
        uspec[:, :llimit, :mlimit] = scaler * torch.sqrt(torch.tensor(4 * torch.pi / llimit / (llimit+1), device=device, dtype=ctype)) * torch.randn_like(uspec[:, :llimit, :mlimit])

        uspec[0] = self.gravity * self.hamp * uspec[0]
        uspec[0, 0, 0] += torch.sqrt(torch.tensor(4 * torch.pi, device=device, dtype=ctype)) * self.havg * self.gravity
        uspec[1:] = mach * uspec[1:] * torch.sqrt(self.gravity * self.havg) / self.radius
     
        return torch.tril(uspec)

    def timestep(self, uspec: torch.Tensor, nsteps: int) -> torch.Tensor:
        """
        Integrate the solution using Adams-Bashforth / forward Euler for nsteps steps.
        """

        dudtspec = torch.zeros(3, 3, self.lmax, self.mmax, dtype=uspec.dtype, device=uspec.device)

        # pointers to indicate the most current result
        inew = 0
        inow = 1
        iold = 2

        for iter in range(nsteps):
            dudtspec[inew] = self.dudtspec(uspec)

            # update vort,div,phiv with third-order adams-bashforth.
            # forward euler, then 2nd-order adams-bashforth time steps to start.
            if iter == 0:
                dudtspec[inow] = dudtspec[inew]
                dudtspec[iold] = dudtspec[inew]
            elif iter == 1:
                dudtspec[iold] = dudtspec[inew]

            uspec = uspec + self.dt*( (23./12.) * dudtspec[inew] - (16./12.) * dudtspec[inow] + (5./12.) * dudtspec[iold] )

            # implicit hyperdiffusion for vort and div.
            uspec[1:] = self.hyperdiff * uspec[1:]

            # cycle through the indices
            inew = (inew - 1) % 3
            inow = (inow - 1) % 3
            iold = (iold - 1) % 3

        return uspec

    def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
        dlon = 2 * torch.pi / self.nlon
        radius = 1 if dimensionless else self.radius
        if polar_opt > 0:
            out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * self.quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
        else:
            out = torch.sum(ugrid * self.quad_weights * dlon * radius**2, dim=(-2, -1))
        return out
    
    def run(self, days, file_name=None, step_per_save=12, output_dir="solver_output"):
        os.makedirs(output_dir, exist_ok=True)
        uspec = self.galewsky_initial_condition() 
        frames = []
        total_steps = days * 24 * 3600 / self.dt
        number_of_frames = int(1 + np.ceil(total_steps / step_per_save))
        print(f" Simulating SWE on Sphere for {days} days (Naive Psuedo-Spectral Solver)")
        print(f" Total frames = {number_of_frames} | Time per frame = {step_per_save * self.dt /3600} (h)")
       
        # 1. Compute raw, physical Potential Vorticity (SI units: 1 / (m * s))
        ugrid = self.spec2grid(uspec)
        phi_physical = ugrid[0] # gh (m^2/s^2)
        h_physical = phi_physical / self.gravity # h (m)
        vrt_physical = ugrid[1] # zeta (1/s)
        
        q_physical = (vrt_physical + self.coriolis) / h_physical # (1 / (m * s))
        q_ref_physical = self.coriolis / self.havg               # (1 / (m * s))
        
        # 2. Compute the characteristic anomaly frequency scale (1/s)
        # This matches Dritschel's numerator: h_avg * max|q - q_ref|
        pv_anomaly_max = torch.max(torch.abs(q_physical - q_ref_physical))
        vrt_scale_dritschel = self.havg * pv_anomaly_max # Units: 1/s
        
        # 3. Handle the spatial scaling matching your dimensional Laplacian
        lambda_max_power_n = (self.lap[-1, 0]) ** 4 # Units: m^-8
        
        # 4. Compute recommended physical hyperviscosity nu (m^8/s)
        nu_recommend = vrt_scale_dritschel / lambda_max_power_n
        
        # 5. Convert this directly to your model's e-folding timescale tau (in hours)
        # At the grid limit, e-fold time in seconds is 1 / (nu * lambda_max^4)
        tau_recommend_seconds = 1.0 / (nu_recommend * lambda_max_power_n)
        tau_recommend_hours = tau_recommend_seconds / 3600.0
        
        # print out in a box
        title_text = f" Hyperdiffusion "
        criteria_text = f"(Dritschel 1997 Criteria):  ν = {nu_recommend.item():.3e} m^8/s | Equivalent e-fold time = {tau_recommend_hours.item():.3f} hours"
        actual_hyerdiffusion_text = f"Actual model e-fold time = {self.tau:.3f} hours" 
        end_text = '-'*98
        print(f"+{title_text.center(98, '-')}+", end='\n')
        print(f"|{criteria_text.center(98)}|", end='\n')
        print(f"|{actual_hyerdiffusion_text.center(98)}|", end='\n')
        print(f"+{end_text.center(98)}+", end='\n')
        
        with torch.no_grad():
            for i in tqdm.trange(number_of_frames, desc=f'Simulation in Progrgess'):
                frames.append(uspec.cpu())
                if i < number_of_frames - 1: # no simulation for the last step
                    uspec = self.timestep(uspec, step_per_save)
        trajectory = torch.stack(frames) 
        data = {
            'metadata' : {
                'dt' : self.dt,
                'nlat' : self.nlat,
                'nlon' : self.nlon,
                'lmax' : self.lmax,
                'mmax' : self.mmax,
                'grid' : self.grid,
                'step_per_save' : step_per_save,
            },
            'trajectory' : trajectory
        }
        if file_name == None:
            file_name = f"galewsky_{days}days"
        save_path  = os.path.join(output_dir, file_name)
        torch.save(data, save_path)
        print(f"Saved simulation data(metadata : dict[dt, nlat, nlon, lmax, mmax, grid, step_per_save], trajectory : tensor({trajectory.shape})) -> {save_path}")


   