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
import time

from src.helpers.print import print_in_box, finish_simulation_log

from src.numerical_solver.solver import AbstractSWSolver


class ShallowWaterSolver(AbstractSWSolver):
    """
    SWE solver class. Interface inspired bu pyspharm and SHTns
    Note: 
    1. uspec is frequency of (ɸ,𝛇,δ)
    """

    def __init__(self, lmax, tau=(10000, 30, 20), cfl=0.25, grid="equiangular",
                 semi_implicit=True, robert_coeff=0.05, umax=120.0, dealias=True):
        # The only resolution knob is lmax. Triangular truncation fixes mmax = lmax, and
        # the grid follows as nlat = 2*lmax, nlon = 2*nlat. dt is derived from CFL below.
        #
        # semi_implicit : integrate the gravity-wave terms with a semi-implicit leapfrog
        #                 (trapezoidal average of levels n+1/n-1) so the timestep is
        #                 limited by the *advective* CFL (max wind `umax`) rather than the
        #                 much faster external gravity-wave speed c = sqrt(g*havg) ~ 313 m/s.
        # robert_coeff  : Robert-Asselin time-filter coefficient that suppresses the
        #                 computational (odd/even) leapfrog mode.
        # dealias       : evaluate the quadratic nonlinear products on a 3/2-padded grid
        #                 so that aliasing of modes > lmax back into the retained band is
        #                 removed (Orszag's 3/2 rule; see dudtspec).
        self.start_time = time.perf_counter()
        print("Initializing the psuedo-spectral solver")
        super().__init__()
        self.solver_type='psuedo_spectral_naive'
        self.cfl = cfl
        self.semi_implicit = semi_implicit
        self.robert_coeff = robert_coeff
        self.umax = umax
        self.dealias = dealias

        # spectral truncation and the grid it implies
        self.lmax = lmax
        self.mmax = lmax
        self.nlat = 2 * lmax
        self.nlon = 2 * self.nlat
        self.grid = grid

        # SHT
        self.sht = harmonics.RealSHT(self.nlat, self.nlon, lmax=self.lmax, mmax=self.mmax, grid=grid, csphase=False)
        self.isht = harmonics.InverseRealSHT(self.nlat, self.nlon, lmax=self.lmax, mmax=self.mmax, grid=grid, csphase=False)
        self.vsht = harmonics.RealVectorSHT(self.nlat, self.nlon, lmax=self.lmax, mmax=self.mmax, grid=grid, csphase=False)
        self.ivsht = harmonics.InverseRealVectorSHT(self.nlat, self.nlon, lmax=self.lmax, mmax=self.mmax, grid=grid, csphase=False)

        # ------------------------------------------------------------------ #
        # De-aliasing (3/2 rule): a second, padded transform pair on which the
        # quadratic nonlinear products are computed. The state is truncated at
        # wavenumber M = lmax; a product of two such fields reaches wavenumber
        # 2M. Evaluating that product on a grid that resolves >= 1.5M and then
        # chopping the forward transform back to M keeps the aliasing tail out
        # of the retained band. We pad the spectrum to 1.5M and use a grid of
        # size 3M (nlat = 2 * lmax_d), matching nlat = 2*lmax of the main grid.
        # ------------------------------------------------------------------ #
        if self.dealias:
            self.lmax_d = int(np.ceil(1.5 * self.lmax))   # pad target ~ 1.5 M
            self.mmax_d = self.lmax_d
            self.nlat_d = 2 * self.lmax_d                  # padded grid ~ 3 M
            self.nlon_d = 2 * self.nlat_d
            self.sht_d = harmonics.RealSHT(self.nlat_d, self.nlon_d, lmax=self.lmax_d, mmax=self.mmax_d, grid=grid, csphase=False)
            self.isht_d = harmonics.InverseRealSHT(self.nlat_d, self.nlon_d, lmax=self.lmax_d, mmax=self.mmax_d, grid=grid, csphase=False)
            self.vsht_d = harmonics.RealVectorSHT(self.nlat_d, self.nlon_d, lmax=self.lmax_d, mmax=self.mmax_d, grid=grid, csphase=False)
            self.ivsht_d = harmonics.InverseRealVectorSHT(self.nlat_d, self.nlon_d, lmax=self.lmax_d, mmax=self.mmax_d, grid=grid, csphase=False)

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

        # CFL-limited timestep. dx_min = pi*a/nlat is the meridional grid spacing.
        #   * explicit scheme  -> limited by the external gravity-wave speed
        #     c = sqrt(g*havg) ~ 313 m/s (the fastest signal).
        #   * semi-implicit    -> gravity waves are handled implicitly, so the limit
        #     becomes the advective CFL set by the fastest wind `umax`; this yields a
        #     substantially larger stable dt.
        c_grav = float(torch.sqrt(self.gravity * self.havg))      # ~313 m/s
        dx_min = float(np.pi * self.radius / self.nlat)           # meridional grid spacing (m)
        if self.semi_implicit:
            self.dt = self.cfl * dx_min / self.umax
            limiter = f"advective umax={self.umax:.0f} m/s (semi-implicit)"
        else:
            self.dt = self.cfl * dx_min / c_grav
            limiter = f"gravity c={c_grav:.0f} m/s (explicit)"


        ## hyperdiffusion: e-fold damping time `tau` (hours) at the truncation scale.
        # `hyperdiff` damps over a single dt; the centred leapfrog advances the field
        # over 2*dt, so `hyperdiff2 = hyperdiff**2` is the factor applied there.
        self.tau = tau
        tau_2, tau_4, tau_8 = self.tau

        # Normalize the Laplacian grid frequencies to peak at 1.0 at the truncation scale
        k_normalized = lap / lap[-1, 0]

        # n=2 (biharmonic damping scales with k^4)
        # n=4 (quad-harmonic damping scales with k^8)
        # n=8 (k^16)
        damping_n2 = (-self.dt / (tau_2 * 3600.)) * (k_normalized ** 2)
        damping_n4 = (-self.dt / (tau_4 * 3600.)) * (k_normalized ** 4)
        damping_n8 = (-self.dt / (tau_8 * 3600.)) * (k_normalized ** 8)

        # Combine the scales in exponent space before applying torch.exp
        hyperdiff = torch.exp(torch.asarray(damping_n2 + damping_n4 + damping_n8))
        hyperdiff2 = hyperdiff ** 2

        # coriolis on the padded de-aliasing grid (needed by the nonlinear products)
        if self.dealias:
            lats_d = self._grid_latitudes(self.nlat_d)
            coriolis_d = 2 * self.omega * torch.sin(lats_d).reshape(self.nlat_d, 1)
            self.register_buffer('coriolis_d', coriolis_d)

        # leapfrog needs the previous (filtered) time level; None triggers a
        # single-sided startup step on the first call. Reset via run() / reset_time().
        self._uspec_prev = None

        # register all
        self.register_buffer('lats', lats)
        self.register_buffer('lons', lons)
        self.register_buffer('l', l)
        self.register_buffer('lap', lap)
        self.register_buffer('invlap', invlap)
        self.register_buffer('coriolis', coriolis)
        self.register_buffer('hyperdiff', hyperdiff)
        self.register_buffer('hyperdiff2', hyperdiff2)
        self.register_buffer('quad_weights', quad_weights)
        
        
        ### Basic Logging ###
        integrator = 'semi_implicit' if self.semi_implicit is True else 'explicit 2nd order Adam-Bashforth'
        
        content_config_log = {
            'title' : "SWE Psuedo-Spectral Solver Configuration" ,
            'lines' : [
                f"Spectral resolution (l_max,mmax) = {self.lmax, self.mmax} | (nlon, nlat) = ({self.nlon}, {self.nlat})",
                f"Integrator is {integrator}| CFL constant is {self.cfl} | dt = {self.dt}",
                f"Numerical Quadrature : {self.grid}",
                f"Dealiasing by 3/2 padding : {str(self.dealias)}",
                f"Device : {self.device}"
            ]
        }
        print_in_box(content_config_log)
        init_end_time = time.perf_counter()
        print(f"Finished initializing solver in {init_end_time - self.start_time:.2f} seconds")
        ######################


    def _grid_latitudes(self, nlat):
        """Latitudes (rad) of the quadrature grid with `nlat` points for self.grid."""
        if self.grid == "legendre-gauss":
            cost, _ = harmonics.quadrature.legendre_gauss_weights(nlat, -1, 1)
        elif self.grid == "lobatto":
            cost, _ = harmonics.quadrature.lobatto_weights(nlat, -1, 1)
        else:  # equiangular
            cost, _ = harmonics.quadrature.clenshaw_curtiss_weights(nlat, -1, 1)
        return -torch.arcsin(cost)

    def reset_time(self):
        """Forget the stored leapfrog level so the next timestep restarts the scheme."""
        self._uspec_prev = None

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
        """map (u,v) to vrtdivspec"""
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

    # ---------------------------------------------------------------------- #
    # De-aliased transform primitives (Orszag 3/2 rule)
    #
    #   spectral (M) --pad zeros--> spectral (1.5M) --iSHT--> physical (3M grid)
    #      -> form the aliasing-free product on the padded physical grid
    #   physical (3M) --SHT--> spectral (1.5M) --truncate--> spectral (M)
    #
    # These mirror grid2spec/spec2grid/getuv/vrtdivspec but route through the
    # padded (self.*_d) transform pair so that quadratic products carry no
    # aliased power back into the retained band [0, M].
    # ---------------------------------------------------------------------- #
    def _pad(self, coeff):
        """Zero-pad spectral coeffs from (.., lmax, mmax) up to the padded (1.5M) band."""
        out = torch.zeros(*coeff.shape[:-2], self.lmax_d, self.mmax_d,
                          dtype=coeff.dtype, device=coeff.device)
        out[..., :self.lmax, :self.mmax] = coeff
        return out

    def _chop(self, coeff):
        """Truncate padded spectral coeffs back down to the retained band (.., M, M)."""
        return coeff[..., :self.lmax, :self.mmax]

    def _spec2grid_d(self, uspec):
        """Inverse SHT onto the padded physical grid (pads high modes with zeros)."""
        return self.isht_d(self._pad(uspec))

    def _getuv_d(self, vrtdivspec):
        """Wind vector on the padded physical grid from vorticity/divergence coeffs."""
        return self.ivsht_d(self._pad(self.invlap * vrtdivspec / self.radius))

    def _grid2spec_d(self, ugrid):
        """Forward SHT from the padded physical grid, chopped back to the retained band."""
        return self._chop(self.sht_d(ugrid))

    def _vrtdivspec_d(self, ugrid):
        """Vorticity/divergence coeffs of a padded-grid vector field, chopped to (M, M)."""
        return self.lap * self.radius * self._chop(self.vsht_d(ugrid))

    def dudtspec(self, uspec):
        """
        Compute time derivatives from solution represented in spectral coefficients.
        When self.dealias is set, the quadratic nonlinear products are evaluated on the
        padded 3/2 grid (self.*_d) and truncated back to wavenumber M; otherwise they are
        formed directly on the main grid.
        """

        dudtspec = torch.zeros_like(uspec)

        # select the (de-aliased) transform primitives / Coriolis grid
        if self.dealias:
            spec2grid, getuv = self._spec2grid_d, self._getuv_d
            grid2spec, vrtdivspec = self._grid2spec_d, self._vrtdivspec_d
            coriolis = self.coriolis_d
        else:
            spec2grid, getuv = self.spec2grid, self.getuv
            grid2spec, vrtdivspec = self.grid2spec, self.vrtdivspec
            coriolis = self.coriolis

        # transform state onto the (padded) physical grid where products are formed
        ugrid = spec2grid(uspec)       # (3, nlat, nlon): phi, vrt, div
        uvgrid = getuv(uspec[1:])      # (2, nlat, nlon): u, v

        tmp = uvgrid * (ugrid[1] + coriolis)
        tmpspec = vrtdivspec(tmp)
        dudtspec[2] = tmpspec[0]
        dudtspec[1] = -1 * tmpspec[1]

        tmp = uvgrid * ugrid[0]
        tmp = vrtdivspec(tmp)
        dudtspec[0] = -1 * tmp[1]

        tmpspec = grid2spec(ugrid[0] + 0.5 * (uvgrid[0]**2 + uvgrid[1]**2))
        dudtspec[2] = dudtspec[2] - self.lap * tmpspec

        return dudtspec
    
    def _si_solve(self, uspec_now, uspec_ref, dtx):
        """
        One semi-implicit step of the gravity-wave subsystem.

          uspec_now : level X^n where the explicit tendencies are evaluated
          uspec_ref : reference level X^ref (n-1 for centred leapfrog, n for the
                      single-sided startup step)
          dtx       : time increment (2*dt for leapfrog, dt for the startup step)

        The linear gravity-wave terms  -lap*phi  (divergence eqn) and  -phibar*div
        (geopotential eqn) are advanced implicitly via the trapezoidal average of
        levels n+1 and ref, which is A-stable and so removes the gravity-wave CFL
        limit. Everything else (advection, kinetic energy, vorticity) is explicit.
        Returns X^{n+1} before hyperdiffusion / Robert-Asselin filtering.
        """
        phibar = self.gravity * self.havg

        # fully-explicit tendency, then strip the linearised gravity-wave part so the
        # remainder R is what stays explicit:  D = R + L,  L_div = -lap*phi^n,
        # L_phi = -phibar*div^n.
        D = self.dudtspec(uspec_now)
        R_div = D[2] + self.lap * uspec_now[0]
        R_phi = D[0] + phibar * uspec_now[2]

        phi_ref = uspec_ref[0]
        div_ref = uspec_ref[2]

        A = div_ref + dtx * R_div
        B = phi_ref + dtx * R_phi

        # solving the coupled trapezoidal update for phi^{n+1} (lap<0 => alpha<0 =>
        # 1-alpha > 1, so the implicit operator is always well conditioned).
        alpha = (dtx ** 2) * phibar * self.lap / 4.0
        phi_new = (B - dtx * phibar / 2.0 * (A + div_ref) + alpha * phi_ref) / (1.0 - alpha)
        div_new = A - dtx * self.lap / 2.0 * (phi_new + phi_ref)
        vrt_new = uspec_ref[1] + dtx * D[1]      # vorticity: plain (leap)frog, no gravity term

        unew = torch.zeros_like(uspec_now)
        unew[0] = phi_new
        unew[1] = vrt_new
        unew[2] = div_new
        return unew

    def _leapfrog_step(self, uspec_now):
        """Advance one dt with the (semi-implicit) leapfrog + Robert-Asselin filter.

        Keeps the previous filtered level in self._uspec_prev across calls so the
        scheme runs continuously (a single-sided startup step bootstraps it when the
        stored level is absent). When semi_implicit is off this reduces to a plain
        explicit leapfrog and requires a gravity-wave-limited dt.
        """
        if self._uspec_prev is None:
            # single-sided (forward) startup: reference is the current level, dt not 2*dt.
            if self.semi_implicit:
                unew = self._si_solve(uspec_now, uspec_now, self.dt)
            else:
                unew = uspec_now + self.dt * self.dudtspec(uspec_now)
            unew[1:] = self.hyperdiff * unew[1:]        # diffusion over one dt
            self._uspec_prev = uspec_now
            return unew

        # centred leapfrog over 2*dt from the stored (filtered) level
        if self.semi_implicit:
            unew = self._si_solve(uspec_now, self._uspec_prev, 2.0 * self.dt)
        else:
            unew = self._uspec_prev + 2.0 * self.dt * self.dudtspec(uspec_now)
        unew[1:] = self.hyperdiff2 * unew[1:]           # diffusion over 2*dt

        # Robert-Asselin time filter on the 'now' level, damping the computational mode:
        #   X-bar^n = X^n + gamma*(X-bar^{n-1} - 2 X^n + X^{n+1})
        uspec_now_f = uspec_now + self.robert_coeff * (self._uspec_prev - 2.0 * uspec_now + unew)

        self._uspec_prev = uspec_now_f
        return unew

    def timestep(self, uspec: torch.Tensor, nsteps: int) -> torch.Tensor:
        """
        Integrate the solution for `nsteps` steps with the semi-implicit leapfrog
        scheme coupled to a Robert-Asselin time filter (see _leapfrog_step). The
        previous time level persists on the instance across calls; use reset_time()
        (called by run()) to restart the scheme from a fresh initial condition.
        """
        for _ in range(nsteps):
            uspec = self._leapfrog_step(uspec)
        return uspec

    def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
        dlon = 2 * torch.pi / self.nlon
        radius = 1 if dimensionless else self.radius
        if polar_opt > 0:
            out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * self.quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
        else:
            out = torch.sum(ugrid * self.quad_weights * dlon * radius**2, dim=(-2, -1))
        return out
    
    def run(self, initial_condition, days, 
            file_name=None, 
            save_interval_minutes=30, 
            output_dir="solver_output"):
        os.makedirs(output_dir, exist_ok=True)
        # save one frame every `save_interval_minutes` of simulated time; infer the
        # number of integration steps between saves from the timestep dt (in seconds).
        step_per_save = max(1, round(save_interval_minutes * 60 / self.dt))
        self.reset_time()   # restart the leapfrog scheme from a clean initial condition
        
        uspec = initial_condition
        total_steps = days * 24 * 3600 / self.dt
        number_of_frames = int(1 + np.ceil(total_steps / step_per_save))
        
        run_log_content = {
            "title" : "Running Psuedo-Spectral Solver",
            "lines" : [
                f"Days : {days} | save_interval : {save_interval_minutes}(minutes) | Total frames = {number_of_frames}",
                f"output_dir = {output_dir} ",
                f"file_name = {file_name}"
            ]
        }
        
        print_in_box(run_log_content)
        
        # preallocate the output on CPU and fill in place: avoids holding both a
        # per-frame list and a stacked copy at once (which ~doubled peak RAM and OOM'd).
        trajectory = torch.empty((number_of_frames, *uspec.shape), dtype=uspec.dtype)
        
        # #### Hyperdiffusion Logs #####
        # # 1. Compute raw, physical Potential Vorticity (SI units: 1 / (m * s))
        # ugrid = self.spec2grid(uspec)
        # phi_physical = ugrid[0] # gh (m^2/s^2)
        # h_physical = phi_physical / self.gravity # h (m)
        # vrt_physical = ugrid[1] # zeta (1/s)
        
        # q_physical = (vrt_physical + self.coriolis) / h_physical # (1 / (m * s))
        # q_ref_physical = self.coriolis / self.havg               # (1 / (m * s))
        
        # # 2. Compute the characteristic anomaly frequency scale (1/s)
        # # This matches Dritschel's numerator: h_avg * max|q - q_ref|
        # pv_anomaly_max = torch.max(torch.abs(q_physical - q_ref_physical))
        # vrt_scale_dritschel = self.havg * pv_anomaly_max # Units: 1/s
        
        # # 3. Handle the spatial scaling matching your dimensional Laplacian
        # lambda_max_power_n = (self.lap[-1, 0]) ** 4 # Units: m^-8
        
        # # 4. Compute recommended physical hyperviscosity nu (m^8/s)
        # nu_recommend = vrt_scale_dritschel / lambda_max_power_n
        
        # # 5. Convert this directly to your model's e-folding timescale tau (in hours)
        # # At the grid limit, e-fold time in seconds is 1 / (nu * lambda_max^4)
        # tau_recommend_seconds = 1.0 / (nu_recommend * lambda_max_power_n)
        # tau_recommend_hours = tau_recommend_seconds / 3600.0
        
        # print out in a box
        content = {
            'title' : " Hyperdiffusion ",
            'lines' : [
                # f"(Dritschel 1997 Criteria):  ν = {nu_recommend.item():.3e} m^8/s | Equivalent e-fold time = {tau_recommend_hours.item():.3f} hours",
                f"model e-fold time (𝝉₂, 𝛕₄, 𝛕₈)= {self.tau} hours" 
            ]
        }
        print_in_box(content)
        ##########

        with torch.no_grad():
            for i in tqdm.trange(number_of_frames, desc=f'Simulation in Progrgess'):
                trajectory[i] = uspec.cpu()
                if i < number_of_frames - 1: # no simulation for the last step
                    uspec = self.timestep(uspec, step_per_save)
        data = {
            'metadata' : {
                'dt' : self.dt,
                'tau' : self.tau,
                'nlat' : self.nlat,
                'nlon' : self.nlon,
                'lmax' : self.lmax,
                'mmax' : self.mmax,
                'grid' : self.grid,
                'step_per_save' : step_per_save,
                'save_interval_minutes' : save_interval_minutes,
            },
            'trajectory' : trajectory
        }
        if file_name == None:
            file_name = f"swe_{days}days"
        save_path  = os.path.join(output_dir, file_name)
        torch.save(data, save_path)
        finsh_time = time.perf_counter()
        finish_simulation_log(file_name, time=(finsh_time - self.start_time))


   