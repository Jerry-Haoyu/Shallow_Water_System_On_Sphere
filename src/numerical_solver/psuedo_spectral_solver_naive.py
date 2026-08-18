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
                 semi_implicit=True, robert_coeff=0.05, umax=120.0, dealias=True,
                 h_avg=None, h_amp=None, non_dimensional=True,
                 rad=False, tau_rad=None):
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
        # h_avg, h_amp  : dataset-wide reference height / height-amplitude (meters, from
        #                 e.g. reanalysis_data/<name>/h_stats.npz); fall back to the
        #                 AbstractSWSolver defaults (10km / 120m) when not given.
        # non_dimensional : rescale radius/gravity/havg/hamp/omega/umax by the length
        #                 scale L=radius, velocity scale U=sqrt(g*havg) and time scale
        #                 T=L/U before anything else is derived from them. The SWE
        #                 tendency equations (dudtspec) are scale-covariant under this
        #                 L/U/T rescaling, so the *same* code with rescaled constants
        #                 integrates the non-dimensional state -- CFL/dt, hyperdiffusion,
        #                 Coriolis and dudtspec all fall out correctly with no further
        #                 changes. Set False to run fully dimensional (physical units).
        # rad, tau_rad  : if rad=True, dudtspec relaxes geopotential toward a
        #                 zonally-symmetric equilibrium phi_eq (see
        #                 set_equilibrium_geopotential) with e-fold time tau_rad
        #                 (days, physical regardless of non_dimensional -- unlike
        #                 `tau`'s hyperdiffusion e-fold times, which are hours,
        #                 since radiative relaxation timescales are naturally
        #                 quoted in days).
        #                 phi_eq itself is IC/dataset-specific, so it is not part of
        #                 this constructor -- it must be set via
        #                 set_equilibrium_geopotential before timestep() is called.
        if rad and tau_rad is None:
            raise ValueError("tau_rad is required when rad=True")
        self.rad = bool(rad)
        self.tau_rad_days = float(tau_rad) if tau_rad is not None else None

        self.start_time = time.perf_counter()
        print("Initializing the psuedo-spectral solver")
        super().__init__()
        self.solver_type='psuedo_spectral_naive'
        self.cfl = cfl
        self.semi_implicit = semi_implicit
        self.robert_coeff = robert_coeff
        self.dealias = dealias

        radius_phys = float(self.radius)
        omega_phys = float(self.omega)
        gravity_phys = float(self.gravity)
        havg_phys = float(h_avg) if h_avg is not None else float(self.havg)
        hamp_phys = float(h_amp) if h_amp is not None else float(self.hamp)

        self.non_dimensional = non_dimensional
        self.U = float(np.sqrt(gravity_phys * havg_phys))   # gravity-wave speed (m/s)
        self.havg_phys = havg_phys                           # physical reference height (m), for
                                                               # rescaling other physical-unit constants
                                                               # (e.g. initial_condition.py's umax/noise)
        self.hamp_phys = hamp_phys                           # physical reference height amplitude (m);
                                                               # kept alongside havg_phys so a checkpoint's
                                                               # model_info.json can record both without
                                                               # threading them through extra call sites.

        if self.non_dimensional:
            self.T = radius_phys / self.U                     # time scale (s); dt ends up in units of T
            self.radius = torch.as_tensor(1.0, dtype=torch.float64)
            self.gravity = torch.as_tensor(1.0, dtype=torch.float64)
            self.havg = torch.as_tensor(1.0, dtype=torch.float64)
            self.hamp = torch.as_tensor(hamp_phys / havg_phys, dtype=torch.float64)
            self.omega = torch.as_tensor(omega_phys * self.T, dtype=torch.float64)
            self.umax = umax / self.U
        else:
            self.T = 1.0                                       # dt is already physical seconds
            self.radius = torch.as_tensor(radius_phys, dtype=torch.float64)
            self.gravity = torch.as_tensor(gravity_phys, dtype=torch.float64)
            self.havg = torch.as_tensor(havg_phys, dtype=torch.float64)
            self.hamp = torch.as_tensor(hamp_phys, dtype=torch.float64)
            self.omega = torch.as_tensor(omega_phys, dtype=torch.float64)
            self.umax = umax

        # relaxation-rate coefficient for dudtspec's rad term, in the solver's own
        # time units (mirrors the hyperdiffusion tau->dt_seconds conversion below;
        # self.T=1 in the dimensional branch, so this formula is unified across both).
        self.inv_tau_rad = (self.T / (self.tau_rad_days * 86400.)) if self.rad else 0.0

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
        # .contiguous(): expand() is a stride-0 view (all columns share memory),
        # which register_buffer accepts fine but load_state_dict's in-place
        # copy_ into it later can't (overlapping destination elements) - give
        # the buffer real, non-aliased storage instead.
        l = l.expand(self.lmax, self.mmax).contiguous()
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
        # tau is a physical e-fold time in hours regardless of non_dimensional, so the
        # ratio needs the *physical* elapsed time per step (dt*T collapses to dt when
        # T=1, i.e. the dimensional case).
        dt_seconds = self.dt * self.T
        damping_n2 = (-dt_seconds / (tau_2 * 3600.)) * (k_normalized ** 2)
        damping_n4 = (-dt_seconds / (tau_4 * 3600.)) * (k_normalized ** 4)
        damping_n8 = (-dt_seconds / (tau_8 * 3600.)) * (k_normalized ** 8)

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
        # zonally-symmetric equilibrium geopotential dudtspec relaxes toward when
        # rad=True; IC/dataset-specific so it starts at zero and is filled in by
        # set_equilibrium_geopotential (not part of the persisted checkpoint config).
        self.register_buffer('phi_eq_spec', torch.zeros(self.lmax, self.mmax, dtype=torch.complex128))
        
        
        ### Basic Logging ###
        integrator = 'semi_implicit' if self.semi_implicit is True else 'explicit 2nd order Adam-Bashforth'
        
        content_config_log = {
            'title' : "SWE Psuedo-Spectral Solver Configuration" ,
            'lines' : [
                f"Spectral resolution (l_max,mmax) = {self.lmax, self.mmax} | (nlon, nlat) = ({self.nlon}, {self.nlat})",
                f"Integrator is {integrator}| CFL constant is {self.cfl} | dt = {self.dt:.4g} ({'non-dim' if self.non_dimensional else 's'})",
                f"non_dimensional = {self.non_dimensional} | U = {self.U:.2f} m/s | T = {self.T:.2f} s",
                f"Numerical Quadrature : {self.grid}",
                f"Dealiasing by 3/2 padding : {str(self.dealias)}",
                f"Radiative relaxation (rad) : {self.rad}" + (f" | tau_rad = {self.tau_rad_days:.4g} days" if self.rad else ""),
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

    def set_equilibrium_geopotential(self, phi_eq_spec):
        """Set phi_eq, the zonally-symmetric background dudtspec's rad term relaxes
        geopotential toward (see dudtspec). Must be called before timestep() on a
        rad=True solver -- phi_eq is IC/dataset-specific (derived from the ERA5
        dataset's own climatology, see initial_condition.radiative_equilibrium_geopotential),
        so it isn't a constructor argument and isn't part of the persisted checkpoint.
        """
        if not self.rad:
            raise RuntimeError("set_equilibrium_geopotential called but solver was constructed with rad=False")
        self.phi_eq_spec.copy_(torch.tril(phi_eq_spec).to(self.phi_eq_spec.dtype))

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
        # vorticity is measured in 1/s so we normalize using sqrt(g h) / r : ratio of earth's radi traveled per second
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

        # Newtonian relaxation of geopotential toward the equilibrium background
        # phi_eq (mimicking radiation) -- linear in phi, so this is applied directly
        # in spectral space rather than round-tripping through grid space.
        if self.rad:
            dudtspec[0] = dudtspec[0] - self.inv_tau_rad * (uspec[0] - self.phi_eq_spec)

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
    
    
    