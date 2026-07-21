"""
    A minimal version of psuedo_spectral solver based on the previous version by torch-harmonics authors from Nvidia
    
    @author Haoyu Tang 
    
    1) Dealising 
    2) Semi-implicit Method 
"""

import numpy as np
import os
import sys

from torch_harmonics.sht import *
import torch_harmonics as harmonics
from torch_harmonics.quadrature import *
from torch_harmonics.quadrature import _precompute_longitudes, _precompute_latitudes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PsuedoSpectralSolver():
    def __init__(self, l_max, dt, N):
        self.l_max = l_max
        self.dt = dt
        self.N = N
        
        ####### physical grid paramters ######
        # 3/2 dealising condition 
        self.nlat = l_max * 3/2 + 20
        self.nlon = 2 * self.nlat
        ######################################
        
        ####### define transformations ########
        # A. Spectral Transformations 
        # Note triangular truncation with mmax = l0
        grid_type = "equitrangular"
        self.ssht = harmonics.RealSHT(self.nlat, self.nlon, lmax=self.l_max, mmax=self.l_max, grid=grid_type, csphase=False)
        self.issht = harmonics.InverseRealSHT(self.nlat, self.nlon, lmax=l0, mmax=l0, grid=grid_type, csphase=False)
        self.vsht = harmonics.RealVectorSHT(self.nlat, self.nlon, lmax=l0, mmax=l0, grid=grid_type, csphase=False)
        self.ivsht = harmonics.InverseRealVectorSHT(self.nlat, self.nlon, lmax=l0, mmax=l0, grid=grid_type, csphase=False)
        
        # B. Diagonalized Operator Transformation 
        # Note with the vSHT,  we expand the field by the basis {ɸ, 𝟁}
        # where 
        #   (a) ɸ is the divergence-free poloidal basis, k⨉∇Y
        #   (b) 𝟁 is the curl-free toroidal basis ∇Y
        # Taking the curl of vSHT(u,v) kills the toroidal terms, furthermore in the poloidal basis
        # the curl is diagonalized with eigenvalues -l(l+1)/a 
        # Likewise, taking the divergence of vSHT(u,v) kilss the poloidal term, in the toroidal basis
        # the divergence is diagonalized with eigenvalues -l(l+1)/a  as well
        # Hence -l(l+a)/a * vSHT(u,v) gives the spectral (𝛇, δ)
        l = torch.linspace(1., self.l_max, steps = self.l_max)
        
        self.eigenvalues = torch.unsqueeze(input=torch.unsqueeze(input=l, dim=-1), dim=0)
        ######################################
        
        
     
    

    def vel_to_vortdiv(self, velocity):
        """Convert (u,v) to (𝜻, δ) 

        Args:
            velocity (_type_): _description_
        """
        return self.ivsht(self.lap(self.vsht(velocity)))

        
        