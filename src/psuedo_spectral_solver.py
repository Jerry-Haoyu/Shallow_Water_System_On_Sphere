# a minimal version of psuedo_spectral solver 

import numpy as np
import os
import sys

from torch_harmonics.sht import *
import torch_harmonics as harmonics
from torch_harmonics.quadrature import *
from torch_harmonics.quadrature import _precompute_longitudes, _precompute_latitudes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PsuedoSpectralSolver():
    def __init__(self, l0, dt, N):
        self.l0 = l0
        self.dt = dt
        self.N = N
        
        ####### physical grid paramters ######
        # 3/2 dealising condition 
        self.nlat = l0 * 3/2 + 20
        self.nlon = 2 * self.nlat
        ######################################
        
        ####### define transformations ########
        # triangular truncation with mmax = l0
        grid_type = "equitrangular"
        self.ssht = harmonics.RealSHT(self.nlat, self.nlon, lmax=l0, mmax=l0, grid=grid_type, csphase=False)
        self.issht = harmonics.InverseRealSHT(self.nlat, self.nlon, lmax=l0, mmax=l0, grid=grid_type, csphase=False)
        self.vsht = harmonics.RealVectorSHT(self.nlat, self.nlon, lmax=l0, mmax=l0, grid=grid_type, csphase=False)
        self.ivsht = harmonics.InverseRealVectorSHT(self.nlat, self.nlon, lmax=l0, mmax=l0, grid=grid_type, csphase=False)
        ######################################

        
        