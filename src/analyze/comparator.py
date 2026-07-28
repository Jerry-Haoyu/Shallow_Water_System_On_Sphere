"""
Given at least 2 of 3 sources from ["rw", "numeric", "learned"], compare the result via
1). Prediction of (φ, δ, 𝛇) 
2). Spectrum 
3). Cyclone Tracking

@author Haoyu Tang hytang2@illinois.edu
"""

class Comparator:
    def __init__(self,
                 netcdf_path,
                 num_model,
                 ml_model,
                 initial_condition
                 ):
        """
            To include L.H.S in benchmarking, make the R.H.S a passed in parameter
                realworld : netcdf_path
                numerical simulation data : num_model and initial_condition 
                SFNO prediction : ml_model and initial_condition
        """
        if (num_model is not None and initial_condition is None) or \
            (ml_model is not None and initial_condition is None):
                raise RuntimeError("Must provided initial condition since either numerical model or ML model requires it!")
        
    
    def realworld_data
    def spectrum(self,):
        raise NotImplementedError