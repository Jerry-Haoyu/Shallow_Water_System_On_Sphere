from abc import ABC, abstractmethod
import torch
import torch.nn as nn

# inherit from nn.Module to allow easy .to(device)
class AbstractSWSolver(ABC, nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        
        # Note this does not move the model actually to device, 
        # which should be done outside after instantiation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # physical sonstants
        self.register_buffer('radius', torch.as_tensor(6.37122E6, dtype=torch.float64))
        self.register_buffer('omega', torch.as_tensor(7.292E-5, dtype=torch.float64))
        self.register_buffer('gravity', torch.as_tensor(9.80616, dtype=torch.float64))
        self.register_buffer('havg', torch.as_tensor(10.e3, dtype=torch.float64))
        self.register_buffer('hamp', torch.as_tensor(120, dtype=torch.float64))
    
    @abstractmethod
    def timestep(self, uspec: torch.Tensor, nsteps: int):
        pass
        
        