import torch
import torch.nn as nn
from .config import *


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        input_dim = 4
        hidden_dims = [50, 50, 50, 50]
        output_dim = 1
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def Se(phi, a, n):
    m = 1.0 - 1.0/n
    mask_sat = (phi >= 0)
    phi_abs = torch.clamp(phi.abs(), min=1e-10, max=1e10)
    term = (a * phi_abs) ** n
    Se_unsat = (1 + term) ** (-m)
    return torch.where(mask_sat, torch.ones_like(phi), Se_unsat).clamp(min=1e-10, max=1.0)


def theta(phi, a, n, tr, ts):
    return tr + (ts - tr) * Se(phi, a, n)


def dtheta_dh(phi, a, n, tr, ts):
    m = 1.0 - 1.0/n
    mask_sat = (phi >= 0)
    phi_abs = torch.clamp(phi.abs(), min=1e-10, max=1e10)
    Se_val = Se(phi, a, n)
    term1 = (a * phi_abs) ** (n - 1)
    term2 = Se_val ** (1/m)
    dSe = -m * term1 * a * n * term2
    dSe *= torch.sign(phi) * -1.0
    return torch.where(mask_sat, torch.zeros_like(phi), (ts - tr) * dSe).clamp(min=-1e10, max=1e10)


def K_unsat(phi, Ks, a, n):
    m = 1.0 - 1.0/n
    Se_val = Se(phi, a, n)
    mask_sat = (phi >= 0)
    term = (1 - (1 - Se_val ** (1/m)) ** m) ** 2
    K_unsat = Ks * Se_val.sqrt() * term
    return torch.where(mask_sat, Ks, K_unsat).clamp(min=1e-30, max=1e30)