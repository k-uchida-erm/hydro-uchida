import torch
import torch.nn as nn
from .config import *

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 重みの初期化を改善
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)

# 有効飽和度
def Se(phi, a, n):
    m = 1.0 - 1.0/n
    
    # 飽和領域
    mask_sat = (phi >= 0)
    
    # 不飽和領域
    mask_unsat = ~mask_sat
    phi_abs = torch.clamp(phi.abs(), min=1e-10, max=1e10)  # 極端な値を防ぐ
    term = (a * phi_abs) ** n
    term = torch.clamp(term, min=1e-10, max=1e10)  # 極端な値を防ぐ
    Se_unsat = (1 + term) ** (-m)
    
    # 結果の結合
    result = torch.where(mask_sat, torch.ones_like(phi), Se_unsat)
    return torch.clamp(result, min=1e-10, max=1.0)  # 有効飽和度の範囲を制限

# 含水率
def theta(phi, a, n, tr, ts):
    Se_val = Se(phi, a, n)
    return tr + (ts-tr)*Se_val

# 含水率の水頭微分
def dtheta_dh(phi, a, n, tr, ts):
    m = 1.0 - 1.0/n
    
    # 飽和領域
    mask_sat = (phi >= 0)
    
    # 不飽和領域
    mask_unsat = ~mask_sat
    phi_abs = torch.clamp(phi.abs(), min=1e-10, max=1e10)  # 極端な値を防ぐ
    Se_val = Se(phi, a, n)
    
    # 微分項の計算
    term1 = (a * phi_abs) ** (n-1)
    term1 = torch.clamp(term1, min=1e-10, max=1e10)  # 極端な値を防ぐ
    term2 = Se_val ** (1/m)
    term2 = torch.clamp(term2, min=1e-10, max=1.0)  # 有効飽和度の範囲を制限
    
    dSe = -m * term1 * a * n * term2
    dSe *= torch.sign(phi) * -1.0
    
    # 結果の結合
    result = torch.where(mask_sat, torch.zeros_like(phi), (ts-tr)*dSe)
    return torch.clamp(result, min=-1e10, max=1e10)  # 極端な値を防ぐ

# 不飽和透水係数
def K_unsat(phi, Ks, a, n):
    m = 1.0 - 1.0/n
    Se_val = Se(phi, a, n)
    
    # 飽和領域
    mask_sat = (phi >= 0)
    
    # 不飽和領域
    mask_unsat = ~mask_sat
    term = (1 - (1-Se_val**(1/m))**m)**2
    K_unsat = Ks * Se_val.sqrt() * term
    
    # 結果の結合
    result = torch.where(mask_sat, Ks, K_unsat)
    return torch.clamp(result, min=1e-30, max=1e30)  # 極端な値を防ぐ
