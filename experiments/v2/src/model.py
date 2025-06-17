# =============================================================================
# モデル定義ファイル
# =============================================================================
# このファイルは、以下の2つの主要なコンポーネントを定義します：
# 1. PINN（物理情報ニューラルネットワーク）の構造
#    - 4次元入力（x, y, z, t）から1次元出力（水頭）への変換
#    - 複数の隠れ層とTanh活性化関数
# 2. van Genuchten-Mualemモデル関連の関数
#    - 有効飽和度の計算
#    - 含水率の計算
#    - 不飽和透水係数の計算
# =============================================================================

import torch
import torch.nn as nn
from .config import *

class PINN(nn.Module):
    """4→1 MLP (隠れ層 4, Tanh)"""
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
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x)

def Se(phi, a, n):
    """有効飽和度の計算（数値的安定性を改善）"""
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

def theta(phi, a, n, tr, ts):
    """含水率の計算"""
    Se_val = Se(phi, a, n)
    return tr + (ts-tr)*Se_val

def dtheta_dh(phi, a, n, tr, ts):
    """含水率の水頭微分（数値的安定性を改善）"""
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

def K_unsat(phi, Ks, a, n):
    """不飽和透水係数の計算（数値的安定性を改善）"""
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
