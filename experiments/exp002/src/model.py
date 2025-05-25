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
from config import *

class PINN(nn.Module):
    """4→1 MLP (隠れ層 4, Tanh)"""
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers = [nn.Linear(4, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def Se(phi, a, n):
    m = 1.0 - 1.0/n
    Se = (1 + (a*phi.abs())**n)**(-m)
    return torch.where(phi<0, Se, torch.ones_like(phi))

def theta(phi, a, n, tr, ts):
    return tr + (ts-tr)*Se(phi,a,n)

def dtheta_dh(phi, a, n, tr, ts):
    m = 1.0-1.0/n
    Se_val = Se(phi,a,n)
    dSe = -m * (a*phi.abs())**(n-1) * a * n * Se_val**(1/m)
    dSe *= torch.sign(phi) * -1.0
    return (ts-tr)*dSe

def K_unsat(phi, Ks, a, n):
    m = 1.0-1.0/n
    Se_val = Se(phi,a,n)
    term = (1 - (1-Se_val**(1/m))**m)**2
    return Ks * Se_val.sqrt()*term
