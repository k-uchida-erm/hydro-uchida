"""
地下水流動シミュレーション用の物理情報ニューラルネットワーク（PINN）の実装
"""

from .config import *
from .model import PINN
from .loss import residual, bc_loss, ic_loss, obs_loss
from .loader import load_soil, load_bc, load_ic, load_obs
from .train import train

__all__ = [
    'PINN',
    'residual',
    'bc_loss',
    'ic_loss',
    'obs_loss',
    'load_soil',
    'load_bc',
    'load_ic',
    'load_obs',
    'train'
] 