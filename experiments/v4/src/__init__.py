"""
地下水流動シミュレーション用の物理情報ニューラルネットワーク（PINN）の実装
"""

__version__ = '0.1.0'

# 設定の明示的なインポート
from .config import (
    DEVICE,
    DTYPE,
    GRID,
    DT,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    LOSS_WEIGHTS,
    DATA_DIR
)

# モデル
from .model import PINN

# 損失関数
from .loss import (
    residual,
    bc_loss,
    ic_loss,
    obs_loss
)

# データローダー
from .loader import (
    load_soil,
    load_bc,
    load_ic,
    load_obs,
    load_all_data,
    load_validation
)

# 学習
from .train import train

__all__ = [
    # バージョン
    '__version__',
    
    # 設定
    'DEVICE',
    'DTYPE',
    'GRID',
    'DT',
    'EPOCHS',
    'BATCH_SIZE',
    'LEARNING_RATE',
    'MAX_GRAD_NORM',
    'LOSS_WEIGHTS',
    'DATA_DIR',
    
    # モデル
    'PINN',
    
    # 損失関数
    'residual',
    'bc_loss',
    'ic_loss',
    'obs_loss',
    
    # データローダー
    'load_soil',
    'load_bc',
    'load_ic',
    'load_obs',
    'load_all_data',
    'load_validation',
    
    # 学習
    'train'
]
