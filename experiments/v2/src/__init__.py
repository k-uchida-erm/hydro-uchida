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
    load_all_data
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
    
    # 学習
    'train'
]

# =============================================================================
# src package
# =============================================================================
# このパッケージは、PINNモデルの実装を含みます：
# 
# 設定 (config)
#   - デバイス設定（CPU/GPU）
#   - 数値計算のパラメータ（グリッドサイズ、時間ステップ）
#   - 学習パラメータ（エポック数、バッチサイズ、学習率）
#   - 損失関数の重み
#   - データディレクトリのパス
# 
# モデル (model)
#   - 物理情報ニューラルネットワークの実装
#   - 入力: (x, y, z, t) → 出力: 水頭 h
# 
# 損失関数 (loss)
#   - 物理方程式の残差（地下水の流動方程式）
#   - 境界条件の損失（Dirichlet/Neumann境界条件）
#   - 初期条件の損失
#   - 観測データの損失（MSE）
# 
# データローダー (loader)
#   - 地盤データの読み込み（透水係数、貯留係数）
#   - 境界条件の読み込み
#   - 初期条件の読み込み
#   - 観測データの読み込み
# 
# 学習 (train)
#   - モデルの学習ループ
#   - 損失の計算と最適化
#   - 学習の進捗表示
#   - モデルの保存
# ============================================================================= 