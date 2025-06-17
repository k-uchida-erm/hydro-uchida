# =============================================================================
# 設定ファイル
# =============================================================================
# このファイルは、シミュレーション全体で使用される設定値を管理します：
# 1. デバイス設定（CPU/GPU）
# 2. 数値計算のパラメータ（グリッドサイズ、時間ステップなど）
# 3. 学習パラメータ（エポック数、バッチサイズ、学習率など）
# 4. 損失関数の重み
# 5. データディレクトリのパス
# =============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
from pathlib import Path

# 乱数シードの固定
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# デバイス設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# ケース設定
CASE = 1  # 1: 鉛直1次元, 2: 鉛直2次元

# グリッド設定
if CASE == 1:
    # ケース1: 鉛直1次元
    GRID = {
        'Nx': 1, 'Ny': 1, 'Nz': 100,  # 空間分割数
        'dx': 0.1, 'dy': 0.1, 'dz': 0.1,  # 格子間隔 (m)
        'Nt': 100  # 時間ステップ数
    }
    DT = 0.01  # 時間ステップ幅 (s)
else:
    # ケース2: 鉛直2次元
    GRID = {
        'Nx': 1, 'Ny': 40, 'Nz': 30,  # 空間分割数
        'dx': 0.05, 'dy': 0.05, 'dz': 0.1,  # 格子間隔 (m)
        'Nt': 100  # 時間ステップ数
    }
    DT = 0.01  # 時間ステップ幅 (s)

# 学習パラメータ
EPOCHS = 10000
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0

# 損失関数の重み
LOSS_WEIGHTS = {
    'pde': 1.0,
    'bc': 1.0,
    'ic': 1.0,
    'obs': 1.0
}

# データディレクトリ
DATA_DIR = Path('data') / f'case{CASE}'