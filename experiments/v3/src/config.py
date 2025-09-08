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

# デバイス設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# グリッド設定
GRID = dict(
    Nx=64, Ny=64, Nz=32,  # グリッド数
    dx=1.0, dy=1.0, dz=0.5,  # グリッド間隔 [m]
    Nt=10  # 時間ステップ数
)
DT = 3600.0  # [s] 時間ステップ

# 学習パラメータ
EPOCHS = 1000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0               # 勾配クリッピングの閾値

# 損失関数の重み
LOSS_WEIGHTS = {
    'pde': 0.1,    # PDE Lossの重みを下げる
    'bc': 1.0,     # 境界条件の重みは維持
    'ic': 1.0,     # 初期条件の重みは維持
    'obs': 10.0    # 観測データの重みを上げる
}

# データディレクトリ（共通data/v3）
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "v3"