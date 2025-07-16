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
EPOCHS = 2000
BATCH_SIZE = 2048
LEARNING_RATE = 5e-5
MAX_GRAD_NORM = 1.0               # 勾配クリッピングの閾値

# 損失関数の重み
LOSS_WEIGHTS = {
    'pde': 1.0,    # PDE Lossの重みを上げる
    'bc': 1.0,     # 境界条件の重みは維持
    'ic': 5.0,     # 初期条件の重みを上げる
    'obs': 5.0     # 観測データの重みを調整
}

# データディレクトリ
DATA_DIR = Path(__file__).parent.parent / "data"