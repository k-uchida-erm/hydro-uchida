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

GRID = dict(Nx=64, Ny=64, Nz=32, dx=1.0, dy=1.0, dz=0.5)
DT = 3600.0                       # [s] オイラー陰ステップ
EPOCHS = 20000
BATCH_INT = 4096                  # PDE 内部点/バッチ
LEARNING_RATE = 1e-3
W = dict(PDE=1.0, BC=10.0, IC=10.0, OBS=1.0)

# データディレクトリ
DATA_DIR = Path(__file__).parent.parent / "data"