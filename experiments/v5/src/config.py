import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

GRID = dict(
    Nx=1, Ny=1, Nz=10,
    dx=1.0, dy=1.0, dz=1.0,
    Nt=50  # 0-10時間の範囲（公式と同じ）
)
DT = 1.0  # 0-10時間の範囲で50ステップ（0.2時間間隔）

EPOCHS = 2000
BATCH_SIZE = 4096
LEARNING_RATE = 5e-5
MAX_GRAD_NORM = 1.0

LOSS_WEIGHTS = {
    'pde': 1.0,
    'bc': 10.0,
    'ic': 10.0,
    'obs': 5.0,
    'theta_obs': 20.0
}

# 共通データ参照
GLOBAL_DATA = Path(__file__).resolve().parents[1] / "global_data"
LOCAL_ANALYTICAL = GLOBAL_DATA / "ddpinns_srivastava" / "analytical_solutions"
LOCAL_PINNS_DATA = GLOBAL_DATA / "ddpinns_srivastava" / "PINNs_codes" / "data"

FALLBACK_PINNS_ROOT = Path("/Users/hydro1/ToshiyukiBandai-DD-PINNs-RRE-5de1644")
FALLBACK_ANALYTICAL = FALLBACK_PINNS_ROOT / "analytical_solutions"
FALLBACK_PINNS_DATA = FALLBACK_PINNS_ROOT / "PINNs_codes" / "data"

ANALYTICAL_DIR = LOCAL_ANALYTICAL if LOCAL_ANALYTICAL.exists() else FALLBACK_ANALYTICAL
PINNS_DATA_DIR = LOCAL_PINNS_DATA if LOCAL_PINNS_DATA.exists() else FALLBACK_PINNS_DATA

NETWORK = dict(
    layers=[2, 50, 50, 50, 50, 1],
    activation='tanh'
)

# ===== Fixed settings to preserve current best behavior =====
CONFIG_VERSION = "v5-fixed-20250908"

# Reproducibility
SEED = 42
DETERMINISTIC = True

# Boundary conditions (v4_ddpinns equivalent)
BC_NEUMANN_FLUX = -0.9

# Loss weights (must match current good run)
FIXED_WEIGHTS = {
    'pde': 1.0,
    'bc': 40.0,
    'ic': 30.0,
    'obs': 8.0,
    'theta_obs': 40.0
}

# Residual sampling
RAR_N_RES = 20000
RAR_FOCUS_RATIO = 0.4  # z∈[7,10], t∈[0,4] 比率