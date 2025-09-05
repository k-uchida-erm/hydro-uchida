import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

GRID = dict(
    Nx=1, Ny=1, Nz=100,
    dx=1.0, dy=1.0, dz=1.0,
    Nt=50
)
DT = 1.0

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

LOCAL_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
LOCAL_ANALYTICAL = LOCAL_DATA_ROOT / "analytical_solutions"
LOCAL_PINNS_DATA = LOCAL_DATA_ROOT / "PINNs_codes" / "data"

FALLBACK_PINNS_ROOT = Path("/Users/hydro1/ToshiyukiBandai-DD-PINNs-RRE-5de1644")
FALLBACK_ANALYTICAL = FALLBACK_PINNS_ROOT / "analytical_solutions"
FALLBACK_PINNS_DATA = FALLBACK_PINNS_ROOT / "PINNs_codes" / "data"

ANALYTICAL_DIR = LOCAL_ANALYTICAL if LOCAL_ANALYTICAL.exists() else FALLBACK_ANALYTICAL
PINNS_DATA_DIR = LOCAL_PINNS_DATA if LOCAL_PINNS_DATA.exists() else FALLBACK_PINNS_DATA

NETWORK = dict(
    layers=[2, 50, 50, 50, 50, 1],
    activation='tanh'
)