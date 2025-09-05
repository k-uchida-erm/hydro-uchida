import numpy as np
import pandas as pd
import torch
from .config import *


def load_all_data():
    psi_true = np.load(ANALYTICAL_DIR / "Srivastava_psi_homogeneous.npy")
    Z_res = np.load(PINNS_DATA_DIR / "residual_points.npy")

    Nt = psi_true.shape[1]
    Nz = psi_true.shape[0]
    data_list = []
    for zi in range(Nz):
        for ti in range(Nt):
            z_val = float(zi)
            h_val = float(psi_true[zi, ti] + z_val)
            data_list.append({
                'x': 0.0,
                'y': 0.0,
                'z': z_val,
                't': float(ti),
                'h': h_val
            })
    df_train = pd.DataFrame(data_list)
    df_obs = df_train.copy()

    psi_ic = psi_true[:, 0]
    X_ic = torch.zeros((Nz, 3), dtype=DTYPE, device=DEVICE)
    X_ic[:, 2] = torch.arange(Nz, dtype=DTYPE, device=DEVICE)
    psi0 = torch.tensor(psi_ic, dtype=DTYPE, device=DEVICE).unsqueeze(1)

    bc_list = []
    
    for ti in range(Nt):
        bc_list.append({
            'type': 'Neumann',
            'x': 0.0, 'y': 0.0, 'z': 0.0, 't': float(ti),
            'value': -0.9,
            'nx': 0.0, 'ny': 0.0, 'nz': 1.0
        })
    
    theta_r = 0.06
    theta_s = 0.40
    alpha = 1.0
    psi_0 = 0.0
    # 下側境界のθ値をより正確に設定（深い部分の再現精度向上）
    theta_lb = theta_r + (theta_s - theta_r) * np.exp(alpha * psi_0)
    
    for ti in range(Nt):
        bc_list.append({
            'type': 'Dirichlet',
            'x': 0.0, 'y': 0.0, 'z': float(Nz-1), 't': float(ti),
            'value': theta_lb,
            'nx': 0.0, 'ny': 0.0, 'nz': -1.0
        })
    
    df_bc = pd.DataFrame(bc_list)

    soil_map = pd.DataFrame([
        {'x': 0.0, 'y': 0.0, 'z': 0.0, 'alpha': 1.0, 'n': 2.0, 'theta_r': 0.06, 'theta_s': 0.40, 'Ks': 1.0, 'Ss': 1e-4}
    ])

    z_col0 = Z_res[:, 0]
    t_col1 = Z_res[:, 1]
    z_col1 = Z_res[:, 1]
    t_col0 = Z_res[:, 0]

    def to_tensor(z_np, t_np):
        z_clamped = np.clip(z_np, 0, Nz - 1)
        t_clamped = np.clip(t_np, 0, Nt - 1)
        X = torch.zeros((len(z_clamped), 4), dtype=DTYPE, device=DEVICE)
        X[:, 2] = torch.tensor(z_clamped, dtype=DTYPE, device=DEVICE)
        X[:, 3] = torch.tensor(t_clamped, dtype=DTYPE, device=DEVICE)
        return X

    X_res_a = to_tensor(z_col0, t_col1)
    X_res_b = to_tensor(z_col1, t_col0)

    use_b = X_res_b[:, 2].std().item() + X_res_b[:, 3].std().item() > X_res_a[:, 2].std().item() + X_res_a[:, 3].std().item()
    X_res = X_res_b if use_b else X_res_a

    return soil_map, df_bc, X_ic, psi0, df_train, df_obs, X_res