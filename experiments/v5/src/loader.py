import numpy as np
import pandas as pd
import torch
import xarray as xr
from .config import *


def load_all_data():
    """NetCDF(dataset.nc) から学習・観測・IC・BCを読み込む。

    - 主量は psi(z,t) [cm]
    - 全水頭は h = psi + z で導出
    """
    # データはコンテナ内で /usr/src/app/global_data にマウントされている
    ds_path = Path(__file__).resolve().parents[1] / 'global_data' / 'vertical_1d_infiltration' / 'dataset.nc'
    ds = xr.load_dataset(ds_path)

    # 座標
    z_vals = ds['z'].to_numpy().astype(float)
    t_vals = ds['t'].to_numpy().astype(float)
    Nz = len(z_vals)
    Nt = len(t_vals)

    # 学習用（教師あり）: v4と同様に全水頭hで監督する
    psi_grid = ds['psi'].to_numpy().astype(float)  # (Nz, Nt)
    # 展開してテーブル化
    rows = []
    for zi in range(Nz):
        for ti in range(Nt):
            z = float(z_vals[zi])
            t = float(t_vals[ti])
            psi = float(psi_grid[zi, ti])
            rows.append({'x': 0.0, 'y': 0.0, 'z': z, 't': t, 'h': psi + z})
    df_train = pd.DataFrame(rows)

    # 時刻ごとの部分ホールドアウト（各tでzの一部を検証に回す）
    # 各時刻での部分ホールドアウト
    # 既定は15%、ただしフロント帯(7-10cm)と表面(<=1cm)は10%に抑制して学習点を厚く残す
    base_val_fraction = 0.15
    front_val_fraction = 0.10
    surface_val_fraction = 0.10
    rng = np.random.default_rng(42)
    rows_val = []
    mask_keep = np.ones(len(df_train), dtype=bool)
    # z座標の代表点を層化（浅/中/深を均等に）
    z_bins = np.linspace(z_vals.min(), z_vals.max(), 11)  # 10区間でより細かく
    for t in np.unique(df_train['t'].values):
        idx_t = np.where(df_train['t'].values == t)[0]
        z_t = df_train['z'].values[idx_t]
        # 各binから一定割合をvalへ
        for b in range(len(z_bins)-1):
            bin_mask = (z_t >= z_bins[b]) & (z_t < z_bins[b+1])
            idx_bin = idx_t[bin_mask]
            if len(idx_bin) == 0:
                continue
            # バンドに応じてval率を変更
            z_mid = 0.5 * (z_bins[b] + z_bins[b+1])
            if z_mid <= 1.0:
                frac = surface_val_fraction
            elif z_mid >= 7.0:
                frac = front_val_fraction
            else:
                frac = base_val_fraction
            k = max(1, int(len(idx_bin) * frac))
            chosen = rng.choice(idx_bin, size=k, replace=False)
            rows_val.extend(chosen.tolist())
    mask_keep[rows_val] = False
    df_obs = df_train.iloc[rows_val].reset_index(drop=True)
    df_train = df_train.iloc[mask_keep.nonzero()[0]].reset_index(drop=True)

    # 初期条件 psi0（圧力水頭）
    if 'ic_psi0' in ds.variables and len(ds['ic_psi0']) == Nz and np.allclose(ds['z_ic'], z_vals):
        psi0_arr = ds['ic_psi0'].to_numpy().astype(float)
    else:
        psi0_arr = psi_grid[:, 0]
    X_ic = torch.zeros((Nz, 3), dtype=DTYPE, device=DEVICE)
    X_ic[:, 2] = torch.tensor(z_vals, dtype=DTYPE, device=DEVICE)
    psi0 = torch.tensor(psi0_arr, dtype=DTYPE, device=DEVICE).unsqueeze(1)

    # 境界条件は v4_ddpinns と同じ規約で合成（データに依存しない）
    # 上端（z=max）: ψ の Neumann フラックス（定数 -0.9）
    # 下端（z=min）: θ の Dirichlet（theta_lb）
    theta_r = 0.06
    theta_s = 0.40
    alpha = 1.0
    psi_0 = 0.0
    theta_lb = theta_r + (theta_s - theta_r) * np.exp(alpha * psi_0)
    bc_rows = []
    z_top = float(z_vals.max())
    z_bot = float(z_vals.min())
    for t in t_vals:
        bc_rows.append({
            'type': 'Neumann',
            'x': 0.0, 'y': 0.0, 'z': z_top, 't': float(t),
            'value': BC_NEUMANN_FLUX,
            'nx': 0.0, 'ny': 0.0, 'nz': 1.0
        })
        bc_rows.append({
            'type': 'Dirichlet',
            'x': 0.0, 'y': 0.0, 'z': z_bot, 't': float(t),
            'value': theta_lb,
            'nx': 0.0, 'ny': 0.0, 'nz': -1.0
        })
    df_bc = pd.DataFrame(bc_rows)

    # 残差点: 湿潤フロント帯を強調（z∈[7,10], t∈[0,4] に50%）
    N_res = RAR_N_RES
    N_focus = int(N_res * RAR_FOCUS_RATIO)
    N_rest = N_res - N_focus
    z_focus = np.random.uniform(7.0, 10.0, size=N_focus)
    t_focus = np.random.uniform(0.0, min(4.0, float(t_vals.max())), size=N_focus)
    # 後半時刻を増やす
    z_rest = np.random.uniform(z_vals.min(), z_vals.max(), size=N_rest)
    t_rest = np.concatenate([
        np.random.uniform(t_vals.min(), min(4.0, float(t_vals.max())), size=N_rest//2),
        np.random.uniform(min(4.0, float(t_vals.max())), t_vals.max(), size=N_rest - N_rest//2)
    ])
    z_res = np.concatenate([z_focus, z_rest])
    t_res = np.concatenate([t_focus, t_rest])
    X_res = torch.zeros((N_res, 4), dtype=DTYPE, device=DEVICE)
    X_res[:, 2] = torch.tensor(z_res, dtype=DTYPE, device=DEVICE)
    X_res[:, 3] = torch.tensor(t_res, dtype=DTYPE, device=DEVICE)

    # 土壌パラメータ（単層・定数）
    soil_map = pd.DataFrame([
        {'x': 0.0, 'y': 0.0, 'z': 0.0, 'alpha': 1.0, 'n': 2.0, 'theta_r': 0.06, 'theta_s': 0.40, 'Ks': 1.0, 'Ss': 1e-4}
    ])

    return soil_map, df_bc, X_ic, psi0, df_train, df_obs, X_res