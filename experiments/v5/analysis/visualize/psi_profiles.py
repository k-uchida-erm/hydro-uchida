import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from src.config import DTYPE, DEVICE


def plot_psi_profiles(model_dir: Path, model, ds_path: Path):
    """psi(z,t) の真値と予測を t={0,1,2,4,10,32} で重ね描き。
    横軸: 圧力水頭 psi [cm]（-10..0）
    縦軸: depth z [cm]（上=10, 下=0）
    """
    ds = xr.load_dataset(ds_path)
    psi_true = ds['psi'].to_numpy().astype(float)  # (Nz, Nt)
    z_vals = ds['z'].to_numpy().astype(float)      # (Nz,)
    t_vals = ds['t'].to_numpy().astype(float)      # (Nt,)

    # 時間のインデックスを決定（存在しない値は最近傍にスナップ）
    target_times = [0, 1, 2, 4, 10, 32]
    def nearest_idx(arr, v):
        return int(np.abs(arr - v).argmin())
    t_indices = [nearest_idx(t_vals, v) for v in target_times]

    z_tensor = torch.tensor(z_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    x0 = torch.zeros_like(z_tensor)
    y0 = torch.zeros_like(z_tensor)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # 真値（psi）
    for i, t_idx in enumerate(t_indices):
        psi_true_t = psi_true[:, t_idx]
        ax.plot(psi_true_t, z_vals, 'k-', lw=2, label='True' if i == 0 else "")

    # 予測（psi）
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    markers = ['o', 'x', '^', 'v', 's', 'D']
    for i, (t_idx, t_val) in enumerate(zip(t_indices, target_times)):
        t_tensor = torch.full_like(z_tensor, float(t_idx))
        X = torch.cat([x0, y0, z_tensor, t_tensor], dim=1)
        with torch.no_grad():
            psi_pred = model(X).cpu().numpy().squeeze()
        ax.plot(psi_pred, z_vals, color=colors[i], marker=markers[i], linestyle='', markersize=3,
                label=f't={t_val}h')

    ax.set_xlabel('Pressure Head ψ [cm]')
    ax.set_ylabel('Depth z [cm]')
    ax.set_xlim(-10, 0)
    ax.set_ylim(0, 10)  # 上が10,下が0（通常の向き）
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_title('Pressure Head Profiles')
    plt.tight_layout()

    out = model_dir / 'plots' / 'psi_profiles.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"saved: {out}")


