import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DTYPE, DEVICE


def compute_theta_numpy(psi: np.ndarray, theta_r: float, theta_s: float, alpha: float, n: float) -> np.ndarray:
    m = 1.0 - 1.0 / n
    mask_sat = (psi >= 0)
    phi_abs = np.clip(np.abs(psi), 1e-10, 1e10)
    Se_unsat = (1.0 + (alpha * phi_abs) ** n) ** (-m)
    Se = np.where(mask_sat, 1.0, Se_unsat)
    Se = np.clip(Se, 1e-10, 1.0)
    theta = theta_r + (theta_s - theta_r) * Se
    return theta


def plot_theta_maps_paper(model_dir: Path, model, psi_true: np.ndarray):
    from src.loader import load_all_data
    
    soil_map, *_ = load_all_data()
    p = soil_map.iloc[0]
    alpha = float(p['alpha'])
    n = float(p['n'])
    theta_r = float(p['theta_r'])
    theta_s = float(p['theta_s'])

    Nz, Nt = psi_true.shape
    z_vals = np.arange(Nz, dtype=float)  # 0-9cm (10点)
    t_vals = np.arange(Nt, dtype=float)

    Z, T = np.meshgrid(z_vals, t_vals, indexing='ij')
    X = np.stack([np.zeros_like(Z), np.zeros_like(Z), Z, T], axis=-1)
    X_torch = torch.tensor(X.reshape(-1, 4), dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        psi_pred = model(X_torch).cpu().numpy().reshape(Nz, Nt)

    theta_pred = compute_theta_numpy(psi_pred, theta_r, theta_s, alpha, n)
    theta_true = compute_theta_numpy(psi_true, theta_r, theta_s, alpha, n)
    theta_diff = theta_pred - theta_true

    t_min, t_max = 0.0, 10.0
    z_min, z_max = -10.0, 0.0
    extent = (t_min, t_max, z_min, z_max)

    vmin, vmax = 0.1, 0.4
    dv = 0.002

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(theta_pred, origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, cmap='jet')
    axes[0].set_title('Predicted $\\theta$')
    axes[0].set_xlabel('$t$ [h]')
    axes[0].set_ylabel('$z$ [cm]')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(theta_true, origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, cmap='jet')
    axes[1].set_title('True $\\theta$')
    axes[1].set_xlabel('$t$ [h]')
    axes[1].set_ylabel('$z$ [cm]')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(theta_diff, origin='lower', aspect='auto', extent=extent, vmin=-dv, vmax=dv, cmap='bwr')
    axes[2].set_title('Predicted $-$ True')
    axes[2].set_xlabel('$t$ [h]')
    axes[2].set_ylabel('$z$ [cm]')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    out = model_dir / 'plots' / 'theta_maps_paper.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"saved: {out}")
