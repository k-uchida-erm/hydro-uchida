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


def plot_theta_profiles_evolution(model_dir: Path, model, psi_true: np.ndarray):
    from src.loader import load_all_data
    
    soil_map, *_ = load_all_data()
    p = soil_map.iloc[0]
    alpha = float(p['alpha'])
    n = float(p['n'])
    theta_r = float(p['theta_r'])
    theta_s = float(p['theta_s'])

    Nz, Nt = psi_true.shape
    z_vals = np.arange(Nz, dtype=float)  # 0-9cm (10点)
    
    t_paper = [0, 0.1, 0.5, 1, 3, 5, 10]
    t_indices = [int(t * (Nt-1) / 10) for t in t_paper]
    
    z_tensor = torch.tensor(z_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    x0 = torch.zeros_like(z_tensor)
    y0 = torch.zeros_like(z_tensor)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for i, t_idx in enumerate(t_indices):
        if t_idx < Nt:
            psi_true_t = psi_true[:, t_idx]
            theta_true_t = compute_theta_numpy(psi_true_t, theta_r, theta_s, alpha, n)
            ax.plot(theta_true_t, z_vals, 'k-', lw=2, label='True' if i == 0 else "")
    
    colors = ['blue', 'pink', 'gray', 'teal', 'lightgreen', 'purple', 'black']
    markers = ['o', 'x', '^', 'v', 's', 'D', '+']
    
    for i, (t_idx, t_val) in enumerate(zip(t_indices, t_paper)):
        if t_idx < Nt:
            t_tensor = torch.full_like(z_tensor, float(t_idx))
            X = torch.cat([x0, y0, z_tensor, t_tensor], dim=1)
            with torch.no_grad():
                psi_pred = model(X).cpu().numpy().squeeze()
            theta_pred = compute_theta_numpy(psi_pred, theta_r, theta_s, alpha, n)
            ax.plot(theta_pred, z_vals, color=colors[i], marker=markers[i], 
                   linestyle='', markersize=4, label=f't = {t_val} [h]')
    
    ax.set_xlabel('Volumetric Water Content θ [-]')
    ax.set_ylabel('Depth z [cm]')
    ax.set_title('PINNs Learning Evolution')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.invert_yaxis()
    plt.tight_layout()
    
    out = model_dir / 'plots' / 'theta_profiles_evolution.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved: {out}")
