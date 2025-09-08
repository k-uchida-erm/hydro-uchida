import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DTYPE, DEVICE


def plot_head_distributions(model_dir: Path, model, psi_true: np.ndarray):
    Nt = psi_true.shape[1]
    Nz = psi_true.shape[0]

    times = [0, max(0, Nt//3), max(0, 2*Nt//3), Nt-1]
    times = sorted(set([t for t in times if t < Nt]))

    z_vals = np.arange(Nz, dtype=float)  # 0-9cm (10点)
    z_tensor = torch.tensor(z_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    x0 = torch.zeros_like(z_tensor)
    y0 = torch.zeros_like(z_tensor)

    out_dir = model_dir / 'plots' / 'head_distribution'
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in times:
        t_tensor = torch.full_like(z_tensor, float(t))
        X = torch.cat([x0, y0, z_tensor, t_tensor], dim=1)
        with torch.no_grad():
            psi_pred = model(X).cpu().numpy().squeeze()
        h_pred = psi_pred + z_vals
        h_true = psi_true[:, t] + z_vals

        plt.figure(figsize=(6,5))
        plt.plot(h_true, z_vals, label='True', lw=2)
        plt.plot(h_pred, z_vals, label='Pred', lw=2, ls='--')
        plt.gca().invert_yaxis()
        plt.xlabel('Head h')
        plt.ylabel('Depth z')
        plt.title(f't = {t}')
        plt.legend()
        plt.tight_layout()
        out = out_dir / f'head_distribution_t{t}.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"saved: {out}")


def plot_head_distributions_paper(model_dir: Path, model, psi_true: np.ndarray, h_limits):
    Nt = psi_true.shape[1]
    Nz = psi_true.shape[0]
    z_vals = np.arange(Nz, dtype=float)  # 0-9cm (10点)
    
    times = [0, max(0, Nt//4), max(0, Nt//2), max(0, 3*Nt//4)]
    times = sorted(set([t for t in times if t < Nt]))
    
    z_tensor = torch.tensor(z_vals, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    x0 = torch.zeros_like(z_tensor)
    y0 = torch.zeros_like(z_tensor)
    
    out_dir = model_dir / 'plots' / 'head_distribution'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for t in times:
        t_tensor = torch.full_like(z_tensor, float(t))
        X = torch.cat([x0, y0, z_tensor, t_tensor], dim=1)
        with torch.no_grad():
            psi_pred = model(X).cpu().numpy().squeeze()
        h_pred = psi_pred + z_vals
        h_true = psi_true[:, t] + z_vals
        
        plt.figure()
        plt.plot(h_true, z_vals, label='True', lw=2, color='black')
        plt.plot(h_pred, z_vals, label='Pred', lw=2, ls='--', color='tab:blue')
        plt.gca().invert_yaxis()
        plt.xlabel('h')
        plt.ylabel('z')
        plt.xlim(h_limits)
        plt.ylim(z_vals.max(), z_vals.min())
        plt.tight_layout()
        out = out_dir / f'head_distribution_t{t}.png'
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"saved: {out}")
