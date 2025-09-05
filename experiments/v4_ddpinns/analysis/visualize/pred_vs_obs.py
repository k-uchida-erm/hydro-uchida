import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DTYPE, DEVICE


def plot_pred_vs_obs(model_dir: Path, model, df_obs: pd.DataFrame):
    if df_obs is None or df_obs.empty:
        return
    X = torch.tensor(df_obs[['x','y','z','t']].values, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        psi_pred = model(X).cpu().numpy().squeeze()
    h_pred = psi_pred + df_obs['z'].values
    h_obs = df_obs['h'].values

    plt.figure(figsize=(5,5))
    lims = [min(h_obs.min(), h_pred.min()), max(h_obs.max(), h_pred.max())]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.scatter(h_obs, h_pred, s=10, alpha=0.6)
    plt.xlabel('Observed h (true)')
    plt.ylabel('Predicted h')
    plt.tight_layout()
    out = model_dir / 'plots' / 'pred_vs_obs.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved: {out}")
