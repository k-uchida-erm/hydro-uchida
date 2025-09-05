import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import PINN, DEVICE, ANALYTICAL_DIR
from src.loader import load_all_data
from visualize import (
    plot_loss_history,
    plot_head_distributions,
    plot_head_distributions_paper,
    plot_pred_vs_obs,
    plot_theta_maps_paper,
    plot_theta_profiles_evolution
)


def load_model(model_dir: Path):
    model = PINN().to(DEVICE)
    state_path = model_dir / 'model.pt'
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='model directory name under result/')
    parser.add_argument('--paper', action='store_true', help='format axes/labels to mimic paper figures')
    args = parser.parse_args()

    exp_root = Path(__file__).resolve().parents[1]
    result_root = exp_root / 'result'

    if args.model:
        model_dir = result_root / args.model
    else:
        candidates = sorted(result_root.glob('model_*'))
        if not candidates:
            print(f"No models found under: {result_root}")
            return
        model_dir = candidates[-1]

    print(f"Using model: {model_dir}")
    model = load_model(model_dir)

    psi_true = np.load(ANALYTICAL_DIR / 'Srivastava_psi_homogeneous.npy')
    *_, df_obs, _ = load_all_data()

    if args.paper:
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.figsize': (6, 4)
        })

    plot_loss_history(model_dir)

    Nt = psi_true.shape[1]
    Nz = psi_true.shape[0]
    z_vals = np.arange(Nz, dtype=float)
    h_all_true = (psi_true + z_vals[:, None])
    hmin = float(h_all_true.min())
    hmax = float(h_all_true.max())
    pad = 0.05 * (hmax - hmin) if hmax > hmin else 1.0
    h_limits = (hmin - pad, hmax + pad)

    if args.paper:
        plot_head_distributions_paper(model_dir, model, psi_true, h_limits)
    else:
        plot_head_distributions(model_dir, model, psi_true)

    plot_pred_vs_obs(model_dir, model, df_obs)
    
    if args.paper:
        plot_theta_maps_paper(model_dir, model, psi_true)
        plot_theta_profiles_evolution(model_dir, model, psi_true)


if __name__ == '__main__':
    main()