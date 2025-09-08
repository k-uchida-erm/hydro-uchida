import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import PINN, DEVICE
from src.loader import load_all_data
from visualize import (
    plot_loss_history,
    plot_head_distributions,
    plot_head_distributions_paper,
    plot_pred_vs_obs,
    plot_theta_maps_paper,
    plot_theta_profiles_evolution
)
from visualize.psi_profiles import plot_psi_profiles


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

    plot_pred_vs_obs(model_dir, model, df_obs)
    
    if args.paper:
        # NetCDFからpsiを読み、ψプロファイルを描画（コンテナ内のglobal_dataを参照）
        ds_path = Path('/usr/src/app/global_data/vertical_1d_infiltration/dataset.nc')
        plot_psi_profiles(model_dir, model, ds_path)


if __name__ == '__main__':
    main()