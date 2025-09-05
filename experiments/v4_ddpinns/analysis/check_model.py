import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import PINN, DEVICE, DTYPE, ANALYTICAL_DIR


def load_model(model_dir: Path) -> PINN:
    model = PINN().to(DEVICE)
    state = torch.load(model_dir / 'model.pt', map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_on_grid(model: PINN, psi_true: np.ndarray):
    Nz, Nt = psi_true.shape
    z_vals = np.arange(Nz, dtype=float)
    t_vals = np.arange(Nt, dtype=float)

    Z, T = np.meshgrid(z_vals, t_vals, indexing='ij')
    X = np.stack([np.zeros_like(Z), np.zeros_like(Z), Z, T], axis=-1)
    X = torch.tensor(X.reshape(-1, 4), dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        psi_pred = model(X).cpu().numpy().reshape(Nz, Nt)

    h_pred = psi_pred + z_vals[:, None]
    h_true = psi_true + z_vals[:, None]
    return h_true, h_pred


def compute_metrics(h_true: np.ndarray, h_pred: np.ndarray) -> dict:
    y_true = h_true.reshape(-1)
    y_pred = h_pred.reshape(-1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return dict(rmse=rmse, mae=mae, r2=r2)


def compute_ic_bc_errors(h_true: np.ndarray, h_pred: np.ndarray) -> dict:
    ic_rmse = np.sqrt(mean_squared_error(h_true[:, 0], h_pred[:, 0]))
    bc_top_rmse = np.sqrt(mean_squared_error(h_true[0, :], h_pred[0, :]))
    return dict(ic_rmse=ic_rmse, bc_top_rmse=bc_top_rmse)


def save_report(model_dir: Path, metrics: dict, details: dict):
    out = model_dir / 'analysis_results.txt'
    lines = []
    lines.append("=== Model Analysis Report ===\n")
    lines.append(f"Model dir: {model_dir}\n\n")
    lines.append("[Overall]\n")
    lines.append(f"RMSE: {metrics['rmse']:.6e}\n")
    lines.append(f"MAE : {metrics['mae']:.6e}\n")
    lines.append(f"R^2 : {metrics['r2']:.6f}\n\n")
    lines.append("[IC / BC]\n")
    lines.append(f"IC  t=0  RMSE: {details['ic_rmse']:.6e}\n")
    lines.append(f"BC  z=0  RMSE: {details['bc_top_rmse']:.6e}\n")

    out.write_text(''.join(lines), encoding='utf-8')
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    exp_root = Path(__file__).resolve().parents[1]
    result_root = exp_root / 'result'

    if args.model:
        model_dir = result_root / args.model
    else:
        cands = sorted(result_root.glob('model_*'))
        if not cands:
            print(f"No models found under: {result_root}")
            return
        model_dir = cands[-1]

    model = load_model(model_dir)
    psi_true = np.load(ANALYTICAL_DIR / 'Srivastava_psi_homogeneous.npy')
    h_true, h_pred = evaluate_on_grid(model, psi_true)

    metrics = compute_metrics(h_true, h_pred)
    details = compute_ic_bc_errors(h_true, h_pred)
    save_report(model_dir, metrics, details)


if __name__ == '__main__':
    main()