import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import PINN, DEVICE, DTYPE, ANALYTICAL_DIR
from src.loss import theta_from_psi_numpy


def load_model(model_dir: Path) -> PINN:
    model = PINN().to(DEVICE)
    state = torch.load(model_dir / 'model.pt', map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_on_grid(model: PINN, psi_true: np.ndarray):
    Nz, Nt = psi_true.shape
    z_vals = np.arange(Nz, dtype=float)  # 0-9cm (10点)
    t_vals = np.arange(Nt, dtype=float)

    Z, T = np.meshgrid(z_vals, t_vals, indexing='ij')
    X = np.stack([np.zeros_like(Z), np.zeros_like(Z), Z, T], axis=-1)
    X = torch.tensor(X.reshape(-1, 4), dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        psi_pred = model(X).cpu().numpy().reshape(Nz, Nt)

    h_pred = psi_pred + z_vals[:, None]
    h_true = psi_true + z_vals[:, None]
    return h_true, h_pred, psi_true, psi_pred, z_vals, t_vals


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


def save_report(model_dir: Path, lines: list):
    out = model_dir / 'analysis_results.txt'
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
    h_true, h_pred, psi_true_grid, psi_pred_grid, z_vals, t_vals = evaluate_on_grid(model, psi_true)

    # Overall metrics (h)
    overall = compute_metrics(h_true, h_pred)
    icbc = compute_ic_bc_errors(h_true, h_pred)

    # Theta metrics per time
    theta_r, theta_s, alpha, n = 0.06, 0.40, 1.0, 2.0

    def nearest_idx(arr, v):
        return int(np.abs(arr - v).argmin())

    times = [0, 0.1, 0.5, 1, 2, 4, 10, 32]
    rows = []
    for t in times:
        ti = nearest_idx(t_vals, t)
        psi_t_true = psi_true_grid[:, ti]
        psi_t_pred = psi_pred_grid[:, ti]
        theta_true = theta_from_psi_numpy(psi_t_true, theta_r, theta_s, alpha, n)
        theta_pred = theta_from_psi_numpy(psi_t_pred, theta_r, theta_s, alpha, n)
        mae = mean_absolute_error(theta_true, theta_pred)
        rmse = np.sqrt(mean_squared_error(theta_true, theta_pred))
        # shallow/deep
        shallow_mask = z_vals <= 1.0
        deep_mask = z_vals >= 7.0
        mae_shal = mean_absolute_error(theta_true[shallow_mask], theta_pred[shallow_mask])
        mae_deep = mean_absolute_error(theta_true[deep_mask], theta_pred[deep_mask])
        # gradient and front position
        dth_true = np.gradient(theta_true, z_vals)
        dth_pred = np.gradient(theta_pred, z_vals)
        grad_mae = mean_absolute_error(dth_true, dth_pred)
        zf_true = float(z_vals[np.argmax(np.abs(dth_true))])
        zf_pred = float(z_vals[np.argmax(np.abs(dth_pred))])
        rows.append((t, mae, rmse, mae_deep, mae_shal, grad_mae, zf_true, zf_pred, zf_pred - zf_true))

    # Build report
    L = []
    L.append("=== Model Analysis Report ===\n")
    L.append(f"Model dir: {model_dir}\n\n")
    L.append("[Overall h-metrics]\n")
    L.append(f"RMSE: {overall['rmse']:.6e}\n")
    L.append(f"MAE : {overall['mae']:.6e}\n")
    L.append(f"R^2 : {overall['r2']:.6f}\n\n")
    L.append("[IC / BC]\n")
    L.append(f"IC  t=0  RMSE: {icbc['ic_rmse']:.6e}\n")
    L.append(f"BC  z=0  RMSE: {icbc['bc_top_rmse']:.6e}\n\n")
    L.append("[Per-time theta metrics]\n")
    L.append("t,  MAE,   RMSE,  MAE(deep z>=7), MAE(shallow z<=1), grad_MAE, z_front_true, z_front_pred, dz\n")
    for r in rows:
        L.append(f"{r[0]:>5.1f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f}, {r[4]:.4f}, {r[5]:.4f}, {r[6]:.2f}, {r[7]:.2f}, {r[8]:+.2f}\n")

    save_report(model_dir, L)


if __name__ == '__main__':
    main()