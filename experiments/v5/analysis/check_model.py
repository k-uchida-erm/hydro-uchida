import argparse
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import PINN, DEVICE, DTYPE


def load_model(model_dir: Path) -> PINN:
    model = PINN().to(DEVICE)
    state = torch.load(model_dir / 'model.pt', map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_on_grid(model: PINN, psi_true: np.ndarray, z_vals: np.ndarray, t_vals: np.ndarray):
    Nz, Nt = psi_true.shape

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
    # NetCDFから真値psiと座標を取得
    ds_path = Path('/usr/src/app/global_data/vertical_1d_infiltration/dataset.nc')
    ds = xr.load_dataset(ds_path)
    psi_true = ds['psi'].to_numpy().astype(float)  # (Nz,Nt)
    z_vals = ds['z'].to_numpy().astype(float)
    t_vals = ds['t'].to_numpy().astype(float)

    h_true, h_pred = evaluate_on_grid(model, psi_true, z_vals, t_vals)

    # 総合指標
    overall = compute_metrics(h_true, h_pred)
    icbc = compute_ic_bc_errors(h_true, h_pred)

    # 時刻別（psiベース）評価
    def nearest_idx(arr, v):
        return int(np.abs(arr - v).argmin())
    times = [0, 1, 2, 4, 10, 32]
    time_rows = []
    for t in times:
        ti = nearest_idx(t_vals, t)
        psi_pred_t = h_pred[:, ti] - z_vals
        psi_true_t = h_true[:, ti] - z_vals
        mae = mean_absolute_error(psi_true_t, psi_pred_t)
        rmse = np.sqrt(mean_squared_error(psi_true_t, psi_pred_t))
        # 深部/浅部
        deep_mask = z_vals >= 7.0
        shallow_mask = z_vals <= 1.0
        mae_deep = mean_absolute_error(psi_true_t[deep_mask], psi_pred_t[deep_mask])
        mae_shal = mean_absolute_error(psi_true_t[shallow_mask], psi_pred_t[shallow_mask])
        # 勾配誤差
        dpsi_true = np.gradient(psi_true_t, z_vals)
        dpsi_pred = np.gradient(psi_pred_t, z_vals)
        grad_mae = mean_absolute_error(dpsi_true, dpsi_pred)
        # フロント位置（|dpsi/dz|最大のz）
        zf_true = float(z_vals[np.argmax(np.abs(dpsi_true))])
        zf_pred = float(z_vals[np.argmax(np.abs(dpsi_pred))])
        time_rows.append((t, mae, rmse, mae_deep, mae_shal, grad_mae, zf_true, zf_pred, zf_pred - zf_true))

    # テキスト生成
    L = []
    L.append("=== Model Analysis Report ===\n")
    L.append(f"Model dir: {model_dir}\n\n")
    L.append("[Overall h-metrics]\n")
    L.append(f"RMSE: {overall['rmse']:.6e}\n")
    L.append(f"MAE : {overall['mae']:.6e}\n")
    L.append(f"R^2 : {overall['r2']:.6f}\n\n")
    L.append("[IC/BC]\n")
    L.append(f"IC t=0 RMSE: {icbc['ic_rmse']:.6e}\n")
    L.append(f"BC top(z={z_vals.min():.1f}) RMSE: {icbc['bc_top_rmse']:.6e}\n\n")
    L.append("[Per-time psi metrics]\n")
    L.append("t,  MAE,   RMSE,  MAE(deep z>=7), MAE(shallow z<=1), grad_MAE, z_front_true, z_front_pred, dz\n")
    for row in time_rows:
        L.append(f"{row[0]:>5.1f}, {row[1]:.4f}, {row[2]:.4f}, {row[3]:.4f}, {row[4]:.4f}, {row[5]:.4f}, {row[6]:.2f}, {row[7]:.2f}, {row[8]:+.2f}\n")

    save_report(model_dir, L)


if __name__ == '__main__':
    main()