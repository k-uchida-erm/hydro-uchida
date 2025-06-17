# =============================================================================
# モデルチェックスクリプト
# =============================================================================
# このスクリプトは、学習済みモデルの状態を確認します：
# 1. モデルの構造とパラメータ数の表示
# 2. 損失履歴の確認
# 3. モデルの保存状態の確認
# =============================================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from .config import CASE, GRID, DT, DEVICE, DTYPE
from .model import PINN
from .loader import (
    load_soil_params, load_boundary_conditions,
    load_initial_conditions, load_observation_data
)

def list_models(result_dir):
    """利用可能なモデルディレクトリの一覧を表示"""
    model_dirs = [d for d in os.listdir(result_dir) if d.startswith("model_")]
    if not model_dirs:
        print("モデルディレクトリが見つかりません")
        return []
    
    print("\n利用可能なモデル:")
    for i, model_dir in enumerate(sorted(model_dirs, reverse=True), 1):
        print(f"{i}. {model_dir}")
    return model_dirs

def check_model(model_dir):
    """モデルの状態を確認"""
    model_path = os.path.join('result', model_dir, 'model.pt')
    loss_path = os.path.join('result', model_dir, 'loss_history.csv')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # モデルの読み込み
    model = PINN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # モデル構造の表示
    print("\nモデル構造:")
    print(model)
    
    # パラメータ数の表示
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    
    # 損失履歴の読み込みと表示
    if os.path.exists(loss_path):
        df_loss = pd.read_csv(loss_path)
        print("\n損失履歴:")
        print(f"最終エポック: {df_loss['epoch'].iloc[-1]}")
        print(f"最終損失値: {df_loss['total_loss'].iloc[-1]:.2e}")
        
        print("\n各損失の最終値:")
        print(f"  PDE Loss: {df_loss['pde_loss'].iloc[-1]:.2e}")
        print(f"  BC Loss: {df_loss['bc_loss'].iloc[-1]:.2e}")
        print(f"  IC Loss: {df_loss['ic_loss'].iloc[-1]:.2e}")
        print(f"  Obs Loss: {df_loss['obs_loss'].iloc[-1]:.2e}")
        
        # 学習率の表示
        if 'learning_rate' in df_loss.columns:
            print(f"  Learning Rate: {df_loss['learning_rate'].iloc[-1]:.2e}")
    
    # 観測データとの比較による評価指標の計算
    try:
        _, _, _, _, df_obs = load_all_data()
        
        # 観測点での予測値を計算
        X_obs = torch.tensor(df_obs[['x', 'y', 'z', 't']].values, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            h_pred = model(X_obs).cpu().numpy()
        
        # 実際の観測値
        h_true = df_obs['h'].values
        
        # 評価指標の計算
        rmse = np.sqrt(np.mean((h_pred - h_true) ** 2))
        mae = np.mean(np.abs(h_pred - h_true))
        max_ae = np.max(np.abs(h_pred - h_true))
        r2 = 1 - np.sum((h_true - h_pred) ** 2) / np.sum((h_true - np.mean(h_true)) ** 2)
        
        print("\n観測データとの比較:")
        print(f"  RMSE: {rmse:.4f} [m]")
        print(f"  MAE: {mae:.4f} [m]")
        print(f"  MaxAE: {max_ae:.4f} [m]")
        print(f"  R²: {r2:.4f}")
        
        # 誤差の分布を表示
        print("\n誤差の統計:")
        errors = h_pred - h_true
        print(f"  平均誤差: {np.mean(errors):.4f} [m]")
        print(f"  標準偏差: {np.std(errors):.4f} [m]")
        print(f"  最小誤差: {np.min(errors):.4f} [m]")
        print(f"  最大誤差: {np.max(errors):.4f} [m]")
        
    except Exception as e:
        print(f"\n観測データとの比較を計算できませんでした: {str(e)}")

def check_pde_residual(model, soil_params):
    """PDEの残差を計算"""
    # グリッド点の生成
    if CASE == 1:
        z = torch.linspace(0, GRID['Nz'] * GRID['dz'], GRID['Nz'] + 1, dtype=DTYPE, device=DEVICE)
        t = torch.linspace(0, GRID['Nt'] * DT, GRID['Nt'] + 1, dtype=DTYPE, device=DEVICE)
        Z, T = torch.meshgrid(z, t, indexing='ij')
        X = torch.zeros_like(Z)
        Y = torch.zeros_like(Z)
    else:
        y = torch.linspace(0, GRID['Ny'] * GRID['dy'], GRID['Ny'] + 1, dtype=DTYPE, device=DEVICE)
        z = torch.linspace(0, GRID['Nz'] * GRID['dz'], GRID['Nz'] + 1, dtype=DTYPE, device=DEVICE)
        t = torch.linspace(0, GRID['Nt'] * DT, GRID['Nt'] + 1, dtype=DTYPE, device=DEVICE)
        Y, Z, T = torch.meshgrid(y, z, t, indexing='ij')
        X = torch.zeros_like(Y)
    
    # 微分を計算
    derivatives = model.compute_derivatives(X.flatten(), Y.flatten(), Z.flatten(), T.flatten())
    
    # 土壌パラメータ
    Ks = soil_params['Ks']
    theta_s = soil_params['theta_s']
    theta_r = soil_params['theta_r']
    Ss = soil_params['Ss']
    alpha = soil_params['alpha']
    n = soil_params['n']
    
    # 有効飽和度を計算
    h = derivatives['h']
    Se = (1 + (alpha * torch.abs(h))**n)**(-(1-1/n))
    
    # 不飽和透水係数を計算
    Kr = Se**0.5 * (1 - (1 - Se**(n/(n-1)))**(1-1/n))**2
    K = Ks * Kr
    
    # 比水分容量を計算
    C = alpha * (theta_s - theta_r) * n * (alpha * torch.abs(h))**(n-1) * Se**(n+1)
    
    # Richards方程式の残差を計算
    if CASE == 1:
        residual = C * derivatives['dh_dt'] - torch.autograd.grad(
            K * derivatives['dh_dz'],
            Z.flatten(),
            grad_outputs=torch.ones_like(K * derivatives['dh_dz']),
            create_graph=True
        )[0]
    else:
        residual = C * derivatives['dh_dt'] - (
            torch.autograd.grad(
                K * derivatives['dh_dy'],
                Y.flatten(),
                grad_outputs=torch.ones_like(K * derivatives['dh_dy']),
                create_graph=True
            )[0] +
            torch.autograd.grad(
                K * derivatives['dh_dz'],
                Z.flatten(),
                grad_outputs=torch.ones_like(K * derivatives['dh_dz']),
                create_graph=True
            )[0]
        )
    
    return {
        'mean': residual.mean().item(),
        'std': residual.std().item(),
        'max': residual.max().item(),
        'min': residual.min().item()
    }

def check_boundary_conditions(model, bc_data):
    """境界条件の満足度を計算"""
    points = bc_data['points']
    values = bc_data['values']
    types = bc_data['types']
    
    x, y, z, t = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    h = model.predict(x, y, z, t)
    
    errors = []
    for i, bc_type in enumerate(types):
        if bc_type == 'head':
            error = (h[i] - values[i])**2
        elif bc_type == 'flux':
            derivatives = model.compute_derivatives(x[i:i+1], y[i:i+1], z[i:i+1], t[i:i+1])
            flux = -derivatives['dh_dz'][0]
            error = (flux - values[i])**2
        errors.append(error.item())
    
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'max': np.max(errors),
        'min': np.min(errors)
    }

def check_initial_conditions(model, ic_data):
    """初期条件の満足度を計算"""
    points = ic_data['points']
    values = ic_data['values']
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    t = torch.zeros_like(x)
    h = model.predict(x, y, z, t)
    
    errors = (h - values)**2
    
    return {
        'mean': errors.mean().item(),
        'std': errors.std().item(),
        'max': errors.max().item(),
        'min': errors.min().item()
    }

def check_observation_data(model, obs_data):
    """観測データとの比較"""
    if obs_data is None:
        return None
    
    points = obs_data['points']
    values = obs_data['values']
    
    x, y, z, t = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    h = model.predict(x, y, z, t)
    
    errors = (h - values)**2
    
    return {
        'mean': errors.mean().item(),
        'std': errors.std().item(),
        'max': errors.max().item(),
        'min': errors.min().item()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    args = parser.parse_args()
    
    # モデルの読み込み
    model = PINN().to(DEVICE)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    
    # データの読み込み
    soil_params = load_soil_params()
    bc_data = load_boundary_conditions()
    ic_data = load_initial_conditions()
    obs_data = load_observation_data()
    
    # 検証の実行
    results = {
        'pde_residual': check_pde_residual(model, soil_params),
        'boundary_conditions': check_boundary_conditions(model, bc_data),
        'initial_conditions': check_initial_conditions(model, ic_data)
    }
    
    if obs_data is not None:
        results['observation_data'] = check_observation_data(model, obs_data)
    
    # 結果の保存
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = result_dir / f'validation_case{CASE}_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 結果の表示
    print("\n=== モデル検証結果 ===")
    print(f"\nPDE残差:")
    for metric, value in results['pde_residual'].items():
        print(f"  {metric}: {value:.2e}")
    
    print(f"\n境界条件の満足度:")
    for metric, value in results['boundary_conditions'].items():
        print(f"  {metric}: {value:.2e}")
    
    print(f"\n初期条件の満足度:")
    for metric, value in results['initial_conditions'].items():
        print(f"  {metric}: {value:.2e}")
    
    if 'observation_data' in results:
        print(f"\n観測データとの比較:")
        for metric, value in results['observation_data'].items():
            print(f"  {metric}: {value:.2e}")
    
    print(f"\n結果を保存しました: {result_file}")

if __name__ == '__main__':
    main() 