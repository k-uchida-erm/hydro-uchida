# =============================================================================
# 結果可視化スクリプト
# =============================================================================
# このスクリプトは、学習結果を可視化します：
# 1. 損失の履歴
# 2. 予測値と観測値の比較
# 3. 水頭分布の3D可視化
# =============================================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
import json
from datetime import datetime
from .config import CASE, GRID, DT, DEVICE, DTYPE
from .model import PINN
from .loader import (
    load_soil_params, load_boundary_conditions,
    load_initial_conditions, load_observation_data
)

def ensure_dir(directory):
    """ディレクトリが存在しない場合は作成"""
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def plot_loss_history(history_file):
    """損失関数の推移をプロット"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    losses = {
        'total': [h['total_loss'] for h in history],
        'pde': [h['pde_loss'] for h in history],
        'bc': [h['bc_loss'] for h in history],
        'ic': [h['ic_loss'] for h in history],
        'obs': [h['obs_loss'] for h in history]
    }
    
    plt.figure(figsize=(10, 6))
    for name, values in losses.items():
        plt.semilogy(epochs, values, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    
    # 保存
    save_path = Path('result') / f'loss_history_case{CASE}.png'
    plt.savefig(save_path)
    plt.close()

def plot_prediction_vs_observation(model, df_obs, output_dir):
    """予測値と観測値の比較をプロット"""
    # 観測点での予測値を計算
    x = torch.tensor(df_obs['x'].values, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(df_obs['y'].values, dtype=torch.float32).to(DEVICE)
    t = torch.tensor(df_obs['t'].values, dtype=torch.float32).to(DEVICE)
    z = torch.tensor(df_obs['z'].values, dtype=torch.float32).to(DEVICE)
    
    # 入力データを結合
    X = torch.stack([x, y, z, t], dim=1)
    
    with torch.no_grad():
        h_pred = model(X).cpu().numpy()
    
    h_obs = df_obs['h'].values
    
    plt.figure(figsize=(10, 6))
    plt.scatter(h_obs, h_pred, alpha=0.5)
    plt.plot([h_obs.min(), h_obs.max()], [h_obs.min(), h_obs.max()], 'r--')
    
    plt.xlabel('Observed Head')
    plt.ylabel('Predicted Head')
    plt.title('Prediction vs Observation')
    plt.grid(True)
    
    # plotsサブディレクトリに保存
    plots_dir = os.path.join(output_dir, 'plots')
    ensure_dir(plots_dir)
    plt.savefig(os.path.join(plots_dir, 'pred_vs_obs.png'))
    plt.close()

def plot_head_distribution(model, t, output_dir):
    """水頭分布の3D可視化"""
    # メッシュの作成
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    
    # 入力データの準備
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    t_array = np.full((len(points), 1), t)
    z_array = np.zeros((len(points), 1))  # z座標を追加
    X_input = np.hstack([points, z_array, t_array])  # (x, y, z, t)の形式
    
    # 予測値の計算
    with torch.no_grad():
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(DEVICE)
        h_pred = model(X_tensor).cpu().numpy()
    
    h_pred = h_pred.reshape(X.shape)
    
    # 3Dプロット
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, h_pred, cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Head')
    ax.set_title(f'Head Distribution at t={t}')
    
    fig.colorbar(surf)
    
    # head_distributionサブディレクトリに保存
    plots_dir = os.path.join(output_dir, 'plots')
    head_dir = os.path.join(plots_dir, 'head_distribution')
    ensure_dir(plots_dir)
    ensure_dir(head_dir)
    plt.savefig(os.path.join(head_dir, f'head_distribution_t{t}.png'))
    plt.close()

def plot_head_distribution_1d(model, soil_params):
    """1次元水頭分布の時系列プロット"""
    # グリッド点の生成
    z = torch.linspace(0, GRID['Nz'] * GRID['dz'], GRID['Nz'] + 1, dtype=DTYPE, device=DEVICE)
    t_points = [0, 0.25, 0.5, 0.75, 1.0]  # 時間ポイント
    
    plt.figure(figsize=(10, 6))
    for t in t_points:
        # 予測
        x = torch.zeros_like(z)
        y = torch.zeros_like(z)
        t_tensor = torch.full_like(z, t)
        h = model.predict(x, y, z, t_tensor).cpu().numpy()
        
        # プロット
        plt.plot(h, z.cpu().numpy(), label=f't = {t:.2f}')
    
    plt.xlabel('Head (m)')
    plt.ylabel('Depth (m)')
    plt.title('1D Head Distribution')
    plt.legend()
    plt.grid(True)
    
    # 保存
    save_path = Path('result') / f'head_1d_case{CASE}.png'
    plt.savefig(save_path)
    plt.close()

def plot_head_distribution_2d(model, soil_params):
    """2次元水頭分布の時系列プロット"""
    # グリッド点の生成
    y = torch.linspace(0, GRID['Ny'] * GRID['dy'], GRID['Ny'] + 1, dtype=DTYPE, device=DEVICE)
    z = torch.linspace(0, GRID['Nz'] * GRID['dz'], GRID['Nz'] + 1, dtype=DTYPE, device=DEVICE)
    t_points = [0, 0.25, 0.5, 0.75, 1.0]  # 時間ポイント
    
    Y, Z = torch.meshgrid(y, z, indexing='ij')
    
    for t in t_points:
        plt.figure(figsize=(10, 6))
        
        # 予測
        x = torch.zeros_like(Y)
        t_tensor = torch.full_like(Y, t)
        h = model.predict(x, Y, Z, t_tensor).cpu().numpy()
        
        # プロット
        plt.contourf(Y.cpu().numpy(), Z.cpu().numpy(), h.reshape(Y.shape))
        plt.colorbar(label='Head (m)')
        plt.xlabel('Y (m)')
        plt.ylabel('Z (m)')
        plt.title(f'2D Head Distribution (t = {t:.2f})')
        
        # 保存
        save_path = Path('result') / f'head_2d_case{CASE}_t{t:.2f}.png'
        plt.savefig(save_path)
        plt.close()

def plot_boundary_conditions(bc_data):
    """境界条件のプロット"""
    points = bc_data['points'].cpu().numpy()
    values = bc_data['values'].cpu().numpy()
    types = bc_data['types']
    
    plt.figure(figsize=(10, 6))
    
    # 時間ごとにグループ化
    unique_times = np.unique(points[:, 3])
    for t in unique_times:
        mask = points[:, 3] == t
        t_points = points[mask]
        t_values = values[mask]
        t_types = [types[i] for i, m in enumerate(mask) if m]
        
        # プロット
        for bc_type in ['head', 'flux']:
            type_mask = [tp == bc_type for tp in t_types]
            if any(type_mask):
                plt.scatter(
                    t_points[type_mask, 2],  # z座標
                    t_values[type_mask],
                    label=f'{bc_type} (t={t:.2f})'
                )
    
    plt.xlabel('Z (m)')
    plt.ylabel('Value')
    plt.title('Boundary Conditions')
    plt.legend()
    plt.grid(True)
    
    # 保存
    save_path = Path('result') / f'boundary_conditions_case{CASE}.png'
    plt.savefig(save_path)
    plt.close()

def plot_soil_parameters(soil_params):
    """土壌パラメータのプロット"""
    params = {
        'Ks': soil_params['Ks'].item(),
        'theta_s': soil_params['theta_s'].item(),
        'theta_r': soil_params['theta_r'].item(),
        'Ss': soil_params['Ss'].item(),
        'alpha': soil_params['alpha'].item(),
        'n': soil_params['n'].item()
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(params.keys(), params.values())
    plt.xticks(rotation=45)
    plt.title('Soil Parameters')
    plt.grid(True)
    
    # 保存
    save_path = Path('result') / f'soil_parameters_case{CASE}.png'
    plt.savefig(save_path)
    plt.close()

def plot_1d_results(model, soil_params, save_dir):
    """1次元ケースの結果を可視化"""
    # グリッド点の生成
    z = torch.linspace(0, GRID['Nz'] * GRID['dz'], GRID['Nz'] + 1, dtype=DTYPE, device=DEVICE)
    t = torch.linspace(0, GRID['Nt'] * DT, GRID['Nt'] + 1, dtype=DTYPE, device=DEVICE)
    Z, T = torch.meshgrid(z, t, indexing='ij')
    X = torch.zeros_like(Z)
    Y = torch.zeros_like(Z)
    
    # 予測値の計算
    h = model.predict(X.flatten(), Y.flatten(), Z.flatten(), T.flatten())
    h = h.reshape(Z.shape)
    
    # 有効飽和度の計算
    Ks = soil_params['Ks']
    theta_s = soil_params['theta_s']
    theta_r = soil_params['theta_r']
    alpha = soil_params['alpha']
    n = soil_params['n']
    
    Se = (1 + (alpha * torch.abs(h))**n)**(-(1-1/n))
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 水頭分布
    im1 = ax1.pcolormesh(T.cpu().numpy(), Z.cpu().numpy(), h.cpu().numpy(), shading='auto')
    ax1.set_xlabel('Time [day]')
    ax1.set_ylabel('Depth [m]')
    ax1.set_title('Pressure Head')
    plt.colorbar(im1, ax=ax1, label='h [m]')
    
    # 有効飽和度分布
    im2 = ax2.pcolormesh(T.cpu().numpy(), Z.cpu().numpy(), Se.cpu().numpy(), shading='auto')
    ax2.set_xlabel('Time [day]')
    ax2.set_ylabel('Depth [m]')
    ax2.set_title('Effective Saturation')
    plt.colorbar(im2, ax=ax2, label='Se [-]')
    
    plt.tight_layout()
    plt.savefig(save_dir / '1d_results.png')
    plt.close()

def plot_2d_results(model, soil_params, save_dir):
    """2次元ケースの結果を可視化"""
    # グリッド点の生成
    y = torch.linspace(0, GRID['Ny'] * GRID['dy'], GRID['Ny'] + 1, dtype=DTYPE, device=DEVICE)
    z = torch.linspace(0, GRID['Nz'] * GRID['dz'], GRID['Nz'] + 1, dtype=DTYPE, device=DEVICE)
    t = torch.linspace(0, GRID['Nt'] * DT, GRID['Nt'] + 1, dtype=DTYPE, device=DEVICE)
    Y, Z, T = torch.meshgrid(y, z, t, indexing='ij')
    X = torch.zeros_like(Y)
    
    # 予測値の計算
    h = model.predict(X.flatten(), Y.flatten(), Z.flatten(), T.flatten())
    h = h.reshape(Y.shape)
    
    # 有効飽和度の計算
    Ks = soil_params['Ks']
    theta_s = soil_params['theta_s']
    theta_r = soil_params['theta_r']
    alpha = soil_params['alpha']
    n = soil_params['n']
    
    Se = (1 + (alpha * torch.abs(h))**n)**(-(1-1/n))
    
    # 時間ステップを選択
    time_indices = [0, GRID['Nt']//4, GRID['Nt']//2, 3*GRID['Nt']//4, GRID['Nt']]
    
    # プロット
    fig, axes = plt.subplots(2, len(time_indices), figsize=(15, 8))
    
    for i, t_idx in enumerate(time_indices):
        # 水頭分布
        im1 = axes[0, i].pcolormesh(Y[:, :, t_idx].cpu().numpy(), 
                                   Z[:, :, t_idx].cpu().numpy(), 
                                   h[:, :, t_idx].cpu().numpy(), 
                                   shading='auto')
        axes[0, i].set_xlabel('y [m]')
        axes[0, i].set_ylabel('z [m]')
        axes[0, i].set_title(f't = {t_idx*DT:.2f} day')
        plt.colorbar(im1, ax=axes[0, i], label='h [m]')
        
        # 有効飽和度分布
        im2 = axes[1, i].pcolormesh(Y[:, :, t_idx].cpu().numpy(), 
                                   Z[:, :, t_idx].cpu().numpy(), 
                                   Se[:, :, t_idx].cpu().numpy(), 
                                   shading='auto')
        axes[1, i].set_xlabel('y [m]')
        axes[1, i].set_ylabel('z [m]')
        plt.colorbar(im2, ax=axes[1, i], label='Se [-]')
    
    plt.tight_layout()
    plt.savefig(save_dir / '2d_results.png')
    plt.close()

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
    
    # 可視化の実行
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    
    # 損失履歴のプロット
    history_file = result_dir / f'history_case{CASE}_*.json'
    history_files = list(result_dir.glob(f'history_case{CASE}_*.json'))
    if history_files:
        plot_loss_history(str(history_files[-1]))  # 最新の履歴ファイルを使用
    
    # 水頭分布のプロット
    if CASE == 1:
        plot_head_distribution_1d(model, soil_params)
    else:
        plot_head_distribution_2d(model, soil_params)
    
    # 境界条件のプロット
    plot_boundary_conditions(bc_data)
    
    # 土壌パラメータのプロット
    plot_soil_parameters(soil_params)
    
    # 結果保存ディレクトリの作成
    save_dir = Path('result/visualization')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ケースに応じた可視化
    if CASE == 1:
        plot_1d_results(model, soil_params, save_dir)
    else:
        plot_2d_results(model, soil_params, save_dir)
    
    print("可視化が完了しました")
    print(f"可視化結果を保存しました: {save_dir}")

if __name__ == '__main__':
    main() 