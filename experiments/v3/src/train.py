# =============================================================================
# 学習ループ定義ファイル
# =============================================================================
# このファイルは、PINNの学習プロセスを定義します：
# 1. 内部点の生成（一様乱数による）
# 2. 各種損失の計算
#    - PDE残差損失
#    - 境界条件損失
#    - 初期条件損失
#    - 観測データ損失
# 3. 勾配降下によるパラメータ更新
# 4. 学習進捗の表示
# =============================================================================

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import pytz
from .config import *
from .loss import residual, bc_loss, ic_loss, obs_loss
from pathlib import Path
import subprocess
import sys
import numpy as np
import json

from .model import PINN
from .loader import (
    load_soil_params, load_boundary_conditions,
    load_initial_conditions, load_observation_data
)
from .loss import compute_total_loss

def ensure_dir(directory):
    """ディレクトリが存在しない場合は作成"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_internal_points():
    """内部点の生成"""
    X_int = torch.rand(BATCH_SIZE, 4, device=DEVICE, dtype=DTYPE)
    X_int[:, 0] *= GRID['Nx'] * GRID['dx']  # x
    X_int[:, 1] *= GRID['Ny'] * GRID['dy']  # y
    X_int[:, 2] *= GRID['Nz'] * GRID['dz']  # z
    X_int[:, 3] *= GRID['Nt'] * DT          # t
    return X_int

def run_analysis(model_dir):
    """モデルの分析と可視化を実行"""
    try:
        # 分析結果を保存するファイル
        analysis_file = os.path.join(model_dir, 'analysis_results.txt')
        
        # 分析コマンドを実行
        check_cmd = f"python3 analysis/check_model.py --model {os.path.basename(model_dir)}"
        visualize_cmd = f"python3 analysis/visualize.py --model {os.path.basename(model_dir)}"
        
        # 分析結果を取得
        check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        
        # 分析結果をファイルに保存
        with open(analysis_file, 'w') as f:
            f.write("=== モデル分析結果 ===\n")
            f.write(f"分析時刻: {datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(check_result.stdout)
            if check_result.stderr:
                f.write("\n=== エラー ===\n")
                f.write(check_result.stderr)
        
        # 可視化を実行
        subprocess.run(visualize_cmd, shell=True)
        
    except Exception as e:
        print(f"分析の実行中にエラーが発生しました: {str(e)}")

def generate_pde_points():
    """PDEの学習点を生成"""
    if CASE == 1:
        # 1次元の場合
        z = torch.linspace(0, GRID['Nz'] * GRID['dz'], GRID['Nz'] + 1, dtype=DTYPE, device=DEVICE)
        t = torch.linspace(0, GRID['Nt'] * DT, GRID['Nt'] + 1, dtype=DTYPE, device=DEVICE)
        
        # メッシュグリッドを作成
        Z, T = torch.meshgrid(z, t, indexing='ij')
        
        # 1次元なのでx, yは0固定
        X = torch.zeros_like(Z)
        Y = torch.zeros_like(Z)
        
    else:
        # 2次元の場合
        y = torch.linspace(0, GRID['Ny'] * GRID['dy'], GRID['Ny'] + 1, dtype=DTYPE, device=DEVICE)
        z = torch.linspace(0, GRID['Nz'] * GRID['dz'], GRID['Nz'] + 1, dtype=DTYPE, device=DEVICE)
        t = torch.linspace(0, GRID['Nt'] * DT, GRID['Nt'] + 1, dtype=DTYPE, device=DEVICE)
        
        # メッシュグリッドを作成
        Y, Z, T = torch.meshgrid(y, z, t, indexing='ij')
        
        # 2次元なのでxは0固定
        X = torch.zeros_like(Y)
    
    return X.flatten(), Y.flatten(), Z.flatten(), T.flatten()

def train():
    """学習を実行"""
    # データを読み込む
    soil_params = load_soil_params()
    bc_data = load_boundary_conditions()
    ic_data = load_initial_conditions()
    obs_data = load_observation_data()
    
    # モデルを初期化
    model = PINN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # PDEの学習点を生成
    pde_points = generate_pde_points()
    
    # 学習ループ
    history = []
    for epoch in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        
        # 損失を計算
        losses = compute_total_loss(
            model, pde_points, bc_data, ic_data, obs_data, soil_params
        )
        
        # 勾配を計算して更新
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        # 履歴を記録
        if epoch % 100 == 0:
            history.append({
                'epoch': epoch,
                'total_loss': losses['total'].item(),
                'pde_loss': losses['pde'].item(),
                'bc_loss': losses['bc'].item(),
                'ic_loss': losses['ic'].item(),
                'obs_loss': losses['obs'].item()
            })
    
    # 学習履歴を保存
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_file = result_dir / f'history_case{CASE}_{timestamp}.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # モデルを保存
    model_file = result_dir / f'model_case{CASE}_{timestamp}.pt'
    torch.save(model.state_dict(), model_file)
    
    return model, history

def main():
    # モデルの初期化
    model = PINN().to(DEVICE)
    
    # データの読み込み
    from loader import load_data
    soil_map, df_bc, X_ic, h0, df_obs = load_data()
    
    # モデルの学習
    model, history = train()
    
    print("学習が完了しました")

if __name__ == "__main__":
    main()
