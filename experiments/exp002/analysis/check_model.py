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
from src import PINN, DEVICE, load_all_data

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='学習済みモデルの状態を確認')
    parser.add_argument('--list', action='store_true', help='利用可能なモデルの一覧を表示')
    parser.add_argument('--model', type=str, help='確認するモデルディレクトリ名')
    args = parser.parse_args()
    
    result_dir = "result"
    model_dirs = [d for d in os.listdir(result_dir) if d.startswith("model_")]
    
    if not model_dirs:
        print("モデルディレクトリが見つかりません")
        exit(1)
    
    if args.list:
        list_models(result_dir)
        exit(0)
    
    if args.model:
        model_dir = args.model
        if not os.path.exists(os.path.join(result_dir, model_dir)):
            print(f"エラー: ディレクトリ {model_dir} が見つかりません")
            list_models(result_dir)
            exit(1)
    else:
        model_dir = sorted(model_dirs)[-1]
    
    check_model(model_dir) 