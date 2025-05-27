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
from src.model import PINN
from src.config import DEVICE

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
    # モデルの読み込み
    model = PINN().to(DEVICE)
    model_path = os.path.join('result', model_dir, 'model.pt')
    
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"\nモデル {model_dir} の状態:")
    
    # モデルの構造とパラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nモデル構造:")
    print(model)
    print(f"\n総パラメータ数: {total_params:,}")
    
    # 損失履歴の確認
    loss_file = os.path.join('result', model_dir, 'loss_history.csv')
    if os.path.exists(loss_file):
        df_loss = pd.read_csv(loss_file)
        print("\n損失履歴:")
        print(f"最終エポック: {df_loss['epoch'].max()}")
        print(f"最終損失値: {df_loss['total_loss'].iloc[-1]:.2e}")
        print("\n各損失の最終値:")
        print(f"  PDE Loss: {df_loss['pde_loss'].iloc[-1]:.2e}")
        print(f"  BC Loss: {df_loss['bc_loss'].iloc[-1]:.2e}")
        print(f"  IC Loss: {df_loss['ic_loss'].iloc[-1]:.2e}")
        print(f"  Obs Loss: {df_loss['obs_loss'].iloc[-1]:.2e}")
    else:
        print("\n警告: 損失履歴ファイルが見つかりません")

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