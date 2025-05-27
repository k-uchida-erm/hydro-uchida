# =============================================================================
# モデル確認スクリプト
# =============================================================================
# このスクリプトは、保存されたモデルの構造とパラメータを確認します：
# 1. モデルの読み込み
# 2. パラメータの表示
# 3. モデル構造の表示
# =============================================================================

import torch
import os
from model import PINN
from config import DEVICE

def load_and_check_model(model_path):
    """保存されたモデルを読み込んで確認"""
    # モデルのインスタンス化
    model = PINN().to(DEVICE)
    
    # 保存されたパラメータの読み込み
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    print(f"\nモデルファイル: {os.path.basename(model_path)}")
    print("\n=== モデルの構造 ===")
    print(model)
    
    print("\n=== パラメータの統計情報 ===")
    for name, param in model.named_parameters():
        print(f"\n{name}:")
        print(f"  形状: {param.shape}")
        print(f"  平均: {param.mean().item():.6f}")
        print(f"  標準偏差: {param.std().item():.6f}")
        print(f"  最小値: {param.min().item():.6f}")
        print(f"  最大値: {param.max().item():.6f}")

if __name__ == "__main__":
    # resultディレクトリ内の最新のモデルファイルを探す
    result_dir = "result"
    model_files = [f for f in os.listdir(result_dir) if f.endswith(".pt")]
    if not model_files:
        print("モデルファイルが見つかりません")
        exit(1)
    
    # 最新のモデルファイルを選択
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(result_dir, latest_model)
    
    # モデルの確認
    load_and_check_model(model_path) 