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

def train(model, soil_map, df_bc, X_ic, h0, df_obs, epochs=EPOCHS):
    """PINNモデルの学習"""
    # 最適化器の設定
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学習率スケジューラーの設定
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=100
    )
    
    # 損失履歴を保存するリスト
    loss_history = []
    
    # 内部点の生成（学習用）
    X_int = generate_internal_points()
    
    # モデル保存用のディレクトリを作成
    timestamp = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join('result', f'model_{timestamp}')
    ensure_dir(model_dir)
    
    try:
        # 学習ループ
        for epoch in tqdm(range(epochs), desc="Training"):
            optimizer.zero_grad()
            
            # 物理方程式の残差
            loss_pde = torch.mean(residual(model, X_int, soil_map, Ss=1e-4) ** 2)
            
            # 境界条件
            loss_bc = bc_loss(model, df_bc)
            
            # 初期条件
            loss_ic = ic_loss(model, X_ic, h0)
            
            # 観測データ
            loss_obs = obs_loss(model, df_obs)
            
            # 重み付き総損失
            loss = LOSS_WEIGHTS['pde'] * loss_pde + \
                   LOSS_WEIGHTS['bc'] * loss_bc + \
                   LOSS_WEIGHTS['ic'] * loss_ic + \
                   LOSS_WEIGHTS['obs'] * loss_obs
            
            # 勾配計算と更新
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()
            
            # 学習率の更新
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(loss.item())
            new_lr = optimizer.param_groups[0]['lr']
            
            # 学習率が変更された場合に表示
            if new_lr != old_lr:
                print(f"\nLearning rate decreased from {old_lr:.2e} to {new_lr:.2e}")
            
            # 損失履歴の記録
            if (epoch + 1) % 10 == 0:  # 10エポックごとに記録
                loss_history.append({
                    'epoch': epoch + 1,
                    'pde_loss': loss_pde.item(),
                    'bc_loss': loss_bc.item(),
                    'ic_loss': loss_ic.item(),
                    'obs_loss': loss_obs.item(),
                    'total_loss': loss.item(),
                    'learning_rate': new_lr
                })
                
                # 進捗の表示
                print(f"\rEpoch {epoch + 1}/{epochs} ({((epoch + 1)/epochs*100):.1f}%)", end="")
                
                # 定期的にモデルを保存
                torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
                pd.DataFrame(loss_history).to_csv(os.path.join(model_dir, 'loss_history.csv'), index=False)
                
                # 分析と可視化を実行
                run_analysis(model_dir)
    
    except KeyboardInterrupt:
        print("\n学習を中断しました。現在のモデルを保存します...")
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        pd.DataFrame(loss_history).to_csv(os.path.join(model_dir, 'loss_history.csv'), index=False)
        
        # 分析と可視化を実行
        run_analysis(model_dir)
        
        print("モデルを保存しました。")
    
    finally:
        # 損失履歴をDataFrameに変換
        df_loss = pd.DataFrame(loss_history)
        
        # モデルと損失履歴を保存
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        df_loss.to_csv(os.path.join(model_dir, 'loss_history.csv'), index=False)
        
        # 最終的な分析と可視化を実行
        run_analysis(model_dir)
        
        print(f"\n学習完了: {model_dir}")
    
    return model

def main():
    # モデルの初期化
    model = PINN().to(DEVICE)
    
    # データの読み込み
    from loader import load_data
    soil_map, df_bc, X_ic, h0, df_obs = load_data()
    
    # モデルの学習
    model = train(model, soil_map, df_bc, X_ic, h0, df_obs)
    
    print("学習が完了しました")

if __name__ == "__main__":
    main()
