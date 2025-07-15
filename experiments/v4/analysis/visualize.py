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
from src import PINN, DEVICE, load_all_data

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

def plot_loss_history(loss_file, output_dir):
    """損失の履歴をプロット"""
    df = pd.read_csv(loss_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['total_loss'], label='Total Loss')
    plt.plot(df['epoch'], df['pde_loss'], label='PDE Loss')
    plt.plot(df['epoch'], df['bc_loss'], label='BC Loss')
    plt.plot(df['epoch'], df['ic_loss'], label='IC Loss')
    
    # 新しいカラム名に対応
    if 'train_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='Training Data Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    elif 'obs_loss' in df.columns:  # 後方互換性のため
        plt.plot(df['epoch'], df['obs_loss'], label='Data Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    
    # plotsサブディレクトリに保存
    plots_dir = os.path.join(output_dir, 'plots')
    ensure_dir(plots_dir)
    plt.savefig(os.path.join(plots_dir, 'loss_history.png'))
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

def plot_validation_profile(model, df_obs, output_dir):
    """
    バリデーション用: 各時刻ごとに観測値とモデル予測値のh-z分布を重ねて描画
    df_obs: x,y,z,t,h（観測値）
    model: PINNモデル
    output_dir: 保存先ディレクトリ
    """
    import matplotlib.pyplot as plt
    import torch
    times = sorted(df_obs['t'].unique())
    plt.figure(figsize=(8, 6))
    for t in times:
        obs_t = df_obs[df_obs['t'] == t]
        # 観測値
        plt.plot(obs_t['h'], obs_t['z'], 'o-', label=f'Obs {int(t)}hr')
        # モデル予測値
        X_pred = torch.tensor(obs_t[['x','y','z','t']].values, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            h_pred = model(X_pred).cpu().numpy().flatten()
        plt.plot(h_pred, obs_t['z'], '--', label=f'Pred {int(t)}hr')
    plt.xlabel('Pressure Head (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Validation: Pressure Head Profile')
    plt.legend()
    plt.gca().invert_yaxis()
    plots_dir = os.path.join(output_dir, 'plots')
    ensure_dir(plots_dir)
    plt.savefig(os.path.join(plots_dir, 'validation_profile.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='学習済みモデルの可視化')
    parser.add_argument('--model', type=str, help='可視化するモデルディレクトリ名')
    args = parser.parse_args()
    
    # モデルディレクトリの取得
    result_dir = "result"
    model_dirs = [d for d in os.listdir(result_dir) if d.startswith("model_")]
    
    if not model_dirs:
        print("モデルディレクトリが見つかりません")
        exit(1)
    
    if args.model:
        model_dir = os.path.join(result_dir, args.model)
        if not os.path.exists(model_dir):
            print(f"エラー: ディレクトリ {model_dir} が見つかりません")
            exit(1)
    else:
        model_dir = os.path.join(result_dir, sorted(model_dirs)[-1])
    
    # 可視化の実行
    print(f"モデル {os.path.basename(model_dir)} の可視化を開始します...")
    
    # モデルの読み込み
    model = PINN().to(DEVICE)
    model_path = os.path.join(model_dir, 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"モデル {os.path.basename(model_dir)} を読み込みました")
    
    # 損失履歴のプロット
    loss_file = os.path.join(model_dir, 'loss_history.csv')
    if os.path.exists(loss_file):
        plot_loss_history(loss_file, model_dir)
        print(f"損失履歴をプロットしました: {os.path.join(model_dir, 'plots/loss_history.png')}")
    else:
        print("警告: 損失履歴ファイルが見つかりません")
    
    # 予測値と観測値の比較
    try:
        df_obs = pd.read_csv('data/verfi_1.csv')
        print("バリデーションデータとして verfi_1.csv を使用")
    except FileNotFoundError:
        try:
            df_obs = pd.read_csv('data/obs.csv')
            print("バリデーションデータとして obs.csv を使用")
        except FileNotFoundError:
            print("警告: バリデーションデータファイルが見つかりません")
            exit(1)
    
    # x, y がなければ0で補完
    if 'x' not in df_obs.columns:
        df_obs['x'] = 0.0
    if 'y' not in df_obs.columns:
        df_obs['y'] = 0.0
    plot_prediction_vs_observation(model, df_obs, model_dir)
    print(f"予測値と観測値の比較をプロットしました: {os.path.join(model_dir, 'plots/pred_vs_obs.png')}")

    # バリデーション用h-z分布グラフ
    plot_validation_profile(model, df_obs, model_dir)
    print(f"バリデーションh-z分布をプロットしました: {os.path.join(model_dir, 'plots/validation_profile.png')}")
    
    # 水頭分布の可視化（複数の時間点）
    for t in [0, 10, 20, 30]:
        plot_head_distribution(model, t, model_dir)
        print(f"水頭分布をプロットしました: {os.path.join(model_dir, 'plots/head_distribution/head_distribution_t{t}.png')}")
    
    # プロットの保存
    print(f"プロットを保存しました: {model_dir}") 