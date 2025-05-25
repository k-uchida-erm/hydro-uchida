# =============================================================================
# データ読み込みファイル
# =============================================================================
# このファイルは、シミュレーションに必要な各種データの読み込み関数を提供します：
# 1. 土壌パラメータの読み込み（van Genuchtenパラメータなど）
# 2. 境界条件データの読み込み（Dirichlet/Neumann条件）
# 3. 初期条件データの読み込み
# 4. 観測データの読み込み（任意）
# =============================================================================

import pandas as pd
import torch
from pathlib import Path
from config import *

def load_all_data():
    """すべてのデータを読み込む（ファイルが存在しない場合はデフォルト値を使用）"""
    # 土壌パラメータ
    try:
        soil = load_soil("soil.csv")
    except FileNotFoundError:
        # Loam デフォルト
        soil = dict(alpha=0.036, n=1.56, theta_r=0.078, theta_s=0.43, Ks=1.2e-5, Ss=1e-4)
        print("[warn] soil.csv が見つからないため Loam パラメータを使用")

    # 境界条件
    try:
        df_bc = load_bc("bc.csv")
    except FileNotFoundError:
        df_bc = pd.DataFrame()
        print("[warn] bc.csv が見つからないため空のDataFrameを使用")

    # 初期条件
    try:
        X_ic, h0 = load_ic("ic.csv")
    except FileNotFoundError:
        # 全セル h0=0 のダミー
        X_ic = torch.zeros((1,3), device=DEVICE, dtype=DTYPE)
        h0   = torch.zeros((1,1), device=DEVICE, dtype=DTYPE)
        print("[warn] ic.csv が無いので h0=0 を使用")

    # 観測データ
    try:
        df_obs = load_obs("obs.csv")
    except FileNotFoundError:
        df_obs = pd.DataFrame()
        print("[warn] obs.csv が見つからないため空のDataFrameを使用")

    return soil, df_bc, X_ic, h0, df_obs

def load_soil(path):
    """土壌パラメータを読み込む"""
    data_path = DATA_DIR / path
    return pd.read_csv(data_path).iloc[0].to_dict()

def load_bc(path):
    """境界条件を読み込む
    CSV: x,y,z,t,type,value,nx,ny,nz  (type: Dirichlet/Neumann)
    """
    data_path = DATA_DIR / path
    return pd.read_csv(data_path)

def load_ic(path):
    """初期条件を読み込む"""
    data_path = DATA_DIR / path
    df = pd.read_csv(data_path)
    X = torch.tensor(df[['x','y','z']].values, dtype=DTYPE, device=DEVICE)
    h0= torch.tensor(df['h0'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    return X,h0

def load_obs(path):
    """観測データを読み込む"""
    data_path = DATA_DIR / path
    return pd.read_csv(data_path)
