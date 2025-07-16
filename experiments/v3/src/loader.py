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
from .config import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 土壌タイプのパラメータを読み込む
SOIL_TYPES = pd.read_csv(DATA_DIR / "soil_types.csv").set_index('soil_type').to_dict('index')

def load_all_data():
    """すべてのデータを読み込む（ファイルが存在しない場合はデフォルト値を使用）"""
    # 土壌パラメータ
    try:
        soil_map = load_soil("soil.csv")
    except FileNotFoundError:
        # デフォルトは全てLoam
        soil_map = pd.DataFrame({
            'x': [0.0], 'y': [0.0], 'z': [0.0],
            'soil_type': ['loam']
        })
        print("[warn] soil.csv が見つからないため全ての地点でLoamパラメータを使用")

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
        try:
            df_obs = load_validation("verfi_1.csv")
            print("[warn] obs.csv が見つからないため verfi_1.csv を観測データとして使用")
        except FileNotFoundError:
            df_obs = pd.DataFrame()
            print("[warn] obs.csv, verfi_1.csv が見つからないため空のDataFrameを使用")

    return soil_map, df_bc, X_ic, h0, df_obs

def load_soil(path):
    """土壌パラメータを読み込む
    CSV: x,y,z,soil_type または x,y,z,alpha,n,theta_r,theta_s,Ks,Ss
    """
    data_path = DATA_DIR / path
    df = pd.read_csv(data_path)
    
    # 文字列カラムの空白を削除
    if 'soil_type' in df.columns:
        df['soil_type'] = df['soil_type'].str.strip()
    
    print("Loaded soil.csv:")
    print(df)
    print("\nUnique soil types:", df['soil_type'].unique())
    print("Available soil types:", list(SOIL_TYPES.keys()))
    
    # soil_typeが指定されている場合
    if 'soil_type' in df.columns:
        # 土壌タイプの存在確認
        for soil_type in df['soil_type'].unique():
            if soil_type not in SOIL_TYPES:
                raise ValueError(f"Unknown soil type: {soil_type}")
    # パラメータが直接指定されている場合
    elif all(param in df.columns for param in ['alpha', 'n', 'theta_r', 'theta_s', 'Ks', 'Ss']):
        pass
    else:
        raise ValueError("soil.csv must contain either soil_type or all soil parameters")
    
    return df

def get_soil_params(x, y, z, soil_map):
    """指定された地点の土壌パラメータを取得"""
    # 入力がTensorの場合はnumpyに変換
    if torch.is_tensor(x):
        x = x.item()
    if torch.is_tensor(y):
        y = y.item()
    if torch.is_tensor(z):
        z = z.item()
    
    # 最も近い地点を探す
    distances = ((soil_map['x'].values - x)**2 + 
                (soil_map['y'].values - y)**2 + 
                (soil_map['z'].values - z)**2)
    nearest_idx = distances.argmin()
    
    # soil_typeが指定されている場合
    if 'soil_type' in soil_map.columns:
        soil_type = soil_map.iloc[nearest_idx]['soil_type']
        return SOIL_TYPES[soil_type]
    # パラメータが直接指定されている場合
    else:
        return {
            'alpha': soil_map.iloc[nearest_idx]['alpha'],
            'n': soil_map.iloc[nearest_idx]['n'],
            'theta_r': soil_map.iloc[nearest_idx]['theta_r'],
            'theta_s': soil_map.iloc[nearest_idx]['theta_s'],
            'Ks': soil_map.iloc[nearest_idx]['Ks'],
            'Ss': soil_map.iloc[nearest_idx]['Ss']
        }

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
    df = pd.read_csv(data_path)
    # x, y がなければ0で補完
    if 'x' not in df.columns:
        df['x'] = 0.0
    if 'y' not in df.columns:
        df['y'] = 0.0
    cols = ['x', 'y', 'z', 't', 'h']
    df = df[[c for c in cols if c in df.columns]]
    return df

def load_validation(path="verfi_1.csv"):
    """バリデーション用観測データを読み込む（z, t, h）"""
    data_path = DATA_DIR / path
    df = pd.read_csv(data_path)
    if 'x' not in df.columns:
        df['x'] = 0.0
    if 'y' not in df.columns:
        df['y'] = 0.0
    cols = ['x', 'y', 'z', 't', 'h']
    df = df[[c for c in cols if c in df.columns]]
    return df
