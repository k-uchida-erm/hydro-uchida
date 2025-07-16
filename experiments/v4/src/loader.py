import pandas as pd
import torch
from pathlib import Path
from .config import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOIL_TYPES = pd.read_csv(DATA_DIR / "soil_types.csv").set_index('soil_type').to_dict('index')

def load_all_data():
    soil_map = load_soil("soil.csv")
    df_bc = load_bc("bc.csv")
    X_ic, h0 = load_ic("ic.csv")
    df_train = load_model_results("model_results.csv")
    df_obs = load_validation("verfi_1.csv")
    return soil_map, df_bc, X_ic, h0, df_train, df_obs

def load_soil(path):
    data_path = DATA_DIR / path
    df = pd.read_csv(data_path)
    
    if 'soil_type' in df.columns:
        df['soil_type'] = df['soil_type'].str.strip()
        for soil_type in df['soil_type'].unique():
            if soil_type not in SOIL_TYPES:
                raise ValueError(f"Unknown soil type: {soil_type}")
    elif not all(param in df.columns for param in ['alpha', 'n', 'theta_r', 'theta_s', 'Ks', 'Ss']):
        raise ValueError("soil.csv must contain either soil_type or all soil parameters")
    
    return df

def get_soil_params(x, y, z, soil_map):
    if torch.is_tensor(x):
        x = x.item()
    if torch.is_tensor(y):
        y = y.item()
    if torch.is_tensor(z):
        z = z.item()
    
    distances = ((soil_map['x'].values - x)**2 + 
                 (soil_map['y'].values - y)**2 + 
                 (soil_map['z'].values - z)**2)
    nearest_idx = distances.argmin()
    
    if 'soil_type' in soil_map.columns:
        soil_type = soil_map.iloc[nearest_idx]['soil_type']
        return SOIL_TYPES[soil_type]
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
    return pd.read_csv(DATA_DIR / path)

def load_ic(path):
    df = pd.read_csv(DATA_DIR / path)
    X = torch.tensor(df[['x', 'y', 'z']].values, dtype=DTYPE, device=DEVICE)
    h0 = torch.tensor(df['h0'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    return X, h0

def load_obs(path):
    df = pd.read_csv(DATA_DIR / path)
    if 'x' not in df.columns:
        df['x'] = 0.0
    if 'y' not in df.columns:
        df['y'] = 0.0
    cols = ['x', 'y', 'z', 't', 'h']
    return df[[c for c in cols if c in df.columns]]

def load_validation(path="verfi_1.csv"):
    return load_obs(path)

def load_model_results(path):
    df = pd.read_csv(DATA_DIR / path, index_col=0)
    depths = [10.0 - i * 0.1 for i in range(99)]
    
    data_list = []
    for time, row in df.iterrows():
        for col_idx, h_value in enumerate(row):
            if col_idx < 99:
                data_list.append({
                    'z': depths[col_idx],
                    't': float(time),
                    'h': float(h_value),
                    'x': 0.0,
                    'y': 0.0
                })
    
    return pd.DataFrame(data_list)
