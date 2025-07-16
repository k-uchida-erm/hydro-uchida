import torch
import torch.autograd as autograd
from .config import *
from .model import K_unsat, dtheta_dh, theta
from .loader import get_soil_params

# 拡散項の計算（∇·(K∇h)）
# ∇hは勾配ベクトル、Kは透水係数、∇·(K∇h)は勾配ベクトルの発散
def div_K_grad(h, K, X):
    grads = []
    for i in range(3):
        # 勾配の計算
        grad_h = autograd.grad(h, X, torch.ones_like(h), create_graph=True)[0]
        # i方向の勾配成分
        grad_h_i = grad_h[:, i:i+1]
        
        # 数値的安定性のためのクリッピング
        grad_h_i = torch.clamp(grad_h_i, min=-1e6, max=1e6)
        
        # K * grad_h_i の勾配（ブロードキャストを考慮）
        K_grad = K * grad_h_i  # 形状: [batch_size, 1]
        
        # 数値的安定性のためのクリッピング
        K_grad = torch.clamp(K_grad, min=-1e6, max=1e6)
        
        # 勾配の計算
        g_i = autograd.grad(K_grad, X, torch.ones_like(K_grad), retain_graph=True)[0][:, i:i+1]
        
        # 数値的安定性のためのクリッピング
        g_i = torch.clamp(g_i, min=-1e6, max=1e6)
        
        grads.append(g_i)
    
    # 各方向の勾配を合計
    result = grads[0] + grads[1] + grads[2]
    
    # 最終的なクリッピング
    result = torch.clamp(result, min=-1e6, max=1e6)
    
    return result

# 物理方程式の残差を計算
def residual(model, X, soil_map, Ss):
    # 各地点の土壌パラメータを取得
    soil_params = []
    for i in range(len(X)):
        x, y, z = X[i, 0], X[i, 1], X[i, 2]
        params = get_soil_params(x, y, z, soil_map)
        soil_params.append(params)
    
    # バッチ処理用にパラメータをまとめる
    a = torch.tensor([p['alpha'] for p in soil_params], device=DEVICE)
    n = torch.tensor([p['n'] for p in soil_params], device=DEVICE)
    tr = torch.tensor([p['theta_r'] for p in soil_params], device=DEVICE)
    ts = torch.tensor([p['theta_s'] for p in soil_params], device=DEVICE)
    Ks = torch.tensor([p['Ks'] for p in soil_params], device=DEVICE)
    
    # 水頭と圧力水頭の計算
    X.requires_grad_(True)
    h = model(X)
    z = X[:,2:3]
    phi = h - z
    
    # 数値的安定性のためのクリッピング
    phi = torch.clamp(phi, min=-1e6, max=1e6)
    
    # 不飽和領域のマスク
    mask_unsat = (phi < 0).float()

    # 有効透水係数と有効比貯留係数の計算
    K_eff = torch.where(mask_unsat.bool(), K_unsat(phi, Ks, a, n), Ks.unsqueeze(1))
    
    # 有効比貯留係数の計算を改善
    S_eff_unsat = dtheta_dh(phi, a, n, tr, ts)
    S_eff_sat = torch.full_like(phi, Ss)
    S_eff = torch.where(mask_unsat.bool(), S_eff_unsat, S_eff_sat)
    
    # 数値的安定性のためのクリッピング（より緩和した範囲）
    K_eff = torch.clamp(K_eff, min=1e-30, max=1e30)
    S_eff = torch.clamp(S_eff, min=1e-30, max=1e30)

    # 時間微分とラプラシアンの計算
    grad_h = autograd.grad(h, X, torch.ones_like(h), create_graph=True)[0]
    dh_dt = grad_h[:,3:4]
    
    # 勾配爆発を防ぐためのクリッピング
    dh_dt = torch.clamp(dh_dt, min=-1e6, max=1e6)
    
    # ラプラシアンの計算
    lap = div_K_grad(h, K_eff, X)
    
    # 最終的な残差の計算
    residual = S_eff * dh_dt - lap
    
    # 数値的安定性のためのクリッピング
    residual = torch.clamp(residual, min=-1e6, max=1e6)
    
    return residual  # q=0

# 境界条件の損失
def bc_loss(model, df):
    if df.empty:
        return torch.tensor(0., device=DEVICE)
    
    # 入力データをテンソルに変換
    X = torch.tensor(df[['x','y','z','t']].values, dtype=DTYPE, device=DEVICE)
    val = torch.tensor(df['value'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    
    # 勾配計算のためにrequires_gradを設定
    X.requires_grad_(True)
    
    # モデルの予測
    pred = model(X)
    
    # Dirichlet条件とNeumann条件のマスク
    dir_mask = torch.tensor((df['type'] == 'Dirichlet').values, device=DEVICE)
    neu_mask = ~dir_mask
    
    loss = 0.
    
    # Dirichlet条件の損失
    if torch.any(dir_mask):
        loss += torch.mean((pred[dir_mask] - val[dir_mask])**2)
    
    # Neumann条件の損失
    if torch.any(neu_mask):  # マスクが空でないことを確認
        # 勾配の計算
        grad = autograd.grad(pred[neu_mask], X[neu_mask], 
                           torch.ones_like(pred[neu_mask]), 
                           create_graph=True,
                           allow_unused=True)[0]
        
        if grad is not None:  # 勾配がNoneでないことを確認
            grad = grad[:, :3]
            # 法線ベクトル
            nvec = torch.tensor(df[['nx','ny','nz']].values, 
                              dtype=DTYPE, device=DEVICE)[neu_mask]
            
            # フラックス
            flux = (grad * nvec).sum(dim=1, keepdim=True)
            loss += torch.mean((flux - val[neu_mask])**2)
    
    return loss

def ic_loss(model,X_ic,h0):
    X=torch.cat([X_ic,torch.zeros_like(h0)],dim=1)
    return torch.mean((model(X)-h0)**2)

# 観測データの損失
def obs_loss(model, df):
    if df is None or df.empty:
        return torch.tensor(0., device=DEVICE)
    
    # 観測データのカラム名を確認
    if 'h' not in df.columns:
        print("Warning: 'h' column not found in observation data. Available columns:", df.columns.tolist())
        return torch.tensor(0., device=DEVICE)
    
    # 入力データをテンソルに変換
    X = torch.tensor(df[['x','y','z','t']].values, dtype=DTYPE, device=DEVICE)
    h_obs = torch.tensor(df['h'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    
    # モデルの予測と損失計算
    h_pred = model(X)
    return torch.mean((h_pred - h_obs)**2)
