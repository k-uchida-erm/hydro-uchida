# =============================================================================
# 損失関数定義ファイル
# =============================================================================
# このファイルは、PINNの学習に使用される損失関数を定義します：
# 1. 偏微分方程式の残差計算
#    - 拡散項（∇·(K∇h)）の自動微分による計算
#    - 時間微分項の計算
# 2. 各種損失関数
#    - 境界条件損失（Dirichlet/Neumann条件）
#    - 初期条件損失
#    - 観測データ損失
# =============================================================================

import torch
import torch.autograd as autograd
from config import *
from model import K_unsat, dtheta_dh

def div_K_grad(h, K, X):
    grads=[]
    for i in range(3):
        g_i = autograd.grad(K*autograd.grad(h,X,torch.ones_like(h),create_graph=True)[0][:,i:i+1],
                            X,torch.ones_like(h),retain_graph=True)[0][:,i:i+1]
        grads.append(g_i)
    return grads[0]+grads[1]+grads[2]

def residual(model,X,soil,Ss):
    X.requires_grad_(True)
    h = model(X)
    z = X[:,2:3]
    phi = h - z
    mask_unsat = (phi<0).float()
    a,n,tr,ts,Ks = soil['alpha'],soil['n'],soil['theta_r'],soil['theta_s'],soil['Ks']

    K_eff = torch.where(mask_unsat.bool(), K_unsat(phi,Ks,a,n), torch.full_like(phi,Ks))
    S_eff = torch.where(mask_unsat.bool(), dtheta_dh(phi,a,n,tr,ts), torch.full_like(phi,Ss))

    grad_h = autograd.grad(h,X,torch.ones_like(h),create_graph=True)[0]
    dh_dt = grad_h[:,3:4]
    lap   = div_K_grad(h,K_eff,X)
    return S_eff*dh_dt - lap  # q=0

def bc_loss(model,df):
    if df.empty: return torch.tensor(0.,device=DEVICE)
    X = torch.tensor(df[['x','y','z','t']].values,dtype=DTYPE,device=DEVICE)
    val=torch.tensor(df['value'].values,dtype=DTYPE,device=DEVICE).unsqueeze(1)
    pred=model(X)
    dir_mask=(df['type']=='Dirichlet').values
    neu_mask=~dir_mask
    loss=0.
    if dir_mask.any():
        loss+=torch.mean((pred[dir_mask]-val[dir_mask])**2)
    if neu_mask.any():
        grad=autograd.grad(pred[neu_mask],X[neu_mask],torch.ones_like(pred[neu_mask]),create_graph=True)[0][:,:3]
        nvec=torch.tensor(df[['nx','ny','nz']].values,dtype=DTYPE,device=DEVICE)[neu_mask]
        flux=(grad*nvec).sum(dim=1,keepdim=True)
        loss+=torch.mean((flux-val[neu_mask])**2)
    return loss

def ic_loss(model,X_ic,h0):
    X=torch.cat([X_ic,torch.zeros_like(h0)],dim=1)
    return torch.mean((model(X)-h0)**2)

def obs_loss(model,df):
    if df.empty: return torch.tensor(0.,device=DEVICE)
    X=torch.tensor(df[['x','y','z','t']].values,dtype=DTYPE,device=DEVICE)
    h_obs=torch.tensor(df['h_obs'].values,dtype=DTYPE,device=DEVICE).unsqueeze(1)
    return torch.mean((model(X)-h_obs)**2)
