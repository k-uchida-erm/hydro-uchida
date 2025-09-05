import torch
import torch.autograd as autograd
import numpy as np
from .config import *
from .model import K_unsat, dtheta_dh


def div_K_grad(h, K, X):
    grads = []
    for i in range(3):
        grad_h = autograd.grad(h, X, torch.ones_like(h), create_graph=True)[0]
        grad_h_i = grad_h[:, i:i+1].clamp(min=-1e6, max=1e6)
        K_grad = (K * grad_h_i).clamp(min=-1e6, max=1e6)
        g_i = autograd.grad(K_grad, X, torch.ones_like(K_grad), retain_graph=True)[0][:, i:i+1]
        grads.append(g_i.clamp(min=-1e6, max=1e6))
    return (grads[0] + grads[1] + grads[2]).clamp(min=-1e6, max=1e6)


def residual(model, X, soil_map, Ss):
    p = soil_map.iloc[0]
    a = torch.tensor([p['alpha']], device=DEVICE)
    n = torch.tensor([p['n']], device=DEVICE)
    tr = torch.tensor([p['theta_r']], device=DEVICE)
    ts = torch.tensor([p['theta_s']], device=DEVICE)
    Ks = torch.tensor([p['Ks']], device=DEVICE)

    X.requires_grad_(True)
    psi = model(X)
    z = X[:, 2:3]
    h = psi + z
    phi = (h - z).clamp(min=-1e6, max=1e6)

    mask_unsat = (phi < 0).float()
    K_eff = torch.where(mask_unsat.bool(), K_unsat(phi, Ks, a, n), Ks.unsqueeze(0))
    S_eff_unsat = dtheta_dh(phi, a, n, tr, ts)
    S_eff_sat = torch.full_like(phi, Ss)
    S_eff = torch.where(mask_unsat.bool(), S_eff_unsat, S_eff_sat)

    grad_h = autograd.grad(h, X, torch.ones_like(h), create_graph=True)[0]
    dh_dt = grad_h[:, 3:4].clamp(min=-1e6, max=1e6)
    lap = div_K_grad(h, K_eff, X)
    res = (S_eff * dh_dt - lap).clamp(min=-1e6, max=1e6)
    return res


def bc_loss(model, df, soil_map):
    if df is None or df.empty:
        return torch.tensor(0., device=DEVICE)
    
    X = torch.tensor(df[['x','y','z','t']].values, dtype=DTYPE, device=DEVICE)
    val = torch.tensor(df['value'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    bc_type = df['type'].values
    X.requires_grad_(True)
    
    psi = model(X)
    h_pred = psi + X[:, 2:3]
    
    loss = torch.tensor(0., device=DEVICE)
    
    dir_mask = torch.tensor(bc_type == 'Dirichlet', device=DEVICE)
    if torch.any(dir_mask):
        phi = psi - X[:, 2:3]
        theta_pred = theta_from_psi(phi, soil_map)
        loss += torch.mean((theta_pred[dir_mask] - val[dir_mask])**2)
    
    neu_mask = torch.tensor(bc_type == 'Neumann', device=DEVICE)
    if torch.any(neu_mask):
        nx = torch.tensor(df['nx'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        ny = torch.tensor(df['ny'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        nz = torch.tensor(df['nz'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        
        grad_h = autograd.grad(h_pred, X, torch.ones_like(h_pred), create_graph=True)[0]
        grad_h_x = grad_h[:, 0:1]
        grad_h_y = grad_h[:, 1:2]
        grad_h_z = grad_h[:, 2:3]
        
        grad_h_n = grad_h_x * nx + grad_h_y * ny + grad_h_z * nz
        
        p = soil_map.iloc[0]
        Ks = torch.tensor([p['Ks']], device=DEVICE)
        alpha = torch.tensor([p['alpha']], device=DEVICE)
        n = torch.tensor([p['n']], device=DEVICE)
        
        phi = psi - X[:, 2:3]
        K_eff = torch.where(phi < 0, K_unsat(phi, Ks, alpha, n), Ks.unsqueeze(0))
        
        flux_pred = -K_eff * grad_h_n
        loss += torch.mean((flux_pred[neu_mask] - val[neu_mask])**2)
    
    return loss


def ic_loss(model, X_ic, psi0):
    X = torch.cat([X_ic, torch.zeros_like(psi0)], dim=1)
    return torch.mean((model(X) - psi0)**2)


def obs_loss(model, df):
    if df is None or df.empty:
        return torch.tensor(0., device=DEVICE)
    X = torch.tensor(df[['x','y','z','t']].values, dtype=DTYPE, device=DEVICE)
    h_obs = torch.tensor(df['h'].values, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    psi = model(X)
    h_pred = psi + X[:, 2:3]
    return torch.mean((h_pred - h_obs)**2)


def theta_from_psi(phi, soil_map):
    p = soil_map.iloc[0]
    theta_r = torch.tensor([p['theta_r']], device=DEVICE)
    theta_s = torch.tensor([p['theta_s']], device=DEVICE)
    alpha = torch.tensor([p['alpha']], device=DEVICE)
    n = torch.tensor([p['n']], device=DEVICE)
    
    m = 1.0 - 1.0 / n
    mask_sat = (phi >= 0)
    phi_abs = torch.clamp(torch.abs(phi), min=1e-10, max=1e10)
    Se_unsat = (1.0 + (alpha * phi_abs) ** n) ** (-m)
    Se = torch.where(mask_sat, torch.ones_like(Se_unsat), Se_unsat)
    Se = torch.clamp(Se, min=1e-10, max=1.0)
    theta = theta_r + (theta_s - theta_r) * Se
    return theta


def theta_from_psi_numpy(psi, theta_r, theta_s, alpha, n):
    m = 1.0 - 1.0 / n
    mask_sat = (psi >= 0)
    phi_abs = np.clip(np.abs(psi), 1e-10, 1e10)
    Se_unsat = (1.0 + (alpha * phi_abs) ** n) ** (-m)
    Se = np.where(mask_sat, 1.0, Se_unsat)
    Se = np.clip(Se, 1e-10, 1.0)
    theta = theta_r + (theta_s - theta_r) * Se
    return theta


def theta_obs_loss(model, df, soil_map):
    if df is None or df.empty:
        return torch.tensor(0., device=DEVICE)
    X = torch.tensor(df[['x','y','z','t']].values, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        psi_pred = model(X).cpu().numpy().squeeze()
    p = soil_map.iloc[0]
    theta_pred = theta_from_psi_numpy(psi_pred, float(p['theta_r']), float(p['theta_s']), float(p['alpha']), float(p['n']))
    theta_obs = theta_from_psi_numpy((df['h'].values - df['z'].values), float(p['theta_r']), float(p['theta_s']), float(p['alpha']), float(p['n']))
    diff = torch.tensor(theta_pred - theta_obs, dtype=DTYPE, device=DEVICE)
    return torch.mean(diff**2)