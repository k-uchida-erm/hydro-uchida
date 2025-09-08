import os
import torch
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from .config import *
from .loss import residual, bc_loss, ic_loss, obs_loss, theta_obs_loss


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def train(model, soil_map, df_bc, X_ic, psi0, df_train, df_obs, X_res, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    fixed_weights = {
        'pde': 1.0,
        'bc': 40.0,  # v5に合わせてBCを強化
        'ic': 30.0,
        'obs': 8.0,   # v5に合わせて観測をやや増
        'theta_obs': 40.0
    }
    
    loss_history = []
    X_res_cached = X_res.detach()
    
    timestamp = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join('result', f'model_{timestamp}')
    ensure_dir(model_dir)

    print("RAR (Residual Adaptive Refinement) を開始...")
    
    for rar_iter in range(5):
        print(f"RAR反復 {rar_iter + 1}/5")
        epochs_per_rar = epochs // 5
        
        for epoch in tqdm(range(epochs_per_rar), desc=f"RAR {rar_iter + 1}"):
            optimizer.zero_grad()

            batch_size = 128
            if X_res_cached.shape[0] <= batch_size:
                X_int = X_res_cached
            else:
                idx_np = np.random.choice(X_res_cached.shape[0], size=batch_size, replace=False)
                idx = torch.tensor(idx_np, device=DEVICE, dtype=torch.long)
                X_int = X_res_cached.index_select(0, idx)
                
            loss_pde = torch.mean(residual(model, X_int, soil_map, Ss=1e-4) ** 2)
            loss_bc_val = bc_loss(model, df_bc, soil_map)
            loss_ic_val = ic_loss(model, X_ic, psi0)
            loss_train = obs_loss(model, df_train)
            loss_obs = obs_loss(model, df_obs)
            loss_theta = theta_obs_loss(model, df_obs, soil_map)

            loss = (fixed_weights['pde'] * loss_pde + 
                   fixed_weights['bc'] * loss_bc_val + 
                   fixed_weights['ic'] * loss_ic_val + 
                   fixed_weights['obs'] * loss_train + 
                   fixed_weights['theta_obs'] * loss_theta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            # 深い部分の学習を強化するため、より頻繁に学習率を調整
            if (epoch + 1) % 200 == 0:
                scheduler.step()

            if (epoch + 1) % 10 == 0:
                loss_history.append({
                    'rar_iter': rar_iter + 1,
                    'epoch': epoch + 1,
                    'pde_loss': loss_pde.item(),
                    'bc_loss': loss_bc_val.item(),
                    'ic_loss': loss_ic_val.item(),
                    'train_loss': loss_train.item(),
                    'val_loss': loss_obs.item(),
                    'theta_loss': loss_theta.item(),
                    'total_loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'pde_weight': fixed_weights['pde'],
                    'bc_weight': fixed_weights['bc'],
                    'ic_weight': fixed_weights['ic'],
                    'obs_weight': fixed_weights['obs'],
                    'theta_weight': fixed_weights['theta_obs']
                })
                torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
                pd.DataFrame(loss_history).to_csv(os.path.join(model_dir, 'loss_history.csv'), index=False)

        if rar_iter < 4:
            print("残差の大きい点を特定中...")
            X_res_grad = X_res_cached.clone().detach().requires_grad_(True)
            
            residuals = residual(model, X_res_grad, soil_map, Ss=1e-4)
            residual_abs = torch.abs(residuals).detach().cpu().numpy().flatten()
            top_indices = np.argsort(residual_abs)[-100:]
            
            # 深い部分（z=8-10cm）の残差を特別に監視
            deep_mask = (X_res_cached[:, 2] >= 8) & (X_res_cached[:, 2] <= 10)
            if deep_mask.sum() > 0:
                deep_residuals = residual_abs[deep_mask.cpu().numpy()]
                deep_top_indices = np.argsort(deep_residuals)[-50:]  # 深い部分の上位50点
                deep_indices = np.where(deep_mask.cpu().numpy())[0][deep_top_indices]
                top_indices = np.concatenate([top_indices, deep_indices])
            
            z_vals = X_res_cached[top_indices, 2].cpu().numpy()
            t_vals = X_res_cached[top_indices, 3].cpu().numpy()
            
            new_points = []
            for i in range(len(top_indices)):
                z_noise = np.random.normal(0, 0.5)
                t_noise = np.random.normal(0, 0.1)
                new_z = np.clip(z_vals[i] + z_noise, 0, 9)
                new_t = np.clip(t_vals[i] + t_noise, 0, 49)  # 0-10時間の範囲
                new_points.append([0.0, 0.0, new_z, new_t])
            
            if rar_iter == 0:
                print("初期時間（t=0-2）の重点サンプリング中...")
                t_early = np.linspace(0, 2, 25)
                z_early = np.linspace(0, 9, 11)
                for t in t_early:
                    for z in z_early:
                        new_points.append([0.0, 0.0, z, t])  # 約275点

                print("湿潤フロント近傍（z=7-9cm）の重点サンプリング中...")
                t_wetfront = np.linspace(0, 2, 60)
                z_wetfront = np.linspace(7, 9, 40)
                for t in t_wetfront:
                    for z in z_wetfront:
                        new_points.append([0.0, 0.0, z, t])  # 約2400点

                print("深い部分（z=8-10cm）の重点サンプリング中...")
                t_deep = np.linspace(0, 10, 100)
                z_deep = np.linspace(8, 10, 40)
                for t in t_deep:
                    for z in z_deep:
                        new_points.append([0.0, 0.0, z, t])  # 約4000点

                print("初期時間の重点点を追加（合計 ≲ 7千点）")
            
            new_X_res = torch.tensor(new_points, dtype=DTYPE, device=DEVICE)
            X_res_cached = torch.cat([X_res_cached, new_X_res], dim=0)
            print(f"新しい残差点 {len(new_points)} 点を追加")

    print("\nL-BFGS-B最適化を開始...")
    model = train_bfgs(model, soil_map, df_bc, X_ic, psi0, df_train, df_obs, X_res_cached, model_dir)

    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    pd.DataFrame(loss_history).to_csv(os.path.join(model_dir, 'loss_history.csv'), index=False)
    print(f"\n学習完了: {model_dir}")

    return model


def train_bfgs(model, soil_map, df_bc, X_ic, psi0, df_train, df_obs, X_res, model_dir):
    iteration_count = 0
    best_loss = float('inf')
    max_iterations = 400
    
    def closure():
        nonlocal iteration_count, best_loss
        optimizer.zero_grad()
        loss_pde = torch.mean(residual(model, X_res, soil_map, Ss=1e-4) ** 2)
        loss_bc_val = bc_loss(model, df_bc, soil_map)
        loss_ic_val = ic_loss(model, X_ic, psi0)
        loss_train = obs_loss(model, df_train)
        loss_obs = obs_loss(model, df_obs)
        loss_theta = theta_obs_loss(model, df_obs, soil_map)

        loss = (LOSS_WEIGHTS['pde'] * loss_pde + 
               LOSS_WEIGHTS['bc'] * loss_bc_val + 
               LOSS_WEIGHTS['ic'] * loss_ic_val + 
               LOSS_WEIGHTS['obs'] * loss_train + 
               LOSS_WEIGHTS['theta_obs'] * loss_theta)

        loss.backward()
        
        # 進行状況を表示
        iteration_count += 1
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            
        # 進捗率を計算
        progress_percent = min(100.0, (iteration_count / max_iterations) * 100)
        
        if iteration_count % 20 == 0 or iteration_count <= 10:
            print(f"L-BFGS反復 {iteration_count}/{max_iterations} ({progress_percent:.1f}%): 損失 = {current_loss:.6f} (最良 = {best_loss:.6f})")
        
        return loss

    optimizer = optim.LBFGS(model.parameters(), 
                           lr=1.0, 
                           max_iter=max_iterations,
                           max_eval=max_iterations * 2,  # 評価回数も調整
                           history_size=50,
                           line_search_fn="strong_wolfe")
    
    print("L-BFGS-B最適化実行中...")
    print("進行状況を20反復ごとに表示します...")
    optimizer.step(closure)
    
    final_progress = min(100.0, (iteration_count / max_iterations) * 100)
    print(f"L-BFGS最適化完了: 総反復数 = {iteration_count}/{max_iterations} ({final_progress:.1f}%), 最終損失 = {best_loss:.6f}")
    
    return model