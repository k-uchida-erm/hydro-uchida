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

import torch
from config import *
from loss import residual, bc_loss, ic_loss, obs_loss

def train(model, soil, df_bc, X_ic, h0, df_obs, epochs=EPOCHS):
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    Ss = soil.get('Ss', 1e-4)
    for ep in range(epochs):
        # ---------------------------------------------------------------
        # 1) PDE 内部点: 一様乱数 (例)  ── LHS 等に置き換えて OK
        # ---------------------------------------------------------------
        X_int = torch.rand(BATCH_INT, 4, device=DEVICE, dtype=DTYPE)
        X_int[:, 0] *= GRID['Nx'] * GRID['dx']      # x [m]
        X_int[:, 1] *= GRID['Ny'] * GRID['dy']      # y [m]
        X_int[:, 2] *= -GRID['Nz'] * GRID['dz']     # z [m] (地下を負値で表す)
        X_int[:, 3] *= DT * 10                      # t [s]   : 任意 horizon

        loss_pde = torch.mean(residual(model, X_int, soil, Ss) ** 2)
        loss_bc  = bc_loss(model, df_bc)
        loss_ic  = ic_loss(model, X_ic, h0)
        loss_obs = obs_loss(model, df_obs)

        loss = (W['PDE'] * loss_pde + W['BC'] * loss_bc +
                W['IC'] * loss_ic + W['OBS'] * loss_obs)

        optim.zero_grad(); loss.backward(); optim.step()

        if ep % 1000 == 0:
            print(f"Ep {ep:5d} | Ltot {loss.item():.2e} (PDE {loss_pde.item():.2e} BC {loss_bc.item():.2e} IC {loss_ic.item():.2e} OBS {loss_obs.item():.2e})")
