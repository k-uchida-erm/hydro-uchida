# -----------------------------------------------------------------------------
# pinn_groundwater.py
# -----------------------------------------------------------------------------
# PyTorch を用いた Physics‑Informed Neural Network (PINN) の骨組み実装例
# 水頭 h(x, y, z, t) を求めると共に、格子点が飽和帯か不飽和帯かを
# 動的に判定し、それぞれの支配方程式を自動で切り替えて損失に組込む。
#
#   ■ 空間離散   : 有限体積法 (FVM) – スタッガード格子想定
#   ■ 時間離散   : オイラー陰解法 (Δt はユーザ入力)
#   ■ 解法        : 非線形部 Newton–Raphson / 線形部 Preconditioned Bi‑CGSTAB
#   ■ コンポーネント
#       1. ニューラルネットワーク            (net)
#       2. van Genuchten 保水曲線             (theta(h), dtheta/dh)
#       3. 不飽和透水係数 K(h)                (Mualem–VG)
#       4. 飽和/不飽和マスク                 (phi = h − z)
#       5. PDE 残差 + データ損失関数          (loss)
#       6. 土壌パラメータ入力スロット         (soil_dict)
#
# ※ 本コードは『構造』を示すための雛形であり、ネットワーク設計／
#   境界条件／データ I/O などはユーザ側で具体化してください。
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.autograd as autograd

# =============================================================================
# ▼ 1. ネットワーク定義
# =============================================================================
class PINN(nn.Module):
    """多層パーセプトロン (隠れ層: 4, 活性化: Tanh)"""
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * 4 + [out_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x = [x, y, z, t] のテンソル
        return self.net(x)

# =============================================================================
# ▼ 2. van Genuchten 保水曲線 & 透水係数
# =============================================================================

def vg_theta(h, alpha, n, theta_r, theta_s):
    """水頭 h (<0) から体積含水率 θ を計算"""
    m = 1.0 - 1.0 / n
    Se = (1 + (alpha * torch.abs(h)) ** n) ** (-m)
    return theta_r + (theta_s - theta_r) * Se

def dtheta_dh(h, alpha, n, theta_r, theta_s):
    """dθ/dh (VG)"""
    m = 1.0 - 1.0 / n
    Se = (1 + (alpha * torch.abs(h)) ** n) ** (-m)
    dSe = -m * Se ** (1/m) * (alpha * torch.abs(h)) ** (n - 1) * alpha * n
    # 符号注意 (h<0) ⇒ abs(h)= -h
    dSe *= torch.sign(h) * -1.0
    return (theta_s - theta_r) * dSe

def vg_K(h, Ks, alpha, n):
    """Mualem–VG 透水係数"""
    m = 1.0 - 1.0 / n
    Se = (1 + (alpha * torch.abs(h)) ** n) ** (-m)
    return Ks * Se ** 0.5 * (1 - (1 - Se ** (1/m)) ** m) ** 2

# =============================================================================
# ▼ 3. 土壌パラメータ受け皿
# =============================================================================
soil_dict = {
    # 例: "Loam": {"alpha":0.036, "n":1.56, "theta_r":0.078, "theta_s":0.43, "Ks":1.2e-5}
}

# =============================================================================
# ▼ 4. PDE 残差計算ユーティリティ
# =============================================================================

def pde_residual(model, x_batch, soil_param, z_gw):
    """PINN 残差 (飽和/不飽和で式切替) を返す
    Parameters
    ----------
    model : PINN ネットワーク
    x_batch : Tensor  (N,4) [x,y,z,t]
    soil_param : dict   土壌固有パラメータ
    z_gw : float        地下水位 (m)
    """
    x_batch.requires_grad_(True)
    h = model(x_batch)
    x, y, z, t = x_batch[:,0], x_batch[:,1], x_batch[:,2], x_batch[:,3]
    phi = h.squeeze() - z              # 圧力水頭 φ = h - z

    # 飽和/不飽和マスク
    sat_mask  = (phi >= 0).float()     # 1 ⇒ 飽和
    unsat_mask = 1.0 - sat_mask        # 1 ⇒ 不飽和

    # VG パラメータ抽出
    a = soil_param["alpha"]; n = soil_param["n"]
    tr = soil_param["theta_r"]; ts = soil_param["theta_s"]
    Ks = soil_param["Ks"]

    # θ, dθ/dh, K 計算 (不飽和のみ使用)
    theta      = vg_theta(h, a, n, tr, ts)
    dtheta_dh_ = dtheta_dh(h, a, n, tr, ts)
    K_unsat    = vg_K(h, Ks, a, n)
    K_sat      = Ks

    # 飽和: S_s dh/dt = ∇·(K∇h) + q   (q=0 としておく)
    # 不飽和: dθ/dh dh/dt = ∇·(K(h)∇h)
    grad_h = autograd.grad(h, x_batch, torch.ones_like(h), create_graph=True)[0]
    dh_dx, dh_dy, dh_dz, dh_dt = grad_h[:,0], grad_h[:,1], grad_h[:,2], grad_h[:,3]

    # 3D ラプラシアン ∇·(K∇h) via autograd（簡易実装例）
    def div_K_grad(h_val, K_val):
        grads = []
        for i in range(3):  # x,y,z
            g = autograd.grad(K_val * grad_h[:,i].unsqueeze(1), x_batch,
                              torch.ones_like(h_val), retain_graph=True)[0][:,i]
            grads.append(g)
        return sum(grads)

    lap_sat  = div_K_grad(h, K_sat)
    lap_unsat = div_K_grad(h, K_unsat)

    Ss = soil_param.get("Ss", 1e-5)

    res_sat   = Ss * dh_dt - lap_sat
    res_unsat = dtheta_dh_ * dh_dt - lap_unsat

    residual = sat_mask * res_sat + unsat_mask * res_unsat
    return residual

# =============================================================================
# ▼ 5. 損失関数
# =============================================================================

def loss_fn(model, x_phys, soil_param, z_gw, x_data=None, h_data=None):
    # PDE 残差損失
    pde_loss = torch.mean(pde_residual(model, x_phys, soil_param, z_gw)**2)
    # 教師ありデータ損失 (オプション)
    if x_data is not None and h_data is not None:
        pred = model(x_data)
        data_loss = torch.mean((pred.squeeze() - h_data)**2)
    else:
        data_loss = torch.tensor(0.0, device=x_phys.device)
    return pde_loss + data_loss

# =============================================================================
# ▼ 6. トレーニング雛形
# =============================================================================

def train(model, optimizer, soil_param, epochs=10000, print_every=1000):
    for ep in range(epochs):
        x_phys = torch.rand(1024,4)  # ダミー物理点
        residual_loss = loss_fn(model, x_phys, soil_param, z_gw=0.0)
        optimizer.zero_grad(); residual_loss.backward(); optimizer.step()
        if ep % print_every == 0:
            print(f"Epoch {ep}: loss={residual_loss.item():.3e}")

if __name__ == "__main__":
    # ▼ 土壌定数（例: Loam）
    soil = dict(alpha=0.036, n=1.56, theta_r=0.078, theta_s=0.43, Ks=1.2e-5, Ss=1e-4)

    pinn = PINN()
    opt = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    train(pinn, opt, soil, epochs=5000, print_every=1000)
