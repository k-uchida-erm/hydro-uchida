# =============================================================
#  pde_pinn.py  –  最小構成の PyTorch PINN（非線形地下水流れ式）
# =============================================================
import torch, torch.nn as nn, torch.optim as optim
from datetime import datetime
import pytz

# 日本時間に変換
jst = pytz.timezone('Asia/Tokyo')
current_time_jst = datetime.now(jst).strftime('%Y%m%d_%H%M%S')

# -------- 1️⃣ 物性関数を「とりあえず」の形で置く -----------------
def theta(phi):         # 含水率 θ(φ)
    return phi          # ここは好きに差し替えて OK
def Sw(phi):            # 飽和度 S_w(φ)
    return phi
def K(phi):             # 透水係数テンソル K(φ) – 1次元だからスカラー扱い
    return 1.0 + 0*phi  # 定数 1 としておく

Ss = 1.0                # 比貯留係数 Ss
q_source = 0.0          # 外部流入 q も 0 と仮定

# -------- 2️⃣ ニューラルネットワーク ----------------------------
class PINN(nn.Module):
    def __init__(self, hidden=20, depth=3):
        super().__init__()
        layers = []
        in_dim, out_dim = 2, 1
        layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth-2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

model = PINN()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# -------- 3️⃣ コロケーション点の生成 ----------------------------
N = 1000
x = torch.rand(N,1, device=device, requires_grad=True)   # x ∈ [0,1]
t = torch.rand(N,1, device=device, requires_grad=True)   # t ∈ [0,1]

# 境界 (x=0,1) と初期 (t=0) の点も少しだけ追加
Nb = 100
xb0 = torch.zeros(Nb,1, device=device, requires_grad=True)
xb1 = torch.ones (Nb,1, device=device, requires_grad=True)
tb  = torch.rand (Nb,1, device=device, requires_grad=True)
ti  = torch.zeros(Nb,1, device=device, requires_grad=True)
xi  = torch.rand (Nb,1, device=device, requires_grad=True)

# -------- 4️⃣ PDE 残差を計算する関数 ----------------------------
def residual():
    phi = model(x, t)                        # φ(x,t)
    h   = phi                                # 例: ここでは h=φ と置く（自由に替えてOK）

    # ∂φ/∂t
    dphi_dt = torch.autograd.grad(
                phi,
                t,
                grad_outputs=torch.ones_like(phi),
                create_graph=True
            )[0]

    # ∂h/∂t
    dh_dt = torch.autograd.grad(
                h, 
                t,
                grad_outputs=torch.ones_like(h),
                create_graph=True
            )[0]

    # ∂h/∂x
    dh_dx = torch.autograd.grad(
                h, 
                x,
                grad_outputs=torch.ones_like(h),
                create_graph=True
            )[0]

    # ∂/∂x [ K(φ) ∂h/∂x ]
    Kphi     = K(phi)
    flux     = Kphi * dh_dx
    dflux_dx = torch.autograd.grad(
                    flux, 
                    x,
                    grad_outputs=torch.ones_like(flux),
                    create_graph=True
                )[0]

    # 残差 r
    r = dphi_dt + Ss * Sw(phi) * dh_dt - dflux_dx - q_source
    return r

# -------- 5️⃣ 学習ループ ---------------------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()

    # PDE 損失
    res  = residual()
    loss_pde = torch.mean(res**2)

    # 境界 & 初期条件損失（とりあえず Dirichlet: φ=0）
    phi_b0 = model(xb0, tb)
    phi_b1 = model(xb1, tb)
    phi_i  = model(xi , ti)
    loss_bc = torch.mean(phi_b0**2) + torch.mean(phi_b1**2) + torch.mean(phi_i**2)

    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f'Epoch {epoch:4d}  Loss={loss.item():.3e}  PDE={loss_pde.item():.3e}')

print('== 学習終了 ==')

# -------- 6️⃣ 可視化（簡単） -----------------------------------
try:
    import matplotlib.pyplot as plt
    with torch.no_grad():
        x_plot = torch.linspace(0,1,100).unsqueeze(1).to(device)
        t_mid  = 0.5*torch.ones_like(x_plot)
        phi_plot = model(x_plot, t_mid).cpu().numpy()
    plt.plot(x_plot.cpu().numpy(), phi_plot)
    plt.title('phi(x, t=0.5)')
    plt.xlabel('x'); plt.ylabel('phi')
    # ファイル名に日本時間を使用
    plt.savefig(f"result_{current_time_jst}.png")
except ImportError:
    print("matplotlib が無いのでグラフはスキップしました。")
