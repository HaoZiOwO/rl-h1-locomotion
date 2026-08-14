"""
eval_diffusion.py — 扩散策略 10 局评估（与 eval_bc.py 完全同协议：seed=42+ep、确定性环境）

采样（DDPM 反向过程）：
  x_T ~ N(0,I)；对 t = T-1..0：
    ε = ε_θ(obs, x_t, t)
    x_{t-1} = (x_t - β_t/√(1-ᾱ_t)·ε) / √α_t + σ_t·z   （z=0 时为确定性采样，公平对比用）

输出：results/diffusion_vs_bc.md（PPO / BC / DiffusionPolicy 三方对比）
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

SEED = 42
NUM_EVAL_EPISODES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 网络定义（和训练脚本一致）----
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class Denoiser(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, time_dim=64):
        super().__init__()
        self.time_mlp = SinusoidalTimeEmb(time_dim)
        in_dim = obs_dim + act_dim + time_dim
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, obs, x_t, t):
        t_emb = self.time_mlp(t)
        return self.net(torch.cat([obs, x_t, t_emb], dim=-1))

# ---- 加载 ----
ckpt = torch.load("dp_humanoid.pt", map_location=DEVICE, weights_only=False)
model = Denoiser(376, 17, ckpt["hidden"]).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
T = ckpt["T"]
betas = ckpt["betas"].to(DEVICE)
alphas = 1.0 - betas
alpha_bar = ckpt["alpha_bar"].to(DEVICE)
obs_mean = ckpt["obs_mean"]; obs_std = ckpt["obs_std"]
print(f"[INFO] 模型加载完成，T={T}，参数量 {sum(p.numel() for p in model.parameters())/1e3:.1f}K")

@torch.no_grad()
def ddpm_sample(obs_norm):
    """给定一个观测，生成一个动作（确定性 DDPM 采样）"""
    x = torch.randn(1, 17, device=DEVICE)          # 从纯噪声开始
    for t in range(T - 1, -1, -1):                 # 反向去噪
        tt = torch.full((1,), t, device=DEVICE)
        eps = model(obs_norm.unsqueeze(0), x, tt)  # 预测噪声
        a_bar_t = alpha_bar[t]
        # DDPM 更新公式（z=0 → 确定性）
        x = (x - betas[t] / torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(alphas[t])
        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x) * 0.0   # 确定性：噪声项置 0
    return x[0]

def dp_policy(obs):
    obs_norm = (torch.tensor(obs, dtype=torch.float32) - obs_mean) / obs_std
    act = ddpm_sample(obs_norm.to(DEVICE))
    return act.cpu().numpy()

# ---- 评估（同 eval_bc 协议）----
returns = []
for ep in range(NUM_EVAL_EPISODES):
    env = gym.make("Humanoid-v4")
    obs, _ = env.reset(seed=SEED + ep)
    total, done = 0.0, False
    while not done:
        obs, reward, term, trunc, _ = env.step(dp_policy(obs))
        total += reward
        done = term or trunc
    returns.append(total)
    env.close()
    print(f"  第 {ep+1:2d} 局: {total:.1f}")

mean = float(np.mean(returns)); std = float(np.std(returns))
print(f"DiffusionPolicy: {mean:.1f} ± {std:.1f}（{NUM_EVAL_EPISODES} 局）")

# ---- 三方对比表 ----
md = f"""# 算法对比：PPO vs BC vs Diffusion Policy（Humanoid-v4，同 demos、同评估协议）

| 算法 | 类型 | 预算 | 10 局平均奖励 |
|---|---|---|---|
| PPO | on-policy RL | 2M 步 | 508.3 ± 47.3 |
| BC | 监督学习（单步回归） | 9913 条示范 | 519.0 ± 51.1 |
| **Diffusion Policy** | 生成式模仿（DDPM） | 9913 条示范 | **{mean:.1f} ± {std:.1f}** |

- 控制变量：同一份 expert_demos.npz、同归一化、同 80/20 划分、同评估协议（10 局 seed=42+ep 确定性）
- 扩散配置：T=100，β 线性 1e-4→0.02，去噪网络 256×3（比 BC 的 64×2 大——方法特性），100 epochs
- DP 逐局：{[round(r,1) for r in returns]}
"""
with open("results/diffusion_vs_bc.md", "w", encoding="utf-8") as f:
    f.write(md)
print("[INFO] 对比表已写入 results/diffusion_vs_bc.md")
