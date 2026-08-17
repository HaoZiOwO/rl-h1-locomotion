"""ddim_latency.py — DP 部署权衡：DDIM 少步采样 步数-延迟-性能 三角表（提前做）

动机（面试金句素材）：DDPM 100 步去噪在实时预算下不可行，DDIM 少步能降到多少？
实时部署必须靠 action chunking 摊销 —— 衔接 ACT 的设计动机。

用法：python scripts/ddim_latency.py
输出：results/ddim_latency.md
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_EPISODES = 10
SEED = 42
MAX_STEPS = 5000  # 保险上限；协议对齐官方 eval_diffusion（走倒为止，不截断正常行走）

# ============ 网络定义（与 train_diffusion.py / eval_diffusion.py 完全一致；
# train 脚本无 __main__ 守卫，不能 import，只能内联定义） ============

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


ckpt = torch.load("dp_humanoid.pt", map_location="cpu", weights_only=False)
model = Denoiser(376, 17, ckpt["hidden"]).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
T = ckpt["T"]
betas = ckpt["betas"].to(DEVICE)
alphas = 1.0 - betas
alpha_bar = ckpt["alpha_bar"].to(DEVICE)
obs_mean = ckpt["obs_mean"]
obs_std = ckpt["obs_std"]
print(f"[INFO] DP 模型加载：T={T}, 参数量 {sum(p.numel() for p in model.parameters())/1e3:.1f}K")


@torch.no_grad()
def ddpm_sample(obs_norm):
    """官方 eval_diffusion.py 的原版 DDPM 确定性采样（基线行）。"""
    x = torch.randn(1, 17, device=DEVICE)
    ob = obs_norm.unsqueeze(0).to(DEVICE)
    for t in range(T - 1, -1, -1):
        tt = torch.full((1,), t, device=DEVICE)
        eps = model(ob, x, tt)
        a_bar_t = alpha_bar[t]
        x = (x - betas[t] / torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(alphas[t])
    return x[0]


@torch.no_grad()
def ddim_sample(obs_norm, n_steps):
    """确定性 DDIM（η=0）：从 T 步训练计划均匀抽 n_steps 个子步做反向去噪。"""
    # 均匀抽子步（含 T-1 和 0），严格降序
    subseq = np.unique(np.linspace(T - 1, 0, n_steps).round().astype(int))[::-1]
    x = torch.randn(1, 17, device=DEVICE)
    ob = obs_norm.unsqueeze(0).to(DEVICE)
    for i, t in enumerate(subseq):
        tt = torch.full((1,), int(t), device=DEVICE)
        eps = model(ob, x, tt)
        a_bar_t = alpha_bar[t]
        # x0 估计 + DDIM 确定性递推（η=0）
        x0 = (x - torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(a_bar_t)
        t_prev = subseq[i + 1] if i + 1 < len(subseq) else -1
        a_bar_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=DEVICE)
        x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1 - a_bar_prev) * eps
    return x[0]


def measure_latency(sampler, trials=50):
    """单动作总延迟 = 一次采样的端到端墙钟（50 次取均值）。"""
    obs_norm = torch.tensor(np.zeros(376), dtype=torch.float32)
    for _ in range(3):
        sampler(obs_norm)                      # 预热
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        sampler(obs_norm)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / trials


def evaluate(sampler):
    """10 局确定性评估（同 eval_diffusion 协议：走倒为止；5000 步为保险上限）。"""
    returns = []
    for ep in range(NUM_EVAL_EPISODES):
        env = gym.make("Humanoid-v4")
        obs, _ = env.reset(seed=SEED + ep)
        total = 0.0
        for _ in range(MAX_STEPS):
            obs_norm = (torch.tensor(obs, dtype=torch.float32) - obs_mean) / obs_std
            act = sampler(obs_norm)
            obs, r, term, trunc, _ = env.step(act.cpu().numpy())
            total += r
            if term or trunc:
                break
        returns.append(float(total))
        env.close()
    return float(np.mean(returns)), float(np.std(returns))


configs = [
    ("DDPM-100（基线）", ddpm_sample),
    ("DDIM-100", lambda o: ddim_sample(o, 100)),
    ("DDIM-20", lambda o: ddim_sample(o, 20)),
    ("DDIM-10", lambda o: ddim_sample(o, 10)),
    ("DDIM-5", lambda o: ddim_sample(o, 5)),
]
rows = []
print(f"{'采样':>16} | {'延迟 (ms)':>10} | {'return':>10}")
for name, sampler in configs:
    lat = measure_latency(sampler) * 1000
    mean, std = evaluate(sampler)
    rows.append((name, lat, mean, std))
    print(f"{name:>16} | {lat:>10.2f} | {mean:>8.1f} ± {std:.1f}")

# 单次前向参考（1 步的去噪网络成本）
with torch.no_grad():
    ob = torch.zeros(1, 376, device=DEVICE)
    x = torch.zeros(1, 17, device=DEVICE)
    tt = torch.zeros(1, dtype=torch.long, device=DEVICE)
    for _ in range(3):
        model(ob, x, tt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        model(ob, x, tt)
    torch.cuda.synchronize()
fwd_ms = (time.perf_counter() - t0) / 100 * 1000

os.makedirs("results", exist_ok=True)
with open("results/ddim_latency.md", "w", encoding="utf-8") as f:
    f.write("# DP 部署权衡：DDIM 少步采样（步数-延迟-性能）\n\n")
    f.write(f"- 基线：DDPM T=100（确定性采样，官方 eval_diffusion 原版采样器）；DDIM η=0 均匀抽子步\n")
    f.write(f"- 延迟口径：单动作端到端采样墙钟（50 次均值，含 GPU 同步）；单次去噪网络前向 {fwd_ms:.3f} ms\n")
    f.write(f"- 性能口径：10 局确定性评估（seed 42+i），与官方 diffusion_vs_bc.md 同协议（走倒为止）\n\n")
    f.write("| 采样器 | 单动作延迟 (ms) | return | vs DDPM-100 |\n|---|---|---|---|\n")
    base = rows[0][2]
    for name, lat, mean, std in rows:
        f.write(f"| {name} | {lat:.2f} | {mean:.1f} ± {std:.1f} | {mean / base * 100:.0f}% |\n")
    f.write("\n**结论**（数字自己说话）：\n")
    f.write(f"- 单次前向 {fwd_ms:.2f} ms × 步数 ≈ 延迟线性；DDPM-100 ≈ {rows[0][1]:.1f} ms/动作，"
            f"1 kHz（1 ms）实时预算下不可行\n")
    f.write(f"- DDIM-10 降到 {rows[3][1]:.2f} ms（性能 {rows[3][2] / base * 100:.0f}%）；"
            f"DDIM-5 {rows[4][1]:.2f} ms（性能 {rows[4][2] / base * 100:.0f}%）\n")
    f.write("- 剩余延迟仍需 chunking 摊销（一次推理出 k 步动作）——这正是 ACT 的设计动机，"
            "本项目 ACT 自实现见 results/act_vs_bc.md\n")
print(f"[INFO] 报告已写入 results/ddim_latency.md（单次前向 {fwd_ms:.3f} ms）")
