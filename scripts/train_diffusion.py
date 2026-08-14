"""
train_diffusion.py — 条件扩散策略（Diffusion Policy）训练 Humanoid-v4

JD 要求 2 点名算法实战：diffusion policy（生成式模仿学习）。
和 BC 用【同一份 expert_demos.npz】、同评估协议（10 局确定性、同种子基）——
直接对比：PPO 508 / BC 519 / DiffusionPolicy ????。

原理（面试能讲）：
  1. 前向扩散 q(x_t|x_0)：给专家动作逐步加噪（T 步，β 调度）
  2. 训练目标：网络 ε_θ(obs, x_t, t) 预测加进去的噪声 ε（MSE）
  3. 推理：从纯噪声 x_T ~ N(0,I) 出发，迭代去噪 T 步 → 生成动作
  4. 条件：观测 obs 通过 concat 注入，时间步 t 用正弦位置编码

产物：
  dp_humanoid.pt（网络+归一化统计）、results/dp_train.log、训练曲线
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_FILE = "expert_demos.npz"
OUT_FILE = "dp_humanoid.pt"

# ---- 扩散配置（DDPM 标准设置）----
T = 100                     # 扩散步数
BETA_MIN, BETA_MAX = 1e-4, 0.02   # 噪声调度范围（线性）
# ---- 训练配置（和 BC 对齐：同数据/同 epoch/同 batch/同 lr）----
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100   # 冒烟测试传 2
BATCH_SIZE = 256
LR = 1e-3
HIDDEN = [256, 256, 256]    # 去噪网络（扩散模型通常比 BC 的 64×2 大——方法特性，对比时如实说明）
VAL_SPLIT = 0.2

# ============ 1. 数据准备（和 BC 完全一致）============
data = np.load(IN_FILE)
obs = data["obs"].astype(np.float32)          # (9913, 376)
actions = data["actions"].astype(np.float32)  # (9913, 17)

obs_mean = obs.mean(axis=0)
obs_std = obs.std(axis=0) + 1e-8
obs_norm = (obs - obs_mean) / obs_std

split = int(len(obs) * (1 - VAL_SPLIT))
obs_tr, obs_va = obs_norm[:split], obs_norm[split:]
act_tr, act_va = actions[:split], actions[split:]
print(f"[INFO] 数据: {len(obs)} 条 | 训练 {len(obs_tr)} / 验证 {len(obs_va)} | obs {obs.shape[1]}D → act {actions.shape[1]}D | {DEVICE}")

# ============ 2. 扩散调度（ᾱ 预计算）============
betas = torch.linspace(BETA_MIN, BETA_MAX, T, device=DEVICE)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)      # ᾱ_t = Π α_s（累积乘积）

def q_sample(x0, t, noise):
    """前向扩散：x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε"""
    a_bar = alpha_bar[t].unsqueeze(1)          # (B,1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

# ============ 3. 去噪网络 ============
class SinusoidalTimeEmb(nn.Module):
    """时间步 t 的正弦位置编码（Transformer 同款）——让网络知道"现在是第几步扩散" """
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
    def forward(self, t):                       # t: (B,) 整数
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

class Denoiser(nn.Module):
    """ε_θ(obs, x_t, t)：输入观测+带噪动作+时间编码 → 预测噪声 ε"""
    def __init__(self, obs_dim, act_dim, hidden, time_dim=64):
        super().__init__()
        self.time_mlp = SinusoidalTimeEmb(time_dim)
        in_dim = obs_dim + act_dim + time_dim   # concat 注入
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]   # SiLU 是扩散模型常用激活
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, obs, x_t, t):
        t_emb = self.time_mlp(t)
        return self.net(torch.cat([obs, x_t, t_emb], dim=-1))

model = Denoiser(obs.shape[1], actions.shape[1], HIDDEN).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print(f"[INFO] 去噪网络参数量: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")

# ============ 4. 训练循环 ============
def make_batch(n):
    idx = np.random.randint(0, len(obs_tr), n)
    return (torch.tensor(obs_tr[idx], device=DEVICE),
            torch.tensor(act_tr[idx], device=DEVICE))

print(f"[INFO] 开始训练（{EPOCHS} epochs，DDPM ε 预测）...")
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches = (len(obs_tr) + BATCH_SIZE - 1) // BATCH_SIZE
    for _ in range(n_batches):
        ob, a0 = make_batch(BATCH_SIZE)
        t = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE)   # 每样本随机一步
        noise = torch.randn_like(a0)
        xt = q_sample(a0, t, noise)                              # 加噪
        pred = model(ob, xt, t)                                  # 预测噪声
        loss = nn.functional.mse_loss(pred, noise)               # ε 预测损失
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    # 验证损失（前向扩散 MSE）
    model.eval()
    with torch.no_grad():
        idx = np.random.randint(0, len(obs_va), 512)
        ob = torch.tensor(obs_va[idx], device=DEVICE)
        a0 = torch.tensor(act_va[idx], device=DEVICE)
        t = torch.randint(0, T, (512,), device=DEVICE)
        noise = torch.randn_like(a0)
        va_loss = nn.functional.mse_loss(model(ob, q_sample(a0, t, noise), t), noise).item()
    print(f"epoch {epoch+1:3d}/{EPOCHS}  train_mse={total_loss/n_batches:.5f}  val_mse={va_loss:.5f}  ({time.time()-t0:.0f}s)")
print(f"[INFO] 训练完成，耗时 {time.time()-t0:.0f}s")

# ============ 5. 保存（网络 + 归一化统计，eval 用）============
torch.save({
    "model": model.state_dict(),
    "obs_mean": obs_mean, "obs_std": obs_std,
    "T": T, "betas": betas.cpu(), "alpha_bar": alpha_bar.cpu(),
    "hidden": HIDDEN,
}, OUT_FILE)
print(f"[INFO] 模型已保存: {OUT_FILE}")
