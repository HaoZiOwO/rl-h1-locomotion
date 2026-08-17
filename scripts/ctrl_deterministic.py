"""ctrl_deterministic.py — 确定性示范对照（SAC 高熵教师：随机采样示范的方差污染 vs 确定性示范）"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMITED_IDX = list(range(0, 5)) + list(range(22, 45))
SEED = 42

# 采集确定性示范（教师走均值动作，即 SAC 评估时的 6416 策略）
teacher = SAC.load("sac_humanoid.zip")
env = gym.make("Humanoid-v4")
obs_all, act_all = [], []
for ep in range(10):
    obs, _ = env.reset(seed=SEED + ep)
    done = False
    while not done:
        a, _ = teacher.predict(obs, deterministic=True)
        obs_all.append(obs)
        act_all.append(a)
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
env.close()
obs_all = np.array(obs_all, dtype=np.float32)
act_all = np.array(act_all, dtype=np.float32)
print(f"[1/2] 确定性示范：{len(obs_all)} 条", flush=True)

X = obs_all[:, LIMITED_IDX]
m, s = X.mean(0), X.std(0) + 1e-8
Xn = torch.tensor((X - m) / s, dtype=torch.float32).to(DEVICE)
Y = torch.tensor(act_all, dtype=torch.float32).to(DEVICE)
n = len(Xn)
idx = np.random.RandomState(SEED).permutation(n)
tr = idx[: int(n * 0.8)]


class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        layers, dd = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(dd, h), nn.ReLU()]
            dd = h
        layers.append(nn.Linear(dd, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = BCNet(len(LIMITED_IDX), 17, [64, 64]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    for b in range(0, len(tr), 512):
        ib = tr[b:b + 512]
        loss = nn.functional.mse_loss(model(Xn[ib]), Y[ib])
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()
print("[2/2] 训练完成，评估中...", flush=True)

env = gym.make("Humanoid-v4")
rewards = []
for ep in range(10):
    obs, _ = env.reset(seed=100 + ep)
    total, done = 0.0, False
    while not done:
        x = torch.tensor((obs[LIMITED_IDX] - m) / s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a = model(x).cpu().numpy()[0]
        obs, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    rewards.append(float(total))
    print(f"  episode {ep+1}: {total:.1f}", flush=True)
env.close()
print(f"[RESULT] 确定性示范学生（28 维受限观测）: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}（教师 6416.8 ± 711.8）", flush=True)
