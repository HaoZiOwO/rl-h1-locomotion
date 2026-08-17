"""ctrl_full_obs.py — 全观测学生对照（分离变量：失败是观测削减导致，还是 SAC 教师特性导致）"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

d = np.load("sac_demos.npz")
X = d["obs"]  # 全 376 维
m, s = X.mean(0), X.std(0) + 1e-8
Xn = torch.tensor((X - m) / s, dtype=torch.float32).to(DEVICE)
Y = torch.tensor(d["actions"], dtype=torch.float32).to(DEVICE)
n = len(Xn)
idx = np.random.RandomState(SEED).permutation(n)
tr, va = idx[: int(n * 0.8)], idx[int(n * 0.8):]


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


model = BCNet(376, 17, [64, 64]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    for b in range(0, len(tr), 512):
        ib = tr[b:b + 512]
        loss = nn.functional.mse_loss(model(Xn[ib]), Y[ib])
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    vl = nn.functional.mse_loss(model(Xn[va]), Y[va]).item()
print(f"[train] full-obs 学生 val mse={vl:.5f}（对照 limited 学生 0.0258）", flush=True)
model.eval()

env = gym.make("Humanoid-v4")
rewards = []
for ep in range(10):
    obs, _ = env.reset(seed=100 + ep)
    total, done = 0.0, False
    while not done:
        x = torch.tensor((obs - m) / s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a = model(x).cpu().numpy()[0]
        obs, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    rewards.append(float(total))
    print(f"  episode {ep+1}: {total:.1f}", flush=True)
env.close()
print(f"[RESULT] 全观测学生: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}（教师 6416.8 ± 711.8）", flush=True)
