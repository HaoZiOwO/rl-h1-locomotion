"""
plot_sac.py — SAC 样本效率 vs PPO 对比图（Humanoid-v4）
输出：isaaclab_h1/assets/sac_sample_efficiency.png
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\jz233\robot-rl-project\isaaclab_h1\assets"
os.makedirs(OUT, exist_ok=True)

d = np.load("C:/Users/jz233/robot-rl-project/results/sac_logs/evaluations.npz")
ts = d["timesteps"]
rs = d["results"].mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(ts / 1e6, rs, "-o", color="#1565c0", markersize=5, linewidth=2, label="SAC（off-policy）")
# PPO 参照线（200 万步 → 508）
ax.axhline(508, color="#e65100", linestyle="--", linewidth=2, label="PPO 200 万步 = 508")
# SAC 首次超过 PPO 的点
cross = np.where(rs > 508)[0]
if len(cross):
    i = cross[0]
    ax.annotate(f"SAC 仅 {ts[i]/1e4:.0f} 万步即超过 PPO",
                xy=(ts[i] / 1e6, rs[i]), xytext=(0.35, 0.75),
                arrowprops=dict(arrowstyle="->", color="#1565c0"),
                fontsize=10, color="#1565c0", transform=ax.transAxes, xycoords="data")
ax.set_xlabel("训练步数（百万）", fontsize=11)
ax.set_ylabel("评估平均奖励（10 局确定性）", fontsize=11)
ax.set_title("SAC vs PPO：样本效率对比（Humanoid-v4，同步数预算）", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sac_sample_efficiency.png"), dpi=150)
plt.close(fig)
print(f"OK: {ts[-1]} 步, SAC 最新 {rs[-1]:.0f} vs PPO 508")
