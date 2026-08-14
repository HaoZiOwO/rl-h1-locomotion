"""
plot_sac.py — SAC sample efficiency vs PPO (Humanoid-v4)
Output: assets/sac_sample_efficiency.png
Data: assets/sac_logs/evaluations.npz (SAC training eval callback, 40 checkpoints)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "assets", "sac_logs", "evaluations.npz")
OUT = os.path.join(HERE, "assets", "sac_sample_efficiency.png")

d = np.load(DATA)
ts = d["timesteps"]
rs = d["results"].mean(axis=1)

PPO_SCORE = 508.3

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(ts / 1e6, rs, "-o", color="#1565c0", markersize=5, linewidth=2, label="SAC (off-policy)")
ax.axhline(PPO_SCORE, color="#e65100", linestyle="--", linewidth=2, label=f"PPO 2M steps = {PPO_SCORE:.0f}")
cross = np.where(rs > PPO_SCORE)[0]
if len(cross):
    i = cross[0]
    ax.annotate(f"SAC exceeds PPO at {ts[i] / 1e4:.0f}0K steps",
                xy=(ts[i] / 1e6, rs[i]), xytext=(0.42, 0.78),
                arrowprops=dict(arrowstyle="->", color="#1565c0"),
                fontsize=10, color="#1565c0", transform=ax.transAxes)
ax.set_xlabel("Training steps (millions)", fontsize=11)
ax.set_ylabel("Mean eval reward (10 deterministic episodes)", fontsize=11)
ax.set_title("SAC vs PPO: Sample Efficiency (Humanoid-v4, equal step budget)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print(f"OK: {ts[-1]} steps, SAC final {rs[-1]:.0f} vs PPO {PPO_SCORE:.0f}, cross at {ts[cross[0]]/1e4:.0f}0K steps")
