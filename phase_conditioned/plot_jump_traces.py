"""plot_jump_traces.py — 相位突变重同步曲线（README 用图，英文标签）。

用法（env_isaaclab python，matplotlib 英文标签防方框）：
  python -u phase_conditioned/plot_jump_traces.py
输出：phase_conditioned/results/jump_resync.png
"""
import glob
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "jump_traces")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "jump_resync.png")

files = sorted(glob.glob(os.path.join(TRACE_DIR, "jump_ep*.npz")))
if not files:
    raise SystemExit(f"no traces in {TRACE_DIR}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)

# --- 左图：全部 8 局的 contact-schedule match（细线）+ 均值（粗线） ---
ax = axes[0]
all_match = []
t_common = None
for f in files:
    d = np.load(f)
    if t_common is None:
        t_common = d["t"]
        jump_at = float(d["jump_at"])
    all_match.append(d["match"])
M = np.vstack(all_match)
ax.plot(t_common, M.T, color="tab:blue", alpha=0.25, lw=0.7)
ax.plot(t_common, M.mean(axis=0), color="tab:blue", lw=1.8, label="mean (n=%d)" % len(files))
ax.axvline(jump_at, color="tab:red", ls="--", lw=1.2, label="phase-rate jump")
ax.axhline(0.9 * M[:, int(jump_at / d["t"][1]) - 100:int(jump_at / d["t"][1])].mean(),
           color="gray", ls=":", lw=1.0, label="0.9 x baseline")
ax.set_ylabel("contact-schedule match")
ax.set_xlabel("time (s)")
ax.set_ylim(0.5, 1.05)
ax.set_title("Phase-rate jump 1.4 -> 1.9 Hz: contact-schedule match")
ax.legend(fontsize=8, loc="lower right")

# --- 右图：突变前后 3s 放大，逐局 ---
ax = axes[1]
window = 3.0
mask = (t_common >= jump_at - 1.0) & (t_common <= jump_at + window)
for i, m in enumerate(all_match):
    ax.plot(t_common[mask] - jump_at, m[mask], color="tab:blue", alpha=0.3, lw=0.7)
ax.plot(t_common[mask] - jump_at, M[:, mask].mean(axis=0), color="tab:blue", lw=1.8)
ax.axvline(0, color="tab:red", ls="--", lw=1.2)
ax.set_ylabel("contact-schedule match")
ax.set_xlabel("time after jump (s)")
ax.set_title("Zoom around jump")
ax.set_ylim(0.5, 1.05)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"saved: {OUT_PATH}")
