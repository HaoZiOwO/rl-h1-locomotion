"""
plot_payload.py — 负载-性能曲线（外骨骼主打图）
数据：flat/rough/nodr 模型 × 0/10/25/40kg 躯干加载（10 局 mean±std）
输出：isaaclab_h1/assets/payload_curve.png
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

OUT = r"C:\Users\jz233\robot-rl-project\isaaclab_h1\assets"
os.makedirs(OUT, exist_ok=True)

# (模型, 负载kg, 奖励, 存活率)
data = {
    "flat（平地训练）":   [(0, 37.2, 1.0), (10, 37.4, 1.0), (25, 34.1, 1.0), (40, 5.0, 0.2)],
    "rough（地形+DR）":  [(0, 35.3, 1.0), (10, 36.1, 1.0), (25, 35.4, 1.0), (40, 29.9, 1.0)],
    "nodr（地形，无DR）": [(0, 35.0, 1.0), (40, 31.8, 1.0)],
}
colors = {"flat（平地训练）": "#e65100", "rough（地形+DR）": "#1565c0", "nodr（地形，无DR）": "#6a1b9a"}

fig, ax = plt.subplots(figsize=(10, 5.5))
for name, pts in data.items():
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    succ = [p[2] for p in pts]
    ax.plot(xs, ys, "-o", color=colors[name], linewidth=2, markersize=7,
            label=f"{name}（{'全部存活' if all(s == 1.0 for s in succ) else '40kg 仅 20% 存活'}）")
    # 标注存活率<100% 的点
    for x, y, s in zip(xs, ys, succ):
        if s < 1.0:
            ax.annotate(f"存活 {s:.0%}", xy=(x, y), xytext=(x + 1.5, y - 3),
                        arrowprops=dict(arrowstyle="->", color="#c62828"), fontsize=9, color="#c62828")
ax.axhline(30, color="#9e9e9e", linestyle=":", linewidth=1)
ax.text(41.5, 30.5, "存活基准线", fontsize=8, color="#757575")
ax.set_xlabel("躯干额外负载（kg，模拟外骨骼承重）", fontsize=11)
ax.set_ylabel("10 局平均奖励（标准奖励评估）", fontsize=11)
ax.set_title("负载-性能曲线：训练环境多样性决定负载鲁棒性", fontsize=12)
ax.set_xlim(-2, 46)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "payload_curve.png"), dpi=150)
plt.close(fig)
print("OK: payload_curve.png")
