"""
plot_curves.py — 从 tensorboard 事件文件绘制 H1 平地 vs 崎岖训练曲线
输出：isaaclab_h1/training_curves.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 两个训练 run 的事件文件目录（本地 Isaac 训练日志）
RUNS = {
    "Flat terrain (1000 iter)": r"C:\Users\jz233\robot-rl-project\isaaclab\logs\rsl_rl\h1_flat\2026-08-06_16-30-59",
    "Rough terrain (3000 iter)": r"C:\Users\jz233\robot-rl-project\isaaclab\logs\rsl_rl\h1_rough\2026-08-06_16-45-01",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "training_curves.png")

# 读取事件文件，自动发现"平均奖励"标签
accs, chosen_tag = {}, None
for name, run_dir in RUNS.items():
    ev_file = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))[0]
    acc = EventAccumulator(ev_file)
    acc.Reload()
    tags = acc.Tags()["scalars"]
    accs[name] = acc
    reward_tags = [t for t in tags if "mean_reward" in t.lower()]
    print(f"{name}: 找到奖励标签 {reward_tags}")
    if reward_tags and chosen_tag is None:
        chosen_tag = reward_tags[0]

if chosen_tag is None:
    raise SystemExit("未找到 mean_reward 标签，请检查事件文件")

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
for name, acc in accs.items():
    events = acc.Scalars(chosen_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    # 平滑（移动平均窗口 10），曲线更易读
    window = 10
    smooth = [
        sum(values[max(0, i - window):i + 1]) / (i - max(0, i - window) + 1)
        for i in range(len(values))
    ]
    ax.plot(steps, smooth, label=name, linewidth=2)
    ax.plot(steps, values, alpha=0.15, linewidth=1)

ax.set_xlabel("Iteration")
ax.set_ylabel("Mean reward")
ax.set_title("Unitree H1 Velocity Tracking — Flat vs Rough Terrain (PPO)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"训练曲线已保存: {OUT}")
