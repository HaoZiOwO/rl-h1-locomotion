"""
plot_results.py — 从 evals_v2 JSON 生成简历图表

输出（isaaclab_h1/assets/）：
  difficulty_curve.png   固定地形难度曲线（reward/success vs level，含误差棒）
  command_track.png      指令跟踪（cmd vs actual，平地+崎岖）
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

EVALS = r"C:\Users\jz233\robot-rl-project\isaaclab\logs\rsl_rl\evals_v2"
OUT = r"C:\Users\jz233\robot-rl-project\isaaclab_h1\assets"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(EVALS, "*.json")))
print(f"找到 {len(files)} 个评估结果文件")

# ============================================================
# 图1：难度-性能曲线
# ============================================================
curve_files = [f for f in files if "curve" in os.path.basename(f)]
if curve_files:
    data = json.load(open(curve_files[-1], encoding="utf-8"))["results"]
    levels, rewards, errs, succs = [], [], [], []
    for key, v in data.items():
        levels.append(v["observed"])
        rewards.append(v["reward_mean"])
        errs.append(v["reward_std"])
        succs.append(v["success_rate"])
    order = np.argsort(levels)
    levels = np.array(levels)[order]; rewards = np.array(rewards)[order]
    errs = np.array(errs)[order]; succs = np.array(succs)[order]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.errorbar(levels, rewards, yerr=errs, fmt="-o", color="#1565c0",
                 capsize=6, capthick=2, markersize=8, linewidth=2,
                 label="平均奖励（10 局）")
    ax1.set_xlabel("地形难度等级（terrain level）", fontsize=11)
    ax1.set_ylabel("平均奖励", color="#1565c0", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax1.set_xticks(levels)
    ax2 = ax1.twinx()
    ax2.plot(levels, succs, "--s", color="#e65100", markersize=8, linewidth=2, label="回合成功率")
    ax2.set_ylabel("回合成功率", color="#e65100", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#e65100")
    ax2.set_ylim(0, 1.1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)
    ax1.set_title("H1 崎岖地形：难度-性能曲线（固定地形等级 × 10 局）", fontsize=12)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "difficulty_curve.png"), dpi=150)
    plt.close(fig)
    print(f"difficulty_curve.png OK: levels={levels.tolist()} rewards={rewards.tolist()} succ={succs.tolist()}")
else:
    print("未找到 curve 结果")

# ============================================================
# 图2：指令跟踪
# ============================================================
track_files = [f for f in files if "track" in os.path.basename(f)]
if track_files:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tf in zip(axes, sorted(track_files)):
        d = json.load(open(tf, encoding="utf-8"))
        name = "平地" if "flat" in os.path.basename(tf) else "崎岖"
        traj = d["results"]["trajectory"]
        t = np.array(traj["t"]) / 200.0          # 步 → 秒（dt=0.005s）
        cmd = np.array(traj["cmd_x"])
        act = np.array(traj["act_x"])
        ax.plot(t, cmd, "--", color="#888", linewidth=2, label="指令速度（1.0 m/s）")
        ax.plot(t, act, color="#1565c0", linewidth=1.5, label="实际速度")
        ax.set_title(f"{name}：指令跟踪（固定 1.0 m/s 直线）", fontsize=11)
        ax.set_xlabel("时间（秒）", fontsize=10)
        ax.set_ylabel("前进速度（m/s）", fontsize=10)
        ax.set_ylim(0, 1.6)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ep = d["results"]["episode"]
        ax.text(0.03, 0.15, f"平均跟踪误差 {ep['err_xy_mean']:.2f} m/s\n奖励 {ep['reward']:.1f} / 存活 {ep['steps']} 步",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.suptitle("H1 指令跟踪测试（命令 vs 实际速度）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "command_track.png"), dpi=150)
    plt.close(fig)
    print("command_track.png OK")
else:
    print("未找到 track 结果")
