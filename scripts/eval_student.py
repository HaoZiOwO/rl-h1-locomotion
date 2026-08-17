"""eval_student.py — 只评估学生（受限观测），教师分数用已知的 508.3±47.3"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

LIMITED_IDX = list(range(0, 5)) + list(range(22, 45))
DEVICE = "cpu"
TEACHER_MEAN, TEACHER_STD = 508.3, 47.3   # 已知：results/ppo_vs_bc.md（10 局确定性）


class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        layers = []
        d_in = obs_dim
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU()]
            d_in = h
        layers.append(nn.Linear(d_in, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


ckpt = torch.load("student_bc_limited.pt", map_location="cpu", weights_only=False)
model = BCNet(len(LIMITED_IDX), 17, ckpt["hidden"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
obs_mean = ckpt["obs_mean"]
obs_std = ckpt["obs_std"]


def student_fn(obs_full):
    x = (obs_full[LIMITED_IDX] - obs_mean) / obs_std
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model(x).numpy()[0]


env = gym.make("Humanoid-v4")
rewards = []
for ep in range(10):
    obs, _ = env.reset(seed=100 + ep)
    total = 0.0
    done = False
    while not done:
        a = student_fn(obs)
        obs, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    rewards.append(total)
    print(f"episode {ep+1}: {total:.1f}")
env.close()

s_mean, s_std = float(np.mean(rewards)), float(np.std(rewards))
print(f"\n[RESULT] 学生(受限观测 28维): {s_mean:.1f} ± {s_std:.1f}")
print(f"[RESULT] 教师(全观测 376维): {TEACHER_MEAN:.1f} ± {TEACHER_STD:.1f}（已知）")
print(f"[RESULT] 性能保持率: {s_mean / TEACHER_MEAN * 100:.1f}%")

with open("results/teacher_student_distill.md", "w", encoding="utf-8") as f:
    f.write("# Teacher-Student 蒸馏结果\n\n")
    f.write("- 教师: PPO 专家，Humanoid-v4 全 376 维观测\n")
    f.write(f"- 学生: BC，受限 {len(LIMITED_IDX)} 维观测（根姿态+全身速度，砍掉关节角度/惯性/力）\n")
    f.write(f"- 蒸馏: 学生 BC 从 9913 条教师示范学习（观测砍到 {len(LIMITED_IDX)} 维）\n\n")
    f.write("## 得分（10 局确定性）\n\n")
    f.write("| 策略 | 观测 | 得分 |\n|---|---|---|\n")
    f.write(f"| 教师 PPO | 全 376 维 | {TEACHER_MEAN:.1f} ± {TEACHER_STD:.1f} |\n")
    f.write(f"| 学生 BC | 受限 {len(LIMITED_IDX)} 维 | {s_mean:.1f} ± {s_std:.1f} |\n\n")
    f.write(f"- 性能保持率: {s_mean / TEACHER_MEAN * 100:.1f}%\n")
    f.write("- 叙事: 传感器最少化（外骨骼 2.7kg 轻量化）下的性能代价；teacher-student 是 sim-to-real 第二条路径（特权信息蒸馏）\n")
print("\n[INFO] 结果已写入 results/teacher_student_distill.md")
