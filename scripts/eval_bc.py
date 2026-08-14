"""
eval_bc.py — 评估 BC 模型，并和 PPO 专家对比

各跑 10 局，报平均分 ± 标准差：
   PPO 专家分 = 上限参考（BC 的目标就是逼近它）
   BC 模型分   = 模仿的效果（抄作业抄到了几成）

输出：results/ppo_vs_bc.md（对比表，直接可以写进简历 / GitHub README）
"""
import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

# ============================================================
# 评估超参数
# ============================================================
NUM_EVAL_EPISODES = 10     # 每模型跑几局（越多越准，10 局够看趋势）
MAX_STEPS = 1000
SEED = 42

# ============================================================
# 第1步：加载 BC 模型
# ============================================================
ckpt = torch.load("bc_humanoid.pt", map_location="cpu", weights_only=False)

class BCNet(torch.nn.Module):   # 结构必须和 train_bc.py 完全一致
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        layers = []
        d_in = obs_dim
        for h in hidden:
            layers += [torch.nn.Linear(d_in, h), torch.nn.ReLU()]
            d_in = h
        layers.append(torch.nn.Linear(d_in, act_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

bc = BCNet(376, 17, ckpt["hidden"])
bc.load_state_dict(ckpt["state_dict"])
bc.eval()

obs_mean = ckpt["obs_mean"]
obs_std = ckpt["obs_std"]

# ============================================================
# 第2步：通用评估函数 —— 跑 N 局，返回每局的总奖励
# ============================================================
def evaluate(policy_fn, name):
    env = gym.make("Humanoid-v4")
    returns = []
    for ep in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        total = 0.0
        for _ in range(MAX_STEPS):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    print(f"{name}: {mean:.1f} ± {std:.1f}（{NUM_EVAL_EPISODES} 局）")
    return mean, std

# ============================================================
# 第3步：定义两种策略的"动作函数"
# ============================================================
def bc_policy(obs):
    # BC 输入前要做和训练时一样的归一化
    x = torch.tensor((obs - obs_mean) / obs_std, dtype=torch.float32)
    with torch.no_grad():
        return bc(x).numpy()

ppo_model = PPO.load("ppo_humanoid")
def ppo_policy(obs):
    # 专家回放用确定性策略（取均值动作，最稳定）
    action, _ = ppo_model.predict(obs, deterministic=True)
    return action

# ============================================================
# 第4步：开跑，把结果写进对比报告
# ============================================================
bc_mean, bc_std = evaluate(bc_policy, "BC 模型")
ppo_mean, ppo_std = evaluate(ppo_policy, "PPO 专家")

os.makedirs("results", exist_ok=True)
with open("results/ppo_vs_bc.md", "w", encoding="utf-8") as f:
    f.write("# PPO vs BC 对比（Humanoid-v4）\n\n")
    f.write(f"- 评估环境：Humanoid-v4（MuJoCo），每模型 {NUM_EVAL_EPISODES} 局\n")
    f.write(f"- PPO 专家：2M 步训练（训练日志 reward ≈ 507 / 2h）\n")
    f.write(f"- BC 模型：MLP {ckpt['hidden']}，100 epoch，训练集来自专家 100 条轨迹\n\n")
    f.write("| 模型 | 平均奖励 | 标准差 |\n|---|---|---|\n")
    f.write(f"| PPO 专家 | {ppo_mean:.1f} | {ppo_std:.1f} |\n")
    f.write(f"| BC 模型 | {bc_mean:.1f} | {bc_std:.1f} |\n\n")
    f.write(f"BC 达到专家的 {bc_mean / max(ppo_mean, 1e-8) * 100:.0f}%。\n")
print("对比报告已写入 results/ppo_vs_bc.md")
