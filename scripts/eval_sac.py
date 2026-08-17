"""
eval_sac.py — SAC 模型 10 局评估（与 eval_bc.py 完全相同的协议：同种子、同局数、确定性推理）
输出：results/sac_vs_ppo.md 对比表
"""
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

SEED_BASE = 1000                     # 与 eval_bc.py 相同的种子基
N_EPS = 10


def evaluate(policy_fn, name):
    """跑 N 局，返回 (mean, std, 逐局列表)。种子与 BC 评估一致 → 公平对比。"""
    returns = []
    for ep in range(N_EPS):
        env = gym.make("Humanoid-v4")
        obs, _ = env.reset(seed=SEED_BASE + ep)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
        returns.append(total)
        env.close()
    arr = np.array(returns)
    return arr.mean(), arr.std(), returns


# 加载 SAC
model = SAC.load("sac_humanoid.zip")  # 完整 2M 步训练模型
sac_mean, sac_std, sac_list = evaluate(
    lambda obs: model.predict(obs, deterministic=True)[0], "SAC"
)

# 已有数据（来自 results/ppo_vs_bc.md）
ppo_mean, ppo_std = 508.3, 47.3
bc_mean, bc_std = 519.0, 51.1

print(f"\n=== 算法对比（各 10 局，确定性评估）===")
print(f"PPO (2M步)  : {ppo_mean:.1f} ± {ppo_std:.1f}")
print(f"BC  (9913条): {bc_mean:.1f} ± {bc_std:.1f}")
print(f"SAC (2M步)  : {sac_mean:.1f} ± {sac_std:.1f}")

md = f"""# 算法对比：PPO vs SAC vs BC（Humanoid-v4）

| 算法 | 类型 | 预算 | 10 局平均奖励 |
|---|---|---|---|
| PPO | on-policy | 2M 步 | {ppo_mean:.1f} ± {ppo_std:.1f} |
| SAC | off-policy | 2M 步 | {sac_mean:.1f} ± {sac_std:.1f} |
| BC | 监督学习（模仿） | 9913 条示范 | {bc_mean:.1f} ± {bc_std:.1f} |

- 控制变量：同步数预算（2M）、同评估协议（10 局确定性、同种子基）
- SAC 逐局：{[round(x,1) for x in sac_list]}
"""
with open("results/sac_vs_ppo.md", "w", encoding="utf-8") as f:
    f.write(md)
print(f"[INFO] 对比表已写入 results/sac_vs_ppo.md")
