"""
collect_demos.py — 用训练好的 PPO 专家，采集"示范数据"（expert demos）

行为克隆（BC）的思路：
   强化学习是"自己摸索"，BC 是"抄作业"——
   让专家（已经训好的 PPO）走很多次，把每一步的
   (观测 obs → 动作 action) 配对记下来，
   然后训练一个"模仿网络"去拟合这些配对。

输出：expert_demos.npz（NumPy 压缩文件，含 obs 和 actions 两个数组）
"""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# ============================================================
# 超参数（想改就改这里）
# ============================================================
NUM_EPISODES = 100      # 采集多少条轨迹（episode）
MAX_STEPS = 1000        # 每条轨迹最多走多少步（Humanoid-v4 自带上限）
SEED = 42               # 随机种子：保证可复现
OUT_FILE = "expert_demos.npz"

# ============================================================
# 第1步：加载环境和专家模型
# ppo_humanoid.zip 就是 train.py 训出来的成果
# ============================================================
env = gym.make("Humanoid-v4")
model = PPO.load("ppo_humanoid")

# ============================================================
# 第2步：循环采集
# obs_list 里每个元素是一条轨迹的所有观测（二维数组 [步数, 376]）
# ============================================================
obs_list, act_list = [], []
total_steps = 0
episode = 0

while episode < NUM_EPISODES:
    obs, _ = env.reset(seed=SEED + episode)
    ep_obs, ep_act = [], []

    for _ in range(MAX_STEPS):
        # deterministic=False：从专家策略的分布里采样（带一点随机）
        # 这样采集到的示范更多样，BC 学到的是"策略的分布"而不是一条死板路线
        action, _ = model.predict(obs, deterministic=False)
        ep_obs.append(obs)
        ep_act.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_steps += 1
        if terminated or truncated:      # 摔倒了或时间到了 → 这条轨迹结束
            break

    # 只保留"至少走了 50 步"的轨迹（一出门就摔的坏示范直接扔掉）
    if len(ep_obs) >= 50:
        obs_list.append(np.array(ep_obs, dtype=np.float32))
        act_list.append(np.array(ep_act, dtype=np.float32))
        episode += 1

    if total_steps >= 150_000:           # 保险丝：最多采 15 万步，防止死循环
        break

env.close()

# ============================================================
# 第3步：把列表拼成一个大数组，存盘
# obs_all: [N, 376]，act_all: [N, 17]（N = 总样本数）
# ============================================================
obs_all = np.concatenate(obs_list, axis=0)
act_all = np.concatenate(act_list, axis=0)
np.savez_compressed(OUT_FILE, obs=obs_all, actions=act_all)

print(f"采集完成：{len(obs_all)} 条样本（{episode} 条轨迹，"
      f"平均每条 {len(obs_all) // max(episode, 1)} 步）")
print(f"观测维度 {obs_all.shape[1]}，动作维度 {act_all.shape[1]}，已保存到 {OUT_FILE}")
