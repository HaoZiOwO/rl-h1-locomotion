"""
train_sac.py — SAC 训练 Humanoid-v4（与 PPO 2M 步控制变量对比）

对比设计：
  PPO  2M 步 → 508.3 ± 47.3（已有，on-policy）
  SAC  2M 步 → ???（本次，off-policy）
  BC   9913 条示范 → 519.0 ± 51.1（已有，监督学习）

产出：
  sac_humanoid.zip（模型）
  results/sac_logs/（训练日志，含周期评估曲线）
  results/sac_tb/（tensorboard）
"""
import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

SEED = 42
TOTAL_TIMESTEPS = 2_000_000          # 与 PPO 完全相同的步数预算（控制变量）
EVAL_EVERY = 50_000                  # 每 5 万步评估一次（出对比曲线用）

os.makedirs("results/sac_logs", exist_ok=True)
os.makedirs("results/sac_tb", exist_ok=True)

# 训练环境（带 Monitor 记录每局奖励）
train_env = Monitor(gym.make("Humanoid-v4"), "results/sac_logs/")
# 独立评估环境（不共享，评估更干净）
eval_env = gym.make("Humanoid-v4")

model = SAC(
    "MlpPolicy",
    train_env,
    seed=SEED,
    verbose=0,
    tensorboard_log="results/sac_tb",
)

# 周期评估回调：每 5 万步跑 10 局，记录 mean reward → 对比曲线数据源
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="results/sac_logs/best",
    log_path="results/sac_logs/",
    eval_freq=EVAL_EVERY,
    n_eval_episodes=10,
    deterministic=True,
)

print(f"[INFO] SAC 训练开始：{TOTAL_TIMESTEPS} 步（与 PPO 同步数）")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
model.save("sac_humanoid.zip")
print("[INFO] SAC 训练完成，模型已保存 sac_humanoid.zip")

train_env.close()
eval_env.close()
