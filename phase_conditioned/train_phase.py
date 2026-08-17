"""train_phase.py — 训练 phase-conditioned H1 平地步态策略（冲刺期）。

用法：
  python train_phase.py                        # 标准训练：2048 envs，1000 iter（与官方 flat 一致）
  python train_phase.py --smoke                # 冒烟测试：16 envs，2 iter（先跑这个验证环境/维度/奖励）
  python train_phase.py --phase_rate 1.9       # 换步频（默认 1.4 Hz）

与官方 h1_flat 训练的唯一差异：
  env 侧（PhaseH1FlatEnvCfg）：+相位时钟 obs（2 维）+ 接触调度 reward（weight 0.5）
  PPO 侧（H1FlatPPORunnerCfg）：完全一致（同网络 128x3、同超参、同 1000 iter、同 2048 envs）
→ 性能差异只能归因于"相位条件化"这一个变量。
"""
import argparse
import os
import sys
import time
from datetime import datetime

# 让本目录（phase_conditioned/）可被 import（gym 注册需要 phase_env / phase_env_cfg）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="冒烟测试：16 envs / 2 iter")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=None)  # None = PPO cfg 默认 1000
parser.add_argument("--phase_rate", type=float, default=1.4)
parser.add_argument("--phase_contact_weight", type=float, default=0.5)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch  # noqa: F401
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import phase_env  # noqa: F401  （注册 Isaac-Velocity-Flat-H1-Phase-v0）
from phase_env_cfg import PhaseH1FlatEnvCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
    H1FlatPPORunnerCfg,
)

# ============ 环境配置（与官方 flat 的差异只在这里） ============
env_cfg = PhaseH1FlatEnvCfg()
env_cfg.phase_rate = args_cli.phase_rate
env_cfg.phase_contact_weight = args_cli.phase_contact_weight
env_cfg.scene.num_envs = 16 if args_cli.smoke else args_cli.num_envs

# ============ PPO 配置（与官方 flat 完全一致） ============
agent_cfg = H1FlatPPORunnerCfg()
agent_cfg.experiment_name = "h1_flat_phase"
agent_cfg.seed = args_cli.seed
env_cfg.seed = agent_cfg.seed
if args_cli.max_iterations:
    agent_cfg.max_iterations = args_cli.max_iterations
if args_cli.smoke:
    agent_cfg.max_iterations = 2

installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

# ============ 日志目录（与 train_custom.py 相同命名规则，独立实验名不覆盖旧资产） ============
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if agent_cfg.run_name:
    log_dir += f"_{agent_cfg.run_name}"
log_dir = os.path.join(log_root, log_dir)
env_cfg.log_dir = log_dir
print(f"[INFO] 日志目录: {log_dir}")
print(f"[INFO] 相位时钟: f={env_cfg.phase_rate} Hz, duty={env_cfg.phase_duty}, "
      f"接触调度权重={env_cfg.phase_contact_weight}")
print(f"[INFO] PPO: num_envs={env_cfg.scene.num_envs}, "
      f"max_iterations={agent_cfg.max_iterations}, seed={agent_cfg.seed}")

# ============ 建环境 + 训练 ============
env = gym.make("Isaac-Velocity-Flat-H1-Phase-v0", cfg=env_cfg, render_mode=None)
print(f"[INFO] obs_dim={env.unwrapped.observation_manager.group_obs_dim['policy']} "
      f"(官方 flat 为 69，本环境应 = 71)")
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
runner.add_git_repo_to_log(__file__)

start_time = time.time()
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
print(f"[INFO] 训练完成，耗时: {round(time.time() - start_time, 2)} 秒")

env.close()
simulation_app.close()
