"""
eval_h1.py — 评估训练好的 H1 模型（平地 / 崎岖），输出简历级数据

用法：
    python eval_h1.py --task flat  --checkpoint logs/rsl_rl/h1_flat/2026-08-06_16-30-59/model_1000.pt
    python eval_h1.py --task rough --checkpoint logs/rsl_rl/h1_rough/2026-08-06_16-45-01/model_3000.pt
"""
import argparse

from isaaclab.app import AppLauncher

# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
args_cli, _ = parser.parse_known_args()

# 以 headless 模式启动 Isaac Sim（不弹窗口）
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import gymnasium as gym
from packaging import version
import importlib.metadata as metadata

from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
import isaaclab_tasks  # noqa: F401

# ============================================================
# 按任务选择 环境配置 + 训练器配置（和训练时完全一致）
# ============================================================
if args_cli.task == "flat":
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg as EnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
        H1FlatPPORunnerCfg as RunnerCfg,
    )
    task_name = "Isaac-Velocity-Flat-H1-v0"
else:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg as EnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
        H1RoughPPORunnerCfg as RunnerCfg,
    )
    task_name = "Isaac-Velocity-Rough-H1-v0"

env_cfg = EnvCfg()
env_cfg.scene.num_envs = 1          # 单环境评估，逐局统计
agent_cfg = RunnerCfg()

installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

# 创建环境并套 rsl_rl 包装器
env = gym.make(task_name, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# 加载检查点（和 play.py 相同的官方加载流程）
resume_path = retrieve_file_path(args_cli.checkpoint)
print(f"[INFO] 加载检查点: {resume_path}")
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)

# ============================================================
# 评估循环：跑 N 局，记录每局奖励和步数
# ============================================================
max_steps = env.unwrapped.max_episode_length
returns, lengths = [], []
ep_return, ep_steps, ep_done = 0.0, 0, 0
obs = env.get_observations()

with torch.inference_mode():
    while ep_done < args_cli.num_episodes:
        actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)
        policy.reset(dones)                     # rsl_rl>=4.0 需要
        ep_return += rewards[0].item()
        ep_steps += 1
        if bool(dones[0]) or ep_steps >= max_steps:
            returns.append(ep_return)
            lengths.append(ep_steps)
            ep_return, ep_steps, ep_done = 0.0, 0, ep_done + 1

returns = np.array(returns)
print(f"\n=== {args_cli.task.upper()} 评估结果（{args_cli.num_episodes} 局）===")
print(f"平均奖励: {returns.mean():.1f} ± {returns.std():.1f}")
print(f"平均存活步数: {np.mean(lengths):.0f}（上限 {max_steps}）")
print(f"逐局奖励: {[round(r, 1) for r in returns]}")

env.close()
simulation_app.close()
