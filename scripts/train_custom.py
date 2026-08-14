"""
train_custom.py — 自定义训练脚本（DR ablation + 多 seed 消融）

用法：
  python train_custom.py --mode nodr  --seed 0   # DR ablation：关掉回合内域随机化事件，保留地形
  python train_custom.py --mode rough --seed 2   # 标准 rough，指定 seed（稳定性实验）
  python train_custom.py --mode flat  --seed 2   # 标准 flat，指定 seed
  # 可选：--num_envs 2048 --max_iterations 3000

原理：
  - 域随机化（DR）的有效性需要 ablation 证明：关掉 push/外力/质量扰动等事件，
    对比训练曲线。若关掉 DR 后性能明显下降 → DR 有用（面试杀手锏图）。
  - 单 seed 结论脆弱，补 2 个 seed 报 mean±std。
"""
import argparse
import os
import time
from datetime import datetime
import importlib.metadata as metadata

import torch  # noqa: F401  负载注入等用到

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="rough", choices=["rough", "flat", "nodr", "anymal_rough"])
parser.add_argument("--reward_abl", type=str, default="none", choices=["none", "no_action_rate", "torque_pen"],
                    help="奖励敏感性实验：移除动作平滑惩罚 / 启用力矩惩罚（H1 默认关）")
parser.add_argument("--payload", type=float, default=0.0, help="训练时给躯干施加的固定负载（kg）")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=None)  # None = 用配置默认值
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import isaaclab_tasks  # noqa: F401

if args_cli.mode == "flat":
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg as EnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
        H1FlatPPORunnerCfg as RunnerCfg,
    )
    task_name = "Isaac-Velocity-Flat-H1-v0"
elif args_cli.mode == "anymal_rough":
    # 四足 Anymal-C 崎岖地形（形态广度：双足→四足）
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
        AnymalCRoughEnvCfg as EnvCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
        AnymalCRoughPPORunnerCfg as RunnerCfg,
    )
    task_name = "Isaac-Velocity-Rough-Anymal-C-v0"
else:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg as EnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
        H1RoughPPORunnerCfg as RunnerCfg,
    )
    task_name = "Isaac-Velocity-Rough-H1-v0"

# ============ 配置 ============
env_cfg = EnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs

if args_cli.mode == "nodr":
    # DR ablation：禁用回合内扰动事件（保留地形随机化和初始状态随机化）
    for term in ["push_robot", "base_external_force_torque", "add_base_mass", "base_com"]:
        if hasattr(env_cfg.events, term):
            setattr(env_cfg.events, term, None)
            print(f"[ABLATION] 已禁用随机化事件: {term}")

# 奖励敏感性实验：展示"奖励工程"理解
if args_cli.reward_abl == "no_action_rate":
    try:
        env_cfg.rewards.action_rate_l2 = None
        print("[REWARD_ABL] 已移除 action_rate_l2（动作平滑惩罚）→ 预期：动作更激进/抖动")
    except Exception as e:
        print(f"[WARN] 奖励消融 no_action_rate 失败: {e}")
elif args_cli.reward_abl == "torque_pen":
    try:
        # H1 默认 dof_torques_l2.weight = 0.0（力矩惩罚关闭）
        # 启用有意义的惩罚 → 预期：动作更省力、速度可能略降（能耗-性能权衡）
        env_cfg.rewards.dof_torques_l2.weight = -0.015
        print(f"[REWARD_ABL] 启用力矩惩罚 weight=-0.015（默认 0.0）→ 预期节能步态")
    except Exception as e:
        print(f"[WARN] 奖励消融 torque_pen 失败: {e}")

agent_cfg = RunnerCfg()
agent_cfg.seed = args_cli.seed
if args_cli.max_iterations:
    agent_cfg.max_iterations = args_cli.max_iterations
env_cfg.seed = agent_cfg.seed
if args_cli.mode == "nodr":
    agent_cfg.experiment_name = "h1_rough_nodr"
elif args_cli.mode == "anymal_rough":
    agent_cfg.experiment_name = "anymal_c_rough"
elif args_cli.mode == "flat":
    if args_cli.reward_abl == "no_action_rate":
        agent_cfg.experiment_name = "h1_flat_noactrate"
    elif args_cli.reward_abl == "torque_pen":
        agent_cfg.experiment_name = "h1_flat_torquepen"
    elif args_cli.payload > 0:
        agent_cfg.experiment_name = "h1_flat_payload"
    else:
        agent_cfg.experiment_name = "h1_flat"
else:
    agent_cfg.experiment_name = "h1_rough"

installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

# ============ 日志目录（与官方 train.py 相同命名规则） ============
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if agent_cfg.run_name:
    log_dir += f"_{agent_cfg.run_name}"
log_dir = os.path.join(log_root, log_dir)
env_cfg.log_dir = log_dir
print(f"[INFO] 日志目录: {log_dir}")

# ============ 建环境 + 训练 ============
env = gym.make(task_name, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# 负载注入（外骨骼训练）：躯干加固定质量，训练"负重行走"策略（官方 API 同款）
if args_cli.payload > 0:
    try:
        art = env.unwrapped.scene["robot"]
        idx = [i for i, n in enumerate(art.body_names) if "torso" in n]
        if idx:
            env_ids = torch.arange(env.unwrapped.num_envs, device="cpu")
            masses = art.root_physx_view.get_masses()
            masses[:, idx] = art.data.default_mass[:, idx].clone() + args_cli.payload
            art.root_physx_view.set_masses(masses, env_ids)
            ratios = masses[:, idx] / art.data.default_mass[:, idx]
            inertias = art.root_physx_view.get_inertias()
            inertias[:, idx] = art.data.default_inertia[:, idx] * ratios[..., None]
            art.root_physx_view.set_inertias(inertias, env_ids)
            print(f"[INFO] 训练负载: 躯干 +{args_cli.payload}kg")
        else:
            print("[WARN] 未找到 torso 连杆，跳过负载注入")
    except Exception as e:
        print(f"[WARN] 负载注入失败: {e}")

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
runner.add_git_repo_to_log(__file__)

start_time = time.time()
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
print(f"[INFO] 训练完成，耗时: {round(time.time() - start_time, 2)} 秒")

env.close()
simulation_app.close()
