"""
eval_h1_v2.py — 升级版评估（专家级协议）

三种模式：
  basic  默认：N 局 mean±std + 能耗代理 + 完整上下文日志
  curve  固定地形难度曲线（rough）：--levels "0,3,6,9,12" --per_level 10
  track  指令跟踪：--fixed_cmd 用固定速度命令场景；否则随机命令。输出 cmd/actual 轨迹 JSON

用法：
  python eval_h1_v2.py --task rough --checkpoint <path> --mode curve --levels 0,3,6,9,12 --per_level 10
  python eval_h1_v2.py --task flat  --checkpoint <path> --mode track --fixed_cmd
  python eval_h1_v2.py --task flat  --checkpoint <path> --mode basic
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--mode", type=str, default="basic", choices=["basic", "curve", "track", "baseline"])
parser.add_argument("--baseline", type=str, default="zero", choices=["zero", "random"], help="baseline 模式的动作类型")
parser.add_argument("--payload", type=float, default=0.0, help="给机器人躯干施加的额外质量（kg），外骨骼载荷模拟")
parser.add_argument("--levels", type=str, default="0,3,6,9,12")
parser.add_argument("--per_level", type=int, default=10)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--fixed_cmd", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
import isaaclab_tasks  # noqa: F401

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

# ============ 配置 ============
env_cfg = EnvCfg()
env_cfg.scene.num_envs = 1
if args_cli.mode == "curve":
    try:
        env_cfg.curriculum.terrain_levels = None      # 关掉地形课程，才能测固定难度
        print("[INFO] 已禁用地形课程（curve 模式）")
    except Exception as e:
        print(f"[WARN] 禁用课程失败: {e}")
if args_cli.mode == "track" and args_cli.fixed_cmd:
    try:
        cmd = env_cfg.commands.velocity_command
        cmd.heading_command = False                    # 固定朝向（不随机转向）
        cmd.vel_range = (1.0, 1.0)                     # 固定速度 1.0 m/s
        print("[INFO] 固定指令场景: 恒速 1.0 m/s 直线前进")
    except Exception as e:
        print(f"[WARN] 设置固定指令失败，改用随机指令: {e}")

agent_cfg = RunnerCfg()
installed_version = __import__("importlib.metadata").metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

env = gym.make(task_name, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# 负载注入（外骨骼载荷模拟）：给躯干加质量（官方 randomize_rigid_body_mass 同款 API）
payload_applied = 0.0
if args_cli.payload > 0:
    try:
        art = env.unwrapped.scene["robot"]
        idx = [i for i, n in enumerate(art.body_names) if "torso" in n]
        if idx:
            env_ids = torch.arange(env.unwrapped.num_envs, device="cpu")
            masses = art.root_physx_view.get_masses()
            masses[:, idx] = art.data.default_mass[:, idx].clone() + args_cli.payload
            art.root_physx_view.set_masses(masses, env_ids)
            # 惯性张量按质量比例缩放（官方 recompute_inertia 同款）
            ratios = masses[:, idx] / art.data.default_mass[:, idx]
            inertias = art.root_physx_view.get_inertias()
            inertias[:, idx] = art.data.default_inertia[:, idx] * ratios[..., None]
            art.root_physx_view.set_inertias(inertias, env_ids)
            payload_applied = args_cli.payload
            print(f"[INFO] 已施加负载: 躯干 +{args_cli.payload}kg")
        else:
            print("[WARN] 未找到 torso 连杆，跳过负载注入")
    except Exception as e:
        print(f"[WARN] 负载注入失败: {e}")

resume_path = retrieve_file_path(args_cli.checkpoint)
print(f"[INFO] 加载检查点: {resume_path}")
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)

max_steps = env.unwrapped.max_episode_length
torch.manual_seed(args_cli.seed)
np.random.seed(args_cli.seed)

# ============ 单局收集 ============
def run_episode(record_traj=False):
    """跑一局，返回统计 + 可选轨迹。obs 布局：lin_vel(3)|ang_vel(3)|gravity(3)|cmd(3)|dof_pos|dof_vel|actions"""
    ep = {"reward": 0.0, "steps": 0, "err_xy": 0.0, "err_yaw": 0.0,
          "torque_pen": 0.0, "action_rate": 0.0, "success": False}
    traj = {"t": [], "cmd_x": [], "cmd_yaw": [], "act_x": [], "act_yaw": []}
    with torch.inference_mode():
        obs, _ = env.reset()          # reset 必须在 inference_mode 内（事件管理器写推理张量）
        act_dim = env.unwrapped.action_space.shape[-1]   # H1 = 19 个关节（shape 第一维是环境数）
        for step in range(max_steps):
            if args_cli.mode == "baseline":
                # baseline：零动作或随机动作（证明训练策略远优于瞎动）
                if args_cli.baseline == "zero":
                    actions = torch.zeros(1, act_dim, device=env.unwrapped.device)
                else:
                    actions = torch.randn(1, act_dim, device=env.unwrapped.device)
            else:
                actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            policy.reset(dones)
            ep["reward"] += rewards[0].item()
            ep["steps"] += 1
            # 从观测解码指令与速度（不依赖 infos，鲁棒）：
            # rsl_rl 5.x 的 obs 是 TensorDict（键如 "policy"），需先取观测组张量
            # obs 布局 = lin_vel(0:3) | ang_vel(3:6) | gravity(6:9) | cmd(9:12) | dof_pos | dof_vel | actions
            obs_keys = list(obs.keys())
            obs_t = obs["policy"] if "policy" in obs_keys else obs[obs_keys[0]]
            o = obs_t[0].cpu().numpy()
            cmd_x, cmd_y, cmd_yaw = o[9], o[10], o[11]
            act_x, act_y, act_yaw = o[0], o[1], o[5]
            ep["err_xy"] += float(((cmd_x - act_x) ** 2 + (cmd_y - act_y) ** 2) ** 0.5)
            ep["err_yaw"] += abs(float(cmd_yaw - act_yaw))
            if record_traj:
                traj["t"].append(step)
                traj["cmd_x"].append(float(cmd_x))
                traj["cmd_yaw"].append(float(cmd_yaw))
                traj["act_x"].append(float(act_x))
                traj["act_yaw"].append(float(act_yaw))
            # 能耗代理（infos 里有才取，防御式）
            if isinstance(infos, dict) and "Episode_Reward/dof_torques_l2" in infos:
                ep["torque_pen"] += float(infos["Episode_Reward/dof_torques_l2"][0])
            if bool(dones[0]):
                break
    n = max(ep["steps"], 1)
    for k in ("err_xy", "err_yaw", "torque_pen", "action_rate"):
        ep[k] = ep[k] / n                      # 转成每步均值
    ep["success"] = ep["steps"] >= max_steps
    return ep, traj

# ============ 上下文 ============
ctx = {
    "task": task_name, "checkpoint": resume_path, "mode": args_cli.mode,
    "seed": args_cli.seed, "num_envs": 1, "max_steps": max_steps,
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "levels_requested": args_cli.levels,
    "payload_kg": payload_applied,   # 实际施加的负载（注入失败则为 0，数据诚实）
}

# ============ 模式分发 ============
results = {}

if args_cli.mode == "curve":
    ctx["per_level"] = args_cli.per_level
    for lvl_str in args_cli.levels.split(","):
        lvl = int(lvl_str)
        # 固定地形等级：写 terrain_levels 张量 → 重置 → 读回真实等级
        # 注意：terrain_levels 是推理张量，inplace 赋值必须在 inference_mode 内
        try:
            with torch.inference_mode():
                t = env.unwrapped.scene.terrain.terrain_levels
                t[:] = min(lvl, 19)          # 防御：防止越界
                env.reset()                  # 重置以应用新地形原点
                observed = int(t[0].item())
        except Exception as e:
            print(f"[WARN] 设置地形等级 {lvl} 失败: {e}，记录实际等级继续")
            env.reset()
            try:
                observed = int(env.unwrapped.scene.terrain.terrain_levels[0].item())
            except Exception:
                observed = -1
        eps, _ = [], None
        for _ in range(args_cli.per_level):
            ep, _ = run_episode()
            eps.append(ep)
        r = np.array([e["reward"] for e in eps])
        succ = np.mean([e["success"] for e in eps])
        err = np.mean([e["err_xy"] for e in eps])
        results[f"level_{observed}"] = {
            "requested": lvl, "observed": observed,
            "reward_mean": round(float(r.mean()), 2), "reward_std": round(float(r.std()), 2),
            "success_rate": round(float(succ), 3),
            "err_xy_mean": round(float(err), 4),
            "torque_pen_mean": round(float(np.mean([e["torque_pen"] for e in eps])), 4),
            "episodes": len(eps),
        }
        print(f"level {observed:>2}: reward {r.mean():6.1f}±{r.std():4.1f}  success {succ:5.1%}  err_xy {err:.3f} m/s")

elif args_cli.mode == "track":
    ctx["fixed_cmd"] = args_cli.fixed_cmd
    ep, traj = run_episode(record_traj=True)
    results["episode"] = {
        "reward": round(ep["reward"], 2), "steps": ep["steps"], "success": ep["success"],
        "err_xy_mean": round(ep["err_xy"], 4), "err_yaw_mean": round(ep["err_yaw"], 4),
    }
    results["trajectory"] = traj
    print(f"track 一局: reward {ep['reward']:.1f}  steps {ep['steps']}  success {ep['success']}  err_xy {ep['err_xy']:.3f} m/s")

else:  # basic
    ctx["num_episodes"] = args_cli.num_episodes
    eps = []
    for _ in range(args_cli.num_episodes):
        ep, _ = run_episode()
        eps.append(ep)
    r = np.array([e["reward"] for e in eps])
    err = np.array([e["err_xy"] for e in eps])
    results["summary"] = {
        "reward_mean": round(float(r.mean()), 2), "reward_std": round(float(r.std()), 2),
        "success_rate": round(float(np.mean([e["success"] for e in eps])), 3),
        "err_xy_mean": round(float(err.mean()), 4),
        "torque_pen_mean": round(float(np.mean([e["torque_pen"] for e in eps])), 4),
        "per_episode": [{k: v for k, v in e.items() if k != "success"} for e in eps],
    }
    print(f"basic {args_cli.num_episodes} 局: reward {r.mean():.1f}±{r.std():.1f}  success {np.mean([e['success'] for e in eps]):.0%}  err_xy {err.mean():.3f} m/s")

# ============ 落盘 ============
out_dir = os.path.join(os.path.dirname(os.path.dirname(resume_path)), "..", "evals_v2")
out_dir = os.path.abspath(out_dir)
os.makedirs(out_dir, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_json = os.path.join(out_dir, f"{args_cli.task}_{args_cli.mode}_{stamp}.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"context": ctx, "results": results}, f, ensure_ascii=False, indent=2)
print(f"[INFO] 结果已保存: {out_json}")

env.close()
simulation_app.close()
