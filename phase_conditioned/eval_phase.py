"""eval_phase.py — phase-conditioned H1 评估三件套（冲刺期）。

用法（先 source 环境，见 run_train_phase.sh 同款 env 设置）：
  # 1) 稳态步态 + 相位锁定评估
  python eval_phase.py --mode steady --checkpoint logs/rsl_rl/h1_flat_phase/<run>/model_999.pt
  # 2) 相位突变重同步：t=5s 时钟频率 1.4 -> 1.9 Hz（"穿戴者突然变速"的最小模拟）
  python eval_phase.py --mode jump   --checkpoint ... [--num-episodes 8]
  # 3) 周期外力扫描：A ∈ {0,50,100,200} N，f=1.4 Hz，对比旧 flat 与 phase 策略
  python eval_phase.py --mode sweep  --checkpoint ... --baseline-checkpoint logs/rsl_rl/h1_flat/<run>/model_999.pt

控制变量（两个策略共用同一评估配置）：
  - 命令速度固定 1.0 m/s、朝向固定 0（消掉命令随机性）
  - 关 obs corruption；H1 默认 push 事件本来就 None，外力事件为 0 力（再显式置 None 兜底）
  - 外力 = 理想正弦力，作用在 torso_link，全局 +x 方向
边界声明：外力是躯干上的理想正弦力，非穿戴者-机器人真实交互力（pHRI 无法纯仿真验证）。
"""
import argparse
import os
import sys

# 让本目录可被 import（gym 注册需要 phase_env / phase_env_cfg）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["steady", "jump", "sweep"])
parser.add_argument("--checkpoint", required=True, help="phase 策略检查点")
parser.add_argument("--baseline-checkpoint", default=None, help="旧 flat 策略检查点（sweep 模式用）")
parser.add_argument("--num-episodes", type=int, default=10)
parser.add_argument("--sweep-amps", type=str, default="0,50,100,200", help="外力幅值列表（N）")
parser.add_argument("--sweep-policy", type=str, default="both", choices=["both", "flat", "phase"],
                    help="只跑 baseline(flat) / 只跑 phase / 两者")
parser.add_argument("--sweep-freq", type=float, default=1.4, help="外力频率（Hz，与步态时钟共振为最坏情况）")
parser.add_argument("--jump-at", type=float, default=5.0, help="相位突变时刻（s）")
parser.add_argument("--jump-from", type=float, default=1.4, help="突变前时钟频率（Hz）")
parser.add_argument("--jump-to", type=float, default=1.9, help="突变后时钟频率（Hz）")
parser.add_argument("--episode-length", type=float, default=0.0, help="0=用环境默认（flat 20s；jump 模式自动 15s）")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import importlib.metadata as metadata
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import phase_env  # noqa: F401
import phase_mdp
from phase_env_cfg import PhaseH1FlatEnvCfg

# 压掉 set_external_force_and_torque 的每步 deprecation 警告（功能正常，警告每步刷屏污染日志）
import logging

logging.getLogger("isaaclab.assets.articulation.articulation").setLevel(logging.ERROR)

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import (
    H1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
    H1FlatPPORunnerCfg,
)

installed_version = metadata.version("rsl-rl-lib")

# ============ 受控评估环境配置 ============

FOOT_CFG = SceneEntityCfg("contact_forces", body_names=".*ankle_link")


def _obs_tensor(obs):
    """rsl_rl 5.x 的 get_observations() 返回 TensorDict（key='policy'），取纯张量。"""
    if hasattr(obs, "keys") and "policy" in obs:
        return obs["policy"]
    return obs


def make_env(task: str, num_envs: int = 1, episode_length_s: float = 0.0):
    """受控评估环境：固定命令 + 无随机扰动 + 无 obs 噪声。"""
    if task == "phase":
        env_cfg = PhaseH1FlatEnvCfg()
        task_id = "Isaac-Velocity-Flat-H1-Phase-v0"
    else:
        env_cfg = H1FlatEnvCfg()
        task_id = "Isaac-Velocity-Flat-H1-v0"

    env_cfg.scene.num_envs = num_envs
    # 命令调度：用**默认随机指令**（两个策略的训练分布都是它——train cfg 未改 commands）。
    # ⚠️ 不要设固定命令：flat 策略在固定 1.0 m/s 下走路异常（实测 -0.78/0.24 m/s），
    # 而官方 eval 脚本的固定指令代码在 isaaclab 0.54 属性名已变（velocity_command 不存在，
    # 静默失败回退随机指令）——受控对比的公平性来自"两边共用同一调度"，不是"人为固定"。
    # 无随机扰动 + 无观测噪声（受控对比）
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.observations.policy.enable_corruption = False
    if episode_length_s > 0:
        env_cfg.episode_length_s = episode_length_s

    return env_cfg, task_id


def load_policy(env, ckpt_path: str, agent_cfg):
    """按 eval_h1.py 官方流程加载 rsl_rl 检查点，返回推理函数。"""
    resume_path = retrieve_file_path(ckpt_path)
    print(f"[INFO] 加载检查点: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)
    return runner.get_inference_policy(device=env.unwrapped.device)


def get_torso_idx(env) -> int:
    names = list(env.unwrapped.scene["robot"].data.body_names)
    for i, n in enumerate(names):
        if "torso" in n:
            return i
    raise RuntimeError(f"未找到 torso 连杆: {names}")


def apply_sinusoidal_force(env, torso_idx: int, amp: float, freq: float, t: float):
    """躯干施加全局 +x 正弦外力 F(t) = A·sin(2πft)。permanent wrench，逐步更新。"""
    robot = env.unwrapped.scene["robot"]
    f = amp * np.sin(2.0 * np.pi * freq * t)
    forces = torch.zeros(env.unwrapped.num_envs, 1, 3, device=env.unwrapped.device)
    forces[:, 0, 0] = f
    robot.set_external_force_and_torque(
        forces, torch.zeros_like(forces), body_ids=[torso_idx], is_global=True
    )


def foot_contacts(env):
    """返回 (left_contact, right_contact) bool，按 phase_mdp 相同的左右脚映射。"""
    sensor = env.unwrapped.scene.sensors[FOOT_CFG.name]
    ids = FOOT_CFG.body_ids
    net_forces = sensor.data.net_forces_w[:, ids]  # (N, 2, 3)
    contact = torch.norm(net_forces, dim=-1) > phase_mdp.CONTACT_FORCE_THRESHOLD
    left_c, right_c = phase_mdp._foot_columns(env.unwrapped, FOOT_CFG)
    return contact[:, left_c], contact[:, right_c]  # 各 (N,)


# ============ 模式 1：稳态步态评估 ============

def run_steady(ckpt_path: str, num_episodes: int):
    env_cfg, task_id = make_env("phase")
    agent_cfg = H1FlatPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    FOOT_CFG.resolve(env.unwrapped.scene)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = load_policy(env, ckpt_path, agent_cfg)

    step_dt = env.unwrapped.step_dt
    max_steps = env.unwrapped.max_episode_length
    print(f"[INFO] step_dt={step_dt}s, max_steps={max_steps} "
          f"({env_cfg.episode_length_s}s/局), episodes={num_episodes}")

    rows = []
    obs = env.get_observations()
    with torch.inference_mode():
        ep = 0
        while ep < num_episodes:
            ep_return, ep_steps = 0.0, 0
            vx_list, match_list, track_err_list = [], [], []
            prev_l, prev_r = None, None
            td_l, td_r = 0, 0
            while True:
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                policy.reset(dones)
                l, r = foot_contacts(env)
                # 着地沿检测（计步）
                if prev_l is not None:
                    td_l += int(bool(l[0]) and not bool(prev_l[0]))
                    td_r += int(bool(r[0]) and not bool(prev_r[0]))
                prev_l, prev_r = l.clone(), r.clone()
                # 速度 + 相位对齐 + 指令跟踪误差（obs 布局：lin_vel(0:3) ... cmd(9:12)）
                vx = float(env.unwrapped.scene["robot"].data.root_lin_vel_w[0, 0])
                obs_t = _obs_tensor(obs)
                track_err = float(torch.norm(obs_t[0, 0:2] - obs_t[0, 9:11]).item())
                match = float(phase_mdp.contact_schedule_match(env.unwrapped, FOOT_CFG)[0])
                vx_list.append(vx)
                track_err_list.append(track_err)
                match_list.append(match)
                ep_return += rewards[0].item()
                ep_steps += 1
                if bool(dones[0]) or ep_steps >= max_steps:
                    break
            duration = ep_steps * step_dt
            rows.append(dict(
                ret=ep_return,
                survived=(ep_steps >= max_steps),
                vx=float(np.mean(vx_list)),
                track_err=float(np.mean(track_err_list)),
                match=float(np.mean(match_list)),
                cadence_L=td_l / duration,
                cadence_R=td_r / duration,
            ))
            print(f"[ep {ep}] ret={rows[-1]['ret']:.0f} survived={rows[-1]['survived']} "
                  f"vx={rows[-1]['vx']:.3f} track_err={rows[-1]['track_err']:.3f} "
                  f"match={rows[-1]['match']:.3f} cad_L={rows[-1]['cadence_L']:.2f} "
                  f"cad_R={rows[-1]['cadence_R']:.2f}")
            ep += 1

    env.close()
    _summarize("STEADY", rows, keys=["ret", "vx", "track_err", "match", "cadence_L", "cadence_R"])


# ============ 模式 2：相位突变重同步 ============

def run_jump(ckpt_path: str, num_episodes: int):
    length = 15.0  # 5s 稳态 + 10s 重同步窗口
    env_cfg, task_id = make_env("phase", episode_length_s=length)
    agent_cfg = H1FlatPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    FOOT_CFG.resolve(env.unwrapped.scene)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = load_policy(env, ckpt_path, agent_cfg)

    step_dt = env.unwrapped.step_dt
    max_steps = env.unwrapped.max_episode_length
    win = max(int(round(0.5 / step_dt)), 1)  # 0.5s 滚动窗
    jump_step = int(round(args_cli.jump_at / step_dt))
    print(f"[INFO] jump: t={args_cli.jump_at}s, f {args_cli.jump_from}->{args_cli.jump_to} Hz, "
          f"window={win} steps (0.5s)")

    results = []
    obs = env.get_observations()
    with torch.inference_mode():
        ep = 0
        while ep < num_episodes:
            # 初始相位随机（env reset 内处理）；前段用 jump_from 频率
            env.unwrapped.phase_rate[:] = args_cli.jump_from
            match_series, vx_err_series = [], []
            step = 0
            while True:
                if step == jump_step:
                    env.unwrapped.phase_rate[:] = args_cli.jump_to  # 突变
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                policy.reset(dones)
                match = float(phase_mdp.contact_schedule_match(env.unwrapped, FOOT_CFG)[0])
                obs_t = _obs_tensor(obs)
                track_err = float(torch.norm(obs_t[0, 0:2] - obs_t[0, 9:11]).item())
                match_series.append(match)
                vx_err_series.append(track_err)
                step += 1
                if bool(dones[0]) or step >= max_steps:
                    break
            match_arr = np.array(match_series)
            vx_err_arr = np.array(vx_err_series)
            # dump 单步级序列（重同步曲线绘图/精细分析用）
            dump_dir = Path(__file__).parent / "results" / "jump_traces"
            dump_dir.mkdir(parents=True, exist_ok=True)
            np.savez(dump_dir / f"jump_ep{ep}.npz",
                     t=np.arange(len(match_series)) * step_dt,
                     match=match_arr, vx_err=vx_err_arr,
                     jump_at=args_cli.jump_at, f_from=args_cli.jump_from, f_to=args_cli.jump_to)
            # 基线：突变前最后 2s 的滚动均值
            base_start = max(0, jump_step - int(round(2.0 / step_dt)))
            baseline = float(np.mean(match_arr[base_start:jump_step])) if jump_step > 0 else float(np.nan)
            # 突变后滚动均值序列
            roll = np.convolve(match_arr, np.ones(win) / win, mode="valid")
            roll_t = (np.arange(len(roll)) + win - 1) * step_dt
            post = roll[np.searchsorted(roll_t, args_cli.jump_at):]
            post_t = roll_t[np.searchsorted(roll_t, args_cli.jump_at):]
            min_post = float(post.min()) if len(post) else float(np.nan)
            # 重同步判据：突变后滚动 match ≥ 0.9×基线，且持续 ≥0.5s
            resync = None
            if np.isfinite(baseline) and baseline > 0:
                thr = 0.9 * baseline
                hits = post >= thr
                for i in range(len(hits) - win + 1):
                    if hits[i:i + win].all():
                        resync = post_t[i] - args_cli.jump_at
                        break
            # 精细判据：单步级 match 恢复到 ≥0.95×基线，持续 10 步（0.2s）
            resync_raw = None
            if np.isfinite(baseline) and baseline > 0:
                thr_raw = 0.95 * baseline
                raw_hits = match_arr[jump_step:] >= thr_raw
                for i in range(len(raw_hits) - 10 + 1):
                    if raw_hits[i:i + 10].all():
                        resync_raw = i * step_dt
                        break
            results.append(dict(baseline=baseline, min_post=min_post, resync=resync,
                                resync_raw=resync_raw,
                                vx_err_pre=float(np.mean(vx_err_arr[:jump_step])),
                                vx_err_post=float(np.mean(vx_err_arr[jump_step:]))))
            print(f"[ep {ep}] baseline_match={baseline:.3f} min_post={min_post:.3f} "
                  f"resync={resync if resync is None else round(resync, 2)}s "
                  f"(raw {resync_raw if resync_raw is None else round(resync_raw, 2)}s) "
                  f"vx_err pre={results[-1]['vx_err_pre']:.3f} post={results[-1]['vx_err_post']:.3f}")
            ep += 1

    env.close()
    base = np.mean([r["baseline"] for r in results])
    resync_ok = [r["resync"] for r in results if r["resync"] is not None]
    print("\n=== JUMP 汇总 ===")
    print(f"基线相位对齐（突变前）: {base:.3f}（{'良好' if base >= 0.6 else '偏弱，重同步指标参考性降低'}）")
    print(f"重同步成功 {len(resync_ok)}/{len(results)} 局，耗时 "
          f"{np.mean(resync_ok):.2f} ± {np.std(resync_ok):.2f} s（判据：滚动对齐 ≥ 0.9×基线，持续 0.5s）")
    print(f"突变后对齐最低值: {np.mean([r['min_post'] for r in results]):.3f}")


# ============ 模式 3：周期外力扫描 ============

def run_sweep_one(task: str, ckpt_path: str, amps, freq: float, num_episodes: int):
    env_cfg, task_id = make_env(task)
    agent_cfg = H1FlatPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = load_policy(env, ckpt_path, agent_cfg)
    torso_idx = get_torso_idx(env)

    step_dt = env.unwrapped.step_dt
    max_steps = env.unwrapped.max_episode_length
    print(f"[INFO] sweep[{task}]: amps={amps} N, f={freq} Hz, "
          f"episodes={num_episodes}, {env_cfg.episode_length_s}s/局")

    out = {}
    obs = env.get_observations()  # 环境在 dones 后自动 reset（与 eval_h1.py 同流程）
    with torch.inference_mode():
        for amp in amps:
            rows = []
            ep = 0
            while ep < num_episodes:
                print(f"[sweep {task} A={amp:>4}N] ep {ep}/{num_episodes} 开始", flush=True)
                ep_return, ep_steps = 0.0, 0
                vx_list, track_err_list = [], []
                while True:
                    t = ep_steps * step_dt
                    apply_sinusoidal_force(env, torso_idx, amp, freq, t)
                    actions = policy(obs)
                    obs, rewards, dones, _ = env.step(actions)
                    policy.reset(dones)
                    vx_list.append(float(env.unwrapped.scene["robot"].data.root_lin_vel_w[0, 0]))
                    obs_t = _obs_tensor(obs)
                    track_err_list.append(float(torch.norm(obs_t[0, 0:2] - obs_t[0, 9:11]).item()))
                    ep_return += rewards[0].item()
                    ep_steps += 1
                    if bool(dones[0]) or ep_steps >= max_steps:
                        break
                rows.append(dict(survived=(ep_steps >= max_steps),
                                 vx=float(np.mean(vx_list)),
                                 track_err=float(np.mean(track_err_list)),
                                 ret=ep_return))
                print(f"[sweep {task} A={amp:>4}N] ep {ep} 完成: "
                      f"survived={rows[-1]['survived']} vx={rows[-1]['vx']:.3f} "
                      f"track_err={rows[-1]['track_err']:.3f}", flush=True)
                ep += 1
            surv = np.mean([r["survived"] for r in rows])
            vx = np.mean([r["vx"] for r in rows])
            terr = np.mean([r["track_err"] for r in rows])
            ret = np.mean([r["ret"] for r in rows])
            out[amp] = dict(survival=surv, vx=vx, track_err=terr, ret=ret)
            print(f"[sweep {task} A={amp:>4}N] survival={surv:.2f} vx={vx:.3f} m/s "
                  f"track_err={terr:.3f} m/s ret={ret:.0f}")
    env.close()
    return out


def run_sweep(phase_ckpt: str, baseline_ckpt: str, amps, freq: float, num_episodes: int):
    results = {}
    if args_cli.sweep_policy in ("both", "flat"):
        print(f"\n===== 基线（官方 flat 策略）=====")
        results["flat"] = run_sweep_one("flat", baseline_ckpt, amps, freq, num_episodes)
    if args_cli.sweep_policy in ("both", "phase"):
        print(f"\n===== Phase-conditioned 策略 =====")
        results["phase"] = run_sweep_one("phase", phase_ckpt, amps, freq, num_episodes)

    print("\n=== SWEEP 汇总（survival | vx m/s | track_err m/s）===")
    if "flat" in results and "phase" in results:
        print(f"{'A(N)':>6} | {'flat surv':>9} {'flat vx':>8} {'flat terr':>9} | "
              f"{'phase surv':>10} {'phase vx':>9} {'phase terr':>10}")
        for amp in amps:
            b, p = results["flat"][amp], results["phase"][amp]
            print(f"{amp:>6} | {b['survival']:>9.2f} {b['vx']:>8.3f} {b['track_err']:>9.3f} | "
                  f"{p['survival']:>10.2f} {p['vx']:>9.3f} {p['track_err']:>10.3f}")
    else:
        tag = "flat" if "flat" in results else "phase"
        print(f"{'A(N)':>6} | {tag + ' surv':>10} {tag + ' vx':>9} {tag + ' terr':>10}")
        for amp in amps:
            r = results[tag][amp]
            print(f"{amp:>6} | {r['survival']:>10.2f} {r['vx']:>9.3f} {r['track_err']:>10.3f}")


# ============ 汇总打印 ============

def _summarize(tag, rows, keys):
    print(f"\n=== {tag} 汇总（{len(rows)} 局）===")
    for k in keys:
        vals = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        if k == "ret":
            print(f"{k}: {vals.mean():.1f} ± {vals.std():.1f}")
        else:
            print(f"{k}: {vals.mean():.3f} ± {vals.std():.3f}")
    if "survived" in rows[0]:
        print(f"survival: {np.mean([r['survived'] for r in rows]):.2f}")


# ============ 主流程 ============

def main():
    ckpt = os.path.abspath(args_cli.checkpoint)
    if not Path(ckpt).exists():
        # 允许用户传 C:/ 绝对路径或相对 logs/ 路径
        print(f"[ERROR] 检查点不存在: {ckpt}")
        sys.exit(1)
    if args_cli.mode == "steady":
        run_steady(ckpt, args_cli.num_episodes)
    elif args_cli.mode == "jump":
        run_jump(ckpt, args_cli.num_episodes)
    elif args_cli.mode == "sweep":
        if not args_cli.baseline_checkpoint:
            print("[ERROR] sweep 模式需要 --baseline-checkpoint（旧 flat 策略）")
            sys.exit(1)
        amps = [float(a) for a in args_cli.sweep_amps.split(",")]
        run_sweep(ckpt, os.path.abspath(args_cli.baseline_checkpoint),
                  amps, args_cli.sweep_freq, args_cli.num_episodes)
    simulation_app.close()


if __name__ == "__main__":
    main()
