"""phase_mdp.py — 相位时钟观测 + 相位锁定接触调度奖励函数。

两个函数都挂在 PhaseH1FlatEnvCfg 上，由 observation/reward manager 调用：
  - phase_clock_obs(env)          -> (N, 2)  [sin(2πφ), cos(2πφ)]
  - phase_contact_schedule(env, sensor_cfg) -> (N,)   接触-相位调度匹配奖励

相位约定（写死，README 边界声明同步）：
  φ ∈ [0, 1)，左足支撑窗 = φ ∈ [0, duty)，右足支撑窗 = φ ∈ [duty, 1)。
  这是"相位锁定"的期望接触模式，属于人为设定的控制约束（不是数据检出）。
"""

import torch

from isaaclab.managers import SceneEntityCfg

# 接触判定阈值（N），与 Isaac Lab 接触传感器常用阈值一致
CONTACT_FORCE_THRESHOLD = 1.0


def _foot_columns(env, sensor_cfg):
    """把 contact_forces 传感器里的双踝 body_ids 映射到 (左足列, 右足列)。

    只在第一次调用时解析并缓存到 env 上；名字里没有 left/right 时退化为 [0,1]
    顺序假设并打警告（此时"左/右"只是标签，相位锁定步态本身不受影响）。
    """
    if hasattr(env, "_phase_foot_map"):
        return env._phase_foot_map
    sensor = env.scene.sensors[sensor_cfg.name]
    names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
    left = right = None
    for col, name in enumerate(names):
        ln = name.lower()
        if "left" in ln or ln.startswith("l_"):
            left = col
        elif "right" in ln or ln.startswith("r_"):
            right = col
    if left is None or right is None:
        if not getattr(env, "_phase_foot_warned", False):
            print(f"[WARN] phase_contact_schedule: 踝部身体名无法判定左右 {names}，"
                  f"按 [left, right] 索引顺序假设")
            env._phase_foot_warned = True
        left, right = 0, 1
    env._phase_foot_map = (left, right)
    return left, right


def phase_clock_obs(env):
    """观测项：单位圆上的相位时钟 [sin(2πφ), cos(2πφ)]，形状 (N, 2)。"""
    phi = env.phase  # (N,)
    return torch.stack(
        [torch.sin(2.0 * torch.pi * phi), torch.cos(2.0 * torch.pi * phi)], dim=-1
    )


def phase_contact_schedule(env, sensor_cfg: SceneEntityCfg):
    """接触调度奖励：接触状态与相位期望支撑窗的匹配度。

    每足：支撑窗内接触 +1；摆动窗内接触 −1（早/晚着地惩罚）；不接触 0。
    返回值范围 [−2, +2]（两足之和），由 RewardTerm 的 weight 缩放。

    设计动机：H1 默认奖励（track_lin_vel + feet_air_time）只能诱导"走得快、脚离地"，
    不约束"何时着地"。本项把着地事件锁到相位时钟上 —— 这正是外骨骼相位控制的最小核心：
    策略必须学会"跟随外部给出的节律"，而不是自定节律。
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    # (N, 2, 3) 当前步双踝净接触力（body_ids 已由 manager 解析）
    net_forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact = (torch.norm(net_forces, dim=-1) > CONTACT_FORCE_THRESHOLD).float()  # (N, 2)

    left_col, right_col = _foot_columns(env, sensor_cfg)
    phi = env.phase  # (N,)
    duty = float(env.cfg.phase_duty)

    left_stance = (phi < duty).float()   # 左足期望支撑窗
    right_stance = (phi >= duty).float()  # 右足期望支撑窗

    # contact·(stance − swing)：支撑窗内接触 +1，摆动窗内接触 −1
    reward = contact[:, left_col] * (left_stance - (1.0 - left_stance)) \
        + contact[:, right_col] * (right_stance - (1.0 - right_stance))
    return reward


def contact_schedule_match(env, sensor_cfg):
    """评估用：接触状态与相位期望窗的二值匹配率（每足 1/0，均值 0..1）。

    与训练 reward 同定义、不同输出口径：reward 是 [−2,+2] 连续值，
    这里输出两足平均的命中率，用于 eval 的"相位对齐分数"。
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact = (torch.norm(net_forces, dim=-1) > CONTACT_FORCE_THRESHOLD).float()
    left_col, right_col = _foot_columns(env, sensor_cfg)
    phi = env.phase
    duty = float(env.cfg.phase_duty)
    expected_left = (phi < duty).float()
    expected_right = (phi >= duty).float()
    match_left = 1.0 - torch.abs(contact[:, left_col] - expected_left)
    match_right = 1.0 - torch.abs(contact[:, right_col] - expected_right)
    return 0.5 * (match_left + match_right)
