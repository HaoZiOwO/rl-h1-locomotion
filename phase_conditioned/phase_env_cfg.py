"""phase_env_cfg.py — PhaseH1FlatEnvCfg：官方 H1 flat 环境 + 相位条件化改造。

与官方 H1FlatEnvCfg 的差异（唯一变量，README 边界声明同步）：
  1. obs 新增 phase_clock：       [sin(2πφ), cos(2πφ)]（+2 维，69 -> 71）
  2. reward 新增 phase_contact：  接触状态与相位期望支撑窗的匹配（weight=0.5）
其余（地形/事件/命令/终止/PPO 配置）与官方 flat 完全一致 —— 对比的公平性来源。
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import (
    H1FlatEnvCfg,
)

import phase_mdp


@configclass
class PhaseH1FlatEnvCfg(H1FlatEnvCfg):
    """H1 平地 + 相位时钟观测 + 相位锁定接触调度奖励。"""

    # 相位时钟参数（plain field，环境 __init__ 读取）
    phase_rate: float = 1.4       # 步态周期时钟频率（Hz），按冲刺计划 ~1.4 Hz
    phase_duty: float = 0.5       # 单足支撑占空比（左足 [0,0.5)，右足 [0.5,1)）
    phase_contact_weight: float = 0.5  # 接触调度奖励权重

    def __post_init__(self):
        super().__post_init__()

        # 1. 观测：追加相位时钟（追加在 policy 组末尾，训练/评估同 cfg 保证顺序一致）
        self.observations.policy.phase_clock = ObsTerm(func=phase_mdp.phase_clock_obs)

        # 2. 奖励：相位锁定的接触调度项
        self.rewards.phase_contact = RewTerm(
            func=phase_mdp.phase_contact_schedule,
            weight=self.phase_contact_weight,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link")},
        )
