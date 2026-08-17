"""phase_env.py — 带相位时钟的环境子类 + gym 注册。

H1PhaseEnv 在官方 ManagerBasedRLEnv 之上只加一样东西：相位状态 φ(t)。
  - step() 前按 phase_rate（Hz）推进 φ，模 1 回绕；
  - reset 时对重置的 env 随机化初始相位（策略不能靠背初始相位走）；
  - phase_rate 是 buffer，评估时可直接改 —— "穿戴者突然变速"的最小模拟入口。
"""

import torch

from isaaclab.envs import ManagerBasedRLEnv

import gymnasium as gym


class H1PhaseEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv + 相位时钟状态。

    新增状态（评估/观测/奖励函数通过 env.phase / env.phase_rate 访问）：
      phase      (N,) 当前相位 φ ∈ [0,1)，由 phase_rate × dt 积分
      phase_rate (N,) 相位时钟频率（Hz），默认取 cfg.phase_rate
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        # 占位 buffer：Observation/Reward manager 在构造期间会调用 obs/reward 函数
        # 探测输出维度，此时 self.phase 必须已存在（用 cfg 里的 device/num_envs 先建零张量）
        self.phase = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.phase_rate = torch.full(
            (cfg.scene.num_envs,), float(cfg.phase_rate), device=cfg.sim.device
        )
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        # super().__init__() 完成后换真随机初始相位（策略必须学会从任意相位入环）
        self.phase = torch.rand(self.num_envs, device=self.device)
        self.phase_rate = torch.full(
            (self.num_envs,), float(self.cfg.phase_rate), device=self.device
        )
        print(f"[PhaseEnv] 相位时钟 f={self.cfg.phase_rate} Hz, "
              f"duty={self.cfg.phase_duty}, num_envs={self.num_envs}")

    def step(self, action):
        # 先推进相位再走官方 step：本步观测里的 φ 已经是推进后的值
        self.phase = torch.remainder(self.phase + self.phase_rate * self.step_dt, 1.0)
        return super().step(action)

    def _reset_idx(self, env_ids=None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            self.phase = torch.rand(self.num_envs, device=self.device)
        else:
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)


gym.register(
    id="Isaac-Velocity-Flat-H1-Phase-v0",
    entry_point="phase_env:H1PhaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "phase_env_cfg:PhaseH1FlatEnvCfg",
    },
)
