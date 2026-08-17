"""probe_env.py — 一次性探测：obs 维度 + 踝部身体名（验证左右脚映射）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import phase_env  # noqa: F401
from phase_env_cfg import PhaseH1FlatEnvCfg

from isaaclab.managers import SceneEntityCfg

env_cfg = PhaseH1FlatEnvCfg()
env_cfg.scene.num_envs = 1
env = gym.make("Isaac-Velocity-Flat-H1-Phase-v0", cfg=env_cfg, render_mode=None)

print(f"[PROBE] obs_dim(policy) = {env.unwrapped.observation_manager.group_obs_dim['policy']}")
print(f"[PROBE] action_dim = {env.unwrapped.action_manager.total_action_dim}")

foot_cfg = SceneEntityCfg("contact_forces", body_names=".*ankle_link")
foot_cfg.resolve(env.unwrapped.scene)
sensor = env.unwrapped.scene.sensors[foot_cfg.name]
print(f"[PROBE] 踝部 body_ids = {foot_cfg.body_ids}")
print(f"[PROBE] 踝部身体名 = {[sensor.body_names[i] for i in foot_cfg.body_ids]}")

import phase_mdp

l, r = phase_mdp._foot_columns(env.unwrapped, foot_cfg)
print(f"[PROBE] 左右脚映射: left_col={l}, right_col={r}")

robot = env.unwrapped.scene["robot"]
torso = [i for i, n in enumerate(robot.data.body_names) if "torso" in n]
print(f"[PROBE] torso body_ids = {torso} ({[robot.data.body_names[i] for i in torso]})")

env.close()
simulation_app.close()
print("[PROBE] done")
