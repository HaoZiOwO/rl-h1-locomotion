"""
export_h1.py — headless 导出训练好的策略为 JIT + ONNX（部署格式）
用法：
    python export_h1.py --task flat  --checkpoint logs/rsl_rl/h1_flat/2026-08-06_16-30-59/model_999.pt
    python export_h1.py --task rough --checkpoint logs/rsl_rl/h1_rough/2026-08-06_16-45-01/model_2999.pt
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--checkpoint", type=str, required=True)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
import importlib.metadata as metadata

import gymnasium as gym
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

env_cfg = EnvCfg()
env_cfg.scene.num_envs = 1
agent_cfg = RunnerCfg()
installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

env = gym.make(task_name, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

resume_path = retrieve_file_path(args_cli.checkpoint)
print(f"[INFO] 加载检查点: {resume_path}")
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
runner.load(resume_path)

export_dir = os.path.join(os.path.dirname(resume_path), "exported")
runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
print(f"[INFO] 已导出到: {export_dir}")

# ---- ONNX 数值一致性校验：同一输入下 JIT vs ONNX Runtime 输出对比 ----
print("[INFO] ONNX 数值一致性校验（JIT vs ONNX Runtime）...")
import onnxruntime as ort
import torch as _torch
ort_sess = ort.InferenceSession(os.path.join(export_dir, "policy.onnx"), providers=["CPUExecutionProvider"])
obs_dim = int(ort_sess.get_inputs()[0].shape[1])
jit_model = _torch.jit.load(os.path.join(export_dir, "policy.pt"), map_location="cpu")
rng = _torch.Generator().manual_seed(0)
x = _torch.randn(8, obs_dim, generator=rng, dtype=_torch.float32)  # 8 个随机观测
with _torch.no_grad():
    y_jit = jit_model(x).cpu().numpy()
y_ort = ort_sess.run(None, {ort_sess.get_inputs()[0].name: x.numpy()})[0]
mse = float(((y_jit - y_ort) ** 2).mean())
max_err = float((abs(y_jit - y_ort)).max())
print(f"[INFO] 一致性校验: obs_dim={obs_dim}  MSE={mse:.2e}  max_err={max_err:.2e}")
assert mse < 1e-5, f"ONNX 与 PyTorch 输出不一致（MSE={mse:.2e}）——部署前必须排查算子支持问题"
print("[INFO] 校验通过：ONNX 与 PyTorch 输出一致，可安全部署")

env.close()
simulation_app.close()
