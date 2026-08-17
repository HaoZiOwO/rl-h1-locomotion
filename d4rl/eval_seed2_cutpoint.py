# -*- coding: utf-8 -*-
"""eval_seed2_cutpoint.py — seed2 离线快照的 0k 切分点评估（只评 0 步，10 局确定性）。"""
import json
import os

os.environ["PYTHONPATH"] = ""

import gymnasium as gym
import h5py
from d3rlpy import load_learnable
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import EnvironmentEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "d3rlpy_logs/rlpd_seed2_offline_20260816124006/model_300000.d3")
OUT = os.path.join(HERE, "eval_rlpd_cutpoints_seed2.json")

with h5py.File(os.path.join(HERE, "halfcheetah_medium-v2.hdf5"), "r") as f:
    dataset = MDPDataset(
        observations=f["observations"][:], actions=f["actions"][:],
        rewards=f["rewards"][:], terminals=f["terminals"][:], timeouts=f["timeouts"][:],
    )

model = load_learnable(SNAP, device="cuda:0")
print("[seed2] 快照已加载", flush=True)
eval_env = gym.make("HalfCheetah-v4")
evaluator = EnvironmentEvaluator(env=eval_env, n_trials=10, epsilon=0.0)
score_0k = float(evaluator(model, dataset))
eval_env.close()
print(f"[seed2] 0k 评估: {score_0k:.1f}", flush=True)

with open(OUT, "w") as f:
    json.dump({"snapshot": SNAP, "0k_raw": score_0k,
               "0k_normalized": round(100.0 * (score_0k + 280.178946) / (12135.0 + 280.178946), 2)}, f, indent=2)
print(f"已保存 {OUT}")
