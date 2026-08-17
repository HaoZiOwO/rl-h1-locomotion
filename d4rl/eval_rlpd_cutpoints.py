# -*- coding: utf-8 -*-
"""eval_rlpd_cutpoints.py — RLPD 离线/在线贡献切分评估（K3 决策 B，2026-08-15）

对 rlpd（10 critics + LN + UTD=10）与 sac_ctl（2 critics 无 LN UTD=1）的
offline 300k 快照做 0k 评估，并继续 fit_online 2000 环境步
（n_steps_per_epoch=1000 → 1k/2k 评估点），评估器与既有 5k..50k 曲线同机制
（fit_online 内置 EnvironmentEvaluator，n_trials=10，epsilon=0）。

快照格式：d3rlpy `save()` 的自包含 pickle（{'torch','config','version'}），
须用 d3rlpy.load_learnable 加载（load_model 只认 torch.save 格式，会报
Invalid magic number——2026-08-15 踩坑记录）。

输出：eval_rlpd_cutpoints.json + 控制台汇总。
"""
import glob
import json
import os

os.environ["PYTHONPATH"] = ""  # 硬覆盖：防 terminal 注入的 hermes venv PYTHONPATH 污染 torch

import gymnasium as gym
import h5py
from d3rlpy import load_learnable
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import EnvironmentEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "halfcheetah_medium-v2.hdf5")


def load_dataset():
    with h5py.File(DATA, "r") as f:
        return MDPDataset(
            observations=f["observations"][:], actions=f["actions"][:],
            rewards=f["rewards"][:], terminals=f["terminals"][:], timeouts=f["timeouts"][:],
        )

SNAPSHOTS = {
    "rlpd": os.path.join(HERE, "d3rlpy_logs/rlpd_offline_20260814235317/model_300000.d3"),
    "sac_ctl": os.path.join(HERE, "d3rlpy_logs/sac_ctl_offline_20260815110132/model_300000.d3"),
}
CONFIGS = {"rlpd": 10, "sac_ctl": 1}  # UTD


def run_one(name, dataset):
    utd = CONFIGS[name]
    model = load_learnable(SNAPSHOTS[name], device="cuda:0")
    print(f"[{name}] 快照已加载：{SNAPSHOTS[name]}", flush=True)

    # 0k 评估（与 fit_online 同评估器机制；dataset 仅用于 action scaler）
    eval_env = gym.make("HalfCheetah-v4")
    evaluator = EnvironmentEvaluator(env=eval_env, n_trials=10, epsilon=0.0)
    score_0k = float(evaluator(model, dataset))
    print(f"[{name}] 0k 评估: {score_0k:.1f}", flush=True)
    eval_env.close()

    # 1k/2k：继续 fit_online 2000 步（协议与主实验一致：random 1000 / update_start 1000）
    env = gym.make("HalfCheetah-v4")
    eval_env2 = gym.make("HalfCheetah-v4")
    print(f"[{name}] 继续 online 2000 步（UTD={utd}，评估点 1k/2k）...", flush=True)
    model.fit_online(
        env, n_steps=2000, n_steps_per_epoch=1000, n_updates=utd,
        eval_env=eval_env2, eval_n_trials=10, eval_epsilon=0.0,
        random_steps=1000, update_start_step=1000,
        experiment_name=f"{name}_cutpoint", show_progress=False,
    )
    env.close(); eval_env2.close()

    # 读回 1k/2k 评估值
    evals = {}
    for csv in glob.glob(os.path.join(HERE, f"d3rlpy_logs/{name}_cutpoint_*/evaluation.csv")):
        with open(csv) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3 and parts[1] in ("1000", "2000"):
                    evals[int(parts[1])] = float(parts[2])
    print(f"[{name}] 切分点：0k={score_0k:.1f}, 1k={evals.get(1000, 'NA')}, 2k={evals.get(2000, 'NA')}", flush=True)
    return {"0k": score_0k, "1k": evals.get(1000), "2k": evals.get(2000)}


def main():
    dataset = load_dataset()
    results = {}
    for name in ("rlpd", "sac_ctl"):
        results[name] = run_one(name, dataset)
    out = os.path.join(HERE, "eval_rlpd_cutpoints.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[ALL DONE] 已保存 {out}\n{json.dumps(results, ensure_ascii=False, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
