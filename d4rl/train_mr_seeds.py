# -*- coding: utf-8 -*-
"""train_mr_seeds.py — medium-replay 边界验证 3-seed 补跑 seed N（K3 决策 C，2026-08-15）

与 train_offline_mr.py 唯一差异 = 种子 + 独立输出名（种子模式同 CQL 3-seed）。
配置一致：CQL/CalQL，alpha 固定（alpha_learning_rate=0.0），offline 400k → online 50k
（每 5k 步 10 局确定性评估）。
输出 {algo}_mr_seed{N}.d3 + d3rlpy_logs/mr_seed{N}_{algo}_{offline,finetune}*。

用法：python train_mr_seeds.py --seed 1 --algo CQL
"""
import argparse
import os

os.environ["PYTHONPATH"] = ""  # 硬覆盖：防 terminal 注入的 hermes venv PYTHONPATH 污染 torch

import gymnasium as gym
import h5py
import numpy as np
import torch
from d3rlpy.algos import CQL, CalQL, CQLConfig, CalQLConfig
from d3rlpy.dataset import MDPDataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "halfcheetah_medium_replay-v2.hdf5")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--algo", required=True, choices=["CQL", "CalQL"])
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

N_OFFLINE, N_FINETUNE = 400_000, 50_000
NAME = f"mr_seed{args.seed}_{args.algo}"


def load_dataset():
    with h5py.File(DATA, "r") as f:
        return MDPDataset(
            observations=f["observations"][:], actions=f["actions"][:],
            rewards=f["rewards"][:], terminals=f["terminals"][:], timeouts=f["timeouts"][:],
        )


def main():
    cfg_cls = CQLConfig if args.algo == "CQL" else CalQLConfig
    algo_cls = CQL if args.algo == "CQL" else CalQL
    print(f"===== {NAME}（medium-replay）offline {N_OFFLINE} 步 =====", flush=True)
    cfg = cfg_cls(alpha_learning_rate=0.0)  # alpha 固定（与 train_offline_cql.py 同修复）
    model = algo_cls(cfg, device="cuda:0", enable_ddp=False)
    model.build_with_dataset(load_dataset())
    model.fit(load_dataset(), n_steps=N_OFFLINE, n_steps_per_epoch=max(N_OFFLINE // 20, 1),
              experiment_name=f"{NAME}_offline", show_progress=False)
    model.save(os.path.join(HERE, f"{args.algo}_mr_seed{args.seed}.d3"))
    print(f"[Phase 1] {NAME} offline 完成，已保存 {args.algo}_mr_seed{args.seed}.d3", flush=True)

    print(f"===== {NAME} offline→online 微调 {N_FINETUNE} 步 =====", flush=True)
    env = gym.make("HalfCheetah-v4")
    eval_env = gym.make("HalfCheetah-v4")
    model.fit_online(
        env, n_steps=N_FINETUNE, n_steps_per_epoch=max(N_FINETUNE // 10, 1),
        eval_env=eval_env, eval_n_trials=10, eval_epsilon=0.0,
        random_steps=1000, update_start_step=1000,
        experiment_name=f"{NAME}_finetune", show_progress=False,
    )
    env.close(); eval_env.close()
    print(f"[DONE] {NAME} 微调完成，曲线见 d3rlpy_logs/{NAME}_finetune_*", flush=True)


if __name__ == "__main__":
    main()
