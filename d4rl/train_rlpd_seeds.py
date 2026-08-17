# -*- coding: utf-8 -*-
"""train_rlpd_seeds.py — RLPD 3-seed 补跑 seed N（K3 决策 C，2026-08-15）

与 train_rlpd.py 主实验唯一差异 = 种子（torch/numpy manual_seed，与 CQL 3-seed 同模式）。
配置完全一致：10 critics + LayerNorm + UTD=10，offline 300k → online 50k（每 5k 步 10 局评估）。
输出 rlpd_seed{N}_halfcheetah_medium.d3 + d3rlpy_logs/rlpd_seed{N}_{offline,finetune}*。

用法：python train_rlpd_seeds.py --seed 2
"""
import argparse
import os

os.environ["PYTHONPATH"] = ""  # 硬覆盖：防 terminal 注入的 hermes venv PYTHONPATH 污染 torch

import gymnasium as gym
import h5py
import numpy as np
import torch
from d3rlpy.algos import SAC, SACConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "halfcheetah_medium-v2.hdf5")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

N_CRITICS, USE_LN, UTD = 10, True, 10
OFFLINE_STEPS, FINETUNE_STEPS = 300_000, 50_000
NAME = f"rlpd_seed{args.seed}"
MODEL_PATH = os.path.join(HERE, f"{NAME}_halfcheetah_medium.d3")


def load_dataset():
    with h5py.File(DATA, "r") as f:
        return MDPDataset(
            observations=f["observations"][:], actions=f["actions"][:],
            rewards=f["rewards"][:], terminals=f["terminals"][:], timeouts=f["timeouts"][:],
        )


def make_sac(n_critics, use_layer_norm):
    enc = VectorEncoderFactory(hidden_units=[256, 256], activation="relu",
                               use_layer_norm=use_layer_norm)
    cfg = SACConfig(
        actor_encoder_factory=enc, critic_encoder_factory=enc,
        n_critics=n_critics, batch_size=256,
        actor_learning_rate=3e-4, critic_learning_rate=3e-4,
    )
    return SAC(cfg, device="cuda:0", enable_ddp=False)


def main():
    print(f"[INFO] {NAME}: n_critics={N_CRITICS}, layer_norm={USE_LN}, UTD={UTD}, "
          f"offline={OFFLINE_STEPS}, finetune={FINETUNE_STEPS}, seed={args.seed}", flush=True)
    dataset = load_dataset()
    model = make_sac(N_CRITICS, USE_LN)
    model.build_with_dataset(dataset)

    print(f"[Phase 1] offline 预训练 {OFFLINE_STEPS} 梯度步", flush=True)
    model.fit(
        dataset, n_steps=OFFLINE_STEPS, n_steps_per_epoch=max(OFFLINE_STEPS // 20, 1),
        experiment_name=f"{NAME}_offline",
    )
    model.save(MODEL_PATH)
    print(f"[Phase 1] 已保存 {MODEL_PATH}", flush=True)

    print(f"[Phase 2] online 微调 {FINETUNE_STEPS} 步（UTD={UTD}）", flush=True)
    env = gym.make("HalfCheetah-v4")
    eval_env = gym.make("HalfCheetah-v4")
    model.fit_online(
        env, n_steps=FINETUNE_STEPS, n_steps_per_epoch=5000, n_updates=UTD,
        eval_env=eval_env, eval_n_trials=10, eval_epsilon=0.0,
        random_steps=1000, update_start_step=1000,
        experiment_name=f"{NAME}_finetune", show_progress=False,
    )
    env.close(); eval_env.close()
    model.save(MODEL_PATH)
    print(f"[ALL DONE] {NAME} 完成，评估曲线见 d3rlpy_logs/{NAME}_finetune_*", flush=True)


if __name__ == "__main__":
    main()
