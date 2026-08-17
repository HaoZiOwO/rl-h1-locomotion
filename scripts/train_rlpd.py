"""train_rlpd.py — RLPD 风格 offline→online 微调 vs 对照（冲刺期）

RLPD 配方（RLPD 论文四要素在 d3rlpy 的落地）：
  1. LayerNorm：VectorEncoderFactory(use_layer_norm=True)（d3rlpy 原生支持，无需自定义工厂）
  2. 10 critics ensemble：SACConfig(n_critics=10)
  3. 高 UTD：fit_online 的 n_steps_per_epoch=20_000 → UTD=20（d3rlpy：每 1000 env 步做
     n_steps_per_epoch 次梯度更新）
  4. offline 预训练 + online 微调：同一份 halfcheetah_medium-v2 数据先离线训 SAC，
     再 HalfCheetah-v4 在线微调 5 万步

对照（隔离"离线算法 SAC vs CQL"的混淆）：SAC 2 critics + 无 LN + UTD=5，与基线 CQL
微调（既有曲线，UTD=5）同口径。指标：达到 55 归一化分（v4 raw ≈ 6548）所需 online 步数。

用法：
  python train_rlpd.py --smoke                        # 冒烟：100 步离线 + 1 轮在线
  python train_rlpd.py --mode rlpd                    # 主实验（10 critics + LN + UTD20）
  python train_rlpd.py --mode sac_ctl                 # 对照（2 critics 无 LN UTD5）
"""
import argparse
import os
import sys

os.environ.setdefault("PYTHONPATH", "")

import gymnasium as gym
import h5py
import numpy as np
from d3rlpy.algos import SAC, SACConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "halfcheetah_medium-v2.hdf5")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="rlpd", choices=["rlpd", "sac_ctl"])
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--offline-steps", type=int, default=300_000, help="RLPD 论文 1M；本机吞吐 23 it/s 降级")
parser.add_argument("--finetune-steps", type=int, default=50_000)
args = parser.parse_args()

N_CRITICS = 10 if args.mode == "rlpd" else 2
USE_LN = args.mode == "rlpd"
# UTD = n_updates（每环境步的梯度更新数）。计划原文"若时间紧先 UTD=10"——
# 本机 d3rlpy 实测 10 critics 吞吐 23 it/s，UTD=20 的 1M 梯度步 ≈ 12h 超时间盒，降级为 10。
# 对照 sac_ctl 用 UTD=1（与既有 CQL 基线微调同协议）。
UTD = 10 if args.mode == "rlpd" else 1
NAME = "rlpd" if args.mode == "rlpd" else "sac_ctl"
MODEL_PATH = os.path.join(HERE, f"{NAME}_halfcheetah_medium.d3")

if args.smoke:
    args.offline_steps = 200
    args.finetune_steps = 5000
    N_CRITICS = 10 if args.mode == "rlpd" else 2
    UTD = 1  # 冒烟用 UTD=1（验证 n_updates 路径即可，不花真时间）


def load_dataset():
    with h5py.File(DATA, "r") as f:
        return MDPDataset(
            observations=f["observations"][:],
            actions=f["actions"][:],
            rewards=f["rewards"][:],
            terminals=f["terminals"][:],
            timeouts=f["timeouts"][:],
        )


def make_sac(n_critics, use_layer_norm):
    enc = VectorEncoderFactory(hidden_units=[256, 256], activation="relu",
                               use_layer_norm=use_layer_norm)
    cfg = SACConfig(
        actor_encoder_factory=enc,
        critic_encoder_factory=enc,
        n_critics=n_critics,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
    )
    return SAC(cfg, device="cuda:0", enable_ddp=False)


def main():
    print(f"[INFO] mode={NAME}: n_critics={N_CRITICS}, layer_norm={USE_LN}, "
          f"UTD={UTD}, offline={args.offline_steps}, finetune={args.finetune_steps}")
    dataset = load_dataset()
    model = make_sac(N_CRITICS, USE_LN)
    model.build_with_dataset(dataset)

    # ---- Phase 1: offline 预训练（RLPD：离线数据上高 UTD 训 SAC）----
    print(f"[Phase 1] offline 预训练 {args.offline_steps} 梯度步")
    # 注：d3rlpy 2.8.1 的 fit() 评估接口已改（EnvironmentEvaluator），离线阶段不做周期
    # 评估——离线预训练后的起点分数 = Phase 2 微调曲线的 epoch 0 评估值，同环境同口径。
    model.fit(
        dataset,
        n_steps=args.offline_steps,
        n_steps_per_epoch=max(args.offline_steps // 20, 1),
        experiment_name=f"{NAME}_offline",
    )
    model.save(MODEL_PATH)
    print(f"[Phase 1] 已保存 {MODEL_PATH}")

    # ---- Phase 2: online 微调（与 CQL 基线同协议：5 万步，每 5k 步评估；UTD=n_updates）----
    print(f"[Phase 2] online 微调 {args.finetune_steps} 步（UTD={UTD}）")
    env = gym.make("HalfCheetah-v4")
    eval_env = gym.make("HalfCheetah-v4")
    model.fit_online(
        env,
        n_steps=args.finetune_steps,
        n_steps_per_epoch=5000,      # 与 CQL 基线评估网格一致（5k/10k/.../50k）
        n_updates=UTD,               # UTD：每环境步梯度更新数
        eval_env=eval_env,
        eval_n_trials=10,
        eval_epsilon=0.0,
        random_steps=1000,
        update_start_step=1000,
        experiment_name=f"{NAME}_finetune",
        show_progress=False,
    )
    env.close()
    eval_env.close()
    model.save(MODEL_PATH)
    print(f"[ALL DONE] {NAME} 完成，评估曲线见 d3rlpy_logs/{NAME}_finetune_*")


if __name__ == "__main__":
    main()
