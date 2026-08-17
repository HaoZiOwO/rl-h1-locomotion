"""train_offline_mr.py — medium-replay 边界验证（K3 计划 6a）

在 halfcheetah-medium-replay（混合质量数据）上重跑 CQL vs Cal-QL 的
offline 训练 + offline→online 微调对照，验证 README 里的边界断言
"Cal-QL 校准优势的主场在稀疏奖励/混合数据"。

用法：python train_offline_mr.py          # 全链 ~2.5h（offline 400k×2 + 微调 50k×2）
      python train_offline_mr.py --smoke  # 冒烟（100 步 offline + 2k 微调）
"""
import argparse
import os

os.environ.setdefault("PYTHONPATH", "")

import gymnasium as gym
import h5py
import numpy as np
from d3rlpy.algos import CQL, CalQL, CQLConfig, CalQLConfig
from d3rlpy.dataset import MDPDataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "halfcheetah_medium_replay-v2.hdf5")

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

N_OFFLINE = 200 if args.smoke else 400_000
N_FINETUNE = 2000 if args.smoke else 50_000


def load_dataset():
    with h5py.File(DATA, "r") as f:
        return MDPDataset(
            observations=f["observations"][:], actions=f["actions"][:],
            rewards=f["rewards"][:], terminals=f["terminals"][:], timeouts=f["timeouts"][:],
        )


def run(algo_cls, cfg_cls, name):
    print(f"\n===== {name}（medium-replay）offline {N_OFFLINE} 步 =====", flush=True)
    cfg = cfg_cls(alpha_learning_rate=0.0)  # alpha 固定（与 train_offline_cql.py 同修复）
    model = algo_cls(cfg, device="cuda:0", enable_ddp=False)
    model.build_with_dataset(load_dataset())
    model.fit(load_dataset(), n_steps=N_OFFLINE, n_steps_per_epoch=max(N_OFFLINE // 20, 1),
              experiment_name=f"mr_{name}_offline", show_progress=False)
    model.save(os.path.join(HERE, f"{name}_mr.d3"))
    print(f"[Phase 1] {name} offline 完成，已保存 {name}_mr.d3", flush=True)

    print(f"===== {name} offline→online 微调 {N_FINETUNE} 步 =====", flush=True)
    env = gym.make("HalfCheetah-v4")
    eval_env = gym.make("HalfCheetah-v4")
    model.fit_online(
        env, n_steps=N_FINETUNE, n_steps_per_epoch=max(N_FINETUNE // 10, 1),
        eval_env=eval_env, eval_n_trials=10, eval_epsilon=0.0,
        random_steps=1000, update_start_step=1000,
        experiment_name=f"mr_{name}_finetune", show_progress=False,
    )
    env.close(); eval_env.close()
    print(f"[DONE] {name} 微调完成，曲线见 d3rlpy_logs/mr_{name}_finetune_*", flush=True)


if __name__ == "__main__":
    run(CQL, CQLConfig, "CQL")
    run(CalQL, CalQLConfig, "CalQL")
    print("\n[ALL DONE] medium-replay 边界验证完成", flush=True)
