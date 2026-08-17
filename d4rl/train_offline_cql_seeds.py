"""train_offline_cql_seeds.py — CQL 多 seed 补跑（K3 计划 6b：零人工成本挂机项）

既有 CQL 48.3 是单 seed（train_offline_cql.py），顶会标准 3-seed。本脚本补 seed 1/2，
配置与 train_offline_cql.py 完全一致（保守权重 5、alpha 固定 1.0、batch 256、400k 步），
唯一差异 = 种子。输出 cql_seed{N}_halfcheetah_medium.d3 + 评估 JSON。

用法：python train_offline_cql_seeds.py --seed 1 [--seed 2]
"""
import argparse
import json
import os

os.environ.setdefault("PYTHONPATH", "")

import h5py
import numpy as np
import torch
import gymnasium as gym

from d3rlpy.algos import CQLConfig, CQL
from d3rlpy.dataset import MDPDataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "halfcheetah_medium-v2.hdf5")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=400_000)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

with h5py.File(DATA, "r") as f:
    dataset = MDPDataset(
        observations=f["observations"][:],
        actions=f["actions"][:],
        rewards=f["rewards"][:],
        terminals=f["terminals"][:],
        timeouts=f["timeouts"][:],
    )

# 配置与 train_offline_cql.py 完全一致（含 alpha 固定修复）
config = CQLConfig(
    conservative_weight=5.0,
    n_critics=2,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    batch_size=256,
    initial_alpha=1.0,
    alpha_learning_rate=0.0,
)
model = CQL(config, device="cuda:0", enable_ddp=False)
model.build_with_dataset(dataset)
print(f"[INFO] CQL seed={args.seed}，{args.steps} 步训练开始")

model.fit(dataset, n_steps=args.steps, n_steps_per_epoch=20_000,
          experiment_name=f"cql_seed{args.seed}", show_progress=False)

out = os.path.join(HERE, f"cql_seed{args.seed}_halfcheetah_medium.d3")
model.save(out)
print(f"[INFO] 已保存 {out}")

# 评估（与既有单 seed 同协议：v4 环境 10 局确定性）
env = gym.make("HalfCheetah-v4")
returns = []
for i in range(10):
    obs, _ = env.reset(seed=100 + i)
    total = 0.0
    for _ in range(1000):
        a = model.predict(np.array([obs]))[0]
        obs, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            break
    returns.append(float(total))
env.close()
res = {"seed": args.seed, "steps": args.steps,
       "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
       "per_episode": returns}
with open(os.path.join(HERE, f"cql_seed{args.seed}_eval.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"[DONE] seed={args.seed} 评估: {res['return_mean']:.1f} ± {res['return_std']:.1f}（raw, v4）")
