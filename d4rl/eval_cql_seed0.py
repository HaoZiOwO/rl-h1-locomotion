"""eval_cql_seed0.py — 用与 train_offline_cql_seeds.py 相同的评估协议评估原单 seed 模型
（凑 3-seed 统计：原模型=seed 0，本脚本 = 同协议同环境的第三个数）"""
import json
import os

os.environ.setdefault("PYTHONPATH", "")

import h5py
import numpy as np
import gymnasium as gym
from d3rlpy.algos import CQL, CQLConfig
from d3rlpy.dataset import MDPDataset

HERE = os.path.dirname(os.path.abspath(__file__))
# 配置与 train_offline_cql.py 一致（保守权重 5、alpha 固定）
config = CQLConfig(
    conservative_weight=5.0,
    n_critics=2,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    batch_size=256,
    initial_alpha=1.0,
    alpha_learning_rate=0.0,
)
with h5py.File(os.path.join(HERE, "halfcheetah_medium-v2.hdf5"), "r") as f:
    dataset = MDPDataset(
        observations=f["observations"][:], actions=f["actions"][:],
        rewards=f["rewards"][:], terminals=f["terminals"][:], timeouts=f["timeouts"][:],
    )
model = CQL(config, device="cuda:0", enable_ddp=False)
model.build_with_dataset(dataset)
model.load_model(os.path.join(HERE, "cql_halfcheetah_medium.d3"))
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
res = {"seed": 0, "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
       "per_episode": returns}
with open(os.path.join(HERE, "cql_seed0_eval.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"[DONE] seed 0: {res['return_mean']:.1f} ± {res['return_std']:.1f} (raw v4)")
