"""
train_offline_calql.py — D4RL halfcheetah-medium-v2 上的 Cal-QL 复现（d3rlpy 2.8.1）
与 CQL 同数据、同评估协议 → 对比实验（Cal-QL 论文：校准的 Q 值，offline→online 更平滑）。
用法: python train_offline_calql.py [--eval]
"""
import os
import sys
import argparse

os.environ.setdefault("PYTHONPATH", "")

import h5py
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halfcheetah_medium-v2.hdf5")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calql_halfcheetah_medium.d3")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_d4rl(path: str):
    from d3rlpy.dataset import MDPDataset
    with h5py.File(path, "r") as f:
        obs = f["observations"][:]
        acts = f["actions"][:]
        rews = f["rewards"][:]
        terminals = f["terminals"][:]
        timeouts = f["timeouts"][:] if "timeouts" in f else np.zeros_like(terminals)
    print(f"[INFO] 数据: {len(obs)} 步, terminals={int(terminals.sum())}, timeouts={int(timeouts.sum())}")
    return MDPDataset(observations=obs, actions=acts, rewards=rews,
                      terminals=terminals, timeouts=timeouts)


def train():
    from d3rlpy.algos import CalQLConfig, CalQL
    dataset = load_d4rl(DATA_PATH)
    # 与 CQL 完全一致的修复配置（alpha 固定，避免自适应坍缩）
    config = CalQLConfig(
        conservative_weight=5.0,
        n_critics=2,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        initial_alpha=1.0,
        alpha_learning_rate=0.0,
    )
    calql = CalQL(config, device="cuda:0", enable_ddp=False)
    print("[INFO] 开始 Cal-QL 训练 400k 步（与 CQL 同配置同数据）...")
    calql.fit(dataset, n_steps=400_000, n_steps_per_epoch=50_000,
              show_progress=True, save_interval=150_000)
    calql.save_model(MODEL_PATH)
    print(f"[INFO] 模型已保存: {MODEL_PATH}")


def evaluate(n_episodes: int = 10):
    import gymnasium as gym
    from d3rlpy.algos import CalQLConfig, CalQL
    # 配置与训练一致（alpha_learning_rate=0，避免 checkpoint 缺 alpha_optim 的 KeyError）
    calql = CalQL(CalQLConfig(alpha_learning_rate=0.0), device="cuda:0", enable_ddp=False)
    calql.build_with_dataset(load_d4rl(DATA_PATH))
    calql.load_model(MODEL_PATH)

    env = gym.make("HalfCheetah-v4")  # v2/v3 不可用（mujoco_py），归一化对比为近似
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=100 + i)
        done = False
        total = 0.0
        while not done:
            action = calql.predict(np.asarray([obs]))[0]
            obs, rew, terminated, truncated, _ = env.step(action)
            total += rew
            done = terminated or truncated
        returns.append(total)
        print(f"[EVAL] episode {i+1}: {total:.1f}")

    raw_mean, raw_std = float(np.mean(returns)), float(np.std(returns))
    random_score, expert_score = -280.178953, 12135.0
    norm = 100.0 * (raw_mean - random_score) / (expert_score - random_score)
    print(f"[RESULT] Cal-QL halfcheetah-medium-v2: raw={raw_mean:.1f} ± {raw_std:.1f}  → normalized≈{norm:.1f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "offline_calql.md"), "w", encoding="utf-8") as f:
        f.write("# D4RL Cal-QL 复现结果\n\n")
        f.write(f"- 数据: D4RL halfcheetah-medium-v2（1M 步，medium 质量）\n")
        f.write(f"- 算法: Cal-QL (d3rlpy 2.8.1, 与 CQL 同配置: conservative_weight=5.0, alpha 固定=1.0)\n")
        f.write(f"- 训练: 400,000 步（batch=256）, GPU (RTX 4090)\n")
        f.write(f"- 评估: {n_episodes} 局确定性推理, mean±std；环境 HalfCheetah-v4（v2/v3 不可用，归一化为近似）\n\n")
        f.write(f"## 得分\n\n| 指标 | 数值 |\n|---|---|\n")
        f.write(f"| 原始 reward | {raw_mean:.1f} ± {raw_std:.1f} |\n")
        f.write(f"| D4RL 归一化(近似) | {norm:.1f} |\n")
        f.write(f"| 论文参考 (halfcheetah-medium-v2) | CQL 47.0 / Cal-QL ~47.7 |\n")
    print(f"[INFO] 结果已写入 {RESULTS_DIR}/offline_calql.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.eval:
        evaluate()
    else:
        train()
