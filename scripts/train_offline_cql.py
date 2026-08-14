"""
train_offline_cql.py — D4RL halfcheetah-medium-v2 上的 CQL 复现（d3rlpy 2.8.1）
目的：把简历里"了解 CQL"变成"标准基准复现实战"。
数据：~/robot-rl-project/d4rl/halfcheetah_medium-v2.hdf5（HF 镜像下载）
用法：
    python train_offline_cql.py            # 训练 + 保存模型
    python train_offline_cql.py --eval     # 仅用已保存模型评估 10 局
"""
import os
import sys
import argparse

os.environ.setdefault("PYTHONPATH", "")  # 防止 hermes venv 包污染

import h5py
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halfcheetah_medium-v2.hdf5")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cql_halfcheetah_medium.d3")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_d4rl(path: str):
    """从 D4RL hdf5 读取并构造 d3rlpy MDPDataset。
    标准处理：episode 边界 = terminals OR timeouts（v2 数据把超时当作边界）。
    """
    from d3rlpy.dataset import MDPDataset

    with h5py.File(path, "r") as f:
        obs = f["observations"][:]
        acts = f["actions"][:]
        rews = f["rewards"][:]
        terminals = f["terminals"][:]
        timeouts = f["timeouts"][:] if "timeouts" in f else np.zeros_like(terminals)

    # 正确语义：terminals=真终止（mujoco 基本为 0），timeouts=截断边界（作为 episode 边界但非 final）
    # MDPDataset 断言至少一个边界非零 → 分开传，不 OR 合并
    print(f"[INFO] 数据: {len(obs)} 步, terminals={int(terminals.sum())}, timeouts={int(timeouts.sum())}, obs_dim={obs.shape[1]}, act_dim={acts.shape[1]}")
    return MDPDataset(observations=obs, actions=acts, rewards=rews,
                      terminals=terminals, timeouts=timeouts)


def train():
    from d3rlpy.algos import CQLConfig, CQL

    dataset = load_d4rl(DATA_PATH)

    # d3rlpy 2.8.1 新式 API：CQLConfig + CQL(config, device=...)
    # 关键修复（v1 失败根因）：d3rlpy 的保守损失 = weight×(raw−threshold)，
    # 后期 raw<threshold 变负 → 自适应 alpha 被压到 0 → 保守项关闭 → 崩。
    # 修复：alpha_learning_rate=0 固定 alpha=1.0（论文实现就是固定 α）。
    config = CQLConfig(
        conservative_weight=5.0,   # CQL 论文默认 α
        n_critics=2,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,            # 论文默认（v1 用的 512 偏离官方）
        initial_alpha=1.0,         # 固定保守缩放（不自动调节）
        alpha_learning_rate=0.0,   # ← 关闭自适应 alpha 坍缩
    )
    cql = CQL(config, device="cuda:0", enable_ddp=False)
    print("[INFO] 开始 CQL 训练 400k 步（batch=256, alpha 固定, 约 4 小时）...")
    cql.fit(
        dataset,
        n_steps=400_000,
        n_steps_per_epoch=50_000,
        show_progress=True,
        evaluators={},  # 评估单独跑，避免训练中中断
        save_interval=150_000,  # 150k/300k 检查点：中途评估验证趋势
    )
    cql.save_model(MODEL_PATH)
    print(f"[INFO] 模型已保存: {MODEL_PATH}")


def evaluate(n_episodes: int = 10):
    import gymnasium as gym
    from d3rlpy.algos import CQLConfig, CQL

    # 配置必须与训练完全一致（alpha_learning_rate=0 → 训练时无 alpha_optim，
    # 评估若用默认配置重建会 KeyError）
    cql = CQL(CQLConfig(alpha_learning_rate=0.0), device="cuda:0", enable_ddp=False)
    cql.build_with_dataset(load_d4rl(DATA_PATH))
    cql.load_model(MODEL_PATH)

    # D4RL 官方基准用 mujoco v2 评估（random=-280, expert=12135）
    # gymnasium 无 v2/v3（v3 需已废弃的 mujoco_py），用 v4：物理与奖励尺度与 v2 有差异，
    # 归一化对比为近似，结果文档中如实标注
    env = gym.make("HalfCheetah-v4")
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=100 + i)
        done = False
        total = 0.0
        while not done:
            action = cql.predict(np.asarray([obs]))[0]  # deterministic (argmax)
            obs, rew, terminated, truncated, _ = env.step(action)
            total += rew
            done = terminated or truncated
        returns.append(total)
        print(f"[EVAL] episode {i+1}: {total:.1f}")

    raw_mean, raw_std = float(np.mean(returns)), float(np.std(returns))
    # D4RL 归一化分数：0=随机策略, 100=专家策略（官方常量, halfcheetah v2）
    # 论文报告的 47.0 就是这个归一化分数，必须同量纲对比
    random_score, expert_score = -280.178953, 12135.0
    norm = 100.0 * (raw_mean - random_score) / (expert_score - random_score)
    print(f"[RESULT] CQL halfcheetah-medium-v2: raw={raw_mean:.1f} ± {raw_std:.1f}  "
          f"→ D4RL normalized={norm:.1f}（论文 CQL=47.0）")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "offline_cql.md"), "w", encoding="utf-8") as f:
        f.write("# D4RL CQL 复现结果\n\n")
        f.write(f"- 数据: D4RL halfcheetah-medium-v2（1M 步，medium 质量）\n")
        f.write(f"- 算法: CQL (d3rlpy 2.8.1, conservative_weight=5.0, n_critics=2)\n")
        f.write(f"- 训练: 400,000 步（batch=256, alpha 固定=1.0）, GPU (RTX 4090)\n")
        f.write(f"- 评估: {n_episodes} 局确定性推理, mean±std\n")
        f.write(f"- 归一化: 100×(raw-random)/(expert-random), random={random_score:.1f}, expert={expert_score:.1f}\n")
        f.write(f"- 环境说明: D4RL 官方用 mujoco v2 评估；此处为 gymnasium HalfCheetah-v4（v2/v3 不可用），归一化对比为近似\n\n")
        f.write(f"## 得分\n\n| 指标 | 数值 |\n|---|---|\n")
        f.write(f"| 原始 reward | {raw_mean:.1f} ± {raw_std:.1f} |\n")
        f.write(f"| D4RL 归一化 | {norm:.1f} |\n")
        f.write(f"| CQL 论文 (halfcheetah-medium-v2) | 47.0（归一化） |\n")
    print(f"[INFO] 结果已写入 {RESULTS_DIR}/offline_cql.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.eval:
        evaluate()
    else:
        train()
