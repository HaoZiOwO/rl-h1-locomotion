"""
finetune_offline_online.py — Cal-QL vs CQL 的 offline→online 微调对比

动机：Cal-QL 论文的核心主张是"校准保守"让 offline→online 微调更快（CQL 把 Q 压太低，
微调时要先恢复 Q 值；Cal-QL 用 max(Q, reference) 校准，保留乐观初始化）。
我们已验证二者 offline 分数（CQL 48.3 / Cal-QL 48.6），但没验证微调优势——
这个实验补上 Cal-QL 的另一半价值，对应外骨骼"offline 预训练 + 少量真机微调"场景。

用法: python finetune_offline_online.py
"""
import os
import gymnasium as gym
import h5py
import numpy as np
from d3rlpy.algos import CQL, CalQL, CQLConfig, CalQLConfig
from d3rlpy.dataset import MDPDataset

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halfcheetah_medium-v2.hdf5")
N_FINETUNE = 50_000          # 每个模型在线微调 5 万步
N_PER_EPOCH = 5_000
RANDOM_STEPS = 1_000         # 先随机收集 1k 步，再开始更新


def load_dataset():
    with h5py.File(DATA, "r") as f:
        obs = f["observations"][:]
        acts = f["actions"][:]
        rews = f["rewards"][:]
        terminals = f["terminals"][:]
        timeouts = f["timeouts"][:]
    return MDPDataset(observations=obs, actions=acts, rewards=rews,
                      terminals=terminals, timeouts=timeouts)


def run(model_cls, config_cls, model_path, name):
    print(f"\n===== {name} offline→online 微调（{N_FINETUNE} 步）=====")
    env = gym.make("HalfCheetah-v4")
    eval_env = gym.make("HalfCheetah-v4")

    # 配置与训练一致（alpha 固定）
    model = model_cls(config_cls(alpha_learning_rate=0.0), device="cuda:0", enable_ddp=False)
    model.build_with_dataset(load_dataset())
    model.load_model(model_path)

    model.fit_online(
        env,
        n_steps=N_FINETUNE,
        n_steps_per_epoch=N_PER_EPOCH,
        eval_env=eval_env,
        eval_n_trials=10,
        eval_epsilon=0.0,          # 确定性评估，与 offline 评估一致
        random_steps=RANDOM_STEPS,
        update_start_step=RANDOM_STEPS,
        experiment_name=f"{name}_finetune",
        show_progress=True,
    )
    env.close()
    eval_env.close()
    print(f"[DONE] {name} 微调完成，评估曲线见 d3rlpy_logs/{name}_finetune_*")


if __name__ == "__main__":
    run(CQL, CQLConfig, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cql_halfcheetah_medium.d3"), "CQL")
    run(CalQL, CalQLConfig, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calql_halfcheetah_medium.d3"), "CalQL")
    print("\n[ALL DONE] 两个模型微调完成")
