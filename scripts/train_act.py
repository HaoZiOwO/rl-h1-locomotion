"""train_act.py — 训练 ACT（CVAE + action chunking）模仿策略。

与 BC 基线共享同一数据与归一化管线（expert_demos.npz、obs 均值/方差），
保证对比公平：唯一变量 = 模型与训练目标（BC: 单步 MSE；ACT: chunk MSE + β·KL）。

输出：act_humanoid.pt（模型权重 + obs 归一化统计量 + 超参数）
用法：
  python scripts/train_act.py                     # 默认 k=25, latent=32, beta=1.0
  python scripts/train_act.py --beta 0.1          # KL 权重消融
  python scripts/train_act.py --smoke             # 冒烟：5 epoch 快速验证
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from act_models import ACTModel, kl_divergence

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, default=25)
parser.add_argument("--latent", type=int, default=32)
parser.add_argument("--beta", type=float, default=1.0, help="KL 权重（官方 ACT 用 1.0）")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke", action="store_true", help="冒烟：5 epoch")
parser.add_argument("--in-file", type=str, default="expert_demos.npz")
parser.add_argument("--out-file", type=str, default="act_humanoid.pt")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============ 数据（与 train_bc.py 同管线：同样的归一化统计量） ============
data = np.load(args.in_file)
obs_all = data["obs"]          # (9913, 376)
acts_all = data["actions"]     # (9913, 17)
N = len(obs_all)
obs_mean = obs_all.mean(axis=0)
obs_std = obs_all.std(axis=0) + 1e-8
obs_norm = (obs_all - obs_mean) / obs_std

K = args.chunk
ACT_DIM = acts_all.shape[1]

# chunk 起点：允许 [0, N-K]；按起点索引 90/10 切训练/验证
valid_starts = np.arange(N - K)
rng = np.random.default_rng(args.seed)
train_starts = valid_starts[: int(len(valid_starts) * 0.9)]
val_starts = valid_starts[int(len(valid_starts) * 0.9):]
print(f"样本 {N}，chunk k={K}，有效起点 {len(valid_starts)}（train {len(train_starts)} / val {len(val_starts)}）")
print(f"设备 {DEVICE}，latent={args.latent}，beta={args.beta}，epochs={args.epochs}")

# 边界注：npz 是平坦转移（无回合边界），少量 chunk 会跨回合拼接——BC 同样忽略
# 回合结构，双方口径一致（见 results/act_vs_bc.md 边界声明）。


def make_batch(starts):
    idx = rng.choice(starts, size=args.batch_size, replace=False)
    obs_b = torch.tensor(obs_norm[idx], dtype=torch.float32, device=DEVICE)          # (B, 376)
    # chunk 动作：每个起点取连续 K 步
    chunks = np.stack([acts_all[i:i + K] for i in idx])                               # (B, K, 17)
    act_b = torch.tensor(chunks, dtype=torch.float32, device=DEVICE)
    return obs_b, act_b


# ============ 模型 ============
model = ACTModel(obs_dim=obs_all.shape[1], act_dim=ACT_DIM,
                 chunk=K, latent_dim=args.latent).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
n_params = sum(p.numel() for p in model.parameters())
print(f"ACT 参数量: {n_params / 1e6:.2f}M（对比 BC MLP[64,64] ≈ 26k 参数）")

# ============ 训练循环 ============
epochs = 5 if args.smoke else args.epochs
for epoch in range(1, epochs + 1):
    model.train()
    n_batches = max(1, len(train_starts) // args.batch_size)
    loss_mse_total = loss_kl_total = 0.0
    for _ in range(n_batches):
        obs_b, act_b = make_batch(train_starts)
        pred, mu, logvar = model(obs_b, act_b.reshape(act_b.shape[0], -1))
        mse = nn.functional.mse_loss(pred.reshape(act_b.shape[0], -1),
                                     act_b.reshape(act_b.shape[0], -1))
        kl = kl_divergence(mu, logvar).mean()
        loss = mse + args.beta * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mse_total += mse.item()
        loss_kl_total += kl.item()
    # 验证（无梯度）
    model.eval()
    with torch.no_grad():
        obs_v, act_v = make_batch(val_starts)
        pred_v, mu_v, logvar_v = model(obs_v, act_v.reshape(act_v.shape[0], -1))
        mse_v = nn.functional.mse_loss(pred_v.reshape(act_v.shape[0], -1),
                                       act_v.reshape(act_v.shape[0], -1)).item()
        kl_v = kl_divergence(mu_v, logvar_v).mean().item()
    print(f"epoch {epoch:3d} | train mse={loss_mse_total / n_batches:.5f} "
          f"kl={loss_kl_total / n_batches:.3f} | val mse={mse_v:.5f} kl={kl_v:.3f}")

# ============ 保存 ============
torch.save({
    "model_state": model.state_dict(),
    "obs_mean": obs_mean,
    "obs_std": obs_std,
    "obs_dim": obs_all.shape[1],
    "act_dim": ACT_DIM,
    "chunk": K,
    "latent": args.latent,
    "beta": args.beta,
    "epochs": epochs,
    "seed": args.seed,
    "n_params": n_params,
    "val_mse": mse_v,
    "val_kl": kl_v,
}, args.out_file)
print(f"已保存 {args.out_file}（val mse={mse_v:.5f}）")
