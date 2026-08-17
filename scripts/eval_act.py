"""eval_act.py — ACT vs BC 对比评估（return + 动作平滑度）。

策略：
  BC            —— 既有基线（bc_humanoid.pt，MLP[64,64]，确定性）
  ACT-TE        —— ACT 完整版：每步重预测 chunk（新采样 z）+ temporal ensemble
                   （指数衰减权重 exp(-m*age)，m=0.01，官方 ACT 推理口径）
  ACT-ChunkOnly —— chunking 执行摊销版：每 k 步预测一次、顺序执行整段（部署延迟口径）

指标（10 局，seed 42+i，Humanoid-v4，≤1000 步）：
  return    —— 平均总奖励 ± std（与 ppo_vs_bc.md 同口径）
  smooth    —— 动作平滑度 Σ‖a_t − a_{t−1}‖²（越小越平滑；DP 部署叙事的配套指标）

输出：results/act_vs_bc.md
用法：python scripts/eval_act.py
"""
import os

import argparse

import numpy as np
import torch
import gymnasium as gym

from act_models import ACTModel

parser = argparse.ArgumentParser()
parser.add_argument("--m-te", type=float, default=0.01, help="temporal ensemble 衰减系数（官方 ACT=0.01）")
parser.add_argument("--ckpt", type=str, default="act_humanoid.pt", help="ACT 检查点")
parser.add_argument("--out-md", type=str, default="results/act_vs_bc_auto.md",
                    help="自动报告路径（完整分析文档 results/act_vs_bc.md 为手写，勿覆盖）")
args = parser.parse_args()

NUM_EVAL_EPISODES = 10
MAX_STEPS = 1000
SEED = 42
M_TE = args.m_te  # temporal ensemble 衰减系数（官方 ACT 值 0.01）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============ 模型加载 ============

# BC 基线
bc_ckpt = torch.load("bc_humanoid.pt", map_location="cpu", weights_only=False)


class BCNet(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        layers = []
        d_in = obs_dim
        for h in hidden:
            layers += [torch.nn.Linear(d_in, h), torch.nn.ReLU()]
            d_in = h
        layers.append(torch.nn.Linear(d_in, act_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


bc = BCNet(bc_ckpt["obs_dim"] if "obs_dim" in bc_ckpt else 376, 17, bc_ckpt["hidden"])
bc.load_state_dict(bc_ckpt["state_dict"])
bc.eval()

# ACT
act_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
act = ACTModel(obs_dim=act_ckpt["obs_dim"], act_dim=act_ckpt["act_dim"],
               chunk=act_ckpt["chunk"], latent_dim=act_ckpt["latent"]).to(DEVICE)
act.load_state_dict(act_ckpt["model_state"])
act.eval()

# ============ 策略 ============

def bc_policy(obs):
    x = torch.tensor((obs - bc_ckpt["obs_mean"]) / bc_ckpt["obs_std"], dtype=torch.float32)
    with torch.no_grad():
        return bc(x).numpy()


class ACTPolicy:
    """ACT 推理策略。mode: 'te'（temporal ensemble，每步重预测）| 'chunk'（每 k 步预测一次）。"""

    def __init__(self, model, ckpt, mode: str, seed: int):
        assert mode in ("te", "chunk")
        self.model = model
        self.ckpt = ckpt
        self.mode = mode
        self.k = ckpt["chunk"]
        self.rng = np.random.default_rng(seed)  # z 采样用固定种子，保证可复现
        self.reset()

    def reset(self):
        self.step_in_chunk = 0
        self.chunk = None          # 当前执行的 chunk（chunk 模式）
        self.queue = []            # (chunk_tensor, age)（te 模式）

    def _predict_chunk(self, obs):
        """给定 obs 预测 (k, 17) 动作 chunk，z 从先验采样。"""
        obs_n = (obs - self.ckpt["obs_mean"]) / self.ckpt["obs_std"]
        x = torch.tensor(obs_n, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z = torch.tensor(self.rng.standard_normal((1, self.ckpt["latent"])),
                         dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            chunk = self.model.decode(self.model.encode_obs(x), z)  # (1, k, 17)
        return chunk[0]

    def __call__(self, obs):
        if self.mode == "chunk":
            if self.step_in_chunk % self.k == 0:
                self.chunk = self._predict_chunk(obs)
            action = self.chunk[self.step_in_chunk % self.k].cpu().numpy()
            self.step_in_chunk += 1
            return action
        # te 模式：每步重预测一个 chunk 入队；对队内所有 chunk 做指数衰减加权平均
        new_chunk = self._predict_chunk(obs)
        self.queue = [(c, age + 1) for (c, age) in self.queue if age + 1 < self.k]
        self.queue.insert(0, (new_chunk, 0))
        weights = [np.exp(-M_TE * age) for (_, age) in self.queue]
        w_sum = sum(weights)
        action = sum(w * c[age].cpu().numpy() for w, (c, age) in zip(weights, self.queue)) / w_sum
        return action


# ============ 评估 ============

def evaluate(policy_fn, name, seed_offset=0):
    env = gym.make("Humanoid-v4")
    returns, smooths = [], []
    for ep in range(NUM_EVAL_EPISODES):
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()
        obs, _ = env.reset(seed=SEED + ep + seed_offset)
        total = 0.0
        smooth = 0.0
        prev_a = None
        for _ in range(MAX_STEPS):
            action = policy_fn(obs)
            if prev_a is not None:
                smooth += float(np.sum((action - prev_a) ** 2))
            prev_a = action
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        smooths.append(smooth)
    env.close()
    mean_r, std_r = float(np.mean(returns)), float(np.std(returns))
    mean_s, std_s = float(np.mean(smooths)), float(np.std(smooths))
    print(f"{name}: return {mean_r:.1f} ± {std_r:.1f} | smooth {mean_s:.2f} ± {std_s:.2f} "
          f"（{NUM_EVAL_EPISODES} 局）")
    return mean_r, std_r, mean_s, std_s


print("=" * 70)
print("ACT vs BC 对比评估（Humanoid-v4，10 局/策略）")
print("=" * 70)
bc_r = evaluate(bc_policy, "BC        ")
act_te_policy = ACTPolicy(act, act_ckpt, mode="te", seed=SEED)
act_te_r = evaluate(act_te_policy, "ACT-TE    ")
act_ck_policy = ACTPolicy(act, act_ckpt, mode="chunk", seed=SEED)
act_ck_r = evaluate(act_ck_policy, "ACT-Chunk ")

# ============ 写报告 ============
os.makedirs("results", exist_ok=True)
with open("results/act_vs_bc.md", "w", encoding="utf-8") as f:
    f.write("# ACT 自实现 vs BC 对比（Humanoid-v4）\n\n")
    f.write(f"- 数据：expert_demos.npz（PPO 教师示范，9913 条转移），与 BC 同管线同归一化\n")
    f.write(f"- ACT 超参：chunk k={act_ckpt['chunk']}，latent={act_ckpt['latent']}，"
            f"KL 权重 β={act_ckpt['beta']}，{act_ckpt['epochs']} epochs，"
            f"参数 {act_ckpt['n_params'] / 1e6:.2f}M\n")
    f.write(f"- 评估：Humanoid-v4，每策略 {NUM_EVAL_EPISODES} 局（seed 42+i），≤1000 步\n")
    f.write(f"- 动作平滑度口径：sum||a_t - a_(t-1)||^2（整局求和，越小越平滑）\n\n")
    f.write("| 策略 | 平均奖励 | ±std | 平滑度 | ±std |\n|---|---|---|---|---|\n")
    f.write(f"| BC | {bc_r[0]:.1f} | {bc_r[1]:.1f} | {bc_r[2]:.2f} | {bc_r[3]:.2f} |\n")
    f.write(f"| ACT-TE（temporal ensemble） | {act_te_r[0]:.1f} | {act_te_r[1]:.1f} "
            f"| {act_te_r[2]:.2f} | {act_te_r[3]:.2f} |\n")
    f.write(f"| ACT-Chunk（k 步摊销执行） | {act_ck_r[0]:.1f} | {act_ck_r[1]:.1f} "
            f"| {act_ck_r[2]:.2f} | {act_ck_r[3]:.2f} |\n\n")
    f.write("## 边界声明\n\n")
    f.write("1. **未复现官方 visuomotor 任务**（ALOHA 双臂图像操作）：官方 repo 依赖\n"
            "   dm_control/labmaze（Windows 安装风险高）且与双足叙事无关，明确不跑；\n"
            "   本实现复用的是 ACT 的三件核心机制（chunking / CVAE / temporal ensemble）。\n")
    f.write("2. 数据 = PPO 教师示范（RL 教师，非遥操作数据）；形态 = state-based\n"
            "   （本体感受），非图像观测——官方 ACT 的视觉编码器在此替换为 MLP。\n")
    f.write("3. npz 为平坦转移（无回合边界），少量 chunk 跨回合拼接；BC 同样忽略\n"
            "   回合结构，双方口径一致。\n")
    f.write("4. 训练 loss 的 val MSE 见 train_act.py 日志；KL 权重 β=1.0 为官方默认值，\n"
            "   如收敛异常会降 β 并如实报告。\n")
print("报告已写入 results/act_vs_bc.md")
