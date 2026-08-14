"""
train_bc.py — 训练行为克隆（BC）网络

BC = 监督学习：输入 obs（376 维）→ 神经网络 → 输出 action（17 维）
训练目标：让网络的输出尽量接近专家的动作（MSE 均方误差）

输出：bc_humanoid.pt（网络权重 + 观测归一化统计量）
"""
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 超参数
# ============================================================
HIDDEN = [64, 64]        # 网络结构：和 PPO 的 MlpPolicy 一致（保证公平对比）
EPOCHS = 100             # 训练轮数
BATCH_SIZE = 256         # 每批样本数
LR = 1e-3                # 学习率（Adam 常用 1e-3）
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_FILE = "expert_demos.npz"
OUT_FILE = "bc_humanoid.pt"

# ============================================================
# 第1步：加载示范数据
# ============================================================
data = np.load(IN_FILE)
obs = data["obs"]          # [N, 376] 专家"看到"的观测
acts = data["actions"]     # [N, 17]  专家"执行"的动作
N = len(obs)
print(f"加载 {N} 条样本，设备：{DEVICE}")

# ============================================================
# 第2步：归一化观测（这一步非常重要！）
#    观测各维度的数值范围差异巨大（角度 ~1，角速度 ~10）
#    不归一化的话梯度下降会走弯路、收敛很慢。
#    做法：减均值、除标准差 → 每维都变成"均值 0、标准差 1"
# ============================================================
obs_mean = obs.mean(axis=0)
obs_std = obs.std(axis=0) + 1e-8          # +1e-8 防止除零
obs_norm = (obs - obs_mean) / obs_std

# 留出 10% 做验证集（监督学习标准做法：验证集误差能看出是否过拟合）
split = int(N * 0.9)
obs_tr, obs_va = obs_norm[:split], obs_norm[split:]
acts_tr, acts_va = acts[:split], acts[split:]

X_tr = torch.tensor(obs_tr, dtype=torch.float32, device=DEVICE)
y_tr = torch.tensor(acts_tr, dtype=torch.float32, device=DEVICE)
X_va = torch.tensor(obs_va, dtype=torch.float32, device=DEVICE)
y_va = torch.tensor(acts_va, dtype=torch.float32, device=DEVICE)

# ============================================================
# 第3步：搭网络：输入 376 → 64 → 64 → 输出 17
#    和 PPO 的策略网络同构，这样"BC 不如 PPO"就不能赖网络太小
# ============================================================
class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        layers = []
        d_in = obs_dim
        for h in hidden:                        # 循环搭隐藏层
            layers += [nn.Linear(d_in, h), nn.ReLU()]
            d_in = h
        layers.append(nn.Linear(d_in, act_dim)) # 输出层不加激活（直接回归力矩值）
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = BCNet(376, 17, HIDDEN).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()          # 均方误差：动作差得越多，惩罚越大

# ============================================================
# 第4步：训练循环（标准监督学习套路：
#    前向算出预测 → 和专家动作比出误差 → 反向传播求梯度 → 更新权重）
# ============================================================
torch.manual_seed(SEED)
n_batches = (len(X_tr) + BATCH_SIZE - 1) // BATCH_SIZE

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr))            # 每轮打乱数据顺序（防过拟合）
    total_loss = 0.0
    for i in range(0, len(X_tr), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        pred = model(X_tr[idx])                 # 前向：网络预测的动作
        loss = loss_fn(pred, y_tr[idx])         # 和专家动作比，算误差
        optimizer.zero_grad()
        loss.backward()                         # 反向传播求梯度
        optimizer.step()                        # 更新权重
        total_loss += loss.item()

    # 每 10 轮打印一次：训练误差 + 验证误差（验证误差不降了 = 开始过拟合）
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(X_va), y_va).item()
        print(f"epoch {epoch+1:3d}/{EPOCHS}  train_mse={total_loss/n_batches:.5f}  val_mse={va_loss:.5f}")

# ============================================================
# 第5步：保存（网络权重 + 归一化统计量——评估时要用同样的归一化）
# ============================================================
torch.save({
    "state_dict": model.state_dict(),
    "obs_mean": obs_mean,
    "obs_std": obs_std,
    "hidden": HIDDEN,
}, OUT_FILE)
print(f"BC 训练完成，已保存到 {OUT_FILE}")
