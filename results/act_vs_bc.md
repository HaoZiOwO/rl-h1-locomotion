# ACT 自实现 vs BC 对比（Humanoid-v4）

> 状态：完成。所有数字可由 `scripts/train_act.py` / `scripts/eval_act.py` 复现。
> 口径：10 局（seed 42+i），Humanoid-v4，≤1000 步；平滑度 = 整局 Σ‖a_t − a_{t−1}‖²。

## 1. 主结果（k=10，短 chunk 版为推荐配置）

| 策略 | 平均奖励 | ±std | 平滑度 | ±std | vs BC return |
|---|---|---|---|---|---|
| BC（MLP[64,64]，26k 参数） | 519.0 | 51.1 | 73.26 | 12.63 | 100% |
| ACT-Chunk k=10（每 10 步预测、顺序执行） | **460.8** | 49.8 | **1.22** | 0.10 | 89% |
| ACT-TE k=10（每步重预测 + 指数加权平均） | 349.7 | 36.5 | 0.10 | 0.02 | 67% |

**chunking 的平滑度收益**：动作平滑度从 73.3 降到 1.2（60×），代价是 11% 的
平均奖励。k=10 是"平滑 vs 跟踪"权衡的合适工作点。

## 2. 敏感性：chunk 长度 k 与 TE 衰减 m（同一模型族，扫参不重训）

| 配置 | 平均奖励 | 平滑度 |
|---|---|---|
| ACT-Chunk k=10 | 460.8 | 1.22 |
| ACT-Chunk k=25 | 296.5 | 0.25 |
| ACT-TE k=10（m=0.01） | 349.7 | 0.10 |
| ACT-TE k=25（m=0.01，官方值） | 270.0 | 0.02 |
| ACT-TE k=25（m=0.1） | 271.5 | 0.03 |
| ACT-TE k=25（m=0.5） | 283.5 | 0.07 |

单调规律：**k 越大越平滑、奖励越低**（开环执行区间越长，漂移越大）；**TE 比同 k 的
Chunk 奖励更低**——每步重预测产生的是不同的自洽 25 步计划，把多个计划加权平均
（m 扫到 0.5 也救不回）说明损失来自"计划混合"而非"预测过时"。

## 3. CVAE 行为：后验坍缩

- 训练 KL 两项 β（1.0 官方值 / 0.1）均快速收敛到 **0.000**，val MSE 相同（0.0908）
  → 坍缩与 KL 权重无关。
- 根因：教师是 PPO 策略（确定性评估采样），同一观测下的动作近乎单模态，z 没有方差可建模。
  **机制实现正确（后验可训练、重参数化、先验采样推理），但在本数据形态下 CVAE
  的多模态收益无法呈现**——这是数据性质，不是实现缺陷。
- 推论：CVAE 的价值在遥操作/人类数据（同状态多解），RL 确定性教师
  数据下 z 退化为噪声通道。

## 4. 边界声明

1. **未复现官方 visuomotor 任务**（ALOHA 双臂图像操作）：官方 repo 依赖
   dm_control/labmaze 且与本项目双足叙事无关，明确不跑。本实现复用的是 ACT 三件
   核心机制：action chunking / CVAE / temporal ensemble，代码全部手写
   （`scripts/act_models.py`，2.57M 参数小 Transformer）。
2. 数据 = PPO 教师示范（RL 教师，非遥操作数据）；形态 = state-based（本体感受），
   非图像观测——官方 ACT 的视觉编码器替换为 MLP。
3. expert_demos.npz 为平坦转移（无回合边界），少量 chunk 跨回合拼接；BC 同样忽略
   回合结构，双方口径一致。
4. 结论适用范围：动态双足 + 确定性教师。chunking 的部署收益（延迟摊销）未在本
   实验体现（本实验测的是平滑度与模仿质量，不是端到端延迟）。

## 5. 复现命令

```bash
python scripts/train_act.py --chunk 10 --out-file act_humanoid_k10.pt   # 训练（~2 分钟 GPU）
python scripts/eval_act.py --ckpt act_humanoid_k10.pt                    # 评估（~3 分钟）
python scripts/train_act.py                                             # k=25 官方默认
python scripts/train_act.py --beta 0.1 --out-file act_humanoid_b01.pt   # KL 敏感性
```
