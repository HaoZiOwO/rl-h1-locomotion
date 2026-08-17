# D4RL CQL 复现结果

- 数据: D4RL halfcheetah-medium-v2（1M 步，medium 质量）
- 算法: CQL (d3rlpy 2.8.1, conservative_weight=5.0, n_critics=2)
- 训练: 400,000 步（batch=256, alpha 固定=1.0）, GPU (RTX 4090)
- 评估: 10 局确定性推理, mean±std
- 归一化: 100×(raw-random)/(expert-random), random=-280.2, expert=12135.0
- 环境说明: D4RL 官方用 mujoco v2 评估；此处为 gymnasium HalfCheetah-v4（v2/v3 不可用），归一化对比为近似

## 得分

| 指标 | 数值 |
|---|---|
| 原始 reward（3 seeds） | 5722.5 / 5698.7 / 5672.6 |
| D4RL 归一化（3 seeds） | 48.2 ± 0.2（48.35 / 48.16 / 47.95） |
| CQL 论文 (halfcheetah-medium-v2) | 47.0（官方自适应 alpha + 1M 步；本实现为固定 alpha 变体 + 400k 步） |
