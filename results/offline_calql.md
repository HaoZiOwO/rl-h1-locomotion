# D4RL Cal-QL 复现结果

- 数据: D4RL halfcheetah-medium-v2（1M 步，medium 质量）
- 算法: Cal-QL (d3rlpy 2.8.1, 与 CQL 同配置: conservative_weight=5.0, alpha 固定=1.0)
- 训练: 400,000 步（batch=256）, GPU (RTX 4090)
- 评估: 10 局确定性推理, mean±std；环境 HalfCheetah-v4（v2/v3 不可用，归一化为近似）

## 得分

| 指标 | 数值 |
|---|---|
| 原始 reward | 5755.0 ± 110.1 |
| D4RL 归一化(近似) | 48.6 |
| 论文参考 (halfcheetah-medium-v2) | CQL 47.0 / Cal-QL ~47.7 |
