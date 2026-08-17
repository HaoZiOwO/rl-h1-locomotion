# medium-replay 边界验证：Cal-QL 校准优势在混合数据上的检验

> 状态：3 次独立训练完成（原跑 + seed1 + seed2，2026-08-16）。**原单 seed"弱阳性"结论下修**——见 §2。
> 统计脚本：`d4rl/mr_stats.py`；曲线：`d3rlpy_logs/mr_seed{N}_{CQL,CalQL}_finetune_*/evaluation.csv`。

## 1. 结果（3 次独立训练 mean±std；offline 400k + 微调 50k，每 5k 步 10 局确定性，raw v4）

| online 步数 | CQL | Cal-QL | Cal-QL 领先 seeds |
|---|---|---|---|
| 5k | 5399.5±73.6 | 5469.8±55.0 | 2/3 |
| 10k | 5522.7±14.8 | 5586.5±70.3 | 2/3 |
| 15k | 5522.6±145.1 | 5572.2±107.2 | 2/3 |
| 20k | 5657.3±105.3 | 5586.2±30.6 | 1/3 |
| 25k | 5714.0±32.1 | 5670.5±106.3 | 2/3 |
| 30k | 5669.5±151.1 | 5737.8±44.0 | 2/3 |
| 35k | 5853.2±199.6 | 5710.9±87.8 | 1/3 |
| 40k | 5837.3±136.3 | 5758.9±143.1 | 0/3 |
| 45k | 5825.9±140.5 | 5836.4±86.4 | 1/3 |
| **50k** | **5908.8±67.1** | **5900.2±136.7** | **2/3** |

- 50k 终值 per-run：CQL 5856 / 5886 / 5984；Cal-QL 5935 / 5749 / 6016（CalQL−CQL：+79 / −136 / +32，均值 **−9** raw ≈ −0.07 归一化）
- 逐点领先不稳健：10 个评估点 Cal-QL 平均仅 1.5/3 runs 领先，无一点 3/3。

## 2. 结论（3-seed 下修，2026-08-16）

1. **原单 seed"弱阳性（9/10 评估点领先、平均 +1 归一化分）"不复现**：50k 终值打平
   （5900±137 vs 5909±67，差 −9 远在 std 内）；各 run 领先互有胜负。单 seed 的
   "9/10 点领先"是**同一条曲线的相关样本假象**，不是 Cal-QL 的真实增量。
2. **dense 侧阴性依旧**：CQL 6826 反超 Cal-QL 6484（未变）。
3. **更新后的边界表述**：
   > "Cal-QL 的校准优势：dense 单一质量数据阴性（CQL 反超）；medium-replay 混合数据
   > 3-seed 打平（5900 vs 5909，无增量）；稀疏奖励（antmaze）未测。"
4. **方法论收获**：单 seed 结论在 3-seed 下被证伪、主动下修——补跑的价值
   就是防止把种子噪声当效应。

## 3. 复现

```bash
cd ~/robot-rl-project/d4rl && python train_offline_mr.py   # 原跑（~7h，隐式种子）
python train_mr_seeds.py --seed 1 --algo CQL               # 3-seed 补跑（seed1/2，过夜队列）
python train_mr_seeds.py --seed 1 --algo CalQL
python train_mr_seeds.py --seed 2 --algo CQL
python train_mr_seeds.py --seed 2 --algo CalQL
python mr_stats.py                                          # 3-seed 统计
```
