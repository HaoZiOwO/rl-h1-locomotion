# d4rl/ — 3-seed 复跑与切分点评估的实跑脚本与结果

> 本目录是 README/results 中 D4RL 数字（3-seed 统计、0k/1k/2k 切分点、mr 边界验证）的**实跑代码与结果**，补齐"每个数字可追溯"的复现链。

- 数据集**不在仓库**（`halfcheetah_medium-v2.hdf5` 227MB / `medium-replay` 59MB，超 GitHub 100MB 限制）：从 D4RL 官网（https://sites.google.com/view/d4rl/home）或镜像（HF: `imone/D4RL`）下载后放入本目录。
- **脚本对照**：`train_rlpd_seeds.py`（UTD=10，10 critics + LN）为 RLPD 3-seed 的实跑脚本；`scripts/train_rlpd.py` 为早期设计版（UTD=20 参数化说明，注释记录了降级为 10 的取舍）——**以 d4rl/ 内实跑脚本为准**。
- 复现命令（数据集就位后）：
  - RLPD 3-seed：`python train_rlpd_seeds.py --seed {0,2,3}`
  - mr 边界验证：`python train_mr_seeds.py`
  - 0k 切分点：`python eval_seed2_cutpoint.py` / `eval_seed3_cutpoint.py`
  - 3-seed 统计：`python mr_stats.py`
