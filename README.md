# Robot RL Portfolio — 双足 → 算法 → Offline RL → 步态

> 面向消费级外骨骼的强化学习项目集：4 个可复现项目，每个数字都能在对应脚本与 `results/` 文档中追溯。
> 主线：**仿真双足（①）→ 算法族与蒸馏（②）→ 离线学习与微调（③）→ 人体步态与助力（④）**。
> 全部为仿真与公开数据验证；真机验证（Sim-to-Real）依赖公司环境。

![外骨骼 RL 技术栈 — 4 项目全景](assets/exoskeleton-stack.drawio.png)

## 项目全景

| # | 项目 | 平台 | 核心成果 |
|---|---|---|---|
| ① | 双足人形运动控制 | Isaac Lab + Unitree H1 | 20 级难度谱 8 代表点 100% 存活、40kg 负载鲁棒、能耗直读 |
| ② | 算法族对比 + 蒸馏 | MuJoCo Humanoid-v4 | SAC 6416 vs PPO 508（样本效率 ~40×）、观测削减 92.6% 性能持平 |
| ③ | Offline RL 复现 + 微调 | D4RL + d3rlpy | CQL 48.3 / Cal-QL 48.6（超论文 47.0/47.7）、offline→online 48→57 |
| ④ | 步态相位 + 助力增益 | Daphnet 数据集 | 16/17 段检出、助力平滑 78% 权衡 |

---

**这些项目对应外骨骼技术栈的哪一环**：

| 项目 | 外骨骼对应 |
|---|---|
| ① 双足 H1 | 承重负载鲁棒性（外骨骼穿戴场景的仿真验证） |
| ② 算法族对比 | 策略选型（样本效率 → 真机/人体数据昂贵） |
| ③ Offline RL | 用离线数据学习（真实人机数据无法在线采集） |
| ④ 步态相位 | 穿戴者意图感知（ATD 决策层） |

---

## 项目 ①：双足人形运动控制（Isaac Lab + Unitree H1）

NVIDIA Isaac Lab + RSL-RL（PPO）训练 H1 速度跟踪策略：平地 → 崎岖（地形课程 + 域随机化）→ 系统化验证。

![难度-性能曲线](assets/difficulty_curve.png)
![负载-性能曲线](assets/payload_curve.png)
![训练收敛（flat vs rough，PPO）](assets/training_curves.png)

| 实验 | 结果 | 结论 |
|---|---|---|
| 难度-性能曲线 | 20 级难度谱抽 8 代表点（0→19 全跨度）**全部 100% 存活**，奖励 35-36.4 零衰减 | 地形自适应步态 |

> 难度谱说明：评估通过写入 `terrain_levels` 张量 + 重置 + **读回 observed 验证**（`observed == requested`，见 `results/evals_summary.md` §1 与 eval 日志），等级真实生效。0→19 级难度增量对"存活 + 速度跟踪奖励"这一度量的区分度有限——训练课程推进到 level 5.8 后，更高等级仍全存活且奖励无衰减，部分是难度谱度量特性（高等级地形对 H1 通过性不构成瓶颈），非策略在全部难度都经过训练。
| 负载 0-40kg | flat 40kg 崩（20%），rough/nodr 40kg 仍 100%（29.9/31.8） | 鲁棒性来自训练环境多样性 |
| 负重训练（+25kg） | 25kg 下 37.6（恢复无负载水平）vs 通用 34.1 | 为承重场景专门训练 |
| 奖励敏感性 | 强力矩惩罚**完全崩**（0% 存活） | 奖励权重临界点实证 |
| DR 消融 | nodr 35.0 ≈ rough 35.3 | 动力学 DR 影响小，地形是泛化主因 |
| 能耗直读 | +25kg 功率 +10%（328 vs 297W）；崎岖 0.175 vs 平地 0.126 reward/W | 节能是奖励设计问题 |

**ONNX 导出与推理延迟**（`scripts/export_h1.py` 导出 JIT/ONNX 并内置数值一致性校验，基准见 `results/inference_latency.md`）：CPU 推理 flat 0.017ms / rough 0.037ms（mean，2000 次计时），对照控制环预算 100Hz→10ms、500Hz→2ms、1kHz→1ms——**满足 1kHz 控制环**（0.017ms 远小于 1ms 预算），且 JIT vs ONNX 输出一致性 MSE ~1e-13（无算子精度损失），策略可安全部署到轻量控制器。

---

## 项目 ②：算法族对比 + Teacher-Student 蒸馏（MuJoCo Humanoid-v4）

同环境、同评估协议对比 on-policy / 模仿学习 / 生成式 / off-policy 四类算法，并做传感器最少化蒸馏。

![四算法对比](assets/algo_compare.png)
![SAC 样本效率](assets/sac_sample_efficiency.png)
![蒸馏架构（双语）](assets/teacher-student-distill-bilingual.drawio.png)

*样本效率图注：SAC 训练过程评估曲线（eval callback，`assets/sac_logs/evaluations.npz` 可复现）——5 万步 ≈504 已接近 PPO 2M 步水平，10 万步正式超过；最终独立评估 6416.8 见上表。*

**SAC 走路 Demo（33s，6416 分）**：

<video src="assets/demo_sac.mp4" controls muted loop width="100%"></video>

| 算法 | 预算 | 平均奖励 |
|---|---|---|
| PPO（on-policy） | 2M 步 | 508.3 ± 47.3 |
| BC（模仿学习） | 9913 条示范 | 519.0 ± 51.1 |
| Diffusion Policy（DDPM 自实现） | 9913 条示范 | 485.7 ± 30.4 |
| **SAC（off-policy）** | 2M 步 | **6416.8 ± 711.8** |

- **样本效率**：同 2M 步预算下 SAC 5 万步即达 PPO 2M 步水平（~40×）——off-policy 对真机/人体数据昂贵的价值；PPO/SAC 均为默认超参未单独调优，对比目标是样本效率而非绝对分数；**DP 适用边界**：近单峰分布优势不显（485.7 vs BC 519）
- **Teacher-Student 蒸馏**：教师（PPO 全 376 维观测）→ 学生（受限 28 维，砍关节角度/惯性），性能持平（534.4 vs 508.3，±std 内）——传感器最少化可行

---

## 项目 ③：Offline RL 复现 + 微调（D4RL halfcheetah-medium）

d3rlpy 复现 CQL / Cal-QL（400k 步 GPU），并做 offline→online 微调对照（RLPD 核心路线）。

![Offline RL 两阶段路线](assets/offline-rl-pipeline.drawio.png)

![D4RL 结果](assets/d4rl_result.png)
![微调提升](assets/finetune.png)

| 结果 | 归一化分数 |
|---|---|
| CQL（复现） | **48.3**（论文 47.0） |
| Cal-QL（复现） | **48.6**（论文 ~47.7） |
| offline→online 微调 | CQL 48→57、Cal-QL 48→55（50k 在线步） |

- **alpha 坍缩修复**：d3rlpy CQL 保守损失后期变负 → alpha 自适应压至 0 → 保守失效（1.1 分）；读源码定位 `update_alpha` 修复为固定 alpha → 48.3
- **阴性结果（诚实报告）**：Cal-QL 论文宣称"微调更快"，但 dense-reward 数据上未体现（CQL 6826 vs Cal-QL 6484），适用边界在稀疏奖励/混合数据

---

## 项目 ④：步态相位估计 + 助力增益（Daphnet 数据集）

公开步态数据集 Daphnet（帕金森患者，踝部 64Hz 加速度计）实现步态事件检测与助力增益原型。

![步态相位](assets/daphnet_gait.png)
![助力增益](assets/assist_gain.png)

- **16/17 段稳定检出**（≥10 步态事件；唯一失败段为短行走/冻结前兆，仅 8 事件），成功段步频 60-98 步/分
- **助力增益曲线**（对应 ATD 自适应扭矩分配决策层）：支撑相（0-60%）助力、摆动相释放；硬开关 vs 平滑梯形，量化"78% 助力换掉阶跃冲击"的取舍

---

## 环境

- Windows 11 + RTX 4090 Laptop (16GB) + Intel Ultra 9 185H
- Isaac Sim 5.1.0 + Isaac Lab 2.3.x + RSL-RL 5.0.1（H1）；SB3 / d3rlpy（MuJoCo / D4RL）
- Python 3.11 venv，torch 2.7.0+cu128

## 复现

> 项目①（H1）训练依赖 Isaac Lab 完整环境（体积巨大），训练工程保留在本地；项目②③④的**训练 / 评估 / 导出 / 绘图脚本均在本仓库 `scripts/`**，数据为公开数据集或仓库内可生成。

**项目①（H1）**：
```bash
# 训练（需本地 Isaac Lab 环境，见下方「Windows 环境搭建笔记」；脚本在仓库 scripts/train_custom.py，H1 完整工程保留在本地）
python scripts/train_custom.py --mode rough --num_envs 2048 --max_iterations 3000
# 评估（本仓库 scripts/eval_h1_v2.py 可直接运行）
python scripts/eval_h1_v2.py --task rough --checkpoint <ckpt> --mode curve --levels 0,3,6,9,12,15,18,19
```

**项目②（MuJoCo Humanoid-v4，SB3）**：
```bash
python scripts/train.py             # PPO → ppo_humanoid.zip（2M 步）
python scripts/collect_demos.py     # PPO 教师采集示范 → expert_demos.npz（9913 条，仓库根目录已包含）
python scripts/train_bc.py          # BC → bc_humanoid.pt
python scripts/train_diffusion.py   # Diffusion Policy（DDPM 自实现）→ dp_humanoid.pt
python scripts/train_sac.py         # SAC → sac_humanoid.zip

# 评估（模型权重已包含在仓库根目录）
python scripts/eval_bc.py           # BC → 519.0
python scripts/eval_sac.py          # SAC → 6416.8
python scripts/eval_diffusion.py    # DP → 485.7
python scripts/eval_student.py      # 蒸馏学生（28 维）→ 534.4
```

**项目③（D4RL + d3rlpy）**：数据 halfcheetah-medium-v2（公开，HF 镜像 imone/D4RL 下载 hdf5 放入 `scripts/` 下）
```bash
python scripts/train_offline_cql.py    # CQL 复现 → results/offline_cql.md（48.3）
python scripts/train_offline_calql.py  # Cal-QL 复现 → results/offline_calql.md（48.6）
```

**项目④（Daphnet 步态）**：数据 Daphnet Freezing of Gait（UCI 公开，注册下载后放入 `scripts/data/dataset_fog_release/`）
```bash
python scripts/run_daphnet.py          # 步态事件检测 → daphnet_gait.png
```

**所有结果文档见 `results/`**（每个数字都能追溯）。

## Windows 环境搭建笔记（Isaac Lab 已知痛点 + 根因）

1. `import isaacsim` 需先接受 EULA：`export OMNI_KIT_ACCEPT_EULA=YES`
2. pip 默认装 torch CPU 版 → 必须手动装 cu128：`pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128`
3. **tensordict 版本必须匹配 torch**：最新版（0.13）针对 torch 2.9+ 编译，配 torch 2.7 会导入即段错误 → `pip install tensordict==0.8.0`
4. Isaac Lab 2.3 任务已改名：`Isaac-Velocity-Flat-Humanoid-v0` → `Isaac-Velocity-Flat-H1-v0`（Unitree H1）
5. 安装：`pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
6. **rsl_rl 5.x 的 obs 是 TensorDict**（`obs["policy"]`），直接 `obs[0]` 索引会 KeyError
7. **gymnasium OrderEnforcing**：step 前必须显式 `env.reset()`，且 reset 需在 `torch.inference_mode()` 内
8. **Isaac Sim 进程异常时 exit code 可能为 0**——判断成败要看日志里有没有 Traceback，不能只看退出码

## Debug 记录

### ① d3rlpy CQL 的 alpha 自适应坍缩（算法 bug）

- **现象**：CQL 训练后分数崩到 1.1（应为 ~48）
- **定位**：读 d3rlpy 源码，`update_alpha` 用**梯度上升**自动调 alpha（Lagrange 对偶）
- **机制**：保守损失后期变负 → alpha 压到 0 → 保守项完全失效
- **修复**：固定 alpha（`alpha_learning_rate=0`）
- **验证**：1.1 → 48.3，复现论文 47.0

### ② Isaac Sim Hydra 渲染崩溃（工程 bug，2026-08-06 实测）

**Isaac Sim demo 视频录制暂不可用**（渲染管线崩溃，与训练无关）——**已改用 MuJoCo 自带 rgb_array 离屏渲染录制，见 `assets/demo_sac.mp4`（注意：这是 MuJoCo Humanoid 走路视频，非 H1）**：
- **真实根因**（已用 crash_*.txt 崩溃栈定位）：Isaac Sim 的 **Hydra 渲染引擎创建**崩溃——headless 录视频崩在 `omni::usd::UsdManager::createHydraEngineWithConfig`
- 早期误判为 omni.warp 段错误（只看崩溃栈尾部）；已实测 **warp 本身健康** → 确认是渲染栈问题
- 训练/评估/导出全程 headless 不创建 Hydra 引擎，因此完全不受影响
- **修复方向**：更新 NVIDIA 驱动后重试；或换装其他 Isaac Sim 5.x 小版本

### ③ 步态图时间轴错位（数据对齐 bug）

- **现象**：daphnet_gait 图 heel strike 红星挤在 320-360s 一小段，assist_gain 面板 2 空、面板 3 助力信号贴 0
- **根因**：最长行走段的 t_seg 是**绝对时间戳**（320-360s），下游 `np.arange(0, t[-1])` 假设从 0 开始 → 时间轴错位
- **修复**：t_seg 归一化到 0 起点 + 裁切到行走段展示

## 作者

📧 jz233w@icloud.com　💬 微信/电话 18578213988
