# 算法对比：PPO vs BC vs Diffusion Policy（Humanoid-v4，同 demos、同评估协议）

| 算法 | 类型 | 预算 | 10 局平均奖励 |
|---|---|---|---|
| PPO | on-policy RL | 2M 步 | 508.3 ± 47.3 |
| BC | 监督学习（单步回归） | 9913 条示范 | 519.0 ± 51.1 |
| **Diffusion Policy** | 生成式模仿（DDPM） | 9913 条示范 | **485.7 ± 30.4** |

- 控制变量：同一份 expert_demos.npz、同归一化、同 80/20 划分、同评估协议（10 局 seed=42+ep 确定性）
- 扩散配置：T=100，β 线性 1e-4→0.02，去噪网络 256×3（比 BC 的 64×2 大——方法特性），100 epochs
- DP 逐局：[np.float64(462.1), np.float64(512.0), np.float64(464.2), np.float64(465.5), np.float64(492.9), np.float64(502.2), np.float64(442.5), np.float64(499.2), np.float64(464.9), np.float64(551.2)]
