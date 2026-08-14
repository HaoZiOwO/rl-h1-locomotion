# ONNX 推理延迟基准

- 工具: onnxruntime 1.28.0（Python API，C 内核）
- 硬件: RTX 4090 笔记本（CPU: 本机 CPU / GPU: CUDA EP）
- 方法: 50 次预热 + 2000 次计时，随机输入
- 控制环预算对照: 100Hz→10ms/步, 500Hz→2ms/步, 1kHz→1ms/步

| 策略 | 推理后端 | mean (ms) | p95 (ms) | max (ms) |
|---|---|---|---|---|
| flat | CPUExecutionProvider | 0.017 | 0.020 | 0.363 |
| rough | CPUExecutionProvider | 0.037 | 0.054 | 0.972 |

## 解读

- 若 CPU mean < 10ms：满足 100Hz 控制环；< 2ms 则满足 500Hz。
- CUDA 对小模型可能有启动开销（每次调用 kernel launch + H2D/D2H 拷贝），如实报告。

## ONNX 数值一致性校验（2026-08-14）

- 方法：同一随机观测输入（seed=0，batch=1），对比 torch.jit `policy.pt` 与 onnxruntime `policy.onnx` 输出（64 位 CPU）
- flat: obs_dim=69, MSE=3.58e-14, max_err=4.77e-07
- rough: obs_dim=256, MSE=5.45e-13, max_err=1.43e-06
- 结论：JIT 与 ONNX 输出一致（浮点误差级），导出无算子精度损失，可安全部署（校验逻辑已内置 `scripts/export_h1.py`）
