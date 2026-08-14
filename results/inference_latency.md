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
