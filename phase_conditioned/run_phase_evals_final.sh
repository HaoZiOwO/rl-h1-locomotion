#!/bin/bash
# run_phase_evals_final.sh — 最终评估链（默认指令调度版）
# 变更：不再固定 1.0 m/s 命令（flat 策略在固定命令下行为异常，且官方 eval 的固定指令代码
#       在 isaaclab 0.54 属性名已变、静默回退随机指令）→ 受控对比 = 两边共用默认随机指令调度。
# 度量：指令跟踪误差 track_err = |obs[0:2] - obs[9:11]|（与训练奖励同口径）。
cd ~/robot-rl-project/isaaclab || exit 1
unset PYTHONPATH
export PATH="/c/Users/jz233/robot-rl-project/env_isaaclab/Scripts:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=1

PHASE_CKPT="logs/rsl_rl/h1_flat_phase/2026-08-14_19-50-04/model_999.pt"
FLAT_SEED42="logs/rsl_rl/h1_flat/2026-08-06_16-30-59/model_999.pt"

echo "===== 1/3 steady（默认指令，10 局）====="
python -u phase_conditioned/eval_phase.py --mode steady --checkpoint "$PHASE_CKPT" --num-episodes 10 > phase_eval_steady3.log 2>&1
echo "steady3: exit=$? traceback=$(grep -c Traceback phase_eval_steady3.log)"

echo "===== 2/3 jump（默认指令，8 局）====="
python -u phase_conditioned/eval_phase.py --mode jump --checkpoint "$PHASE_CKPT" --num-episodes 8 > phase_eval_jump3.log 2>&1
echo "jump3: exit=$? traceback=$(grep -c Traceback phase_eval_jump3.log)"

echo "===== 3/3 sweep（seed42 flat vs phase，各 10 局 × 4 幅度）====="
python -u phase_conditioned/eval_phase.py --mode sweep --checkpoint "$PHASE_CKPT" --baseline-checkpoint "$FLAT_SEED42" --num-episodes 10 > phase_eval_sweep4.log 2>&1
echo "sweep4: exit=$? traceback=$(grep -c Traceback phase_eval_sweep4.log)"

echo "===== FINAL CHAIN DONE ====="
