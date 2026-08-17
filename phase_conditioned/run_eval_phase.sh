#!/bin/bash
# run_eval_phase.sh — phase 评估入口（K3 冲刺 D2）
# 用法: ./phase_conditioned/run_eval_phase.sh --mode steady --checkpoint <path> [更多参数]
cd ~/robot-rl-project/isaaclab || exit 1
unset PYTHONPATH
export PATH="/c/Users/jz233/robot-rl-project/env_isaaclab/Scripts:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=1

python -u phase_conditioned/eval_phase.py "$@"
