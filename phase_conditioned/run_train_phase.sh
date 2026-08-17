#!/bin/bash
# run_train_phase.sh — phase-conditioned H1 训练入口（K3 冲刺 D1）
# 用法: ./phase_conditioned/run_train_phase.sh [--smoke | train_phase.py 的任意参数]
cd ~/robot-rl-project/isaaclab || exit 1
unset PYTHONPATH
export PATH="/c/Users/jz233/robot-rl-project/env_isaaclab/Scripts:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=1

python -u phase_conditioned/train_phase.py "$@"
