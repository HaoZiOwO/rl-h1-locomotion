# -*- coding: utf-8 -*-
"""plot_sweep_survival.py — 周期外力存活率对比图（数据源：phase_closure.md §6 权威表，10 局/格）。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

A = [0, 50, 100, 200]
flat = [1.00, 1.00, 0.70, 0.20]
phase = [1.00, 1.00, 1.00, 0.60]

x = np.arange(len(A))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w/2, flat, w, label='Flat baseline (seed 42)', color='#9e9e9e', alpha=0.85)
b2 = ax.bar(x + w/2, phase, w, label='Phase-conditioned', color='#2e7d32', alpha=0.9)

for b in (b1, b2):
    for r in b:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.02,
                f'{r.get_height():.2f}', ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([f'A={a} N' for a in A])
ax.set_ylabel('Survival rate (10 episodes)')
ax.set_xlabel('Periodic force amplitude F(t)=A·sin(2π·1.4·t) on torso, resonant with gait clock')
ax.set_ylim(0, 1.2)
ax.set_title('Survival under periodic force: phase-conditioned H1 vs flat baseline\n'
             '(3× at A=200N: 0.60 vs 0.20 — directional signal, Fisher p≈0.09, not significant)', fontsize=11)
ax.legend(loc='lower left')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
out = r'C:/Users/jz233/robot-rl-project/isaaclab_h1/phase_conditioned/results/sweep_survival.png'
fig.savefig(out, dpi=150)
print('saved', out)
