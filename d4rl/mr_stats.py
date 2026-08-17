# -*- coding: utf-8 -*-
"""mr_stats.py — medium-replay 3-seed 统计（原跑 + seed1 + seed2，每 5k 步 10 局确定性）。
输出：每评估点 3-seed mean±std、Cal-QL 领先计数、50k 终值统计。
"""
import csv, os
import numpy as np

BASE = r'C:/Users/jz233/robot-rl-project/d4rl/d3rlpy_logs'
RUNS = {
    'CQL': [
        'mr_CQL_finetune_20260815152354',      # 原跑（隐式种子）
        'mr_seed1_CQL_finetune_20260816010014',
        'mr_seed2_CQL_finetune_20260816082700',
    ],
    'CalQL': [
        'mr_CalQL_finetune_20260815185431',    # 原跑（隐式种子）
        'mr_seed1_CalQL_finetune_20260816051516',
        'mr_seed2_CalQL_finetune_20260816121207',
    ],
}

def load(d):
    pts = {}
    with open(os.path.join(BASE, d, 'evaluation.csv'), newline='') as f:
        for row in csv.reader(f):
            epoch, step, val = int(row[0]), int(row[1]), float(row[2])
            pts[step] = val
    return pts

curves = {a: [load(d) for d in ds] for a, ds in RUNS.items()}
steps = sorted(curves['CQL'][0].keys())

print(f"{'step':>6} | {'CQL mean±std':>16} | {'CalQL mean±std':>17} | CalQL leads (n/3)")
for s in steps:
    c = [r[s] for r in curves['CQL']]
    q = [r[s] for r in curves['CalQL']]
    cm, cs = float(np.mean(c)), float(np.std(c, ddof=1))
    qm, qs = float(np.mean(q)), float(np.std(q, ddof=1))
    leads = sum(1 for a, b in zip(q, c) if b > a)
    print(f"{s:>6} | {cm:>8.1f}±{cs:<6.1f} | {qm:>8.1f}±{qs:<6.1f} | {leads}/3")

# 50k 终值
c50 = [r[steps[-1]] for r in curves['CQL']]
q50 = [r[steps[-1]] for r in curves['CalQL']]
print(f"\n50k final: CQL {np.mean(c50):.1f}±{np.std(c50, ddof=1):.1f}  CalQL {np.mean(q50):.1f}±{np.std(q50, ddof=1):.1f}")
print(f"50k per-seed: CQL {[f'{v:.0f}' for v in c50]}  CalQL {[f'{v:.0f}' for v in q50]}")
diff = [b - a for a, b in zip(c50, q50)]
print(f"50k CalQL-CQL per seed: {[f'{v:+.0f}' for v in diff]}  mean {np.mean(diff):+.0f}")
# 全曲线逐点平均差（per-seed 均值差，先逐 seed 算曲线差再平均）
avg_lead = np.mean([np.mean([q[s] - c[s] for s in steps]) for q, c in zip(curves['CalQL'], curves['CQL'])])
print(f"avg per-seed curve diff (CalQL-CQL): {avg_lead:+.1f}")
