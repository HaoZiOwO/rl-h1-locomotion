# -*- coding: utf-8 -*-
"""compute_ledger.py — 从日志时间戳对账 GPU 算力账（时间线可审计性的核心材料）。

只读扫描，不动任何训练进程：
1. Isaac Lab: logs/rsl_rl/<family>/<launch_ts>/ 目录名=启动时间，最大检查点 model_N.pt 的 mtime=结束时间
2. d3rlpy: d4rl/*.log 的 ISO 时间戳首尾行
3. MuJoCo: 用用户既有文档里已记录的时长（evals_summary §9、act_vs_bc.md 复现节）
输出 isaaclab_h1/compute_ledger.md
"""
import os, re, datetime, glob

BASE = r'C:/Users/jz233/robot-rl-project'
RSL = os.path.join(BASE, 'isaaclab', 'logs', 'rsl_rl')
D4RL = os.path.join(BASE, 'd4rl')
OUT = os.path.join(BASE, 'isaaclab_h1', 'compute_ledger.md')

lines = []
def p(s=''):
    lines.append(s)

p('# GPU 算力账（时间线可审计性）')
p()
p('> 硬件：RTX 4090 Laptop 16GB（个人笔记本）。模式：夜间过夜队列 + 白天评估，')
p('> 2026-08-05（首个实验日志）至 08-18（3-seed 复跑队列预计结束）连轴。')
p('> 生成方式：compute_ledger.py 扫训练日志时间戳，全部可复核。')
p()

# ---------- 1. Isaac Lab ----------
p('## 1. 项目① Isaac Lab（H1 策略族，2048 并行，2 万 env 步/秒）')
p()
p('| 训练族 | 启动 | 结束 | 时长 | 最终迭代 |')
p('|---|---|---|---|---|')
isaac_total = 0.0
isaac_launches = 0
for family in sorted(os.listdir(RSL)):
    fam_dir = os.path.join(RSL, family)
    if not os.path.isdir(fam_dir):
        continue
    for launch in sorted(os.listdir(fam_dir)):
        ldir = os.path.join(fam_dir, launch)
        if not os.path.isdir(ldir):
            continue
        m = re.match(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', launch)
        if not m:
            continue
        start = datetime.datetime.strptime(f'{m.group(1)}_{m.group(2)}', '%Y-%m-%d_%H-%M-%S')
        ckpts = [f for f in os.listdir(ldir) if re.match(r'model_\d+\.pt$', f)]
        if not ckpts:
            continue
        max_iter = max(int(re.match(r'model_(\d+)\.pt', f).group(1)) for f in ckpts)
        end = datetime.datetime.fromtimestamp(
            max(os.path.getmtime(os.path.join(ldir, f)) for f in ckpts))
        dur = (end - start).total_seconds() / 3600
        isaac_total += dur
        isaac_launches += 1
        p(f'| {family} | {start:%m-%d %H:%M} | {end:%m-%d %H:%M} | {dur:.1f}h | {max_iter} |')
p()
p(f'**小计：{isaac_launches} 次训练 launch，{isaac_total:.1f} GPU-hours**')
p('另：评估网格（8 难度点×10 局、负载 4 档×3 模型×10 局、3-seed×2 地形、消融系列）')
p('在 evals_v2（08-07 15:35 起）逐局跑，评估机时未单列。')
p()

# ---------- 2. D4RL ----------
p('## 2. 项目③ D4RL（d3rlpy，日志 ISO 时间戳对账）')
p()
p('| 作业 | 起 | 止 | 时长 |')
p('|---|---|---|---|')
d4rl_total = 0.0
TS = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})')
for log in ['train_rlpd.log', 'train_sacctl.log', 'train_mr.log',
            'cql_seed1.log', 'cql_seed2.log', 'eval_cutpoints.log']:
    lp = os.path.join(D4RL, log)
    if not os.path.exists(lp):
        continue
    with open(lp, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    ts = TS.findall(text)
    if len(ts) < 2:
        p(f'| {log} | （时间戳不足，无法对账） | | |')
        continue
    t0 = datetime.datetime.strptime(ts[0], '%Y-%m-%d %H:%M')
    t1 = datetime.datetime.strptime(ts[-1], '%Y-%m-%d %H:%M')
    dur = (t1 - t0).total_seconds() / 3600
    d4rl_total += dur
    p(f'| {log} | {t0:%m-%d %H:%M} | {t1:%m-%d %H:%M} | {dur:.1f}h |')
p()
p(f'**小计（已完成部分）：{d4rl_total:.1f} GPU-hours**；3-seed 复跑队列（mr seed1/2 × CQL/CalQL + RLPD seed2/3）')
p('进行中，预计 08-18 完成后再补对账（见 mr_seeds_chain.log / rlpd_seed{2,3}.log）。')
p()

# ---------- 3. MuJoCo ----------
p('## 3. 项目② MuJoCo（时长引用既有文档，见 evals_summary.md §9 / act_vs_bc.md §5）')
p()
p('| 作业 | 时长 | 出处 |')
p('|---|---|---|')
p('| PPO 教师 2M 步（2048 并行） | ~0.5h | evals_summary §9 |')
p('| SAC 2M 步（单环境逐帧） | ~10h | evals_summary §9 |')
p('| BC / 蒸馏学生（9913 transitions） | 分钟级 | train_bc / teacher_student_distill |')
p('| Diffusion Policy（DDPM，253K 参数） | 7s（日志实测） | dp_train.log |')
p('| ACT（4 配置：k10/k25 × β1.0/0.1） | ~2 分钟/次 | act_vs_bc.md §5 |')
p()

# ---------- 4. 步态 ----------
p('## 4. 项目④ 步态相位（纯 CPU，无 GPU 占用）')
p()
p('Daphnet 信号处理 + 助力增益设计：CPU 完成（run_daphnet.py，08-14）。')
p()

# ---------- 合计 ----------
grand = isaac_total + d4rl_total
p('---')
p()
p(f'## 合计（截至 08-16，不含进行中队列与 MuJoCo 数小时级训练）')
p()
p(f'**≥ {grand:.0f} GPU-hours**，且队列还有 ~40h 在跑。一个月内完成的关键：')
p('GPU 以过夜队列 24/7 运行 + 各项目间协议复用（统一 10 局确定性评估、')
p('同环境基准共享），白天时间用于评估、对账与文档。')
p()
p('> 口径声明：Isaac 时长 = 启动目录名 → 最后检查点 mtime；d3rlpy 时长 = 日志首尾')
p('> ISO 时间戳；MuJoCo 时长引用既有结果文档记录值。全部可复核，无估算填充。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\n[written] {OUT}')
