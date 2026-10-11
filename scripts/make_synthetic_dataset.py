# -*- coding: utf-8 -*-
"""生成仿真特征序列数据集，用于在真实数据到位前打通训练链路。

这不是真实运动员数据，而是按 7 类击球动作的关节角度/相对位置运动规律，
生成“准备 -> 击球 -> 随挥”三阶段特征轨迹，并叠加个体差异与噪声。
用途：验证 数据生成 -> 训练 -> 评估 -> 导出 整条链路可用。
真实数据（BFMD 或自采）到位后，用 prepare_bfmd.py / 自采脚本替换即可。
"""

import argparse
import csv
import os

import numpy as np

from common import PROCESSED_DIR, REPO_ROOT, save_dataset
from app.classes import SHOT_CLASSES
from app.features import FEATURE_NAMES, F


# 中性准备姿态（15 维，顺序见 FEATURE_NAMES）。
READY = np.array([
    25.0,   # arm_raise
    150.0,  # elbow
    170.0,  # knee
    165.0,  # hip
    6.0,    # trunk_lean
    88.0,   # shoulder_turn（90 度为双肩水平）
    0.02,   # com_ratio
    -0.70,  # wrist_above_shoulder
    0.35,   # wrist_reach
    -1.10,  # wrist_above_head
    0.06,   # head_forward
    0.55,   # ankle_width
    145.0,  # non_dom_elbow
    22.0,   # non_dom_arm_raise
    170.0,  # non_dom_knee
], dtype=np.float64)


def vec(**overrides):
    out = READY.copy()
    for name, value in overrides.items():
        out[FEATURE_NAMES.index(name)] = value
    return out


# 每类动作的“引拍 / 击球 / 随挥”三阶段特征（只写与 READY 不同的项）。
PROFILES = {
    "高远球": (
        vec(arm_raise=75, elbow=125, wrist_above_shoulder=-0.3, shoulder_turn=62, trunk_lean=10),
        vec(arm_raise=156, elbow=172, wrist_above_shoulder=1.25, wrist_reach=1.15,
            wrist_above_head=0.35, shoulder_turn=40, trunk_lean=8, knee=150, hip=145),
        vec(arm_raise=85, elbow=145, wrist_above_shoulder=0.55, shoulder_turn=18, trunk_lean=5),
    ),
    "杀球": (
        vec(arm_raise=90, elbow=100, wrist_above_shoulder=0.0, shoulder_turn=55, knee=145, trunk_lean=20),
        vec(arm_raise=172, elbow=160, wrist_above_shoulder=1.55, wrist_above_head=0.65,
            wrist_reach=1.25, shoulder_turn=32, trunk_lean=26, knee=130, hip=120, com_ratio=0.18),
        vec(arm_raise=70, elbow=130, wrist_above_shoulder=0.6, shoulder_turn=12, trunk_lean=22, knee=150),
    ),
    "吊球": (
        vec(arm_raise=72, elbow=130, wrist_above_shoulder=-0.25, shoulder_turn=65, trunk_lean=9),
        vec(arm_raise=148, elbow=142, wrist_above_shoulder=1.05, wrist_above_head=0.28,
            wrist_reach=1.0, shoulder_turn=42, trunk_lean=11, knee=152, hip=148),
        vec(arm_raise=78, elbow=148, wrist_above_shoulder=0.45, shoulder_turn=22),
    ),
    "挑球": (
        vec(arm_raise=40, elbow=120, wrist_above_shoulder=-1.05, wrist_above_head=-1.55,
            shoulder_turn=70, knee=135, trunk_lean=14),
        vec(arm_raise=138, elbow=118, wrist_above_shoulder=0.75, wrist_above_head=-0.25,
            wrist_reach=0.7, shoulder_turn=48, knee=128, trunk_lean=12),
        vec(arm_raise=60, elbow=150, wrist_above_shoulder=0.2, shoulder_turn=30, knee=150),
    ),
    "搓球": (
        vec(arm_raise=55, elbow=135, wrist_above_shoulder=-0.45, shoulder_turn=78, trunk_lean=12, knee=155),
        vec(arm_raise=78, elbow=132, wrist_above_shoulder=0.05, wrist_above_head=0.05,
            wrist_reach=0.75, shoulder_turn=68, trunk_lean=16, knee=150, com_ratio=0.10),
        vec(arm_raise=60, elbow=145, wrist_above_shoulder=-0.1, shoulder_turn=74, trunk_lean=14),
    ),
    "正手发高远球": (
        vec(arm_raise=30, elbow=150, wrist_above_shoulder=-1.15, wrist_above_head=-1.7,
            shoulder_turn=52, trunk_lean=12, knee=150),
        vec(arm_raise=150, elbow=165, wrist_above_shoulder=1.0, wrist_reach=0.85,
            wrist_above_head=0.0, shoulder_turn=58, trunk_lean=7, knee=140),
        vec(arm_raise=95, elbow=150, wrist_above_shoulder=0.5, shoulder_turn=34, trunk_lean=4),
    ),
    "反手发网前小球": (
        vec(arm_raise=20, elbow=95, wrist_above_shoulder=-0.35, wrist_reach=0.55,
            wrist_above_head=-0.9, shoulder_turn=82, trunk_lean=9, knee=162),
        vec(arm_raise=28, elbow=108, wrist_above_shoulder=-0.15, wrist_reach=0.7,
            wrist_above_head=-0.75, shoulder_turn=84, trunk_lean=10, knee=160),
        vec(arm_raise=25, elbow=120, wrist_above_shoulder=-0.25, wrist_reach=0.6,
            shoulder_turn=84, trunk_lean=9),
    ),
}


def _interp_phase(t, t0, t1, v0, v1):
    """在 [t0, t1] 之间用平滑 s 曲线从 v0 过渡到 v1。"""
    if t1 <= t0:
        return v1
    u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    s = u * u * (3.0 - 2.0 * u)  # smoothstep
    return v0 + (v1 - v0) * s


def _one_sample(windup, contact, follow, T, rng):
    # 随机化击球时刻（在 45% ~ 65% 区间），产生个体节奏差异。
    t_contact = int(T * rng.uniform(0.45, 0.65))
    t_windup_end = max(1, int(t_contact * 0.55))
    seq = np.zeros((T, F), dtype=np.float64)
    for t in range(T):
        if t <= t_windup_end:
            seq[t] = _interp_phase(t, 0, t_windup_end, windup, contact)
        else:
            seq[t] = _interp_phase(t, t_windup_end, T - 1, contact, follow)
    # 个体差异：每个样本一个缓变偏移。
    person_offset = rng.normal(0.0, 1.5, size=(F,))
    person_offset[:7] *= 3.0  # 角度类特征差异更大
    seq = seq + person_offset
    # 逐帧观测噪声。
    noise_scale = np.array([3.0] * 6 + [0.04] + [0.06] * 4 + [0.04] + [3.0] * 3)
    seq = seq + rng.normal(0.0, noise_scale, size=(T, F))
    return seq.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=200)
    parser.add_argument("--T", type=int, default=24, help="每个样本的帧数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=PROCESSED_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y, rows = [], [], []
    for cls_id, cls_name in enumerate(SHOT_CLASSES):
        windup, contact, follow = PROFILES[cls_name]
        for i in range(args.n_per_class):
            X.append(_one_sample(windup, contact, follow, args.T, rng))
            y.append(cls_id)
            rows.append([f"syn_{cls_id:02d}_{i:04d}", cls_name, "synthetic"])

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    save_dataset(args.out, X, y, SHOT_CLASSES, manifest_rows=rows)
    print(f"已生成仿真数据集：{X.shape} -> {args.out}")


if __name__ == "__main__":
    main()

