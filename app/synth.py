# -*- coding: utf-8 -*-
"""合成演示源：按 7 类动作生成特征轨迹，并渲染简易火柴人画面。

用于在没有摄像头/视频时演示整条“检测->识别->打分->建议”链路。
特征生成逻辑与 scripts/make_synthetic_dataset.py 保持一致。
"""

import math

import cv2
import numpy as np

from app.classes import SHOT_CLASSES
from app.features import FEATURE_NAMES, F


def _vec(**overrides):
    # 中性准备姿态（顺序见 FEATURE_NAMES）
    ready = np.array([
        25.0, 150.0, 170.0, 165.0, 6.0, 88.0, 0.02,
        -0.70, 0.35, -1.10, 0.06, 0.55, 145.0, 22.0, 170.0,
    ], dtype=np.float64)
    for name, value in overrides.items():
        ready[FEATURE_NAMES.index(name)] = value
    return ready


PROFILES = {
    "高远球": (
        _vec(arm_raise=75, elbow=125, wrist_above_shoulder=-0.3, shoulder_turn=62, trunk_lean=10),
        _vec(arm_raise=156, elbow=172, wrist_above_shoulder=1.25, wrist_reach=1.15,
             wrist_above_head=0.35, shoulder_turn=40, trunk_lean=8, knee=150, hip=145),
        _vec(arm_raise=85, elbow=145, wrist_above_shoulder=0.55, shoulder_turn=18, trunk_lean=5),
    ),
    "杀球": (
        _vec(arm_raise=90, elbow=100, wrist_above_shoulder=0.0, shoulder_turn=55, knee=145, trunk_lean=20),
        _vec(arm_raise=172, elbow=160, wrist_above_shoulder=1.55, wrist_above_head=0.65,
             wrist_reach=1.25, shoulder_turn=32, trunk_lean=26, knee=130, hip=120, com_ratio=0.18),
        _vec(arm_raise=70, elbow=130, wrist_above_shoulder=0.6, shoulder_turn=12, trunk_lean=22, knee=150),
    ),
    "吊球": (
        _vec(arm_raise=72, elbow=130, wrist_above_shoulder=-0.25, shoulder_turn=65, trunk_lean=9),
        _vec(arm_raise=148, elbow=142, wrist_above_shoulder=1.05, wrist_above_head=0.28,
             wrist_reach=1.0, shoulder_turn=42, trunk_lean=11, knee=152, hip=148),
        _vec(arm_raise=78, elbow=148, wrist_above_shoulder=0.45, shoulder_turn=22),
    ),
    "挑球": (
        _vec(arm_raise=40, elbow=120, wrist_above_shoulder=-1.05, wrist_above_head=-1.55,
             shoulder_turn=70, knee=135, trunk_lean=14),
        _vec(arm_raise=138, elbow=118, wrist_above_shoulder=0.75, wrist_above_head=-0.25,
             wrist_reach=0.7, shoulder_turn=48, knee=128, trunk_lean=12),
        _vec(arm_raise=60, elbow=150, wrist_above_shoulder=0.2, shoulder_turn=30, knee=150),
    ),
    "搓球": (
        _vec(arm_raise=55, elbow=135, wrist_above_shoulder=-0.45, shoulder_turn=78, trunk_lean=12, knee=155),
        _vec(arm_raise=78, elbow=132, wrist_above_shoulder=0.05, wrist_above_head=0.05,
             wrist_reach=0.75, shoulder_turn=68, trunk_lean=16, knee=150, com_ratio=0.10),
        _vec(arm_raise=60, elbow=145, wrist_above_shoulder=-0.1, shoulder_turn=74, trunk_lean=14),
    ),
    "正手发高远球": (
        _vec(arm_raise=30, elbow=150, wrist_above_shoulder=-1.15, wrist_above_head=-1.7,
             shoulder_turn=52, trunk_lean=12, knee=150),
        _vec(arm_raise=150, elbow=165, wrist_above_shoulder=1.0, wrist_reach=0.85,
             wrist_above_head=0.0, shoulder_turn=58, trunk_lean=7, knee=140),
        _vec(arm_raise=95, elbow=150, wrist_above_shoulder=0.5, shoulder_turn=34, trunk_lean=4),
    ),
    "反手发网前小球": (
        _vec(arm_raise=20, elbow=95, wrist_above_shoulder=-0.35, wrist_reach=0.55,
             wrist_above_head=-0.9, shoulder_turn=82, trunk_lean=9, knee=162),
        _vec(arm_raise=28, elbow=108, wrist_above_shoulder=-0.15, wrist_reach=0.7,
             wrist_above_head=-0.75, shoulder_turn=84, trunk_lean=10, knee=160),
        _vec(arm_raise=25, elbow=120, wrist_above_shoulder=-0.25, wrist_reach=0.6,
             shoulder_turn=84, trunk_lean=9),
    ),
}


def _smooth(t, t0, t1, v0, v1):
    if t1 <= t0:
        return v1
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    s = u * u * (3.0 - 2.0 * u)
    return v0 + (v1 - v0) * s


def generate_sequence(cls_id, T=24, rng=None):
    """生成 (T, F) 特征序列（含随机节奏与噪声），与训练数据同分布。"""
    rng = rng or np.random.default_rng()
    windup, contact, follow = PROFILES[SHOT_CLASSES[cls_id]]
    t_contact = int(T * rng.uniform(0.45, 0.65))
    t_windup_end = max(1, int(t_contact * 0.55))
    seq = np.zeros((T, F), dtype=np.float64)
    for t in range(T):
        if t <= t_windup_end:
            seq[t] = _smooth(t, 0, t_windup_end, windup, contact)
        else:
            seq[t] = _smooth(t, t_windup_end, T - 1, contact, follow)
    # 与训练数据的逐帧噪声一致：角度类大、比值类小。
    noise_scale = np.array([3.0] * 6 + [0.04] + [0.06] * 4 + [0.04] + [3.0] * 3)
    seq += rng.normal(0.0, noise_scale, size=(T, F))
    return seq.astype(np.float32)


def render_stick(features, cls_name=""):
    """把一帧 15 维特征画成简易火柴人（640x480 BGR）。"""
    img = np.full((480, 640, 3), 32, dtype=np.uint8)
    f = np.asarray(features, dtype=np.float64)
    arm_raise = float(f[0]) if not math.isnan(f[0]) else 25.0
    elbow = float(f[1]) if not math.isnan(f[1]) else 150.0
    trunk = float(f[4]) if not math.isnan(f[4]) else 6.0
    knee = float(f[2]) if not math.isnan(f[2]) else 170.0

    cx = 320
    lean = int((trunk / 90.0) * 42)
    shoulder_mid = (cx + lean, 285)
    hip_mid = (cx, 392)

    head = (shoulder_mid[0] + 12, shoulder_mid[1] - 48)
    sh_l = (cx - 30, 292)
    sh_r = (cx + 30, 292)
    hip_l = (cx - 24, 392)
    hip_r = (cx + 24, 392)

    def _leg(hip_x):
        fwd = int((180.0 - knee) / 90.0 * 22)
        kn = (hip_x + fwd, 450)
        an = (hip_x + fwd + 4, 508)
        return kn, an

    kn_l, an_l = _leg(hip_l[0])
    kn_r, an_r = _leg(hip_r[0])

    # 主臂（右侧）：arm_raise 为从竖直向下起算的上抬角，elbow 为肘角。
    phi = math.radians(arm_raise)
    L1, L2 = 58.0, 58.0
    el = (sh_r[0] + L1 * math.sin(phi), sh_r[1] + L1 * math.cos(phi))
    bend = math.radians(180.0 - elbow)
    wr = (el[0] + L2 * math.sin(phi + bend), el[1] + L2 * math.cos(phi + bend))

    # 非主臂（左侧）：自然下垂微屈。
    el2 = (sh_l[0] - 14, sh_l[1] + 62)
    wr2 = (sh_l[0] - 22, sh_l[1] + 118)

    white = (220, 220, 220)
    green = (90, 220, 120)
    for a, b in [
        (head, shoulder_mid), (shoulder_mid, hip_mid), (sh_l, sh_r),
        (hip_l, hip_r), (sh_l, el2), (el2, wr2),
        (sh_r, (int(el[0]), int(el[1]))), ((int(el[0]), int(el[1])), (int(wr[0]), int(wr[1]))),
        (hip_l, kn_l), (kn_l, an_l), (hip_r, kn_r), (kn_r, an_r),
    ]:
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), white, 3, cv2.LINE_AA)
    for p in [head, sh_l, sh_r, el2, wr2, hip_l, hip_r, kn_l, an_l, kn_r, an_r,
              (int(el[0]), int(el[1])), (int(wr[0]), int(wr[1]))]:
        cv2.circle(img, tuple(map(int, p)), 5, green, -1, cv2.LINE_AA)

    if cls_name:
        cv2.putText(img, cls_name, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 220, 255), 2)
    return img

