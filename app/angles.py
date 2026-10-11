# -*- coding: utf-8 -*-
"""从 YOLOv8-Pose 的 17 个 COCO 关键点计算评分所需关节角度。"""

import numpy as np

# COCO 17 关键点索引
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def _angle_at(a, b, c):
    """返回 b 点处、向量 ba 与 bc 之间的夹角（度）。"""
    ba = np.asarray(a[:2], dtype=float) - np.asarray(b[:2], dtype=float)
    bc = np.asarray(c[:2], dtype=float) - np.asarray(b[:2], dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cos = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _angle_with_vertical(a, b, upward=True):
    """返回向量 a->b 与竖直方向的夹角（度）；upward=True 表示相对竖直向上。"""
    v = np.asarray(b[:2], dtype=float) - np.asarray(a[:2], dtype=float)
    ref = np.array([0.0, -1.0]) if upward else np.array([0.0, 1.0])
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return None
    cos = float(np.dot(v, ref) / norm)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compute_angles(kpts, handedness="right"):
    """kpts 为 (17,3) 数组 (x, y, conf)，返回关键角度字典。

    指标说明（对应《评分规则设计》第 3 节）：
    - arm_raise：上臂抬角，躯干与上臂夹角，越大表示手臂抬得越高。
    - elbow：肘角，肩-肘-腕夹角。
    - trunk_lean：躯干倾角，躯干相对竖直向上的偏离。
    - shoulder_turn：转体角，双肩连线相对水平方向的偏离（近似）。
    - knee：膝角，髋-膝-踝夹角。
    - hip：髋角，肩-髋-膝夹角。
    - com_ratio：重心投影，髋中点相对踝中点的水平偏移（占身高比例）。
    """
    kpts = np.asarray(kpts, dtype=float)
    if handedness == "left":
        sh, el, wr = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        hip, kn, an = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
    else:
        sh, el, wr = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        hip, kn, an = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE

    shoulder_mid = (kpts[LEFT_SHOULDER, :2] + kpts[RIGHT_SHOULDER, :2]) / 2.0
    hip_mid = (kpts[LEFT_HIP, :2] + kpts[RIGHT_HIP, :2]) / 2.0
    ankle_mid = (kpts[LEFT_ANKLE, :2] + kpts[RIGHT_ANKLE, :2]) / 2.0

    elbow = _angle_at(kpts[sh], kpts[el], kpts[wr])
    arm_raise = _angle_at(kpts[hip], kpts[sh], kpts[el])
    knee = _angle_at(kpts[hip], kpts[kn], kpts[an])
    hip_angle = _angle_at(kpts[sh], kpts[hip], kpts[kn])
    trunk_lean = _angle_with_vertical(hip_mid, shoulder_mid, upward=True)
    shoulder_turn = _angle_with_vertical(kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER], upward=False)

    height = float(np.linalg.norm(shoulder_mid - ankle_mid))
    com_ratio = None
    if height > 1e-6:
        com_ratio = float((hip_mid[0] - ankle_mid[0]) / height)

    return {
        "arm_raise": arm_raise,
        "elbow": elbow,
        "trunk_lean": trunk_lean,
        "shoulder_turn": shoulder_turn,
        "knee": knee,
        "hip": hip_angle,
        "com_ratio": com_ratio,
    }
