# -*- coding: utf-8 -*-
"""从 YOLOv8-Pose 的 17 个 COCO 关键点提取尺度/位置不变的动作特征向量。

这些特征既用于“击球类型识别”，也用于“动作标准度打分”：
  - 关节角度类特征对相机远近不敏感；
  - 相对位置类特征用肩宽（或躯干长）归一化，对身高不敏感。
"""

import numpy as np

from app.angles import (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    _angle_at,
    _angle_with_vertical,
)

HEAD = 0

# 特征向量维度与顺序（顺序固定，模型输入依赖它）。
FEATURE_NAMES = [
    "arm_raise",            # 0  持拍臂上抬角（躯干-肩-肘）
    "elbow",                # 1  持拍臂肘角
    "knee",                 # 2  持拍侧膝角
    "hip",                  # 3  持拍侧髋角
    "trunk_lean",           # 4  躯干相对竖直方向的前倾角
    "shoulder_turn",        # 5  双肩连线的倾斜角（近似转体）
    "com_ratio",            # 6  重心水平偏移（髋中点相对踝中点的偏移 / 身高）
    "wrist_above_shoulder", # 7  (肩高 - 腕高) / scale
    "wrist_reach",          # 8  |腕x - 肩x| / scale
    "wrist_above_head",     # 9  (头顶 - 腕高) / scale
    "head_forward",         # 10 (头x - 髋中点x) / scale
    "ankle_width",          # 11 两踝间距 / scale
    "non_dom_elbow",        # 12 非持拍臂肘角
    "non_dom_arm_raise",    # 13 非持拍臂上抬角
    "non_dom_knee",         # 14 非持拍侧膝角
]

F = len(FEATURE_NAMES)

# 时间导数（速度/加速度）特征的顺序，附加在 15 维静态特征之后。
# 速度与加速度用于区分“挥拍快慢”（如高远球 vs 吊球），
# 对挥拍手（持拍腕）在垂直/前伸/过顶三个方向上的运动 + 上臂上抬角做差分。
KINEMATIC_NAMES = [
    "wrist_v_vertical",   # d(wrist_above_shoulder)/dt
    "wrist_v_reach",      # d(wrist_reach)/dt
    "wrist_v_above_head", # d(wrist_above_head)/dt
    "arm_raise_v",        # d(arm_raise)/dt
    "wrist_a_vertical",   # d2(wrist_above_shoulder)
    "wrist_a_reach",      # d2(wrist_reach)
    "wrist_a_above_head", # d2(wrist_above_head)
    "arm_raise_a",        # d2(arm_raise)
]

# 参与差分计算的静态特征下标：wrist_above_shoulder=7, wrist_reach=8,
# wrist_above_head=9, arm_raise=0
KINEMATIC_IDX = (7, 8, 9, 0)
FULL_F = F + len(KINEMATIC_NAMES)


def add_kinematics(static_seq):
    """在 (T, F) 静态特征序列上追加速度与加速度特征，返回 (T, FULL_F)。

    用 np.gradient（二阶中心差分）沿时间轴求导，输入不应含 NaN
    （调用方应先做插值清洗，见 scripts/prepare_bfmd.py 的 _clean_sequence）。
    """
    arr = np.asarray(static_seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != F:
        raise ValueError(f"static_seq 应为 (T, {F})，实际 {arr.shape}")
    vel = np.gradient(arr, axis=0)
    acc = np.gradient(vel, axis=0)
    extra = np.concatenate([vel[:, KINEMATIC_IDX], acc[:, KINEMATIC_IDX]], axis=1)
    return np.concatenate([arr, extra], axis=1).astype(np.float32)


def _scale_from(kpts):
    """用肩宽作为尺度基准；肩宽不可用时退化为躯干长度，再退化到 1。"""
    shoulder_width = float(
        np.linalg.norm(kpts[RIGHT_SHOULDER, :2] - kpts[LEFT_SHOULDER, :2])
    )
    if shoulder_width > 1e-6:
        return shoulder_width
    hip_mid = (kpts[LEFT_HIP, :2] + kpts[RIGHT_HIP, :2]) / 2.0
    shoulder_mid = (kpts[LEFT_SHOULDER, :2] + kpts[RIGHT_SHOULDER, :2]) / 2.0
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    return torso if torso > 1e-6 else 1.0


def compute_features(kpts, handedness="right"):
    """把 (17,3) 关键点转成形状为 (F,) 的 float32 特征向量。

    缺失关键点（置信度 0）会被置为 NaN，调用方自行处理（训练时丢弃或插值）。
    """
    kpts = np.asarray(kpts, dtype=float)
    if handedness == "left":
        sh, el, wr = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        hip, kn, an = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        nsh, nel, nkn = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_KNEE
    else:
        sh, el, wr = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        hip, kn, an = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        nsh, nel, nkn = LEFT_SHOULDER, LEFT_ELBOW, LEFT_KNEE

    shoulder_mid = (kpts[LEFT_SHOULDER, :2] + kpts[RIGHT_SHOULDER, :2]) / 2.0
    hip_mid = (kpts[LEFT_HIP, :2] + kpts[RIGHT_HIP, :2]) / 2.0
    ankle_mid = (kpts[LEFT_ANKLE, :2] + kpts[RIGHT_ANKLE, :2]) / 2.0

    scale = _scale_from(kpts)
    height = float(np.linalg.norm(shoulder_mid - ankle_mid))

    arm_raise = _angle_at(kpts[hip], kpts[sh], kpts[el])
    elbow = _angle_at(kpts[sh], kpts[el], kpts[wr])
    knee = _angle_at(kpts[hip], kpts[kn], kpts[an])
    hip_angle = _angle_at(kpts[sh], kpts[hip], kpts[kn])
    trunk_lean = _angle_with_vertical(hip_mid, shoulder_mid, upward=True)
    shoulder_turn = _angle_with_vertical(
        kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER], upward=False
    )
    non_dom_elbow = _angle_at(kpts[nsh], kpts[nel], kpts[RIGHT_WRIST if handedness == "left" else LEFT_WRIST])
    non_dom_arm_raise = _angle_at(kpts[RIGHT_HIP if handedness == "left" else LEFT_HIP], kpts[nsh], kpts[nel])
    non_dom_knee = _angle_at(
        kpts[RIGHT_HIP if handedness == "left" else LEFT_HIP],
        kpts[nkn],
        kpts[RIGHT_ANKLE if handedness == "left" else LEFT_ANKLE],
    )

    com_ratio = None
    if height > 1e-6:
        com_ratio = float((hip_mid[0] - ankle_mid[0]) / height)

    feat = np.full((F,), np.nan, dtype=np.float32)
    feat[0] = arm_raise if arm_raise is not None else np.nan
    feat[1] = elbow if elbow is not None else np.nan
    feat[2] = knee if knee is not None else np.nan
    feat[3] = hip_angle if hip_angle is not None else np.nan
    feat[4] = trunk_lean if trunk_lean is not None else np.nan
    feat[5] = shoulder_turn if shoulder_turn is not None else np.nan
    feat[6] = com_ratio if com_ratio is not None else np.nan
    feat[7] = float((kpts[sh, 1] - kpts[wr, 1]) / scale)
    feat[8] = float(abs(kpts[wr, 0] - kpts[sh, 0]) / scale)
    feat[9] = float((kpts[HEAD, 1] - kpts[wr, 1]) / scale)
    feat[10] = float((kpts[HEAD, 0] - hip_mid[0]) / scale)
    feat[11] = float(abs(kpts[RIGHT_ANKLE, 0] - kpts[LEFT_ANKLE, 0]) / scale)
    feat[12] = non_dom_elbow if non_dom_elbow is not None else np.nan
    feat[13] = non_dom_arm_raise if non_dom_arm_raise is not None else np.nan
    feat[14] = non_dom_knee if non_dom_knee is not None else np.nan
    return feat


def feature_dict(kpts, handedness="right"):
    """把关键点转成 {特征名: 值} 字典，便于评分规则按名取用。"""
    feat = compute_features(kpts, handedness=handedness)
    return {
        name: (None if np.isnan(feat[i]) else float(feat[i]))
        for i, name in enumerate(FEATURE_NAMES)
    }

