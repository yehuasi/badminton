# -*- coding: utf-8 -*-
"""单帧动作标准度打分（草案）。

STD_RANGES / ERROR_RULES 为第一层“定性要领”映射到 15 维特征后的占位区间，
正式数值待专业标准库样本（第二层）在相同特征空间统计后替换。
详见《标准库-技术要领.md》。
"""

# 占位区间：含义见《标准库-技术要领.md》。全部为草案值，不代表最终标准。
STD_RANGES = {
    "高远球": {
        "arm_raise": (140.0, 180.0),
        "elbow": (145.0, 180.0),
        "wrist_above_head": (0.0, 1.0),
    },
    "杀球": {
        "arm_raise": (145.0, 180.0),
        "elbow": (140.0, 180.0),
        "wrist_above_head": (0.1, 1.2),
        "com_ratio": (0.0, 0.3),
    },
    "吊球": {
        "arm_raise": (130.0, 180.0),
        "elbow": (125.0, 175.0),
        "wrist_above_head": (0.0, 0.9),
    },
    "挑球": {
        "arm_raise": (60.0, 140.0),
        "elbow": (110.0, 180.0),
        "wrist_above_shoulder": (-1.2, 0.5),
    },
    "搓球": {
        "arm_raise": (30.0, 100.0),
        "elbow": (100.0, 160.0),
        "wrist_reach": (0.4, 1.2),
    },
    "正手发高远球": {
        "arm_raise": (20.0, 90.0),
        "elbow": (130.0, 180.0),
        "wrist_above_shoulder": (-1.2, -0.1),
    },
    "反手发网前小球": {
        "arm_raise": (10.0, 60.0),
        "elbow": (80.0, 140.0),
        "wrist_above_shoulder": (-1.2, -0.2),
    },
}

ERROR_RULES = {
    "高远球": [
        ("elbow", "low", "疑似屈肘击球（肘角偏小）", 10),
        ("arm_raise", "low", "疑似击球点偏低（臂上抬不足）", 10),
        ("wrist_above_head", "low", "疑似击球点低于头顶", 10),
    ],
    "杀球": [
        ("elbow", "low", "疑似屈肘击球", 10),
        ("arm_raise", "low", "疑似击球点偏低", 10),
        ("com_ratio", "low", "疑似重心未跟进（击球点偏后）", 10),
    ],
    "吊球": [
        ("arm_raise", "low", "疑似击球点偏低", 10),
        ("wrist_above_head", "low", "疑似击球点低于头顶", 10),
    ],
    "挑球": [
        ("wrist_above_shoulder", "high", "疑似击球点过高（未低位引拍）", 10),
        ("elbow", "low", "疑似未跟送（肘部僵硬）", 8),
    ],
    "搓球": [
        ("wrist_reach", "low", "疑似离网太远（手臂未前伸）", 10),
        ("arm_raise", "high", "疑似动作过大（搓成挑球）", 8),
    ],
    "正手发高远球": [
        ("wrist_above_shoulder", "high", "疑似击球点过高（高于腰）", 10),
        ("elbow", "low", "疑似未充分随挥", 8),
    ],
    "反手发网前小球": [
        ("wrist_above_shoulder", "high", "疑似拍头高于腰（违例风险）", 10),
        ("arm_raise", "high", "疑似动作过大", 8),
    ],
}


def _checkpoint_score(value, low, high, tolerance=15.0):
    """单检查点打分：区间内满分 100，超出区间按比例扣分，超出 tolerance 记 0。"""
    if value is None:
        return 0.0
    if low <= value <= high:
        return 100.0
    if value < low:
        over = low - value
    else:
        over = value - high
    return max(0.0, 100.0 * (1.0 - over / tolerance))


def score_frame(kpts, action, handedness="right"):
    """单帧打分入口：关键点 -> 特征字典 -> 标准度得分与错误列表。"""
    from app.features import feature_dict

    return score_standardness(feature_dict(kpts, handedness=handedness), action=action)


def score_standardness(angles, action="高远球"):
    """返回 (得分, 命中的错误列表)。得分 0-100，仅动作标准度维度。"""
    ranges = STD_RANGES.get(action)
    if ranges is None:
        return 0.0, []

    scores = []
    for name, (lo, hi) in ranges.items():
        # 容差按区间宽度自适应：角度类区间宽、容差大；比值类区间窄、容差小。
        tolerance = max(5.0, (hi - lo) * 0.3)
        scores.append(_checkpoint_score(angles.get(name), lo, hi, tolerance=tolerance))
    base = sum(scores) / len(scores)

    errors = []
    penalty = 0
    for name, mode, message, weight in ERROR_RULES.get(action, []):
        value = angles.get(name)
        if value is None:
            continue
        lo, hi = ranges[name]
        hit = (mode == "low" and value < lo) or (mode == "high" and value > hi)
        if hit:
            errors.append(message)
            penalty += weight

    return max(0.0, base - penalty), errors
