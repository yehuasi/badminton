# -*- coding: utf-8 -*-
"""单帧动作标准度打分。

STD_RANGES / ERROR_RULES 把动作要领映射到 24 维特征（15 维基础 + 9 维扩展）后的区间。
区间来源分两层：
  - 第一层：15 维基础特征，结合《标准库-技术要领.md》的定性要领与占位值；
  - 第二层：用自采标准集（233 条正确动作）统计的分位区间回填；扩展特征（肩髋跨度、
    前臂指向、腕肘有符号位移等）的区间完全来自标准集 p10~p90，对应《自采数据训练基线.md》。

说明：扩展特征对应的错误规则样本量较小（每族 3~8 条），属于数据初标，已用较低权重；
待自拍数据到位后统一复核。
"""

# 占位区间：含义见《标准库-技术要领.md》。全部为草案值，不代表最终标准。
STD_RANGES = {
    "高远球": {
        "arm_raise": (140.0, 180.0),
        "elbow": (145.0, 180.0),
        "wrist_above_head": (0.0, 1.0),
        "shoulder_hip_ratio": (1.0, 2.1),
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
        "wrist_x_signed": (-0.26, 0.13),
        "elbow_x_signed": (-0.24, 0.10),
    },
    "挑球": {
        "arm_raise": (60.0, 140.0),
        "elbow": (110.0, 180.0),
        "wrist_above_shoulder": (-1.2, 0.5),
        "forearm_angle": (43.0, 116.0),
    },
    "搓球": {
        "arm_raise": (30.0, 100.0),
        "elbow": (100.0, 160.0),
        "wrist_reach": (0.4, 1.2),
        "wrist_x_signed": (-0.48, 0.11),
        "elbow_x_signed": (-0.33, 0.08),
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
        "elbow_x_signed": (-0.25, 0.08),
    },
}

ERROR_RULES = {
    "高远球": [
        ("elbow", "low", "疑似屈肘击球（肘角偏小）", 10),
        ("arm_raise", "low", "疑似击球点偏低（臂上抬不足）", 10),
        ("wrist_above_head", "low", "疑似击球点低于头顶", 10),
        ("shoulder_hip_ratio", "low", "疑似侧身不足（身体正对球网）", 6),
    ],
    "杀球": [
        ("elbow", "low", "疑似屈肘击球", 10),
        ("arm_raise", "low", "疑似击球点偏低", 10),
        ("com_ratio", "low", "疑似重心未跟进（击球点偏后）", 10),
    ],
    "吊球": [
        ("arm_raise", "low", "疑似击球点偏低", 10),
        ("wrist_above_head", "low", "疑似击球点低于头顶", 10),
        ("elbow_x_signed", "high", "疑似肘部过度前伸（拍面控制差）", 6),
        ("wrist_x_signed", "high", "疑似手腕过度前伸（拍面控制差）", 6),
    ],
    "挑球": [
        ("wrist_above_shoulder", "high", "疑似击球点过高（未低位引拍）", 10),
        ("elbow", "low", "疑似未跟送（肘部僵硬）", 8),
        ("forearm_angle", "low", "疑似手腕未充分外展（拍面控制不足）", 6),
    ],
    "搓球": [
        ("wrist_reach", "low", "疑似离网太远（手臂未前伸）", 10),
        ("arm_raise", "high", "疑似动作过大（搓成挑球）", 8),
        ("elbow_x_signed", "high", "疑似肘部过度前伸（拍面控制差）", 6),
        ("wrist_x_signed", "high", "疑似手腕过度前伸（拍面控制差）", 6),
    ],
    "正手发高远球": [
        ("wrist_above_shoulder", "high", "疑似击球点过高（高于腰）", 10),
        ("elbow", "low", "疑似未充分随挥", 8),
    ],
    "反手发网前小球": [
        ("wrist_above_shoulder", "high", "疑似拍头高于腰（违例风险）", 10),
        ("arm_raise", "high", "疑似动作过大", 8),
        ("elbow_x_signed", "low", "疑似引拍时肘部过度后拉", 5),
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
        if angles.get(name) is None:
            continue  # 该检查点关键点缺失，不参与本帧打分
        tolerance = max(5.0, (hi - lo) * 0.3)
        scores.append(_checkpoint_score(angles.get(name), lo, hi, tolerance=tolerance))
    if not scores:
        return 0.0, []
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
