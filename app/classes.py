# -*- coding: utf-8 -*-
"""击球类型与标签映射（全局唯一来源）。"""

# 本项目 7 类击球类型，顺序即模型分类输出的类别顺序。
SHOT_CLASSES = [
    "正手发高远球",
    "反手发网前小球",
    "高远球",
    "杀球",
    "吊球",
    "挑球",
    "搓球",
]

CLASS_TO_ID = {name: i for i, name in enumerate(SHOT_CLASSES)}

# 两阶段分类的粗分组：过顶击球 vs 非过顶击球。
# 过顶（击球点在头顶附近/上方）：高远球、杀球、吊球；
# 非过顶（击球点在肩/腰以下）：发球、挑球、搓球。
OVERHEAD_CLASSES = ["高远球", "杀球", "吊球"]
NON_OVERHEAD_CLASSES = ["正手发高远球", "反手发网前小球", "挑球", "搓球"]


def shot_group(shot):
    """返回 'overhead' / 'non_overhead'。"""
    if shot in OVERHEAD_CLASSES:
        return "overhead"
    return "non_overhead"


def group_allowed_classes(group):
    """返回该分组内的类别名列表。"""
    return OVERHEAD_CLASSES if group == "overhead" else NON_OVERHEAD_CLASSES


# BFMD 数据集标签 -> 本项目类别 id 的映射。
# 说明：BFMD 不区分正/反手发球；serve 近似反手发网前小球，
# flick_serve 近似正手发高远球（仅作初版占位，真实数据以自采为准）。
BFMD_LABEL_MAP = {
    "flick_serve": CLASS_TO_ID["正手发高远球"],
    "serve": CLASS_TO_ID["反手发网前小球"],
    "clear": CLASS_TO_ID["高远球"],
    "smash": CLASS_TO_ID["杀球"],
    "drop": CLASS_TO_ID["吊球"],
    "lift": CLASS_TO_ID["挑球"],
    "net_shot": CLASS_TO_ID["搓球"],
}

