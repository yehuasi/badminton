# -*- coding: utf-8 -*-
"""序列级动作分析：击球帧检测 + 四项评分（标准度/类型/连贯性/步伐）。"""

import math

import numpy as np

from app.advice import generate_advice
from app.classes import SHOT_CLASSES, group_allowed_classes
from app.features import FEATURE_NAMES, F
from app.score import score_standardness

# 四项评分权重（依据：AIMS 2026 可解释羽毛球击球质量评估框架、AHP 层次分析法、
# 国内外羽毛球技术测试标准与教练评价量表；完整论证见《评分权重依据.md》）。
# 关键结论：动作类型识别是“前提/门控”而非质量维度，故从 40% 大幅下调为 10%；
# 技术标准度是核心（50%），步伐是羽毛球根基（25%），连贯协调 15%。
WEIGHTS = {
    "standardness": 0.50,  # 动作标准度（技术规范 / 生物力学质量）
    "footwork": 0.25,      # 步伐效率（重心转移、站距、起动）
    "coherence": 0.15,     # 连贯协调性（平滑度、节奏、完整性）
    "action": 0.10,        # 动作类型识别（置信度加权的完成度，低置信度作门控）
}

# 识别门控阈值：低于该置信度时，动作类型维度按“未完成”处理并显著降权提示。
RECOGNITION_GATE = 0.5

# 分组门控阈值：整段序列“腕相对头顶高度”（feat[9]）的峰值达到该值判为过顶击球。
# 注意：正手发高远球触球在腰下、随挥才过顶，属已知歧义，待两阶段模型用真实数据替换此启发式。
OVERHEAD_THRESHOLD = 0.15

# 挥速强度闸：用“原始收集帧”的平均挥速判断（快慢挥拍均适用）。
# 真实动作约 2.5~8.6，站立抖动约 1.3；取 2.0 留余量。
SWING_MEAN_THRESHOLD = 2.0

# 活动触发式切分参数（帧不进固定窗口，检测到挥拍起止才收集）。
# 阈值按真实摄像头数据上调：站立中位运动量约 1.7，走动/调整可达 3~8，
# 故取 3.5 减少把非挥拍动作当击球；最短 8 帧过滤噪声短窗。
MOTION_FRAME_THRESHOLD = 3.5  # 单帧运动量阈值（臂抬角 + 腕相对肩高 + 腕前伸的变化）
ONSET_FRAMES = 3              # 连续多少帧运动判定“挥拍开始”
OFFSET_FRAMES = 5             # 连续多少帧静止判定“挥拍结束”
MIN_STROKE_FRAMES = 8         # 至少收集多少帧才算一拍
MAX_STROKE_FRAMES = 90        # 最多收集多少帧（防无限收集）

# 置信度闸：分类置信度低于该值视为“动作不明确”，不强行归类。
NO_ACTION_CONF = 0.45

def _feat_dict(feat):
    return {
        name: (None if math.isnan(v) else float(v))
        for name, v in zip(FEATURE_NAMES, feat)
    }


def detect_hit_frame(seq):
    """在 (T,F) 特征序列中定位击球帧：挥臂速度峰值帧。"""
    arr = np.nan_to_num(np.asarray(seq, dtype=np.float64))
    if arr.shape[0] < 3:
        return int(arr.shape[0] // 2)
    # 持拍臂上抬角(0) + 腕相对肩高度(7) 的一阶差分绝对值，作为挥速。
    speed = np.abs(np.diff(arr[:, 0])) + np.abs(np.diff(arr[:, 7]))
    return int(np.argmax(speed)) + 1


def _swing_speed(seq):
    """挥速代理：持拍臂上抬角 + 腕相对肩高的一阶差绝对值之和。"""
    arr = np.nan_to_num(np.asarray(seq, dtype=np.float64))
    if arr.shape[0] < 3:
        return np.zeros(arr.shape[0], dtype=np.float64)
    return np.abs(np.diff(arr[:, 0])) + np.abs(np.diff(arr[:, 7]))


def _has_swing(seq):
    """是否出现有效挥拍：窗口平均挥速达到阈值。"""
    speed = _swing_speed(seq)
    if speed.size == 0:
        return False
    return float(speed.mean()) >= SWING_MEAN_THRESHOLD


def _frame_motion(prev, cur):
    """单帧运动量：臂抬角 + 腕相对肩高 + 腕前伸的变化。"""
    if prev is None:
        return 0.0
    a = np.nan_to_num(np.asarray(prev, dtype=np.float64))
    b = np.nan_to_num(np.asarray(cur, dtype=np.float64))
    return float(abs(b[0] - a[0]) + abs(b[7] - a[7]) + abs(b[8] - a[8]))


def _resample(seq, target):
    """把变长序列沿时间轴线性重采样到 target 帧。"""
    seq = np.asarray(seq, dtype=np.float32)
    n = seq.shape[0]
    if n <= 0 or n == target:
        return seq
    src = np.linspace(0, n - 1, target)
    out = np.empty((target, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(src, np.arange(n), np.nan_to_num(seq[:, j]))
    return out


def _hit_group(seq, hit_idx):
    """依据整段序列“腕相对头顶高度”的峰值判定过顶 / 非过顶。"""
    arr = np.nan_to_num(np.asarray(seq, dtype=np.float64))
    peak = float(np.max(arr[:, 9]))
    return "overhead" if peak >= OVERHEAD_THRESHOLD else "non_overhead"


def coherence_score(seq):
    """连贯性（0-100）：动作平滑（低抖动）+ 引拍幅度完整。"""
    arr = np.nan_to_num(np.asarray(seq, dtype=np.float64))
    if arr.shape[0] < 4:
        return 50.0
    jerk = np.abs(np.diff(arr, n=2, axis=0)).mean()
    jerk_penalty = min(40.0, jerk * 2.0)
    # 引拍完整性：上臂上抬角的运动范围。
    arm_range = float(np.ptp(arr[:, 0]))
    completeness = min(1.0, arm_range / 70.0)
    return float(np.clip(100.0 - jerk_penalty - (1.0 - completeness) * 30.0, 0.0, 100.0))


def footwork_score(seq):
    """步伐（0-100，摄像头前专门动作的初版）：重心转移 + 站距变化。"""
    arr = np.nan_to_num(np.asarray(seq, dtype=np.float64))
    if arr.shape[0] < 2:
        return 50.0
    com_range = float(np.ptp(arr[:, 6]))      # com_ratio 变化 = 前后重心转移
    stance_range = float(np.ptp(arr[:, 11]))  # ankle_width 变化 = 步幅/站距变化
    # 有一定转移和步幅为佳；完全静止扣分。阈值为草案值。
    score = (com_range / 0.25 + stance_range / 0.45) * 40.0
    return float(np.clip(score, 0.0, 100.0))


def _weighted_total(standardness, action_score, coherence, footwork):
    return (
        WEIGHTS["standardness"] * standardness
        + WEIGHTS["action"] * action_score
        + WEIGHTS["coherence"] * coherence
        + WEIGHTS["footwork"] * footwork
    )


def _build_result(pred, conf, seq, hit_idx):
    std, errors = score_standardness(_feat_dict(seq[hit_idx]), action=pred)
    coh = coherence_score(seq)
    foot = footwork_score(seq)
    action_score = conf * 100.0
    if conf < RECOGNITION_GATE:
        action_score *= 0.5  # 低置信度：动作类型维度减半（视为完成度不足）
    total = _weighted_total(std, action_score, coh, foot)
    advice = generate_advice(pred, conf, std, errors)
    if coh < 60:
        advice.append("连贯性偏弱：动作起伏偏大，建议保持匀速、完整做完整套动作。")
    if foot < 60:
        advice.append("步伐偏弱：重心转移或步幅不足，注意上步/蹬转发力。")
    return {
        "shot_type": pred,
        "confidence": round(conf, 3),
        "action_score": round(action_score, 1),
        "standardness": round(std, 1) if std is not None else None,
        "coherence": round(coh, 1),
        "footwork": round(foot, 1),
        "hit_frame": hit_idx,
        "errors": errors,
        "advice": advice,
        "total": round(total, 1),
    }


class ShotAnalyzer:
    """活动触发式切分：帧不进固定窗口，逐帧检测挥拍起止。

    单帧运动量连续若干帧达到阈值判定“挥拍开始”，开始收集帧；连续若干帧
    静止判定“挥拍结束”，把收集到的帧重采样到 window 后做一次识别。
    快拍窗口短、慢拍窗口长，一拍恰好一次结果，且分类只在收拍时运行一次。
    """

    def __init__(self, classifier=None, window=24, debug=False):
        self.window = window
        self.classifier = classifier
        self.debug = debug
        self.buffer = []  # 正在收集的帧
        self.collecting = False
        self._prev = None
        self._moving_streak = 0
        self._still_streak = 0

    def reset(self):
        self.buffer = []
        self.collecting = False
        self._prev = None
        self._moving_streak = 0
        self._still_streak = 0

    def update_feature(self, feat):
        cur = np.asarray(feat, dtype=np.float32)
        motion = _frame_motion(self._prev, cur)
        self._prev = cur
        if self.debug:
            print(
                f"[motion] {motion:.2f} collect={int(self.collecting)} "
                f"ms={self._moving_streak} ss={self._still_streak} n={len(self.buffer)}",
                flush=True,
            )

        if not self.collecting:
            if motion >= MOTION_FRAME_THRESHOLD:
                self._moving_streak += 1
                self.buffer.append(cur)
            else:
                self._moving_streak = 0
                self.buffer = []
            if self._moving_streak >= ONSET_FRAMES:
                self.collecting = True
                self._still_streak = 0
            return None

        # 正在收集
        self.buffer.append(cur)
        if motion < MOTION_FRAME_THRESHOLD:
            self._still_streak += 1
        else:
            self._still_streak = 0
        if self._still_streak >= OFFSET_FRAMES or len(self.buffer) >= MAX_STROKE_FRAMES:
            return self._finalize()
        return None

    def _finalize(self):
        frames = self.buffer
        n = len(frames)
        self.reset()
        if n < MIN_STROKE_FRAMES:
            if self.debug:
                print(f"[finalize] too short frames={n}", flush=True)
            return None
        if not _has_swing(np.stack(frames, axis=0)):
            if self.debug:
                mean = float(_swing_speed(np.stack(frames, axis=0)).mean())
                print(f"[finalize] rejected mean={mean:.2f}", flush=True)
            return None  # 轻量安全闸：原始帧平均挥速须达标，过滤纯抖动
        seq = _resample(frames, self.window)
        result = self.analyze_sequence(seq)
        if self.debug:
            print(
                f"[finalize] frames={n} shot={result.get('shot_type')!r} "
                f"conf={result.get('confidence')} is_action={result.get('is_action')}",
                flush=True,
            )
        return result

    def analyze_sequence(self, seq):
        seq = np.asarray(seq, dtype=np.float32)
        hit_idx = detect_hit_frame(seq)
        if self.classifier is not None:
            group = _hit_group(seq, hit_idx)
            top = self.classifier.predict_sequence(seq, allowed=group_allowed_classes(group))
            pred, conf = top[0][0], top[0][1]
        else:
            group = None
            pred, conf = SHOT_CLASSES[0], 1.0
        if self.classifier is not None and conf < NO_ACTION_CONF:
            return {
                "shot_type": "",
                "is_action": False,
                "confidence": round(conf, 3),
                "action_score": round(conf * 100.0, 1),
                "standardness": None,
                "coherence": None,
                "footwork": None,
                "hit_frame": hit_idx,
                "errors": [],
                "advice": ["动作不明确，请完整规范地完成引拍-击球-随挥。"],
                "total": None,
                "top3": [[n, round(p, 3)] for n, p in top],
                "group": group,
            }
        result = _build_result(pred, conf, seq, hit_idx)
        if self.classifier is not None:
            result["top3"] = [[n, round(p, 3)] for n, p in top]
            result["group"] = group
        result["is_action"] = True
        return result

