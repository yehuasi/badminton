# -*- coding: utf-8 -*-
"""双视角（正面 + 侧面）评分融合。

单摄像头分时采集时，正面与侧面是两次独立挥拍（同人、同动作类型）。
融合规则（草案，数据到位后按视角校准权重）：
  - 动作类型：取置信度更高的那一侧；两侧一致时置信度略提升；
  - 动作标准度：侧面为主（挥拍平面 / 肘 / 躯干更可靠），正面为辅（55% / 45%）；
  - 步伐：仅取正面（步伐只在正面可靠）；
  - 连贯性：两侧取均值；
  - 综合得分：按 analyzer.WEIGHTS 重新加权。
"""

from app.analyzer import RECOGNITION_GATE, WEIGHTS

FRONT_STD_WEIGHT = 0.45
SIDE_STD_WEIGHT = 0.55


def _mean(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return round((a + b) / 2.0, 1)


def fuse_views(front, side):
    front = front or {}
    side = side or {}

    fc = front.get("confidence") or 0.0
    sc = side.get("confidence") or 0.0
    primary = front if fc >= sc else side
    secondary = side if fc >= sc else front

    shot_type = primary.get("shot_type") or secondary.get("shot_type")
    conf = max(fc, sc)
    if front.get("shot_type") and front.get("shot_type") == side.get("shot_type"):
        conf = min(1.0, conf + 0.05)

    std_f, std_s = front.get("standardness"), side.get("standardness")
    if std_f is None and std_s is None:
        standardness = None
    elif std_f is None:
        standardness = std_s
    elif std_s is None:
        standardness = std_f
    else:
        standardness = round(FRONT_STD_WEIGHT * std_f + SIDE_STD_WEIGHT * std_s, 1)

    coherence = _mean(front.get("coherence"), side.get("coherence"))
    footwork = front.get("footwork")  # 步伐只看正面

    action_score = round(conf * 100.0, 1)
    if conf < RECOGNITION_GATE:
        action_score = round(action_score * 0.5, 1)

    total = None
    if standardness is not None and coherence is not None and footwork is not None:
        total = round(
            WEIGHTS["standardness"] * standardness
            + WEIGHTS["footwork"] * footwork
            + WEIGHTS["coherence"] * coherence
            + WEIGHTS["action"] * action_score,
            1,
        )

    errors = []
    for r in (front, side):
        for e in r.get("errors") or []:
            if e not in errors:
                errors.append(e)

    advice = list(primary.get("advice") or [])
    for a in secondary.get("advice") or []:
        if a not in advice:
            advice.append(a)

    return {
        "shot_type": shot_type,
        "confidence": round(conf, 3),
        "action_score": action_score,
        "standardness": standardness,
        "coherence": coherence,
        "footwork": footwork,
        "errors": errors,
        "advice": advice,
        "total": total,
        "view": "front+side",
    }
