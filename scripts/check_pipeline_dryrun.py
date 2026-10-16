# -*- coding: utf-8 -*-
"""自采数据流水线试跑（只读，不改动原始数据）。

对每个视频跑一遍与运行时一致的链路：
    逐帧姿态检测 → 15 维特征 → 活动触发式切分 → 取挥拍窗口
统计：关键点检出率、关键点平均置信度、切出的窗口数与长度、是否走了兜底路径、
挥速峰值、单视频耗时。用于判断这批数据能否直接进入训练与标准库回填。

用法：
    .venv\\Scripts\\python scripts\\check_pipeline_dryrun.py --root "self-build dataset"
    .venv\\Scripts\\python scripts\\check_pipeline_dryrun.py --root "self-build dataset" --limit 5
"""

import argparse
import collections
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT  # noqa: E402
from app.analyzer import ShotAnalyzer, _swing_speed  # noqa: E402
from app.features import F, compute_features  # noqa: E402
from prepare_selfcollected import (  # noqa: E402
    _build_estimator,
    _clean_sequence,
    _pick_person,
    _windows_for_video,
)

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def window_stats(seq, T):
    """返回 (切分捕获的窗口数, 最终窗口长度列表, 是否用了兜底)。"""
    analyzer = ShotAnalyzer(classifier=None, window=T, capture=True)
    for t in range(seq.shape[0]):
        analyzer.update_feature(seq[t])
    captured = len(analyzer.captured)
    wins = _windows_for_video(seq, T)
    used_fallback = captured == 0 and len(wins) > 0
    return captured, [int(w.shape[0]) for w in wins], used_fallback


def analyze_video(path, estimator, handedness, T, decode_width):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False}
    feats, n, det, confs = [], 0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if decode_width and frame.shape[1] > decode_width:
            s = decode_width / float(frame.shape[1])
            frame = cv2.resize(frame, (decode_width, max(1, int(round(frame.shape[0] * s)))),
                               interpolation=cv2.INTER_AREA)
        person = _pick_person(estimator.predict(frame))
        if person is None:
            feats.append(np.full(F, np.nan, dtype=np.float32))
        else:
            det += 1
            confs.append(float(np.mean(person[:17, 2])))
            feats.append(compute_features(person[:17], handedness=handedness))
    cap.release()

    out = {"ok": True, "frames": n, "det_frames": det,
           "det_rate": round(det / n, 3) if n else 0.0,
           "kpt_conf": round(float(np.mean(confs)), 3) if confs else 0.0}
    if n == 0:
        out.update({"windows": None, "reason": "空视频"})
        return out

    seq = _clean_sequence(np.stack(feats))
    if seq is None:
        out.update({"windows": None, "reason": "整段关键点缺失过半"})
        return out

    speed = _swing_speed(seq)
    out["swing_max"] = round(float(speed.max()), 2) if speed.size else 0.0
    captured, lens, fallback = window_stats(seq, T)
    out.update({"captured": captured, "windows": len(lens), "win_lengths": lens,
                "fallback": fallback})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "self-build dataset"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "pipeline_dryrun.json"))
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="每类最多处理多少个视频（0=全部）")
    ap.add_argument("--classes", default="", help="只处理这些类别，逗号分隔")
    ap.add_argument("--backend", default="onnx", choices=["torch", "onnx"])
    ap.add_argument("--pose-model", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--decode-width", type=int, default=1280,
                    help="读帧后先缩到这个宽度再送姿态模型（0=保持原始分辨率）")
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if args.pose_model is None:
        args.pose_model = os.path.join(
            REPO_ROOT, "yolov8n-pose-480.onnx" if args.backend == "onnx" else "yolov8n-pose.pt")
    estimator = _build_estimator(args.backend, args.pose_model, args.threads)

    wanted = [c for c in args.classes.split(",") if c]
    videos = []
    for dirpath, _d, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXT:
                videos.append(os.path.join(dirpath, fn))
    videos.sort()
    if wanted:
        videos = [v for v in videos if any(w in v for w in wanted)]

    records, t_start = [], time.time()
    for i, path in enumerate(videos, 1):
        if args.limit:
            cls_dir = os.path.relpath(path, root).split(os.sep)[0]
            done = sum(1 for r in records if r["top"] == cls_dir)
            if done >= args.limit:
                continue
        t0 = time.time()
        res = analyze_video(path, estimator, args.handedness, args.T, args.decode_width)
        res.update({"path": os.path.relpath(path, root).replace("\\", "/"),
                    "top": os.path.relpath(path, root).split(os.sep)[0],
                    "seconds": round(time.time() - t0, 2)})
        records.append(res)
        if i % 20 == 0 or i == len(videos):
            el = time.time() - t_start
            print(f"  {len(records)}/{len(videos)} 视频，用时 {el/60:.1f} 分钟，"
                  f"均 {el/max(1,len(records)):.2f}s/条", flush=True)

    print("\n=== 每类切分结果 ===")
    print(f"{'类别':<16}{'视频':>5}{'单窗口':>7}{'零窗口':>7}{'多窗口':>7}"
          f"{'检出率':>8}{'关键点置信':>10}{'耗时/条':>8}")
    tops = sorted({r["top"] for r in records})
    for top in tops:
        rs = [r for r in records if r["top"] == top]
        w1 = sum(1 for r in rs if r.get("windows") == 1)
        w0 = sum(1 for r in rs if r.get("windows") == 0)
        wm = sum(1 for r in rs if (r.get("windows") or 0) > 1)
        bad = sum(1 for r in rs if r.get("windows") is None)
        det = statistics.median([r["det_rate"] for r in rs])
        kc = statistics.median([r["kpt_conf"] for r in rs])
        sec = statistics.median([r["seconds"] for r in rs])
        print(f"{top:<16}{len(rs):>5}{w1:>7}{w0:>7}{wm:>7}{det:>8.2f}{kc:>10.3f}{sec:>8.2f}"
              + (f"   （另 {bad} 条无法分析）" if bad else ""))

    print("\n=== 零窗口 / 多窗口 / 检出率偏低 的问题样本（前 30）===")
    bad = [r for r in records if not r.get("windows") or (r.get("windows") or 0) != 1
           or r["det_rate"] < 0.5]
    for r in bad[:30]:
        print(f"  {r['path']}  窗口={r.get('windows')}  检出={r['det_rate']}  "
              f"关键点={r.get('kpt_conf')}  帧={r.get('frames')}  挥速={r.get('swing_max')}"
              + (f"  ({r.get('reason')})" if r.get("reason") else ""))
    print(f"  共 {len(bad)} 条待人工确认，占 {len(bad)/max(1,len(records)):.1%}")

    fb = [r for r in records if r.get("fallback")]
    print(f"\n走兜底取窗的视频：{len(fb)}（活动触发式切分没抓到挥拍）")
    zero = [r for r in records if r.get("windows") == 0]
    print(f"完全取不到窗口的视频：{len(zero)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"root": root, "count": len(records), "records": records},
                  f, ensure_ascii=False, indent=1)
    print(f"\n逐条结果已写入：{args.out}")


if __name__ == "__main__":
    main()
