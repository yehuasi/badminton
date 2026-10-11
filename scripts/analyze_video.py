# -*- coding: utf-8 -*-
"""离线录像分析：对整段视频逐帧分析，输出每次击球的 JSON 报告。"""

import argparse
import json
import os
import sys

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.analyzer import ShotAnalyzer  # noqa: E402
from app.features import compute_features  # noqa: E402


def build_runtime(args):
    if args.backend == "onnx":
        from app.classifier_onnx import ONNXShotClassifier
        from app.pose_onnx import ONNXPoseEstimator

        estimator = ONNXPoseEstimator(args.pose_model, conf=args.conf, threads=args.threads)
        classes = None
        if os.path.isfile(args.classes_file):
            with open(args.classes_file, "r", encoding="utf-8") as f:
                classes = json.load(f)
        classifier = ONNXShotClassifier(args.classifier_model, classes=classes, threads=args.threads)
    else:
        from app.classifier import ShotClassifierRunner
        from app.pose import PoseEstimator

        estimator = PoseEstimator(args.pose_model)
        classifier = ShotClassifierRunner(
            args.classifier_model, T=args.window, handedness=args.handedness
        )
    return estimator, ShotAnalyzer(classifier=classifier, window=args.window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    parser.add_argument("--pose-model", default=None)
    parser.add_argument("--classifier-model", default=None)
    parser.add_argument("--classes-file", default=None)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--handedness", default="right", choices=["left", "right"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--out", default=None, help="报告输出路径，默认 视频名_report.json")
    args = parser.parse_args()

    if args.backend == "onnx":
        args.pose_model = args.pose_model or os.path.join(REPO_ROOT, "yolov8n-pose-480.onnx")
        args.classifier_model = args.classifier_model or os.path.join(REPO_ROOT, "models", "shot_classifier.onnx")
        args.classes_file = args.classes_file or os.path.join(REPO_ROOT, "models", "shot_classifier_classes.json")
    else:
        args.pose_model = args.pose_model or os.path.join(REPO_ROOT, "yolov8n-pose.pt")
        args.classifier_model = args.classifier_model or os.path.join(REPO_ROOT, "models", "shot_classifier_bfmd.pt")
        args.classes_file = args.classes_file or os.path.join(REPO_ROOT, "models", "shot_classifier_classes.json")

    estimator, analyzer = build_runtime(args)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{args.video}")

    shots = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        persons = estimator.predict(frame)
        if not persons:
            analyzer.buffer.clear()
            continue
        feat = compute_features(persons[0], handedness=args.handedness)
        res = analyzer.update_feature(feat)
        if res is not None:
            shots.append(res)
    cap.release()

    out = args.out or os.path.splitext(args.video)[0] + "_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"video": args.video, "shots": shots}, f, ensure_ascii=False, indent=2)

    print(f"共识别 {len(shots)} 次击球，报告已写入 {out}")
    for i, s in enumerate(shots, 1):
        print(f"  {i}. {s['shot_type']}  综合={s['total']}  标准度={s['standardness']}  "
              f"连贯性={s['coherence']}  步伐={s['footwork']}")


if __name__ == "__main__":
    main()
