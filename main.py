# -*- coding: utf-8 -*-
"""MVP 主入口：视频输入 -> 姿态检测 -> 角度计算 -> 单帧打分 -> 叠加显示。

用法：
    .venv\\Scripts\\python main.py --source 0            # 默认摄像头
    .venv\\Scripts\\python main.py --source 视频.mp4      # 本地视频
    .venv\\Scripts\\python main.py --handedness left      # 左手持拍
"""

import argparse

import cv2

from app.classifier import ShotClassifierRunner
from app.pose import PoseEstimator
from app.score import score_frame


def parse_source(value):
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description="羽毛球姿态分析 MVP")
    parser.add_argument("--source", default="0", help="摄像头索引或视频文件路径")
    parser.add_argument("--weights", default="yolov8n-pose.pt", help="姿态模型权重")
    parser.add_argument("--handedness", default="right", choices=["left", "right"])
    parser.add_argument("--action", default="高远球", help="当前分析的动作类型")
    parser.add_argument("--classifier", default="", help="击球类型识别模型 checkpoint 路径（可选）")
    parser.add_argument("--window", type=int, default=24, help="识别窗口帧数")
    args = parser.parse_args()

    estimator = PoseEstimator(args.weights)
    classifier = None
    if args.classifier:
        classifier = ShotClassifierRunner(
            args.classifier, T=args.window, handedness=args.handedness
        )
        print(f"已加载击球类型识别模型：{args.classifier}")
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"无法打开输入源: {args.source}")

    print("按 q 退出。")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        persons = estimator.predict(frame)
        if persons:
            score, errors = score_frame(
                persons[0], action=args.action, handedness=args.handedness
            )
            print(f"动作: {args.action} | 标准度: {score:.1f} | 错误: {errors}")
            cv2.putText(frame, f"score: {score:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
            if classifier is not None:
                classifier.update(persons[0])
                if classifier.ready:
                    top = classifier.predict()
                    label = top[0][0] + f" {top[0][1]*100:.0f}%"
                    cv2.putText(frame, label, (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 150, 255), 2)
                    for i, (name, p) in enumerate(top[1:], start=1):
                        cv2.putText(frame, f"{name} {p*100:.0f}%", (20, 80 + 30 * i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "no person", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("badminton pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
