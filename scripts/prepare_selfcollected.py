# -*- coding: utf-8 -*-
"""把用户自采视频转成本项目规范数据集（T 帧窗口 + 15 维特征）。

输入目录结构（每类一个子目录，视频文件名任意）：
    data/self_collected/
        高远球/   xxx.mp4
        杀球/     xxx.mp4
        ...

输出：data/processed_selfcollected 下的 X.npz / y.npy / classes.json / manifest.csv

没有人工击球帧标注，脚本用与运行时完全相同的“活动触发式切分”定位挥拍并取窗口
（保证训练分布和运行时一致）；若某视频切不出窗口，则退回“挥速峰值帧居中取窗”。
"""

import argparse
import glob
import os

import cv2
import numpy as np

from common import REPO_ROOT, save_dataset
from app.analyzer import ShotAnalyzer, detect_hit_frame, _swing_speed
from app.classes import SHOT_CLASSES
from app.features import add_kinematics, compute_features, F

VIDEO_EXT = ("mp4", "avi", "mov", "mkv", "MP4", "AVI", "MOV", "MKV")


def _build_estimator(backend, weights, threads):
    if backend == "onnx":
        from app.pose_onnx import ONNXPoseEstimator

        return ONNXPoseEstimator(weights, threads=threads)
    from app.pose import PoseEstimator

    return PoseEstimator(weights)


def _pick_person(persons):
    """多人时取关键点包围盒面积最大者（最靠近镜头）。"""
    if not persons:
        return None
    if len(persons) == 1:
        return persons[0]

    def area(p):
        xs, ys = p[:, 0], p[:, 1]
        return float((xs.max() - xs.min()) * (ys.max() - ys.min()))

    return max(persons, key=area)


def _clean_sequence(seq):
    """沿时间线性插值 NaN；整段过半 NaN 则返回 None（与 prepare_bfmd 一致）。"""
    if np.isnan(seq).mean() > 0.5:
        return None
    df = np.asarray(seq, dtype=np.float64)
    n, m = df.shape
    x_idx = np.arange(n)
    for j in range(m):
        col = df[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any() and not nan_mask.all():
            col[nan_mask] = np.interp(x_idx[nan_mask], x_idx[~nan_mask], col[~nan_mask])
        col[:] = np.where(np.isnan(col), np.nanmean(col), col)
    return df.astype(np.float32)


def _features_from_video(path, estimator, handedness):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    feats = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        person = _pick_person(estimator.predict(frame))
        if person is None:
            feats.append(np.full(F, np.nan, dtype=np.float32))
        else:
            feats.append(compute_features(person[:17], handedness=handedness))
    cap.release()
    if not feats:
        return None
    return _clean_sequence(np.stack(feats))


def _window_around(seq, center, T):
    """以 center 为中心取 T 帧窗口，越界用首/末帧补齐。"""
    half = T // 2
    lo, hi = center - half, center - half + T
    if lo < 0:
        seq = np.concatenate([np.tile(seq[0], (-lo, 1)), seq], axis=0)
        hi += -lo
        lo = 0
    if hi > seq.shape[0]:
        seq = np.concatenate([seq, np.tile(seq[-1], (hi - seq.shape[0], 1))], axis=0)
    return seq[lo:hi]


def _windows_for_video(seq, T):
    """用与运行时一致的活动触发式切分取窗口；无窗口则退回挥速峰值居中取窗。"""
    analyzer = ShotAnalyzer(classifier=None, window=T, capture=True)
    for t in range(seq.shape[0]):
        analyzer.update_feature(seq[t])
    windows = list(analyzer.captured)
    if not windows and seq.shape[0] >= T:
        fallback = _window_around(seq, detect_hit_frame(seq), T)
        # 数据准备阶段视频均为已标注的挥拍，退回判定放宽到“挥速峰值”层面，
        # 避免小动作（反手发球/搓球）被运行时那套偏严的阈值丢弃。
        speed = _swing_speed(fallback)
        if speed.size and float(speed.max()) >= 3.0:
            windows = [fallback]
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "data", "self_collected"))
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--handedness", default="right", choices=["left", "right"])
    parser.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    parser.add_argument("--pose-model", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--kinematics", action="store_true",
                        help="追加手腕速度/加速度等时间导数特征（15 -> 23 维）")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "processed_selfcollected"))
    args = parser.parse_args()

    if args.pose_model is None:
        args.pose_model = os.path.join(
            REPO_ROOT,
            "yolov8n-pose-480.onnx" if args.backend == "onnx" else "yolov8n-pose.pt",
        )

    if not os.path.isdir(args.root):
        raise SystemExit(
            f"未找到自采数据目录：{args.root}\n"
            f"请按 “<类别名>/<视频文件>” 组织，例如 {args.root}\\高远球\\a.mp4"
        )

    estimator = _build_estimator(args.backend, args.pose_model, args.threads)

    X, y, rows = [], [], []
    for cls_id, cls_name in enumerate(SHOT_CLASSES):
        cls_dir = os.path.join(args.root, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"跳过（无目录）：{cls_name}")
            continue
        videos = []
        for ext in VIDEO_EXT:
            videos += glob.glob(os.path.join(cls_dir, f"*.{ext}"))
        videos = sorted(set(videos))
        print(f"{cls_name}: {len(videos)} 个视频")

        for vid in videos:
            seq = _features_from_video(vid, estimator, args.handedness)
            if seq is None:
                print(f"  跳过（无法提取特征）：{os.path.basename(vid)}")
                continue
            windows = _windows_for_video(seq, args.T)
            if not windows:
                print(f"  跳过（未切出窗口）：{os.path.basename(vid)}")
                continue
            for k, w in enumerate(windows):
                if args.kinematics:
                    w = add_kinematics(w)
                X.append(w)
                y.append(cls_id)
                rows.append([f"{os.path.splitext(os.path.basename(vid))[0]}_{k}",
                             cls_name, os.path.basename(vid)])
    if not X:
        raise SystemExit("未转换出任何样本，请检查数据目录与视频。")

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    save_dataset(args.out, X, y, SHOT_CLASSES, manifest_rows=rows)
    print(f"已转换 {len(X)} 个样本到 {args.out}，X.shape={X.shape}")


if __name__ == "__main__":
    main()
