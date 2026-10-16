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
import csv
import glob
import os

import cv2
import numpy as np

from common import REPO_ROOT, save_dataset
from app.analyzer import ShotAnalyzer, detect_hit_frame, _swing_speed
from app.classes import SHOT_CLASSES
from app.features import (
    F,
    F_FULL,
    add_kinematics,
    compute_features,
    compute_features_full,
)

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


class _SubjectPicker:
    """跨帧跟踪同一主体：优先延续上一帧选中的人，避免在教练/学员之间跳。

    教学视频里常有多人同框，逐帧“选最大框”会在两人之间来回切换，导致同一段视频里
    混进两个人的骨架。这里用“中心距离 + 尺寸差异”打分，选与上一帧最连续的那一个。
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.center = None
        self.size = None

    @staticmethod
    def _box(p):
        xs, ys = p[:, 0], p[:, 1]
        return ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0,
                max(1.0, xs.max() - xs.min()), max(1.0, ys.max() - ys.min()))

    def __call__(self, persons):
        if not persons:
            return None
        if len(persons) == 1 or not self.enabled or self.center is None:
            chosen = _pick_person(persons)
        else:
            cx, cy, w, h = self.center[0], self.center[1], self.size[0], self.size[1]
            diag = max(1.0, (w * w + h * h) ** 0.5)

            def cost(p):
                bx, by, bw, bh = self._box(p)
                dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / diag
                scale = abs(bw * bh - w * h) / max(1.0, w * h)
                return dist + 0.5 * scale

            chosen = min(persons, key=cost)
        bx, by, bw, bh = self._box(chosen)
        self.center, self.size = (bx, by), (bw, bh)
        return chosen


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


def _features_from_video(path, estimator, handedness, decode_width=0, tracker=None,
                         extended=False):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    feats = []
    dim = F_FULL if extended else F
    pick = tracker if tracker is not None else _SubjectPicker(False)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decode_width and frame.shape[1] > decode_width:
            s = decode_width / float(frame.shape[1])
            frame = cv2.resize(frame, (decode_width, max(1, int(round(frame.shape[0] * s)))),
                               interpolation=cv2.INTER_AREA)
        person = pick(estimator.predict(frame))
        if person is None:
            feats.append(np.full(dim, np.nan, dtype=np.float32))
        else:
            feat = (compute_features_full(person[:17], handedness=handedness) if extended
                    else compute_features(person[:17], handedness=handedness))
            feats.append(feat)
    cap.release()
    if not feats:
        return None
    return _clean_sequence(np.stack(feats))


def _videos_from_index(index_path, root, classes, keep_only=True, min_frames=0):
    """按索引表收集视频：返回 [(类别, 绝对路径), ...]，类别顺序与 SHOT_CLASSES 一致。"""
    by_class = {c: [] for c in classes}
    meta = {}
    skipped = {"重复": 0, "帧数不足": 0, "类别不匹配": 0}
    with open(index_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cls = row["class"].strip()
            if cls not in by_class:
                skipped["类别不匹配"] += 1
                continue
            if keep_only and row.get("keep", "是").strip() == "否":
                skipped["重复"] += 1
                continue
            try:
                if min_frames and int(float(row.get("frames") or 0)) < min_frames:
                    skipped["帧数不足"] += 1
                    continue
            except ValueError:
                pass
            abs_path = os.path.join(root, row["path"].replace("/", os.sep))
            by_class[cls].append(abs_path)
            meta[abs_path] = {k: row.get(k, "") for k in
                              ("split", "label", "error_family", "view", "variant", "player")}
    for c in by_class:
        by_class[c] = sorted(set(by_class[c]))
    return by_class, skipped, meta


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
    """用与运行时一致的活动触发式切分取窗口；无窗口则退回挥速峰值居中取窗。

    窗口本身由 analyzer 统一成“重采样到 T 帧 + 击球帧居中”，因此这里产出的分布
    与运行时推理、以及以触球帧居中的公开数据（BFMD）完全一致，不再另做居中处理。
    """
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
    parser.add_argument("--index", default=None,
                        help="索引表 CSV（scripts/build_selfcollected_index.py 生成）；"
                             "给定时按表内的类别与 keep 标记收录视频，支持深层目录结构")
    parser.add_argument("--min-frames", type=int, default=24,
                        help="帧数少于该值的视频直接跳过（凑不满一个窗口）")
    parser.add_argument("--decode-width", type=int, default=1280,
                        help="读帧后先缩到这个宽度再送姿态模型（0=原始分辨率）")
    parser.add_argument("--no-track", action="store_true",
                        help="关闭跨帧主体跟踪（默认开启，多人同框时锁定同一人）")
    parser.add_argument("--extended", action="store_true",
                        help="计算 24 维扩展特征（15 维 + 躯干朝向/转体/拍臂指向/非持拍手等 9 维）")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "processed_selfcollected"))
    parser.add_argument("--cache", default=None,
                        help="把每个视频的完整特征序列缓存到该目录（.npz），"
                             "后续回填标准库、调阈值不必重复跑姿态检测")
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

    by_index = None
    if args.index:
        if not os.path.isfile(args.index):
            raise SystemExit(f"未找到索引表：{args.index}")
        by_index, skipped, idx_meta = _videos_from_index(
            args.index, args.root, SHOT_CLASSES, min_frames=args.min_frames)
        print(f"按索引表收录：跳过重复 {skipped['重复']}，"
              f"帧数不足 {skipped['帧数不足']}，类别不匹配 {skipped['类别不匹配']}")
    else:
        idx_meta = {}

    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    estimator = _build_estimator(args.backend, args.pose_model, args.threads)

    X, y, rows = [], [], []
    for cls_id, cls_name in enumerate(SHOT_CLASSES):
        if by_index is not None:
            videos = by_index.get(cls_name, [])
        else:
            cls_dir = os.path.join(args.root, cls_name)
            videos = []
            for ext in VIDEO_EXT:
                videos += glob.glob(os.path.join(cls_dir, f"*.{ext}"))
            videos = sorted(set(videos))
            if not videos:
                print(f"跳过（无目录或无视频）：{cls_name}")
                continue
        print(f"{cls_name}: {len(videos)} 个视频")

        for vid in videos:
            seq = _features_from_video(vid, estimator, args.handedness,
                                       decode_width=args.decode_width,
                                       tracker=_SubjectPicker(not args.no_track),
                                       extended=args.extended)
            if seq is None:
                print(f"  跳过（无法提取特征）：{os.path.basename(vid)}")
                continue
            if args.cache:
                import hashlib

                key = hashlib.md5(vid.encode("utf-8")).hexdigest()[:16]
                md = idx_meta.get(vid, {})
                np.savez_compressed(
                    os.path.join(args.cache, key + ".npz"),
                    seq=seq.astype(np.float32),
                    path=np.array(vid), class_name=np.array(cls_name),
                    split=np.array(md.get("split", "")), label=np.array(md.get("label", "")),
                    view=np.array(md.get("view", "")), variant=np.array(md.get("variant", "")),
                    player=np.array(md.get("player", "")),
                    error_family=np.array(md.get("error_family", "")),
                    decode_width=np.array(args.decode_width),
                )
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
