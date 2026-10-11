# -*- coding: utf-8 -*-
"""把 BFMD 标注 JSON 转成本项目规范数据集（T 帧窗口 + 15 维特征）。

输入：data/raw/BFMD_data（annotations/pose、annotations/hit_inferred、annotations/court）
输出：data/processed 下的 X.npz / y.npy / classes.json / manifest.csv

击球者判定：hit_inferred 给出 side（top/bottom），逐帧取中心点 y 更小者为 top。
"""

import argparse
import glob
import json
import os

import numpy as np

from common import PROCESSED_DIR, REPO_ROOT, save_dataset
from app.classes import SHOT_CLASSES, BFMD_LABEL_MAP
from app.features import add_kinematics, compute_features, F


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rally_ranges(pose_data):
    """返回 [(start, end, rally_key, frames_dict)]，rally_key 形如 rally_045_74939_75211。"""
    out = []
    for rally_key, rally in pose_data.get("videos", {}).items():
        parts = rally_key.split("_")
        if len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
            out.append((int(parts[2]), int(parts[3]), rally_key, rally.get("frames", {})))
    return out


def _pick_player(players, side):
    if len(players) == 0:
        return None
    if len(players) == 1:
        return players[0]
    centers = [(p["bbox"][1] + p["bbox"][3]) / 2.0 for p in players]
    idx = int(np.argmin(centers)) if side == "top" else int(np.argmax(centers))
    return players[idx]


def _clean_sequence(seq):
    """seq: (T, F)，含 NaN；沿时间线性插值，若仍有过半 NaN 则返回 None。"""
    if np.isnan(seq).mean() > 0.5:
        return None
    df = np.asarray(seq, dtype=np.float64)
    # 沿时间逐特征做线性插值（首尾用最近邻填充）。
    n, m = df.shape
    x_idx = np.arange(n)
    for j in range(m):
        col = df[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any() and not nan_mask.all():
            col[nan_mask] = np.interp(x_idx[nan_mask], x_idx[~nan_mask], col[~nan_mask])
        col[:] = np.where(np.isnan(col), np.nanmean(col), col)
    return df.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "data", "raw", "BFMD_data"))
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--handedness", default="right", choices=["left", "right"])
    parser.add_argument("--out", default=PROCESSED_DIR)
    parser.add_argument("--matches", default=None, help="匹配名子串，逗号分隔；默认全部")
    parser.add_argument("--kinematics", action="store_true",
                        help="追加手腕速度/加速度等时间导数特征（15 维 -> 23 维）")
    args = parser.parse_args()

    pose_dir = os.path.join(args.root, "annotations", "pose")
    hit_dir = os.path.join(args.root, "annotations", "hit_inferred")
    if not os.path.isdir(pose_dir) or not os.path.isdir(hit_dir):
        raise SystemExit(f"未找到标注目录，请先运行 download_bfmd.py。检查：{args.root}")

    match_filter = None
    if args.matches:
        match_filter = [s.strip() for s in args.matches.split(",") if s.strip()]

    X, y, rows = [], [], []
    half = args.T // 2

    for hit_path in sorted(glob.glob(os.path.join(hit_dir, "*.json"))):
        match_name = os.path.basename(hit_path)[:-5]
        if match_filter and not any(s in match_name for s in match_filter):
            continue
        pose_path = os.path.join(pose_dir, os.path.basename(hit_path))
        if not os.path.isfile(pose_path):
            print(f"跳过（无 pose）：{match_name}")
            continue

        hits = _load_json(hit_path).get("hits", [])
        pose_data = _load_json(pose_path)
        ranges = _rally_ranges(pose_data)
        print(f"{match_name}: {len(hits)} hits, {len(ranges)} rallies")

        for hit in hits:
            frame = int(hit.get("frame", -1))
            shot_type = hit.get("shot_type")
            cls_id = BFMD_LABEL_MAP.get(shot_type)
            if cls_id is None or frame < 0:
                continue
            # 找到包含该击球帧的 rally。
            rally = None
            for start, end, key, frames in ranges:
                if start <= frame <= end:
                    rally = (key, frames)
                    break
            if rally is None:
                continue
            key, frames = rally
            seq = []
            for t in range(frame - half, frame + half):
                players = frames.get(str(t))
                if not players:
                    seq.append(np.full(F, np.nan, dtype=np.float32))
                    continue
                player = _pick_player(players, hit.get("side", "top"))
                if player is None or "keypoints" not in player:
                    seq.append(np.full(F, np.nan, dtype=np.float32))
                    continue
                kpts = np.asarray(player["keypoints"], dtype=float)
                if kpts.ndim != 2 or kpts.shape[0] < 17:
                    seq.append(np.full(F, np.nan, dtype=np.float32))
                    continue
                seq.append(compute_features(kpts[:17], handedness=args.handedness))
            seq = np.vstack(seq)
            seq = _clean_sequence(seq)
            if seq is None:
                continue
            if args.kinematics:
                seq = add_kinematics(seq)
            X.append(seq)
            y.append(cls_id)
            rows.append([f"{match_name}_{frame}", SHOT_CLASSES[cls_id], match_name])

    if not X:
        raise SystemExit("未转换出任何样本，请检查数据路径与匹配名。")
    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    save_dataset(args.out, X, y, SHOT_CLASSES, manifest_rows=rows)
    print(f"已转换 {len(X)} 个样本到 {args.out}，X.shape={X.shape}")


if __name__ == "__main__":
    main()

