# -*- coding: utf-8 -*-
"""从特征缓存（data/feature_cache）重建数据集，不必重跑姿态检测。

用途：同一批视频要按不同筛选条件反复出数据集时（标准集回填、正确/错误分档、
按视角切分、换窗口长度），只跑一次姿态检测，之后都从这里瞬间生成。

用法：
    # 标准集（用于回填 STD_RANGES）
    .venv\\Scripts\\python scripts\\build_dataset_from_cache.py --split 标准 --label 正确 --out data\\processed_std
    # 验证集里的错误样本（用于检验扣分是否生效）
    .venv\\Scripts\\python scripts\\build_dataset_from_cache.py --split 验证 --label 错误 --out data\\processed_err
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT, save_dataset  # noqa: E402
from app.classes import SHOT_CLASSES  # noqa: E402
from app.analyzer import _swing_speed  # noqa: E402
from app.features import add_kinematics  # noqa: E402
from prepare_selfcollected import _windows_for_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(REPO_ROOT, "data", "feature_cache"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="", help="标准 / 验证，留空为不限")
    ap.add_argument("--label", default="", help="正确 / 错误，留空为不限")
    ap.add_argument("--view", default="", help="正侧面 / 正面 / 侧面，留空为不限")
    ap.add_argument("--classes", default="", help="只要这些类别，逗号分隔")
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--kinematics", action="store_true", help="追加速度/加速度特征（15→23 维）")
    ap.add_argument("--per-video", type=int, default=0,
                    help="每个视频最多取几个窗口（0=全部，1=只取挥速峰值最强的一个窗口）")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    if not files:
        raise SystemExit(f"缓存目录为空：{args.cache}\n先跑 scripts/prepare_selfcollected.py --cache ...")

    wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
    X, y, rows = [], [], []
    used = {}
    for fp in files:
        z = np.load(fp, allow_pickle=False)
        cls = str(z["class_name"])
        if cls not in SHOT_CLASSES:
            continue
        if wanted and cls not in wanted:
            continue
        if args.split and str(z["split"]) != args.split:
            continue
        if args.label and str(z["label"]) != args.label:
            continue
        if args.view and str(z["view"]) != args.view:
            continue
        seq = z["seq"]
        wins = _windows_for_video(seq, args.T)
        if not wins:
            continue
        if args.per_video:
            if args.per_video == 1:
                # 一条一个动作的短录像，活动触发切分可能把“准备/起步”也切成一段，
                # 取挥速峰值最强的窗口（真正击球那一拍），避免低运动量片段变成标签噪声。
                best = max(range(len(wins)), key=lambda i: float(_swing_speed(wins[i]).max()))
                wins = [wins[best]]
            else:
                wins = wins[: args.per_video]
        used[cls] = used.get(cls, 0) + 1
        for k, w in enumerate(wins):
            if args.kinematics:
                w = add_kinematics(w)
            X.append(w)
            y.append(SHOT_CLASSES.index(cls))
            rows.append([os.path.splitext(os.path.basename(str(z["path"])))[0] + f"_{k}",
                         cls, os.path.basename(str(z["path"])),
                         str(z["split"]), str(z["label"]), str(z["view"])])

    if not X:
        raise SystemExit("筛选后没有任何样本，放宽条件再试。")
    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    save_dataset(args.out, X, y, SHOT_CLASSES, manifest_rows=rows)
    print("各类使用视频数：" + "，".join(f"{k} {v}" for k, v in sorted(used.items())))
    print(f"已生成 {args.out}：X.shape={X.shape}，样本 {len(y)}，类别 {len(SHOT_CLASSES)}")
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump({"cache": args.cache, "filters": vars(args), "used": used},
                      f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
