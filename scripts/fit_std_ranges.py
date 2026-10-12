# -*- coding: utf-8 -*-
"""用专业示范数据集拟合标准库角度区间（回填 app/score.py 的 STD_RANGES）。

输入：处理好的数据集目录（含 X.npz / classes.json，由 prepare_selfcollected.py
      或 prepare_bfmd.py 产出）。
做法：对每个样本窗口定位击球帧，取击球帧的各特征值，按动作类别聚合，
      对 STD_RANGES 里列出的每个目标特征统计“均值±k×σ”，作为建议区间。
输出：打印一份可直接替换的 STD_RANGES 草案，并可写入 JSON 供复核。
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.analyzer import detect_hit_frame  # noqa: E402
from app.features import FEATURE_NAMES  # noqa: E402
from app.score import STD_RANGES  # noqa: E402
from scripts.common import load_dataset  # noqa: E402


def _round(value):
    return round(float(value), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="处理好的数据集目录")
    parser.add_argument("--k", type=float, default=2.0, help="区间半宽 = k×σ，默认 2.0")
    parser.add_argument("--min-samples", type=int, default=5, help="每类最少样本数")
    parser.add_argument("--stat", default="sigma", choices=["sigma", "percentile"],
                        help="区间来源：sigma=均值±kσ；percentile=分位区间[lo%%,hi%%]（更抗离群）")
    parser.add_argument("--lo", type=float, default=10.0, help="percentile 模式下限分位，默认 10")
    parser.add_argument("--hi", type=float, default=90.0, help="percentile 模式上限分位，默认 90")
    parser.add_argument("--hit", default="detect", choices=["detect", "center"],
                        help="击球帧取法：detect=挥速峰值（自采切分窗口）；"
                             "center=窗口中心（BFMD 等以击球帧居中的窗口）")
    parser.add_argument("--out", default=None, help="可选：把建议区间写入 JSON")
    args = parser.parse_args()

    x, y, classes = load_dataset(args.data)
    print(f"数据集：{x.shape}，类别 {len(classes)}")

    # 收集每个类别、每个目标特征在击球帧的取值
    values = {
        cls: {feat: [] for feat in STD_RANGES.get(cls, {})}
        for cls in classes
    }
    for i in range(len(y)):
        cls = classes[int(y[i])]
        if cls not in STD_RANGES:
            continue
        seq = x[i]
        hit = (seq.shape[0] // 2) if args.hit == "center" else detect_hit_frame(seq)
        for feat in STD_RANGES[cls]:
            if feat in FEATURE_NAMES:
                values[cls][feat].append(float(seq[hit, FEATURE_NAMES.index(feat)]))

    proposed = {}
    print("\n建议区间（feature: mean ± kσ -> [low, high]，对比当前占位值）：")
    for cls in classes:
        if cls not in STD_RANGES:
            continue
        row = {}
        print(f"\n[{cls}]")
        for feat, cur in STD_RANGES[cls].items():
            vals = np.asarray(values[cls].get(feat, []), dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size < args.min_samples:
                print(f"  {feat}: 样本不足（{vals.size}）-> 保留当前 {cur}")
                row[feat] = list(cur)
                continue
            mean, std = float(vals.mean()), float(vals.std())
            if args.stat == "percentile":
                lo = float(np.percentile(vals, args.lo))
                hi = float(np.percentile(vals, args.hi))
                desc = f"p{args.lo:g}~p{args.hi:g} -> [{_round(lo)}, {_round(hi)}]"
            else:
                lo, hi = mean - args.k * std, mean + args.k * std
                desc = f"{_round(mean)} ± {args.k}×{_round(std)} -> [{_round(lo)}, {_round(hi)}]"
            row[feat] = [_round(lo), _round(hi)]
            print(f"  {feat}: {desc}  (当前 {list(cur)})")
        proposed[cls] = row

    print("\n\n建议 STD_RANGES（可直接替换 score.py 中的表）：")
    print(json.dumps(proposed, ensure_ascii=False, indent=4))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(proposed, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
