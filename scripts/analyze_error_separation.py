# -*- coding: utf-8 -*-
"""数据驱动的错误规则发现：哪些特征能把「正确」和「错误」样本分开。

只读特征缓存（data/feature_cache_ext，24 维扩展特征），逐类逐错误族计算：
正确样本 vs 错误样本在击球帧（窗口中心）各特征上的差异（Cohen's d 效应量）。
据此决定给评分规则加哪些扩展特征、方向取 low 还是 high，避免拍脑袋定规则。

用法：
    .venv\\Scripts\\python scripts\\analyze_error_separation.py
    .venv\\Scripts\\python scripts\\analyze_error_separation.py --min-n 3 --top 6
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT  # noqa: E402
from app.features import FEATURE_NAMES_FULL  # noqa: E402


def effect_size(a, b):
    """Cohen's d（带符号，b 为错误组），正值表示错误组偏高。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt((sa * sa + sb * sb) / 2.0) if (sa + sb) > 0 else 0.0
    if pooled < 1e-9:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(REPO_ROOT, "data", "feature_cache_ext"))
    ap.add_argument("--min-n", type=int, default=3, help="错误族样本少于该数则跳过")
    ap.add_argument("--top", type=int, default=6, help="每类每族显示前几个特征")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "error_separation.json"))
    args = ap.parse_args()

    by_class = {}
    for fp in glob.glob(os.path.join(args.cache, "*.npz")):
        z = np.load(fp, allow_pickle=False)
        cls = str(z["class_name"])
        split = str(z["split"])
        label = str(z["label"])
        family = str(z["error_family"]) if str(z["error_family"]) != "" else ""
        seq = z["seq"]
        if seq.shape[0] < 1:
            continue
        hit = seq[seq.shape[0] // 2]
        by_class.setdefault(cls, {"正确": [], "错误": {}})
        if label == "错误":
            by_class[cls]["错误"].setdefault(family, []).append(hit)
        elif label == "正确":
            by_class[cls]["正确"].append(hit)

    report = {}
    print("=== 各错误族 vs 正确样本 的击球帧特征差异（Cohen's d，|d|>0.5 才算有意义）===\n")
    for cls in sorted(by_class):
        correct = np.asarray(by_class[cls]["正确"], dtype=float)
        if correct.ndim != 2:
            continue
        for family, errs in sorted(by_class[cls]["错误"].items()):
            err = np.asarray(errs, dtype=float)
            if err.shape[0] < args.min_n:
                continue
            row = {}
            for j, name in enumerate(FEATURE_NAMES_FULL):
                row[name] = round(effect_size(correct[:, j], err[:, j]), 3)
            ranked = sorted(row.items(), key=lambda kv: -abs(kv[1]))[: args.top]
            report.setdefault(cls, {})[family] = row
            print(f"[{cls} / {family}] 正确 {correct.shape[0]} vs 错误 {err.shape[0]}")
            for name, d in ranked:
                mark = "↑错误偏高" if d > 0.5 else ("↓错误偏低" if d < -0.5 else "≈")
                print(f"    {name:<24} d={d:+.2f}  {mark}")
            print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"完整效应量已写入：{args.out}")


if __name__ == "__main__":
    main()
