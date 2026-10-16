# -*- coding: utf-8 -*-
"""用验证集检验标准度打分：正确样本应得高分少误报，错误样本应被扣分并命中错误提示。

从特征缓存读取验证集（含正确/错误标签与错误族），在击球帧上跑 score_standardness，
输出：正确 vs 错误样本的平均得分、命中任意错误提示的比例，以及两族新增错误
（侧身与转体、手腕与拍面）各自被命中的情况。

用法：
    .venv\\Scripts\\python scripts\\eval_scoring.py
"""

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT  # noqa: E402
from app.features import FEATURE_NAMES_FULL  # noqa: E402
from app.score import score_standardness  # noqa: E402

NEW_KEYS = ("侧身", "拍面", "手腕", "肘部")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(REPO_ROOT, "data", "feature_cache_ext"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "scoring_validation.json"))
    args = ap.parse_args()

    rows = []
    for fp in glob.glob(os.path.join(args.cache, "*.npz")):
        z = np.load(fp, allow_pickle=False)
        if str(z["split"]) != "验证":
            continue
        cls = str(z["class_name"])
        label = str(z["label"])
        family = str(z["error_family"]) or ""
        seq = z["seq"]
        hit = seq[seq.shape[0] // 2]
        feat = {name: (None if np.isnan(float(hit[j])) else float(hit[j]))
                for j, name in enumerate(FEATURE_NAMES_FULL)}
        score, errors = score_standardness(feat, action=cls)
        rows.append({
            "class": cls, "label": label, "family": family,
            "score": round(score, 1), "errors": errors,
            "new_hit": any(k in e for e in errors for k in NEW_KEYS),
        })

    correct = [r for r in rows if r["label"] == "正确"]
    wrong = [r for r in rows if r["label"] == "错误"]

    def stat(rs):
        if not rs:
            return (0, 0, 0)
        scores = [r["score"] for r in rs]
        flagged = sum(1 for r in rs if r["errors"])
        return (float(np.mean(scores)), float(np.median(scores)), flagged / len(rs))

    print(f"验证集：正确 {len(correct)} / 错误 {len(wrong)}")
    cm, cmed, cf = stat(correct)
    wm, wmed, wf = stat(wrong)
    print(f"正确样本：均分 {cm:.1f}  中位 {cmed:.1f}  命中错误提示 {cf:.1%}")
    print(f"错误样本：均分 {wm:.1f}  中位 {wmed:.1f}  命中错误提示 {wf:.1%}")
    print(f"分离度（正确均分 - 错误均分）：{cm - wm:+.1f} 分\n")

    print("=== 各错误族命中率（该族错误样本中被提示的比例）===")
    by_family = collections.defaultdict(list)
    for r in wrong:
        by_family[r["family"] or "(未归族)"].append(r)
    for fam in sorted(by_family, key=lambda f: -len(by_family[f])):
        rs = by_family[fam]
        hit = sum(1 for r in rs if r["errors"]) / len(rs)
        new = sum(1 for r in rs if r["new_hit"]) / len(rs)
        print(f"  {fam:<20} n={len(rs):>3}  任意提示 {hit:.1%}  含新增规则 {new:.1%}")

    print("\n=== 两类新增覆盖的命中明细 ===")
    for fam in ("侧身与转体", "手腕与拍面控制"):
        rs = [r for r in wrong if r["family"] == fam]
        if not rs:
            print(f"  {fam}：无样本")
            continue
        for r in rs:
            hit = [e for e in r["errors"] if any(k in e for k in NEW_KEYS)]
            print(f"  [{r['class']}] 得分 {r['score']:>5.1f} 命中 {len(hit)} 条：{'；'.join(hit) if hit else '（未命中）'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"correct": {"mean": cm, "median": cmed, "flagged": cf},
                   "error": {"mean": wm, "median": wmed, "flagged": wf},
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"\n逐条结果：{args.out}")


if __name__ == "__main__":
    main()
