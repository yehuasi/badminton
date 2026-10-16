# -*- coding: utf-8 -*-
"""在指定数据集上**全量**评估一个 checkpoint（不做内部划分），用于跨集合评测。

例如：用标准集训练、拿验证集全部样本测试，得到干净的跨域指标。

用法：
    .venv\\Scripts\\python scripts\\eval_cross.py --ckpt models\\shot_classifier_std.pt --data data\\processed_valid2
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_dataset  # noqa: E402
from app.model import load_checkpoint  # noqa: E402
from app.analyzer import _hit_group  # noqa: E402
from app.classes import group_allowed_classes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="可选：逐条预测写入 JSON")
    ap.add_argument("--no-gate", action="store_true",
                    help="关闭过顶/非过顶分组门控，直接 7 类取最大")
    args = ap.parse_args()

    x, y, classes = load_dataset(args.data)
    model, ckpt_classes = load_checkpoint(args.ckpt, map_location="cpu")
    if list(ckpt_classes) != list(classes):
        print(f"注意：checkpoint 类别顺序与数据集不同，按 checkpoint 顺序重排。")
    import torch

    gate_ok = 0
    pred, true = [], []
    with torch.no_grad():
        for i in range(len(y)):
            seq = np.nan_to_num(x[i], nan=0.0)
            logits = model(torch.from_numpy(seq)[None])[0]
            if not args.no_gate:
                allowed = group_allowed_classes(_hit_group(seq, 12))
                idxs = [list(classes).index(c) for c in allowed if c in list(classes)]
                if idxs:
                    mask = torch.full_like(logits, float("-inf"))
                    mask[idxs] = logits[idxs]
                    logits = mask
            p = int(logits.argmax().item())
            pred.append(classes[p])
            true.append(classes[int(y[i])])

    pred = np.asarray(pred)
    true = np.asarray(true)
    acc = float((pred == true).mean())
    print(f"数据集 {args.data}：{len(true)} 个样本，类别顺序 {'一致' if list(ckpt_classes)==list(classes) else '不一致'}")
    print(f"全量准确率：{acc*100:.1f}%  ({int((pred==true).sum())}/{len(true)})"
          + ("（已启用过顶/非过顶门控）" if not args.no_gate else "（无门控）"))
    print(f"\n{'类别':<12}{'样本':>5}{'精确率':>8}{'召回率':>8}")
    for c in classes:
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        n_true = int((true == c).sum())
        print(f"{c:<12}{n_true:>5}{(tp/(tp+fp)*100 if tp+fp else 0):>7.1f}%"
              f"{(tp/(tp+fn)*100 if tp+fn else 0):>7.1f}%")

    conf = collections.Counter(zip(true, pred))
    print("\n主要混淆（真实→预测，前 8）：")
    for (t, p), n in conf.most_common(8):
        if t != p:
            print(f"  {t} → {p}：{n}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"data": args.data, "ckpt": args.ckpt, "accuracy": acc,
                       "pairs": [{"true": t, "pred": p} for t, p in zip(true.tolist(), pred.tolist())]},
                      f, ensure_ascii=False, indent=1)
        print(f"\n逐条结果：{args.out}")


if __name__ == "__main__":
    main()
