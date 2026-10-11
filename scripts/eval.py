# -*- coding: utf-8 -*-
"""评估训练好的击球类型识别模型：总体准确率 + 各类别精确率/召回率。"""

import argparse
import os

import numpy as np
import torch

from common import (
    REPO_ROOT,
    load_dataset,
    load_manifest,
    make_loaders_with_indices,
)
from app.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--ckpt", default=os.path.join(REPO_ROOT, "models", "shot_classifier.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-by-match", action="store_true")
    args = parser.parse_args()

    data_dir = args.data or os.path.join(REPO_ROOT, "data", "processed")
    x, y, classes = load_dataset(data_dir)
    model, ckpt_classes = load_checkpoint(args.ckpt, map_location="cpu")

    manifest = load_manifest(data_dir) if args.split_by_match else None
    _, val_loader = make_loaders_with_indices(
        x, y, batch_size=args.batch_size, val_frac=0.25, seed=0,
        manifest=manifest, split_by_match=args.split_by_match,
    )

    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            all_pred.append(logits.argmax(dim=1).numpy())
            all_true.append(yb.numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    total = len(true)
    print(f"总体准确率：{(pred == true).mean()*100:.1f}%  ({int((pred==true).sum())}/{total})")
    print("各类别指标：")
    print(f"{'类别':<12} {'样本':>5} {'精确率':>8} {'召回率':>8}")
    for cls_id, name in enumerate(classes):
        tp = int(((pred == cls_id) & (true == cls_id)).sum())
        fp = int(((pred == cls_id) & (true != cls_id)).sum())
        fn = int(((pred != cls_id) & (true == cls_id)).sum())
        n_true = int((true == cls_id).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"{name:<12} {n_true:>5} {precision*100:>7.1f}% {recall*100:>7.1f}%")


if __name__ == "__main__":
    main()

