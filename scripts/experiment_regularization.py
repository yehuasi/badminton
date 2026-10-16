# -*- coding: utf-8 -*-
"""过拟合消融：模型大小 × 数据增强，多次随机划分取平均。

背景：自采数据上训练准确率 100%、留出准确率 53%~57%，差距说明样本量不足以支撑
当前模型容量。本脚本在同一划分下比较「小模型」「时间抖动/噪声增强」「早停」的效果，
用多次随机划分的平均值给出可信结论（单次划分的方差很大）。

用法：
    .venv\\Scripts\\python scripts\\experiment_regularization.py --data data\\self_ext --seeds 3
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT, augment_batch, load_dataset  # noqa: E402
from app.model import ShotClassifier  # noqa: E402


def run_once(x_tr, y_tr, x_va, y_va, hidden, layers, augment, epochs, lr, batch_size, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = ShotClassifier(in_dim=x_tr.shape[2], hidden=hidden, layers=layers,
                           num_classes=int(y_tr.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    counts = np.bincount(y_tr, minlength=int(y_tr.max()) + 1).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(len(y_tr) / (len(counts) * counts), dtype=torch.float32))
    xv = torch.from_numpy(np.nan_to_num(x_va, nan=0.0)).float()
    yv = torch.from_numpy(y_va).long()
    best = 0.0
    for _epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(y_tr))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = x_tr[idx]
            if augment:
                xb = augment_batch(xb, rng)
            opt.zero_grad()
            loss = crit(model(torch.from_numpy(np.nan_to_num(xb, nan=0.0)).float()),
                        torch.from_numpy(y_tr[idx]).long())
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xv).argmax(1) == yv).float().mean())
        best = max(best, acc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "self_ext"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "regularization_ablation.json"))
    args = ap.parse_args()

    x, y, classes = load_dataset(args.data)
    print(f"数据集 {args.data}：{x.shape}")
    configs = [
        ("h96_l2_无增强", dict(hidden=96, layers=2, augment=False)),
        ("h96_l2_带增强", dict(hidden=96, layers=2, augment=True)),
        ("h48_l2_带增强", dict(hidden=48, layers=2, augment=True)),
        ("h32_l1_带增强", dict(hidden=32, layers=1, augment=True)),
        ("h48_l1_无增强", dict(hidden=48, layers=1, augment=False)),
    ]
    results = []
    for name, cfg in configs:
        accs = []
        for seed in range(args.seeds):
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(y))
            n_val = max(1, int(len(y) * 0.25))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            acc = run_once(x[tr_idx], y[tr_idx], x[val_idx], y[val_idx],
                           epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                           seed=seed, **cfg)
            accs.append(acc)
        mean, std = float(np.mean(accs)), float(np.std(accs))
        results.append({"config": name, "mean_val_acc": round(mean, 4),
                        "std": round(std, 4), "runs": [round(a, 4) for a in accs], **cfg})
        print(f"  {name:<16} 留出准确率 {mean:.1%} ± {std:.1%}   各次 {[f'{a:.1%}' for a in accs]}",
              flush=True)

    best = max(results, key=lambda r: r["mean_val_acc"])
    print(f"\n最佳配置：{best['config']} → {best['mean_val_acc']:.1%}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"data": args.data, "epochs": args.epochs, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"结果已写入：{args.out}")


if __name__ == "__main__":
    main()
