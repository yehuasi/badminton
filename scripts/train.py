# -*- coding: utf-8 -*-
"""训练击球类型识别模型（GRU）。"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from common import (
    REPO_ROOT,
    load_dataset,
    load_manifest,
    make_loaders_with_indices,
)
from app.model import ShotClassifier, save_checkpoint


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="数据集目录，默认 data/processed")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-by-match", action="store_true",
                        help="按来源比赛划分验证集，避免同场数据泄漏")
    parser.add_argument("--class-weight", action="store_true",
                        help="用逆频率类别权重缓解类别不均衡")
    parser.add_argument("--kinematics", action="store_true",
                        help="用 23 维运动学数据集（15 维 + 速度/加速度）训练，"
                             "专治高远球/吊球混淆；数据目录切换为 data/processed_bfmd_kin")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "models", "shot_classifier.pt"))
    parser.add_argument("--extra", default=None,
                        help="额外数据集目录（逗号分隔），与 --data 合并训练；类别须一致")
    parser.add_argument("--init", default=None,
                        help="从已有 checkpoint 初始化权重（微调用），例如 models/shot_classifier_bfmd.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data or (
        os.path.join(REPO_ROOT, "data", "processed_bfmd_kin")
        if args.kinematics
        else os.path.join(REPO_ROOT, "data", "processed")
    )
    x, y, classes = load_dataset(data_dir)
    print(f"数据集：{x.shape}，类别 {len(classes)}")
    if args.extra:
        for extra_dir in [s.strip() for s in args.extra.split(",") if s.strip()]:
            xe, ye, ce = load_dataset(extra_dir)
            if list(ce) != list(classes):
                raise SystemExit(f"类别不一致：{extra_dir} 与主数据集不同，请先统一类别顺序。")
            x = np.concatenate([x, xe], axis=0)
            y = np.concatenate([y, ye], axis=0)
            print(f"已合并 {extra_dir}：+{xe.shape[0]} 个样本 -> 共 {x.shape[0]}")

    manifest = load_manifest(data_dir) if args.split_by_match else None
    train_loader, val_loader = make_loaders_with_indices(
        x, y, batch_size=args.batch_size, val_frac=0.25, seed=args.seed,
        manifest=manifest, split_by_match=args.split_by_match,
    )

    model = ShotClassifier(in_dim=x.shape[2], hidden=args.hidden, layers=args.layers,
                           num_classes=len(classes))
    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        print(f"已从 {args.init} 初始化权重（微调）")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weights = None
    if args.class_weight:
        counts = np.bincount(y, minlength=len(classes)).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        class_weights = torch.tensor(len(y) / (len(classes) * counts), dtype=torch.float32)
        print(f"类别权重：{class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_acc, n = 0.0, 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_acc += accuracy(logits, yb) * len(yb)
            n += len(yb)
        scheduler.step()

        model.eval()
        val_loss, val_acc, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                val_acc += accuracy(logits, yb) * len(yb)
                nv += len(yb)

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss/n:.3f} train_acc={train_acc/n:.3f}  "
            f"val_loss={val_loss/nv:.3f} val_acc={val_acc/nv:.3f}  "
            f"({time.time()-t0:.1f}s)"
        )

    config = {
        "in_dim": x.shape[2],
        "hidden": args.hidden,
        "layers": args.layers,
        "num_classes": len(classes),
        "T": x.shape[1],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_checkpoint(args.out, model, config)
    with open(os.path.splitext(args.out)[0] + "_classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"已保存模型：{args.out}")


if __name__ == "__main__":
    main()

