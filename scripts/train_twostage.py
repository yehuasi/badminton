# -*- coding: utf-8 -*-
"""两阶段击球类型识别：先判「过顶 / 非过顶」，再在组内细分类别。

背景：单用「腕相对头顶高度」这一个量就能把 7 类分对约 84%，说明「先分组」比「七类平推」
更稳；组内细分才是特征区分度的难点。本脚本把两阶段做成可训练、可评测的版本，
并与七类平推做同口径对比（同一划分、同一轮数）。

用法：
    .venv\\Scripts\\python scripts\\train_twostage.py --data data\\self_ext --out-prefix models\\ts_self
    .venv\\Scripts\\python scripts\\train_twostage.py --data data\\std_ext --test data\\val_ext --out-prefix models\\ts_std
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT, load_dataset  # noqa: E402
from app.classes import (  # noqa: E402
    NON_OVERHEAD_CLASSES,
    OVERHEAD_CLASSES,
    shot_group,
)
from app.model import ShotClassifier, save_checkpoint  # noqa: E402

GROUPS = ["overhead", "non_overhead"]


def train_model(x, y, num_classes, epochs, lr, batch_size, seed, class_weight=True):
    torch.manual_seed(seed)
    model = ShotClassifier(in_dim=x.shape[2], hidden=96, layers=2, num_classes=num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if class_weight:
        counts = np.bincount(y, minlength=num_classes).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        weight = torch.tensor(len(y) / (num_classes * counts), dtype=torch.float32)
    else:
        weight = None
    crit = nn.CrossEntropyLoss(weight=weight)
    xt = torch.from_numpy(np.nan_to_num(x, nan=0.0)).float()
    yt = torch.from_numpy(np.asarray(y)).long()
    n = len(yt)
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def predict_probs(model, x):
    with torch.no_grad():
        logits = model(torch.from_numpy(np.nan_to_num(x, nan=0.0)).float())
        return torch.softmax(logits, dim=-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="训练集目录")
    ap.add_argument("--test", default=None, help="可选：用于跨来源全量评测的数据集目录")
    ap.add_argument("--out-prefix", default=os.path.join(REPO_ROOT, "models", "twostage"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    args = ap.parse_args()

    x, y, classes = load_dataset(args.data)
    classes = list(classes)
    print(f"训练集 {args.data}：{x.shape}，类别 {classes}")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(y))
    n_val = max(1, int(len(y) * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[val_idx], y[val_idx]

    flat = train_model(x_tr, y_tr, len(classes), args.epochs, args.lr, args.batch_size, args.seed)
    flat_pred_va = predict_probs(flat, x_va).argmax(1)
    flat_acc_va = float((flat_pred_va == y_va).mean())

    g_ids = np.array([GROUPS.index(shot_group(c)) for c in classes])
    group_y = g_ids[y]
    group_model = train_model(x_tr, group_y[tr_idx], len(GROUPS), args.epochs, args.lr,
                              args.batch_size, args.seed)

    within_models, within_classes = {}, {}
    for gname, gcls in (("overhead", OVERHEAD_CLASSES), ("non_overhead", NON_OVERHEAD_CLASSES)):
        gid = GROUPS.index(gname)
        mask = group_y[tr_idx] == gid
        sub_y = np.array([gcls.index(classes[y[i]]) for i in tr_idx[mask]])
        within_models[gname] = train_model(x_tr[mask], sub_y, len(gcls), args.epochs,
                                           args.lr, args.batch_size, args.seed)
        within_classes[gname] = gcls

    def two_stage_probs(xx):
        pg = predict_probs(group_model, xx)
        out = np.zeros((len(xx), len(classes)), dtype=np.float64)
        for gname, gcls in within_classes.items():
            pw = predict_probs(within_models[gname], xx)
            for j, cname in enumerate(gcls):
                out[:, classes.index(cname)] = pg[:, GROUPS.index(gname)] * pw[:, j]
        return out

    def report(tag, xx, yy, prefix=""):
        probs = two_stage_probs(xx)
        ts_pred = probs.argmax(1)
        f_acc = float((predict_probs(flat, xx).argmax(1) == yy).mean())
        t_acc = float((ts_pred == yy).mean())
        g_pred = predict_probs(group_model, xx).argmax(1)
        g_acc = float((g_pred == g_ids[yy]).mean())
        right_group = g_pred == g_ids[yy]
        within = float((ts_pred[right_group] == yy[right_group]).mean()) if right_group.any() else 0.0
        print(f"{prefix}{tag}：七类平推 {f_acc:.1%}　两阶段 {t_acc:.1%}　"
              f"分组准确率 {g_acc:.1%}　组内条件准确率 {within:.1%}（n={len(yy)}）")
        return {"seven_class": f_acc, "two_stage": t_acc, "group_acc": g_acc,
                "within_acc": within, "n": int(len(yy))}

    metrics = {"data": args.data, "shape": list(x.shape), "in_distribution": report("同分布留出", x_va, y_va)}
    if args.test:
        xt, yt, ct = load_dataset(args.test)
        if list(ct) != classes:
            raise SystemExit(f"类别顺序不一致：{args.test}")
        metrics["cross_source"] = report("跨来源全量", xt, yt, prefix="")

    for name, model, cls_names in (
        ("flat", flat, classes),
        ("group", group_model, GROUPS),
        ("overhead", within_models["overhead"], OVERHEAD_CLASSES),
        ("non_overhead", within_models["non_overhead"], NON_OVERHEAD_CLASSES),
    ):
        path = f"{args.out_prefix}_{name}.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_checkpoint(path, model, {"in_dim": x.shape[2], "hidden": 96, "layers": 2,
                                      "num_classes": model.num_classes, "T": x.shape[1],
                                      "stage": name}, classes=cls_names)
        print(f"已保存 {path}")

    with open(args.out_prefix + "_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"指标已写入 {args.out_prefix}_metrics.json")


if __name__ == "__main__":
    main()
