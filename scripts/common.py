# -*- coding: utf-8 -*-
"""训练链路共用工具：数据集存取、划分、DataLoader。"""

import json
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes import SHOT_CLASSES  # noqa: E402
from app.features import F  # noqa: E402

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")


class SkeletonDataset(Dataset):
    """x: (N, T, F) float32，y: (N,) int64。"""

    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def augment_batch(xb, rng, noise=0.02, jitter=0.12):
    """训练增强：时间抖动（模拟快慢挥拍）+ 小幅噪声（模拟关键点抖动）。

    自采数据只有 600 多条，训练准确率能到 100%、留出只有 53%~59%，属于样本量不足。
    消融实验（reports/regularization_ablation.json）显示，这项增强把留出准确率从
    57.1% 抬到 58.7%（3 次随机划分平均），是当前最稳的正则化手段。
    """
    out = np.empty_like(xb)
    T = xb.shape[1]
    for i in range(xb.shape[0]):
        rate = 1.0 + rng.uniform(-jitter, jitter)
        src = np.clip(np.arange(T) * rate - (rate - 1.0) * T / 2.0, 0, T - 1)
        lo = np.floor(src).astype(int)
        hi = np.clip(lo + 1, 0, T - 1)
        w = (src - lo)[:, None]
        out[i] = xb[i][lo] * (1 - w) + xb[i][hi] * w
    std = out.std(axis=(0, 1), keepdims=True)
    out = out + rng.normal(0.0, noise, size=out.shape).astype(np.float32) * std
    return out


def save_dataset(directory, x, y, classes, manifest_rows=None):
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        os.path.join(directory, "X.npz"),
        X=x.astype(np.float32),
        y=y.astype(np.int64),
    )
    with open(os.path.join(directory, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    if manifest_rows:
        import csv

        with open(os.path.join(directory, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "class", "source"])
            for row in manifest_rows:
                writer.writerow(row)


def load_dataset(directory=PROCESSED_DIR):
    with np.load(os.path.join(directory, "X.npz")) as data:
        x = data["X"]
        y = data["y"]
    with open(os.path.join(directory, "classes.json"), "r", encoding="utf-8") as f:
        classes = json.load(f)
    return x, y, classes


def load_manifest(directory=PROCESSED_DIR):
    """读取 manifest.csv，返回 list[dict]；没有该文件时返回空列表。"""
    path = os.path.join(directory, "manifest.csv")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_split(y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    val_idx = np.array(sorted(val_idx), dtype=np.int64)
    return train_idx, val_idx


def split_by_source(manifest, val_frac=0.25, seed=0):
    """按 source（如比赛名）划分：部分来源的样本整体放入验证集，避免同场泄漏。"""
    rng = np.random.default_rng(seed)
    sources = sorted({row["source"] for row in manifest})
    rng.shuffle(sources)
    n_val_sources = max(1, int(round(len(sources) * val_frac)))
    val_sources = set(sources[:n_val_sources])
    train_idx, val_idx = [], []
    for i, row in enumerate(manifest):
        (val_idx if row["source"] in val_sources else train_idx).append(i)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def make_loaders(x, y, batch_size=32, val_frac=0.2, seed=0, num_workers=0):
    return make_loaders_with_indices(x, y, batch_size, val_frac, seed, num_workers)


def make_loaders_with_indices(x, y, batch_size=32, val_frac=0.2, seed=0,
                              num_workers=0, manifest=None, split_by_match=False):
    if split_by_match and manifest:
        train_idx, val_idx = split_by_source(manifest, val_frac, seed)
    else:
        train_idx, val_idx = stratified_split(y, val_frac, seed)
    train_loader = DataLoader(
        SkeletonDataset(x[train_idx], y[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        SkeletonDataset(x[val_idx], y[val_idx]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader

