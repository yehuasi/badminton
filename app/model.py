# -*- coding: utf-8 -*-
"""击球类型识别模型：轻量 GRU 时序分类器。

输入：(B, T, F) 的逐帧特征序列；输出：(B, 7) 的类别 logits。
CPU 可训练、易部署，后续若真实数据精度不足可升级为 ST-GCN。
"""

import json

import torch
from torch import nn

from app.classes import SHOT_CLASSES
from app.features import F


class ShotClassifier(nn.Module):
    def __init__(self, in_dim=F, hidden=96, layers=2, num_classes=len(SHOT_CLASSES)):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.num_classes = num_classes
        self.gru = nn.GRU(
            in_dim,
            hidden,
            layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))

    def forward(self, x):
        # x: (B, T, F)
        _, h = self.gru(x)          # h: (layers, B, hidden)
        return self.head(h[-1])     # (B, num_classes)


def save_checkpoint(path, model, config, classes=None):
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "classes": list(classes) if classes is not None else SHOT_CLASSES,
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = payload.get("config", {})
    model = ShotClassifier(
        in_dim=config.get("in_dim", F),
        hidden=config.get("hidden", 96),
        layers=config.get("layers", 2),
        num_classes=config.get("num_classes", len(SHOT_CLASSES)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    classes = payload.get("classes", SHOT_CLASSES)
    return model, classes

