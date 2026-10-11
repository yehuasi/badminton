# -*- coding: utf-8 -*-
"""实时击球类型识别封装：滚动窗口缓冲特征，输出 Top-K 类别与置信度。"""

from collections import deque

import numpy as np
import torch

from app.features import compute_features
from app.model import load_checkpoint


class ShotClassifierRunner:
    def __init__(self, ckpt, T=24, handedness="right", topk=3, device="cpu"):
        self.model, self.classes = load_checkpoint(ckpt, map_location=device)
        self.model.to(device).eval()
        self.T = T
        self.topk = topk
        self.handedness = handedness
        self.device = device
        self.buffer = deque(maxlen=T)

    def update(self, kpts):
        feat = compute_features(kpts, handedness=self.handedness)
        self.buffer.append(feat)

    def update_features(self, feat):
        """直接推入一帧特征向量（合成/演示模式用）。"""
        self.buffer.append(np.asarray(feat, dtype=np.float32))

    @property
    def ready(self):
        return len(self.buffer) == self.T

    def _topk(self, x, allowed=None):
        """x: (T, F) numpy 数组，返回 [(类别, 概率), ...]。

        allowed 为允许的类别名集合时，在完整 logits 上屏蔽其它类别并重新归一化，
        用于两阶段分类的“过顶 / 非过顶”门控。
        """
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(x)[None].to(self.device)  # (1, T, F)
        with torch.no_grad():
            logits = self.model(x)[0]
            if allowed is not None:
                mask = torch.zeros_like(logits, dtype=torch.bool)
                idxs = [self.classes.index(c) for c in allowed if c in self.classes]
                mask[idxs] = True
                logits = logits.masked_fill(~mask, float("-inf"))
            prob = torch.softmax(logits, dim=-1)
        order = prob.argsort(descending=True)[: self.topk]
        return [(self.classes[i], float(prob[i])) for i in order]

    def predict(self):
        x = np.stack(list(self.buffer), axis=0).astype(np.float32)  # (T, F)
        return self._topk(x)

    def predict_sequence(self, seq, allowed=None):
        """直接对一段完整序列 (T, F) 做预测（演示模式用）。"""
        return self._topk(np.asarray(seq, dtype=np.float32), allowed=allowed)

