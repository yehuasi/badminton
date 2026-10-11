# -*- coding: utf-8 -*-
"""基于 ONNX Runtime 的动作分类器推理（端侧部署用，不依赖 torch）。"""

import json

import numpy as np

from app.classes import SHOT_CLASSES


class ONNXShotClassifier:
    def __init__(self, weights, classes=None, topk=3, threads=4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(weights, providers=["CPUExecutionProvider"], sess_options=opts)
        self.input_name = self.sess.get_inputs()[0].name
        self.classes = classes or SHOT_CLASSES
        self.topk = topk

    def predict_sequence(self, seq, allowed=None):
        x = np.asarray(seq, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim == 2:
            x = x[None]  # (1,T,F)
        logits = self.sess.run(None, {self.input_name: x})[0][0]  # (C,)
        logits = logits - logits.max()
        if allowed is not None:
            keep = [i for i, c in enumerate(self.classes) if c in allowed]
            masked = np.full_like(logits, -np.inf)
            masked[keep] = logits[keep]
            logits = masked - masked.max()
        prob = np.exp(logits) / np.exp(logits).sum()
        order = np.argsort(prob)[::-1][: self.topk]
        return [(self.classes[i], float(prob[i])) for i in order]

