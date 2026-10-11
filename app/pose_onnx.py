# -*- coding: utf-8 -*-
"""基于 ONNX Runtime 的 YOLOv8-Pose 推理（端侧部署用，不依赖 ultralytics/torch）。"""

import cv2
import numpy as np


def _detect_imgsz(shape):
    if not shape or len(shape) < 3:
        return None
    for dim in (shape[2], shape[3]):
        if isinstance(dim, int) and dim > 0:
            return int(dim)
    return None


class ONNXPoseEstimator:
    def __init__(self, weights, conf=0.3, imgsz=None, threads=4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(weights, providers=["CPUExecutionProvider"], sess_options=opts)
        self.input_name = self.sess.get_inputs()[0].name
        self.conf = conf
        # 从模型固定输入尺寸自动推断 imgsz；动态输入则回退到显式值或 640
        self.imgsz = int(imgsz) if imgsz is not None else _detect_imgsz(self.sess.get_inputs()[0].shape) or 640

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        r = self.imgsz / max(h, w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        pad_w = (self.imgsz - new_w) / 2
        pad_h = (self.imgsz - new_h) / 2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.copyMakeBorder(
            resized,
            int(np.floor(pad_h)),
            int(np.ceil(pad_h)),
            int(np.floor(pad_w)),
            int(np.ceil(pad_w)),
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return canvas, r, (pad_w, pad_h), (h, w)

    def predict(self, frame):
        """返回 list of (17,3) 关键点数组，坐标为原图像素，与 app/pose.py 接口一致。"""
        canvas, r, (pad_w, pad_h), (oh, ow) = self._letterbox(frame)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # (1,3,H,W)
        out = self.sess.run(None, {self.input_name: blob})[0]  # (1,300,57)

        persons = []
        for row in out[0]:
            if float(row[4]) < self.conf:
                continue
            kpts = row[6:].reshape(17, 3).copy()
            kpts[:, 0] = (kpts[:, 0] - pad_w) / r
            kpts[:, 1] = (kpts[:, 1] - pad_h) / r
            kpts[:, 0] = np.clip(kpts[:, 0], 0, ow - 1)
            kpts[:, 1] = np.clip(kpts[:, 1], 0, oh - 1)
            persons.append(kpts)
        return persons
