# -*- coding: utf-8 -*-
"""YOLOv8-Pose 姿态检测封装。"""

from ultralytics import YOLO


class PoseEstimator:
    """加载 YOLOv8-Pose 模型并对单帧图像做人形关键点检测。"""

    def __init__(self, weights="yolov8n-pose.pt", device="cpu", conf=0.3):
        self.weights = weights
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def predict(self, frame):
        """返回检测到的所有人：list of (17,3) 数组 (x, y, conf)。"""
        results = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)
        persons = []
        for result in results:
            if result.keypoints is None or result.keypoints.data is None:
                continue
            data = result.keypoints.data
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            for keypoints in data:
                persons.append(keypoints)
        return persons
