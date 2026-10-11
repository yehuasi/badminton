# -*- coding: utf-8 -*-
"""回看持久化：分析结果与代表帧落盘，支持按动作类型检索。"""

import json
import os
import time


class ReplayStore:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.frames_dir = os.path.join(root_dir, "frames")
        self.records_path = os.path.join(root_dir, "records.jsonl")
        os.makedirs(self.frames_dir, exist_ok=True)

    def append(self, result, jpg_bytes=None):
        """追加一条分析结果；jpg_bytes 为可选的代表帧 JPEG。"""
        ts = time.time()
        frame_name = None
        if jpg_bytes:
            frame_name = f"{int(ts)}_{int((ts % 1) * 1000):03d}.jpg"
            try:
                with open(os.path.join(self.frames_dir, frame_name), "wb") as f:
                    f.write(jpg_bytes)
            except OSError:
                frame_name = None

        record = {
            "time": round(ts, 1),
            "frame": frame_name,
            "shot_type": result.get("shot_type", ""),
            "confidence": result.get("confidence", 0.0),
            "standardness": result.get("standardness"),
            "coherence": result.get("coherence"),
            "footwork": result.get("footwork"),
            "action_score": result.get("action_score"),
            "total": result.get("total"),
            "group": result.get("group"),
            "errors": result.get("errors", []),
        }
        try:
            with open(self.records_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def load_recent(self, limit=500):
        return self._read_all()[-limit:]

    def query(self, class_name=None, limit=50):
        rows = self._read_all()
        if class_name:
            rows = [r for r in rows if r.get("shot_type") == class_name]
        return rows[-limit:]

    def _read_all(self):
        if not os.path.isfile(self.records_path):
            return []
        rows = []
        with open(self.records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
