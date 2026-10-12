# -*- coding: utf-8 -*-
"""局域网 Web 演示服务：MJPEG 视频流 + 实时结果（轮询 /api/status）。

用法：
    .venv\\Scripts\\python app\\server.py                       # 合成演示（无需摄像头）
    .venv\\Scripts\\python app\\server.py --source 0            # 摄像头
    .venv\\Scripts\\python app\\server.py --source demo.mp4     # 本地视频
浏览器打开 http://127.0.0.1:8000
"""

import argparse
import asyncio
import os
import threading
import time

import cv2
import numpy as np

from app.analyzer import ShotAnalyzer
from app.classes import SHOT_CLASSES
from app.features import compute_features
from app.fusion import fuse_views
from app.store import ReplayStore
from app.synth import generate_sequence, render_stick

# 显示帧最大宽度与 JPEG 质量：网页面板约 500px，缩小 + 降质以降低带宽，
# 让视频流在手机热点等抖动链路上更抗卡
DISPLAY_MAX_W = 400
DISPLAY_JPEG_QUALITY = 70


def _open_capture(index_or_path):
    """打开摄像头或视频文件；Windows 上摄像头用 DirectShow 后端更稳。"""
    s = str(index_or_path)
    if s.isdigit():
        idx = int(s)
        if os.name == "nt":
            return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        return cv2.VideoCapture(idx)
    return cv2.VideoCapture(index_or_path)


def list_cameras():
    found = []
    for i in range(6):
        cap = _open_capture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, w, h))
            cap.release()
    return found


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame_jpg = None
        self.history = []
        self.store = None  # ReplayStore，由 make_app 注入
        self.current_stroke = None  # 当前这一拍：{result, frame, ts}
        self.pending_views = {"front": None, "side": None}
        self.frame_history = []  # 回看用：[(ts, result, jpg_bytes), ...]
        self.frame_seq = 0       # 帧序号，供 MJPEG 流判断“是否有新帧”
        self.max_frames = 240
        self.result = {
            "source": "starting",
            "shot_type": "",
            "confidence": 0.0,
            "standardness": None,
            "errors": [],
            "advice": ["正在初始化…"],
            "top3": [],
            "fps": 0.0,
        }

    def update(self, frame, result):
        h, w = frame.shape[:2]
        if w > DISPLAY_MAX_W:
            scale = DISPLAY_MAX_W / float(w)
            frame = cv2.resize(
                frame, (DISPLAY_MAX_W, int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, jpg = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), DISPLAY_JPEG_QUALITY]
        )
        with self.lock:
            if ok:
                jpg_bytes = jpg.tobytes()
                self.frame_jpg = jpg_bytes
                self.frame_seq += 1
                self.frame_history.append({
                    "ts": time.time(),
                    "result": dict(result),
                    "jpg": jpg_bytes,
                })
                if len(self.frame_history) > self.max_frames:
                    self.frame_history = self.frame_history[-self.max_frames:]
            self.result = result

    def snapshot(self):
        with self.lock:
            return self.frame_jpg, dict(self.result), self.frame_seq

    def record(self, result):
        with self.lock:
            item = {"time": round(time.time(), 1), **dict(result)}
            self.history.append(item)
            if len(self.history) > 500:
                self.history = self.history[-500:]

    def set_current_stroke(self, result, frame):
        """锁定当前这一拍：结果 + 代表帧，供前端展示与双视角捕获。"""
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        with self.lock:
            self.current_stroke = {
                "result": dict(result),
                "frame": jpg.tobytes() if ok else None,
                "ts": round(time.time(), 1),
            }

    def clear_current_stroke(self):
        with self.lock:
            self.current_stroke = None

    def current_stroke_snapshot(self):
        with self.lock:
            if self.current_stroke is None:
                return None
            return {
                "result": dict(self.current_stroke["result"]),
                "frame": self.current_stroke["frame"],
                "ts": self.current_stroke["ts"],
            }

    def persist(self, result, frame):
        """把一次完整分析结果 + 代表帧写入持久化存储。"""
        if self.store is None:
            return
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            self.store.append(result, jpg.tobytes())

    def history_snapshot(self):
        with self.lock:
            return list(self.history)

    def capture_view(self, view):
        if view not in ("front", "side"):
            return None
        with self.lock:
            if self.current_stroke is None:
                return None
            self.pending_views[view] = dict(self.current_stroke["result"])
            return dict(self.pending_views)

    def reset_views(self):
        with self.lock:
            self.pending_views = {"front": None, "side": None}
            return dict(self.pending_views)

    def view_status(self):
        with self.lock:
            front = self.pending_views.get("front")
            side = self.pending_views.get("side")
            fused = fuse_views(front, side) if front and side else None
            cs = self.current_stroke
            return {
                "front": front,
                "side": side,
                "fused": fused,
                "current_stroke": (
                    {
                        "shot_type": cs["result"].get("shot_type", ""),
                        "confidence": cs["result"].get("confidence", 0.0),
                        "total": cs["result"].get("total"),
                        "ts": cs["ts"],
                        "has_frame": cs["frame"] is not None,
                    }
                    if cs is not None
                    else None
                ),
            }

    def replay_meta(self):
        """回看元数据（不含图片），供前端绘制趋势与选择帧。"""
        with self.lock:
            rows = []
            for i, item in enumerate(self.frame_history):
                r = item["result"]
                rows.append({
                    "idx": i,
                    "ts": round(item["ts"], 1),
                    "shot_type": r.get("shot_type", ""),
                    "confidence": r.get("confidence", 0.0),
                    "total": r.get("total"),
                    "standardness": r.get("standardness"),
                })
            return rows

    def replay_frame(self, idx):
        with self.lock:
            if 0 <= idx < len(self.frame_history):
                return self.frame_history[idx]["jpg"]
        return None


def _summarize(history):
    from collections import defaultdict

    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for r in history:
        cls = r.get("shot_type")
        if not cls:
            continue
        cell = agg[cls]
        cell[0] += 1
        if r.get("standardness") is not None:
            cell[1] += r["standardness"]
        if r.get("total") is not None:
            cell[2] += r["total"]
    rows = []
    for cls, (cnt, std_sum, total_sum) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        rows.append({
            "class": cls,
            "count": cnt,
            "avg_standardness": round(std_sum / cnt, 1) if cnt else None,
            "avg_total": round(total_sum / cnt, 1) if cnt else None,
        })
    return rows


def _run_demo(state, analyzer, window):
    cls_id = 0
    t0, frames = time.time(), 0
    while not state.stop_event.is_set():
        cls_name = SHOT_CLASSES[cls_id]
        seq = generate_sequence(cls_id, T=window)
        result = analyzer.analyze_sequence(seq)
        result["source"] = "demo"
        result["demo_class"] = cls_name
        result["fps"] = 0.0
        state.record(result)
        hit = result.get("hit_frame", window // 2)
        hit_frame = render_stick(seq[hit], cls_name)
        state.persist(result, hit_frame)
        state.set_current_stroke(result, hit_frame)
        for t in range(window):
            if state.stop_event.is_set():
                break
            feat = seq[t]
            frame = render_stick(feat, cls_name)
            frames += 1
            now = time.time()
            if now - t0 >= 1.0:
                result["fps"] = round(frames / (now - t0), 1)
                frames, t0 = 0, now
            state.update(frame, result)
            time.sleep(1 / 25)
        cls_id = (cls_id + 1) % len(SHOT_CLASSES)


def _draw_skeleton(frame, kpts):
    edges = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (0, 5), (0, 6),
    ]
    for a, b in edges:
        pa, pb = kpts[a][:2], kpts[b][:2]
        if kpts[a][2] > 0.2 and kpts[b][2] > 0.2:
            cv2.line(frame, tuple(map(int, pa)), tuple(map(int, pb)), (0, 220, 120), 2)
    for i in range(17):
        if kpts[i][2] > 0.2:
            cv2.circle(frame, tuple(map(int, kpts[i][:2])), 3, (0, 160, 255), -1)
    return frame


def _run_video(state, source, estimator, analyzer, handedness):
    cap = _open_capture(source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开输入源：{source}")

    # 显示与推理分离：显示循环按摄像头原生帧率跑（画面流畅，不随推理卡顿），
    # 推理在独立线程按自己的节奏跑，使用最近一帧；骨架叠加可能滞后半拍但无感知。
    frame_lock = threading.Lock()
    latest_frame = None
    last_kpts = None
    last_result = None
    last_result_time = 0.0
    last_lock_time = 0.0
    last_locked_type = None

    waiting = {
        "source": str(source), "shot_type": "", "confidence": 0.0,
        "standardness": None, "errors": [], "advice": ["无有效击球动作，请完整挥拍一次。"],
        "top3": [], "fps": 0.0,
    }
    no_person = {
        "source": str(source), "shot_type": "", "confidence": 0.0,
        "standardness": None, "errors": [], "advice": ["未检测到人体，请全身入镜。"],
        "top3": [], "fps": 0.0,
    }

    def analysis_loop():
        nonlocal latest_frame, last_kpts, last_result, last_result_time
        nonlocal last_lock_time, last_locked_type
        while not state.stop_event.is_set():
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.005)
                continue
            persons = estimator.predict(frame)
            if persons:
                last_kpts = persons[0]
                feat = compute_features(last_kpts, handedness=handedness)
                analyzed = analyzer.update_feature(feat)
                if analyzed is not None:
                    analyzed["source"] = str(source)
                    now = time.time()
                    if analyzed.get("is_action", True):
                        if (now - last_lock_time >= 1.5
                                or analyzed["shot_type"] != last_locked_type):
                            last_lock_time = now
                            last_locked_type = analyzed["shot_type"]
                            state.record(analyzed)
                            state.persist(analyzed, frame)
                            state.set_current_stroke(analyzed, frame)
                    last_result = analyzed
                    last_result_time = now
                if last_result is not None and time.time() - last_result_time > 3.0:
                    last_result = None
                    state.clear_current_stroke()
            else:
                last_kpts = None
                analyzer.reset()
                last_result = None
                state.clear_current_stroke()

    threading.Thread(target=analysis_loop, daemon=True).start()

    t0, frames = time.time(), 0
    fps = 0.0
    last_max_gap = 0.0
    prev_frame_time = None
    max_gap = 0.0
    while not state.stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if prev_frame_time is not None:
            max_gap = max(max_gap, now - prev_frame_time)
        prev_frame_time = now
        with frame_lock:
            latest_frame = frame.copy()
        # 用最近一次骨架叠加（可能滞后半拍，但画面保持流畅）
        if last_kpts is not None:
            frame = _draw_skeleton(frame, last_kpts)
        if last_kpts is None:
            result = dict(no_person)
        elif last_result is not None:
            result = dict(last_result)
        else:
            result = dict(waiting)
        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = round(frames / (now - t0), 1)
            last_max_gap = round(max_gap * 1000.0, 1)
            frames, t0 = 0, now
            max_gap = 0.0
        result["fps"] = fps
        result["max_frame_gap_ms"] = last_max_gap
        state.update(frame, result)
        time.sleep(0.001)
    cap.release()


def _load_classes(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_runtime(args):
    if args.backend == "onnx":
        from app.classifier_onnx import ONNXShotClassifier
        from app.pose_onnx import ONNXPoseEstimator

        estimator = ONNXPoseEstimator(args.pose_model, conf=args.conf, threads=args.threads)
        classes = _load_classes(args.classes_file) if os.path.isfile(args.classes_file) else None
        classifier = ONNXShotClassifier(args.classifier_model, classes=classes, threads=args.threads)
    else:
        from app.classifier import ShotClassifierRunner
        from app.pose import PoseEstimator

        estimator = PoseEstimator(args.pose_model)
        classifier = ShotClassifierRunner(
            args.classifier_model, T=args.window, handedness=args.handedness
        )
    return estimator, ShotAnalyzer(
        classifier=classifier, window=args.window, debug=args.debug_motion
    )


def make_app(args, page_html):
    from fastapi import FastAPI
    from contextlib import asynccontextmanager
    from fastapi.responses import StreamingResponse, JSONResponse, Response

    state = SharedState()
    replay_dir = getattr(args, "replay_dir", None)
    if replay_dir:
        state.store = ReplayStore(replay_dir)
        state.history = state.store.load_recent(500)

    @asynccontextmanager
    async def lifespan(app):
        estimator, analyzer = _build_runtime(args)
        if args.source == "demo":
            target = _run_demo
            worker_args = (state, analyzer, args.window)
        else:
            target = _run_video
            worker_args = (state, args.source, estimator, analyzer, args.handedness)
        worker = threading.Thread(target=target, args=worker_args, daemon=True)
        worker.start()
        yield
        state.stop_event.set()

    app = FastAPI(title="羽毛球动作分析演示", lifespan=lifespan)
    app.state.app_state = state

    @app.get("/")
    def index():
        from fastapi.responses import HTMLResponse
        return HTMLResponse(page_html)

    @app.get("/video_feed")
    async def video_feed():
        async def gen():
            last_seq = -1
            while not state.stop_event.is_set():
                jpg, _, seq = state.snapshot()
                if jpg and seq != last_seq:
                    last_seq = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(0.005)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def status():
        _, result, _ = state.snapshot()
        return JSONResponse(result)

    @app.get("/api/session")
    def session():
        history = state.history_snapshot()
        return JSONResponse({
            "summary": _summarize(history),
            "records": history[-20:],
            "total_records": len(history),
        })

    @app.get("/api/replay")
    def replay():
        return JSONResponse({"frames": state.replay_meta()})

    @app.get("/api/frame/{idx}")
    def frame(idx: int):
        jpg = state.replay_frame(idx)
        if jpg is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return Response(content=jpg, media_type="image/jpeg")

    @app.post("/api/view/capture")
    def view_capture(view: str):
        pending = state.capture_view(view)
        if pending is None:
            return JSONResponse({"error": "尚无分析结果可记录"}, status_code=400)
        return JSONResponse(state.view_status())

    @app.get("/api/view/status")
    def view_status():
        return JSONResponse(state.view_status())

    @app.get("/api/current_stroke")
    def current_stroke():
        cs = state.current_stroke_snapshot()
        if cs is None:
            return JSONResponse({"result": None, "ts": None})
        return JSONResponse({
            "result": cs["result"],
            "ts": cs["ts"],
            "has_frame": cs["frame"] is not None,
        })

    @app.get("/api/current_frame")
    def current_frame():
        cs = state.current_stroke_snapshot()
        if cs is None or cs["frame"] is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return Response(content=cs["frame"], media_type="image/jpeg")

    @app.post("/api/view/reset")
    def view_reset():
        return JSONResponse({"pending": state.reset_views()})

    @app.get("/api/history")
    def history(class_name: str = None, limit: int = 50):
        if state.store is None:
            return JSONResponse({"records": [], "total": 0})
        records = state.store.query(class_name=class_name, limit=limit)
        return JSONResponse({"records": records, "total": len(records)})

    return app


class StopOnShutdownServer:
    """在开始优雅关闭前先置停止标志，让 MJPEG 流自行结束，避免超时取消。"""

    @classmethod
    def run(cls, app, host, port, stop_event, log_level="info"):
        import uvicorn

        class _Server(uvicorn.Server):
            async def shutdown(self, sockets=None):
                stop_event.set()
                await super().shutdown(sockets)

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level,
            timeout_graceful_shutdown=2,
        )
        _Server(config).run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source", default="demo", help="demo / 摄像头索引 / 视频文件路径")
    parser.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    parser.add_argument("--pose-model", default=None)
    parser.add_argument("--classifier-model", default=None)
    parser.add_argument("--classes-file", default=None)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--handedness", default="right", choices=["left", "right"])
    parser.add_argument("--list-cameras", action="store_true", help="枚举可用摄像头后退出")
    parser.add_argument("--threads", type=int, default=4, help="ONNX 推理线程数（板子实测 4 核最优）")
    parser.add_argument("--conf", type=float, default=0.3, help="姿态检测置信度阈值")
    parser.add_argument("--debug-motion", action="store_true", help="打印逐帧运动量调试日志")
    args = parser.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("未检测到可用摄像头。")
        else:
            print("可用摄像头（索引, 宽, 高）：")
            for idx, w, h in cams:
                print(f"  index {idx}: {w}x{h}")
        return

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.backend == "onnx":
        if args.pose_model is None:
            args.pose_model = os.path.join(root, "yolov8n-pose-480.onnx")
        if args.classifier_model is None:
            args.classifier_model = os.path.join(root, "models", "shot_classifier.onnx")
        if args.classes_file is None:
            args.classes_file = os.path.join(root, "models", "shot_classifier_classes.json")
    else:
        if args.pose_model is None:
            args.pose_model = os.path.join(root, "yolov8n-pose.pt")
        if args.classifier_model is None:
            name = "shot_classifier.pt" if args.source == "demo" else "shot_classifier_bfmd.pt"
            args.classifier_model = os.path.join(root, "models", name)
        args.classes_file = args.classes_file or os.path.join(root, "models", "shot_classifier_classes.json")
    args.pose_model = os.path.abspath(args.pose_model)
    args.classifier_model = os.path.abspath(args.classifier_model)
    args.classes_file = os.path.abspath(args.classes_file)
    args.replay_dir = os.path.join(root, "replay")

    page_path = os.path.join(root, "web", "index.html")
    with open(page_path, "r", encoding="utf-8") as f:
        page_html = f.read()

    app = make_app(args, page_html)
    stop_event = app.state.app_state.stop_event
    StopOnShutdownServer.run(app, args.host, args.port, stop_event)


if __name__ == "__main__":
    main()

