# -*- coding: utf-8 -*-
"""自拍动作采集脚本：一条一个动作，自动按类别分文件夹保存 mp4。

用途：为「域适配微调」采集目标机位数据（约每类 12~15 条，共约 100 条）。
拍摄要求：全身入镜、站距 2~3 米、机位固定、每条只做一个完整动作
（站定 → 引拍 → 击球 → 随挥 → 停顿半秒）。

操作（在预览窗口上）：
    空格 / 回车   开始：倒计时后自动录制一段，然后自动停止保存
    D / 退格     删除上一条（拍坏了就删掉重拍）
    Q / Esc      退出

用法：
    .venv\\Scripts\\python scripts\\record_selfclips.py --camera 1
    .venv\\Scripts\\python scripts\\record_selfclips.py --class 高远球 --camera 1
    .venv\\Scripts\\python scripts\\record_selfclips.py --list     # 先看摄像头在哪个索引
"""

import argparse
import os
import re
import sys
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 关掉 OpenCV 探测摄像头的刷屏警告

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT  # noqa: E402
from app.classes import SHOT_CLASSES  # noqa: E402

OUT_ROOT = os.path.join(REPO_ROOT, "self_filmed")

# 预览窗口只能画 ASCII，中文会变方块，故用短英文标签；完整中文名打在终端里。
CLASS_LABEL = {
    "正手发高远球": "forehand_serve",
    "反手发网前小球": "backhand_serve",
    "高远球": "clear",
    "杀球": "smash",
    "吊球": "drop",
    "挑球": "lift",
    "搓球": "net",
}


def _open_camera(index, width, height):
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if cap.isOpened() and width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_cameras():
    """探测 0~5 号摄像头，打印可用的索引与分辨率。"""
    print("探测摄像头索引 0~5 ...")
    found = False
    for i in range(6):
        cap = _open_camera(i, None, None)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  索引 {i} 可用：{w}x{h} @ {fps:.0f}fps")
            found = True
            cap.release()
        else:
            cap.release()
    if not found:
        print("  没有找到摄像头。请确认摄像头已插入、驱动正常，再用 --list 重试。")


def _next_index(cls_dir, cls_name):
    """按既有文件名里的序号算下一个编号，避免覆盖。"""
    max_idx = 0
    for fn in os.listdir(cls_dir):
        m = re.match(re.escape(cls_name) + r"_(\d+)\.mp4$", fn)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _overlay(frame, lines):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (16, 40 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="摄像头索引（默认 0，外接摄像头常为 1）")
    ap.add_argument("--class", "--class-name", dest="cls", default="",
                    help="类别名，默认启动后交互选择")
    ap.add_argument("--duration", type=float, default=4.0, help="每条录制时长（秒），默认 4")
    ap.add_argument("--countdown", type=int, default=3, help="开始前倒计时秒数，默认 3")
    ap.add_argument("--width", type=int, default=1280, help="录制宽度（0=保持原生）")
    ap.add_argument("--height", type=int, default=720, help="录制高度（0=保持原生）")
    ap.add_argument("--list", action="store_true", help="只探测摄像头索引，不录制")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return

    if args.width <= 0 or args.height <= 0:
        args.width = args.height = 0

    cap = _open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        print(f"打不开摄像头索引 {args.camera}。先用 --list 看哪个索引可用。")
        raise SystemExit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    print(f"摄像头已打开：{w}x{h} @ {fps:.0f}fps，每条录制 {args.duration:.0f} 秒")

    cls_name = args.cls.strip()
    if cls_name and cls_name not in SHOT_CLASSES:
        raise SystemExit(f"类别名必须是：{' / '.join(SHOT_CLASSES)}")
    if not cls_name:
        print("请选择类别：")
        for i, c in enumerate(SHOT_CLASSES, 1):
            print(f"  {i}. {c}")
        pick = input("输入编号（1-7）：").strip()
        try:
            cls_name = SHOT_CLASSES[int(pick) - 1]
        except (ValueError, IndexError):
            raise SystemExit("编号无效。")

    cls_dir = os.path.join(OUT_ROOT, cls_name)
    os.makedirs(cls_dir, exist_ok=True)
    idx = _next_index(cls_dir, cls_name)
    print(f"保存目录：{cls_dir}")
    print(f"类别：{cls_name}，从编号 {idx:02d} 开始。")
    print("预览窗口：空格开始录制；D 删除上一条；Q 退出。")

    state = "preview"       # preview / countdown / rec
    countdown_end = 0.0
    rec_end = 0.0
    writer = None
    last_saved = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("读帧失败，退出。")
            break
        now = time.time()

        if state == "countdown":
            remain = max(0, int(countdown_end - now) + 1)
            _overlay(frame, [f"{CLASS_LABEL.get(cls_name, 'shot')}  #{idx:02d}",
                             f"START IN {remain}", ""])
            if now >= countdown_end:
                out_path = os.path.join(cls_dir, f"{cls_name}_{idx:02d}.mp4")
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    print("写文件失败，退出。")
                    break
                rec_end = now + args.duration
                state = "rec"
        elif state == "rec":
            left = max(0, rec_end - now)
            _overlay(frame, [f"{CLASS_LABEL.get(cls_name, 'shot')}  #{idx:02d}",
                             f"REC {left:.1f}s", ""])
            writer.write(frame)
            if now >= rec_end:
                writer.release()
                writer = None
                last_saved.append(os.path.join(cls_dir, f"{cls_name}_{idx:02d}.mp4"))
                print(f"已保存 {cls_name}_{idx:02d}.mp4（本类累计 {idx} 条）")
                idx += 1
                state = "preview"
        else:
            _overlay(frame, [f"{CLASS_LABEL.get(cls_name, 'shot')}  #{idx:02d}",
                             "SPACE start   D del last   Q quit",
                             f"saved: {idx - 1} clips"])

        cv2.imshow("record", frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (32, 13) and state == "preview":
            countdown_end = now + args.countdown
            state = "countdown"
        if key in (ord("d"), 8) and state == "preview" and last_saved:
            p = last_saved.pop()
            if os.path.exists(p):
                os.remove(p)
                idx -= 1
                print(f"已删除上一条：{os.path.basename(p)}")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("结束。视频保存在", OUT_ROOT)


if __name__ == "__main__":
    main()
