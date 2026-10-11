# -*- coding: utf-8 -*-
"""导出端侧部署所需 ONNX：动作分类器（固定 T 窗口），可选导出 YOLOv8-Pose。"""

import argparse
import os

import torch

from common import REPO_ROOT
from app.model import load_checkpoint


def export_classifier(ckpt, out, T=None):
    model, classes = load_checkpoint(ckpt, map_location="cpu")
    model.eval()
    in_dim = model.in_dim
    if T is None:
        T = 24
    dummy = torch.randn(1, T, in_dim)
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"已导出分类器 ONNX：{out}（输入 (B,{T},{in_dim})，输出 {len(classes)} 类 logits）")
    return out


def export_pose(weights, out):
    from ultralytics import YOLO

    model = YOLO(weights)
    path = model.export(format="onnx", imgsz=640, opset=17)
    print(f"已导出 YOLOv8-Pose ONNX：{path}")
    return path


def verify_onnx(path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("未安装 onnxruntime，跳过校验。")
        return
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for inp in sess.get_inputs():
        print(f"  输入：{inp.name} {inp.shape} {inp.type}")
    for out in sess.get_outputs():
        print(f"  输出：{out.name} {out.shape} {out.type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(REPO_ROOT, "models", "shot_classifier_bfmd.pt"))
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "models", "shot_classifier.onnx"))
    parser.add_argument("--pose", action="store_true", help="同时导出 YOLOv8-Pose")
    parser.add_argument("--pose-weights", default=os.path.join(REPO_ROOT, "yolov8n-pose.pt"))
    args = parser.parse_args()

    path = export_classifier(args.ckpt, args.out, T=args.T)
    verify_onnx(path)
    if args.pose:
        export_pose(args.pose_weights, args.out.replace(".onnx", "_pose.onnx"))


if __name__ == "__main__":
    main()

