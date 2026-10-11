# -*- coding: utf-8 -*-
"""下载 BFMD 数据集标注包（Google Drive，约 1.8GB）。

用法（需能访问 Google Drive，或使用可用的代理/VPN）：
    .venv\\Scripts\\python scripts\\download_bfmd.py
下载到 data/raw/BFMD_data。

注意：BFMD 的标注包不含视频，只有 JSON 标注（含 17 关键点 pose 与击球类型）。
"""

import argparse
import os
import sys

from common import REPO_ROOT

DRIVE_URL = "https://drive.google.com/drive/folders/1wQr4DpMbx-e8jFnvOH9WUJXxfXgTC1IJ"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "raw", "BFMD_data"))
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("缺少 gdown，请先执行：.venv\\Scripts\\python -m pip install gdown")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print(f"开始下载 BFMD 标注包到 {args.out} ...")
    gdown.download_folder(DRIVE_URL, output=args.out, quiet=False, use_cookies=False)
    print("下载完成。若部分文件失败，可重复运行本脚本续传。")


if __name__ == "__main__":
    main()

