# -*- coding: utf-8 -*-
"""按需下载 BFMD 中我们真正需要的标注子目录（pose / hit_inferred / metadata）。

相较 download_bfmd.py 的全量下载（含 shuttle、court_perframe、caption 等约 1.8GB），
本脚本只下载动作识别所需的三个子目录，逐文件下载、失败重试、已存在则跳过，
更适合不稳定的代理网络，可反复运行续传。
"""

import argparse
import os
import sys
import time

from common import REPO_ROOT

DRIVE_URL = "https://drive.google.com/drive/folders/1wQr4DpMbx-e8jFnvOH9WUJXxfXgTC1IJ"
WANT_PREFIXES = (
    "annotations/pose/",
    "annotations/hit_inferred/",
    "annotations/metadata/",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "raw", "BFMD_data"))
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--matches", default=None, help="匹配名子串，逗号分隔；默认全部")
    parser.add_argument("--min-size", type=int, default=1024, help="视为已完成的文件最小字节数")
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("缺少 gdown，请先执行：.venv\\Scripts\\python -m pip install gdown")
        sys.exit(1)

    print("读取 Google Drive 文件夹清单 ...")
    files = gdown.download_folder(DRIVE_URL, skip_download=True, quiet=True)
    rows = [(f.id, f.path.replace(os.sep, "/")) for f in files]

    match_filter = None
    if args.matches:
        match_filter = [s.strip() for s in args.matches.split(",") if s.strip()]

    selected = []
    for fid, path in rows:
        if not path.startswith(WANT_PREFIXES):
            continue
        if path.endswith(".bak"):
            continue
        if match_filter and not any(s in path for s in match_filter):
            continue
        selected.append((fid, path))

    print(f"待下载 {len(selected)} 个文件：")
    for _, path in selected:
        print("  " + path)

    manifest = os.path.join(args.out, ".downloaded.txt")
    os.makedirs(args.out, exist_ok=True)
    done_set = set()
    if os.path.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as f:
            done_set = {line.strip() for line in f if line.strip()}

    ok = fail = skip = 0
    for fid, path in selected:
        dest = os.path.join(args.out, path.replace("/", os.sep))
        dest_dir = os.path.dirname(dest)
        base = os.path.basename(dest)
        has_part = (
            os.path.isdir(dest_dir)
            and any(f.startswith(base) and f.endswith(".part") for f in os.listdir(dest_dir))
        )
        already_complete = (
            os.path.isfile(dest)
            and os.path.getsize(dest) >= args.min_size
            and not has_part
        )
        if already_complete:
            skip += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        success = False
        for attempt in range(1, args.retries + 1):
            try:
                print(f"[{path}] 下载中 (attempt {attempt}/{args.retries}) ...")
                gdown.download(
                    id=fid,
                    output=dest,
                    quiet=False,
                    use_cookies=False,
                    resume=True,
                    timeout=(15, 600),
                    retries=2,
                )
                if os.path.isfile(dest) and os.path.getsize(dest) >= args.min_size:
                    success = True
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"[{path}] 失败：{exc}")
                time.sleep(5 * attempt)
        if success:
            ok += 1
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(path + "\n")
        else:
            fail += 1
            print(f"[{path}] 多次失败，已跳过（重跑本脚本可续传）")

    print(f"完成：成功 {ok}，跳过 {skip}，失败 {fail}")


if __name__ == "__main__":
    main()

