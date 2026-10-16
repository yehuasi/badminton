# -*- coding: utf-8 -*-
"""自采数据体检：结构、命名、元信息、重复与数据泄漏检查。

只读扫描，不修改原始数据。输出：
    reports/selfcollected_audit.json   机器可读的逐条记录与统计
    控制台                             分层汇总（类别 × 划分 × 标签 × 视角）

用法：
    .venv\\Scripts\\python scripts\\audit_selfcollected.py --root "self-build dataset"
    .venv\\Scripts\\python scripts\\audit_selfcollected.py --root "self-build dataset" --no-hash
"""

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import statistics
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT  # noqa: E402

SPLIT_SUFFIX = ("标准集", "标准库", "验证集", "验证库")
VIEW_KEYS = ("正侧面", "正面", "侧面", "front", "side")


def parse_record(root, path):
    root = pathlib.Path(root)
    path = pathlib.Path(path)
    rel = path.relative_to(root)
    dirs = list(rel.parts[:-1])
    name = path.name
    stem = name
    while True:
        base, ext = os.path.splitext(stem)
        if base and ext.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            stem = base
        else:
            break
    blob = "/".join(dirs) + "/" + stem

    top = dirs[0] if dirs else ""
    split = "其他"
    cls = top
    for suf in SPLIT_SUFFIX:
        if top.endswith(suf):
            cls = top[: -len(suf)].rstrip("_- 　")
            split = "标准" if suf.startswith("标准") else "验证"
            break
    else:
        cls = cls.rstrip("_- 　")

    label = "未标注"
    if "错误" in blob:
        label = "错误"
    elif "正确" in blob:
        label = "正确"

    error_reason = ""
    m = re.search(r"错误[_-]([^_/]+)", blob)
    if m:
        error_reason = m.group(1)
    else:
        m = re.search(r"错误[_-]?([\u4e00-\u9fff]+)", blob)
        if m:
            error_reason = m.group(1)

    view = "未标注"
    last_dir = dirs[-1] if dirs else ""
    if "正侧面" in blob:
        view = "正侧面"
    elif "正面" in last_dir:
        view = "正面"
    elif "侧面" in last_dir:
        view = "侧面"
    else:
        for key, val in (("front", "正面"), ("side", "侧面")):
            if re.search(r"[_.-]" + key + r"([._-]|$)", stem):
                view = val
                break
        else:
            if "正面" in "/".join(dirs):
                view = "正面"
            elif "侧面" in "/".join(dirs):
                view = "侧面"

    variant = ""
    for key in ("正手", "反手"):
        if key in dirs or key in stem:
            variant = key
            break

    m = re.search(r"[Pp](\d{1,3})", blob)
    player = f"p{int(m.group(1)):02d}" if m else ""

    seq = ""
    nums = re.findall(r"(?<![0-9A-Za-z])(\d{1,3})(?![0-9A-Za-z])", stem)
    if nums:
        seq = nums[-1]

    rec = {
        "path": str(rel).replace("\\", "/"),
        "class": cls,
        "split": split,
        "label": label,
        "error_reason": error_reason,
        "view": view,
        "variant": variant,
        "player": player,
        "seq": seq,
        "size": path.stat().st_size,
        "name": name,
    }
    rec["dup_ext"] = bool(re.search(r"\.(mp4|avi|mov|mkv)\.(mp4|avi|mov|mkv)$", name, re.I))
    rec["per_video_dir"] = bool(last_dir) and bool(re.search(r"[Pp]\d{1,3}", last_dir))
    return rec


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False}
    info = {
        "ok": True,
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    info["duration"] = round(info["frames"] / info["fps"], 2) if info["fps"] > 0 else 0.0
    ok, frame = cap.read()
    cap.release()
    info["first_frame_ok"] = bool(ok)
    if ok and frame is not None:
        info["w"] = info["w"] or int(frame.shape[1])
        info["h"] = info["h"] or int(frame.shape[0])
    return info


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "self-build dataset"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "selfcollected_audit.json"))
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"未找到数据目录：{root}")

    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in (".mp4", ".avi", ".mov", ".mkv"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    print(f"数据根目录：{root}")
    print(f"视频总数：{len(paths)}\n")

    records = []
    for i, p in enumerate(paths, 1):
        rec = parse_record(os.path.abspath(root), os.path.abspath(p))
        if not args.no_probe:
            rec.update(probe(p))
        if not args.no_hash:
            rec["md5"] = md5_of(p)
        records.append(rec)
        if i % 100 == 0:
            print(f"  ... 已扫描 {i}/{len(paths)}")

    classes = sorted({r["class"] for r in records})
    print("\n=== 一、视频分布（类别 × 划分）===")
    header = f"{'类别':<12}{'标准':>6}{'验证':>6}{'其他':>6}{'合计':>6}   标签（正确/错误/未标注）"
    print(header)
    for c in classes:
        rs = [r for r in records if r["class"] == c]
        n_std = sum(1 for r in rs if r["split"] == "标准")
        n_val = sum(1 for r in rs if r["split"] == "验证")
        n_oth = len(rs) - n_std - n_val
        n_ok = sum(1 for r in rs if r["label"] == "正确")
        n_bad = sum(1 for r in rs if r["label"] == "错误")
        n_unk = len(rs) - n_ok - n_bad
        print(f"{c:<12}{n_std:>6}{n_val:>6}{n_oth:>6}{len(rs):>6}   {n_ok}/{n_bad}/{n_unk}")
    print(f"{'合计':<12}{sum(1 for r in records if r['split']=='标准'):>6}"
          f"{sum(1 for r in records if r['split']=='验证'):>6}"
          f"{sum(1 for r in records if r['split'] not in ('标准','验证')):>6}{len(records):>6}")

    print("\n=== 二、视角分布（类别 × 视角）===")
    views = ["正面", "侧面", "正侧面", "未标注"]
    print(f"{'类别':<12}" + "".join(f"{v:>8}" for v in views))
    for c in classes:
        rs = [r for r in records if r["class"] == c]
        row = "".join(f"{sum(1 for r in rs if r['view']==v):>8}" for v in views)
        print(f"{c:<12}{row}")

    print("\n=== 三、划分内标签 × 视角（标准集 / 验证集）===")
    for split in ("标准", "验证"):
        rs = [r for r in records if r["split"] == split]
        nb_ok = sum(1 for r in rs if r["label"] == "正确")
        nb_bad = sum(1 for r in rs if r["label"] == "错误")
        print(f"  {split}集：共 {len(rs)}，正确 {nb_ok}，错误 {nb_bad}")
        byc = collections.Counter((r["class"], r["label"]) for r in rs)
        for c in classes:
            print(f"     {c:<12} 正确 {byc[(c,'正确')]:>3}  错误 {byc[(c,'错误')]:>3}")

    print("\n=== 四、错误类型（仅验证集里的错误样本）===")
    err = collections.Counter(r["error_reason"] or "(未写明)" for r in records if r["label"] == "错误")
    for k, v in err.most_common():
        print(f"  {k:<24}{v:>4}")

    print("\n=== 五、人物编号分布与跨划分重合 ===")
    by_pl = collections.defaultdict(lambda: collections.Counter())
    for r in records:
        if r["player"]:
            by_pl[r["class"]][r["player"]] += 1
    for c in classes:
        std = {r["player"] for r in records if r["class"] == c and r["split"] == "标准" and r["player"]}
        val = {r["player"] for r in records if r["class"] == c and r["split"] == "验证" and r["player"]}
        nopl = sum(1 for r in records if r["class"] == c and not r["player"])
        print(f"  {c:<12} 标准人数 {len(std):>2}  验证人数 {len(val):>2}  "
              f"重合 {len(std & val):>2}  无编号 {nopl:>3}  "
              f"重合编号 {','.join(sorted(std & val)) or '无'}")

    print("\n=== 六、视频技术属性 ===")
    ok = [r for r in records if r.get("ok")]
    bad = [r for r in records if not r.get("ok")]
    print(f"  可解码 {len(ok)} / 无法打开 {len(bad)}")
    for r in bad[:20]:
        print(f"    打不开：{r['path']}")
    if ok:
        durs = [r["duration"] for r in ok if r.get("duration")]
        print(f"  时长：min {min(durs):.2f}s  中位 {statistics.median(durs):.2f}s  max {max(durs):.2f}s")
        res = collections.Counter(f"{r['w']}x{r['h']}" for r in ok)
        print("  分辨率：" + "  ".join(f"{k}×{v}" for k, v in res.most_common(8)))
        fr = collections.Counter(round(r["fps"]) for r in ok if r.get("fps"))
        print("  帧率：" + "  ".join(f"{k}fps×{v}" for k, v in fr.most_common(8)))
        nf = collections.Counter(r["frames"] for r in ok)
        print(f"  帧数：min {min(nf)}  中位 {int(statistics.median(list(nf.elements())))}  max {max(nf)}")
        portrait = [r for r in ok if r["h"] > r["w"]]
        print(f"  竖屏：{len(portrait)}")
        short = [r for r in ok if r.get("duration", 0) < 0.8]
        longv = [r for r in ok if r.get("duration", 0) > 20]
        print(f"  过短(<0.8s)：{len(short)}   过长(>20s)：{len(longv)}")
        for r in (short + longv)[:10]:
            print(f"     {r['duration']:.2f}s  {r['path']}")

    print("\n=== 七、重复与命名异常 ===")
    if not args.no_hash:
        by_hash = collections.defaultdict(list)
        for r in records:
            by_hash[r["md5"]].append(r)
        dups = {h: v for h, v in by_hash.items() if len(v) > 1}
        print(f"  内容重复的视频组：{len(dups)}（涉及 {sum(len(v) for v in dups.values())} 个文件）")
        cross = [v for v in dups.values()
                 if len({x["split"] for x in v}) > 1]
        print(f"  其中跨「标准 / 验证」的重复组：{len(cross)}  ← 会直接造成评测虚高")
        for v in cross[:10]:
            print("     " + "  ==  ".join(f"[{x['split']}]{x['path']}" for x in v))
        same_split = [v for v in dups.values() if len({x["split"] for x in v}) == 1]
        for v in same_split[:10]:
            print("     同划分重复：" + "  ==  ".join(x["path"] for x in v))
    de = [r for r in records if r["dup_ext"]]
    print(f"  双扩展名（.mp4.mp4 等）：{len(de)}")
    nested = [r for r in records if r.get("per_video_dir")]
    print(f"  每个视频单独一层子目录：{len(nested)}")
    nolabel = [r for r in records if r["label"] == "未标注"]
    print(f"  未标注正确/错误：{len(nolabel)}")
    for r in nolabel[:10]:
        print(f"     {r['path']}")

    print("\n=== 八、与《自采数据采集要求》对照 ===")
    for c in classes:
        rs = [r for r in records if r["class"] == c]
        std = [r for r in rs if r["split"] == "标准"]
        val = [r for r in rs if r["split"] == "验证"]
        v_front = sum(1 for r in std if r["view"] == "正面")
        v_side = sum(1 for r in std if r["view"] == "侧面")
        v_fs = sum(1 for r in std if r["view"] == "正侧面")
        flag = []
        if len(std) < 50:
            flag.append(f"标准集不足 50（{len(std)}）")
        if len(val) < 50:
            flag.append(f"验证集不足 50（{len(val)}）")
        if v_front == 0 and v_side == 0:
            flag.append("标准集缺正面/侧面单视角")
        print(f"  {c:<12} 标准 {len(std):>3}(正{v_front}/侧{v_side}/正侧{v_fs})  验证 {len(val):>3}  "
              f"{'；'.join(flag) if flag else '达标'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"root": root, "count": len(records), "records": records},
                  f, ensure_ascii=False, indent=1)
    print(f"\n逐条记录已写入：{args.out}")


if __name__ == "__main__":
    main()
