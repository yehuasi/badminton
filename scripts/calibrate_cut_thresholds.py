# -*- coding: utf-8 -*-
"""用真实自采数据重新标定“活动触发式切分”的阈值。

背景：现有阈值（MOTION_FRAME_THRESHOLD=3.5、SWING_MEAN_THRESHOLD=2.0 等）是在合成
数据上定的，全量试跑显示 57% 的真实视频切分没命中、要靠兜底取窗，必须按真实分布重标。

做法：每个视频只包含一个完整击球动作，因此“切出恰好 1 个窗口”就是正确行为。
在特征缓存上网格搜索阈值组合，直接数每个组合下的命中率，取最优。

用法：
    .venv\\Scripts\\python scripts\\calibrate_cut_thresholds.py
    .venv\\Scripts\\python scripts\\calibrate_cut_thresholds.py --quick
"""

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT  # noqa: E402
import app.analyzer as A  # noqa: E402
from app.analyzer import ShotAnalyzer, _frame_motion, _swing_speed  # noqa: E402


def cut_count(seq, motion_thr, onset, offset, min_frames, swing_thr, window):
    """按给定阈值复现一遍切分，返回窗口数（不跑分类）。"""
    A.MOTION_FRAME_THRESHOLD = motion_thr
    A.ONSET_FRAMES = onset
    A.OFFSET_FRAMES = offset
    A.MIN_STROKE_FRAMES = min_frames
    A.SWING_MEAN_THRESHOLD = swing_thr
    analyzer = ShotAnalyzer(classifier=None, window=window, capture=True)
    for t in range(seq.shape[0]):
        analyzer.update_feature(seq[t])
    return len(analyzer.captured), analyzer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(REPO_ROOT, "data", "feature_cache"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "cut_threshold_calibration.json"))
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--quick", action="store_true", help="只搜索推荐候选组合")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个视频（0=全部）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    if not files:
        raise SystemExit(f"缓存目录为空：{args.cache}")
    if args.limit:
        files = files[: args.limit]

    seqs, motion_stats, swing_means = [], [], []
    for fp in files:
        z = np.load(fp, allow_pickle=False)
        seq = z["seq"]
        seqs.append(seq)
        prev, mv = None, []
        for t in range(seq.shape[0]):
            mv.append(_frame_motion(prev, seq[t]))
            prev = seq[t]
        mv = np.asarray(mv, dtype=float)
        motion_stats.append(np.percentile(mv, [50, 75, 90, 95, 99]).tolist())
        sp = _swing_speed(seq)
        swing_means.append(float(sp.mean()) if sp.size else 0.0)
    motion_stats = np.asarray(motion_stats)
    swing_means = np.asarray(swing_means)

    print(f"视频数：{len(seqs)}")
    print("单帧运动量分位（中位视频）：p50=%.2f p75=%.2f p90=%.2f p95=%.2f p99=%.2f"
          % tuple(np.median(motion_stats, axis=0)))
    print("全片平均挥速：p10=%.2f p50=%.2f p90=%.2f"
          % (np.percentile(swing_means, 10), np.percentile(swing_means, 50),
             np.percentile(swing_means, 90)))
    print(f"（当前阈值 运动量 {A.MOTION_FRAME_THRESHOLD} / 挥速均值闸 {A.SWING_MEAN_THRESHOLD}）\n")

    if args.quick:
        motion_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        onset_grid = [3]
        offset_grid = [5]
        swing_grid = [0.5, 1.0, 1.5, 2.0]
    else:
        motion_grid = [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5]
        onset_grid = [2, 3, 4]
        offset_grid = [3, 5, 8]
        swing_grid = [0.5, 1.0, 2.0, 3.0]

    results = []
    for motion_thr in motion_grid:
        for onset in onset_grid:
            for offset in offset_grid:
                for swing_thr in swing_grid:
                    hit = multi = zero = 0
                    for seq in seqs:
                        n, _ = cut_count(seq, motion_thr, onset, offset, 8, swing_thr, args.window)
                        if n == 1:
                            hit += 1
                        elif n == 0:
                            zero += 1
                        else:
                            multi += 1
                    results.append({
                        "motion_frame_threshold": motion_thr,
                        "onset_frames": onset,
                        "offset_frames": offset,
                        "swing_mean_threshold": swing_thr,
                        "exactly_one": hit,
                        "zero": zero,
                        "multi": multi,
                        "hit_rate": round(hit / len(seqs), 4),
                    })
                    print(f"  运动量 {motion_thr:<4} onset {onset} offset {offset} "
                          f"挥速闸 {swing_thr:<4} → 一拍一击 {hit:>3} 零窗口 {zero:>3} "
                          f"多窗口 {multi:>3}  命中率 {hit/len(seqs):.1%}", flush=True)

    best = sorted(results, key=lambda r: (-r["exactly_one"], r["multi"], r["zero"]))[:10]
    print("\n最佳组合（按“恰好一个窗口”排序）：")
    for r in best:
        print(f"  运动量 {r['motion_frame_threshold']} onset {r['onset_frames']} "
              f"offset {r['offset_frames']} 挥速闸 {r['swing_mean_threshold']} → "
              f"命中率 {r['hit_rate']:.1%}（多窗口 {r['multi']}，零窗口 {r['zero']}）")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"videos": len(seqs), "results": results, "best": best}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{args.out}")


if __name__ == "__main__":
    main()
