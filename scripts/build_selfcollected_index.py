# -*- coding: utf-8 -*-
"""把自采数据目录整理成一张索引表，供后续入库/训练/评测使用。

输入：自采数据原始目录（结构不统一，见 reports/selfcollected_audit.json）。
输出：
    reports/selfcollected_index.csv   每条视频一行，带类别、划分、标签、错误族、视角、去重结论
控制台：错误族 × 类别分布、去重与不可用样本清单。

只读原始数据，不移动、不改名、不删除任何视频。
"""

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import REPO_ROOT  # noqa: E402

# 数据集里的教练错误标签 → 系统实际能检查的维度（错误族）
ERROR_FAMILY = {
    # 击球点与空间位置：对应 arm_raise / elbow / wrist_above_head / wrist_above_shoulder / wrist_reach
    "击球点错误": "击球点与空间位置",
    "击球点偏低": "击球点与空间位置",
    "击球点靠后": "击球点与空间位置",
    "击球点过近": "击球点与空间位置",
    "击球点过远": "击球点与空间位置",
    "击球点不稳定": "击球点与空间位置",
    "抢点击球": "击球点与空间位置",
    "放点过低": "击球点与空间位置",
    "发球点过高": "击球点与空间位置",
    "架拍太低": "击球点与空间位置",
    "仰着接球": "击球点与空间位置",
    # 侧身与转体：现有 15 维特征没有躯干朝向指标，需要新增
    "未侧身站位": "侧身与转体",
    "未旋转身体": "侧身与转体",
    "击球未侧身": "侧身与转体",
    "侧身不足": "侧身与转体",
    "身体正对球网": "侧身与转体",
    "未侧身蹬转": "侧身与转体",
    "转身架拍和挥拍脱节": "侧身与转体",
    # 发力链与手臂主导：肘部相对位置、是否大臂代偿
    "甩大臂": "发力链与手臂主导",
    "抡大臂": "发力链与手臂主导",
    "抡大臂挑球": "发力链与手臂主导",
    "大臂主导击球": "发力链与手臂主导",
    "肩部主导发力": "发力链与手臂主导",
    "掉肘击球": "发力链与手臂主导",
    "持拍手肘提前到身体前方": "发力链与手臂主导",
    "持拍手肘位置错误": "发力链与手臂主导",
    "肘关节不随挥拍带动": "发力链与手臂主导",
    "小臂发力，引拍过大": "发力链与手臂主导",
    "平抽式挑球": "发力链与手臂主导",
    "引拍位置靠后": "发力链与手臂主导",
    # 手腕与拍面控制：需要新增腕/拍面角度特征
    "勾手腕": "手腕与拍面控制",
    "压手腕": "手腕与拍面控制",
    "拧腕": "手腕与拍面控制",
    "锁腕": "手腕与拍面控制",
    "冗余甩腕": "手腕与拍面控制",
    "过度依赖手腕": "手腕与拍面控制",
    "手腕不外展": "手腕与拍面控制",
    "拍面角度不当": "手腕与拍面控制",
    "拍面角度不对": "手腕与拍面控制",
    "拍面控制差": "手腕与拍面控制",
    "拍面横放": "手腕与拍面控制",
    "手握拍过紧": "手腕与拍面控制",
    "握拍大拇指碾压过多": "手腕与拍面控制",
    "挥拍角度错误": "手腕与拍面控制",
    "向下发力过多": "手腕与拍面控制",
    # 步伐与支撑：对应 footwork_score
    "站姿错误": "步伐与支撑",
    "后腿支撑不稳": "步伐与支撑",
    "手脚不协调": "步伐与支撑",
    # 连贯与节奏：对应 coherence_score 与动作幅度
    "随挥停顿": "连贯与节奏",
    "发力不连贯": "连贯与节奏",
    "动作幅度过大": "连贯与节奏",
    "忽略非持拍手": "连贯与节奏",
    # 发球专项：持球与触点，属于发球类动作的规则性错误
    "持球部位错误+扔球": "发球专项",
    "持球部位错误+向上扔球": "发球专项",
    "发球时扔球": "发球专项",
    "击球触点错误": "发球专项",
}

FAMILIES = ["击球点与空间位置", "侧身与转体", "发力链与手臂主导", "手腕与拍面控制",
            "步伐与支撑", "连贯与节奏", "发球专项", "未归族"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(REPO_ROOT, "reports", "selfcollected_audit.json"))
    ap.add_argument("--dryrun", default=os.path.join(REPO_ROOT, "reports", "pipeline_dryrun.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "selfcollected_index.csv"))
    args = ap.parse_args()

    audit = json.load(open(args.audit, encoding="utf-8"))
    dry = {}
    if os.path.exists(args.dryrun):
        for r in json.load(open(args.dryrun, encoding="utf-8"))["records"]:
            dry[r["path"]] = r

    records = audit["records"]

    # 同一内容出现在多个位置时：保留标准集那份，验证集那份标记为剔除
    by_hash = collections.defaultdict(list)
    for r in records:
        if r.get("md5"):
            by_hash[r["md5"]].append(r)
    drop = {}
    for h, group in by_hash.items():
        if len(group) < 2:
            continue
        keep = sorted(group, key=lambda x: (x["split"] != "标准", x["path"]))[0]
        for r in group:
            if r is not keep:
                drop[r["path"]] = "与 " + keep["path"] + " 内容完全相同"

    rows = []
    for r in records:
        d = dry.get(r["path"], {})
        reason = r["error_reason"]
        family = ERROR_FAMILY.get(reason, "未归族") if reason else ""
        notes = []
        if r["path"] in drop:
            notes.append("重复：" + drop[r["path"]])
        if r["frames"] and r["frames"] < 24:
            notes.append(f"帧数不足 24（{r['frames']}），凑不满一个窗口")
        if r["view"] == "未标注":
            notes.append("未标注视角")
        if d.get("windows") == 0:
            notes.append("切分取不到挥拍窗口")
        elif (d.get("windows") or 0) > 1:
            notes.append(f"切出 {d['windows']} 个窗口，需人工确认是否只做一个动作")
        elif d.get("fallback"):
            notes.append("活动触发式切分未命中，走了兜底取窗")
        if d.get("det_rate") is not None and d.get("det_rate", 1) < 0.6:
            notes.append(f"人体检出率偏低（{d['det_rate']}）")
        rows.append({
            "path": r["path"],
            "class": r["class"],
            "split": r["split"],
            "label": r["label"],
            "error_reason": reason,
            "error_family": family,
            "view": r["view"],
            "variant": r["variant"],
            "player": r["player"],
            "frames": r.get("frames", ""),
            "duration": r.get("duration", ""),
            "fps": round(r.get("fps", 0), 2),
            "resolution": f"{r.get('w','')}x{r.get('h','')}",
            "windows": d.get("windows", ""),
            "det_rate": d.get("det_rate", ""),
            "kpt_conf": d.get("kpt_conf", ""),
            "swing_max": d.get("swing_max", ""),
            "md5": r.get("md5", ""),
            "keep": "否" if r["path"] in drop else "是",
            "notes": "；".join(notes),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("=== 错误族 × 类别（验证集里的错误样本）===")
    classes = sorted({r["class"] for r in rows})
    cnt = collections.Counter((r["error_family"], r["class"]) for r in rows if r["label"] == "错误")
    print(f"{'错误族':<20}" + "".join(f"{c[:6]:>8}" for c in classes) + f"{'合计':>8}")
    for fam in FAMILIES:
        row = [cnt[(fam, c)] for c in classes]
        if sum(row) == 0:
            continue
        print(f"{fam:<20}" + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")

    print("\n=== 处理结论 ===")
    print(f"  索引条目：{len(rows)}")
    print(f"  建议剔除（内容重复且已在标准集保留）：{sum(1 for r in rows if r['keep']=='否')}")
    print(f"  帧数不足 24：{sum(1 for r in rows if r['frames'] and r['frames']<24 and '帧数不足' in r['notes'])}")
    print(f"  未标注视角：{sum(1 for r in rows if r['view']=='未标注')}")
    print(f"  切分取不到窗口：{sum(1 for r in rows if '切分取不到挥拍窗口' in r['notes'])}")
    print(f"  切出多个窗口：{sum(1 for r in rows if '切出' in r['notes'])}")
    print(f"  走兜底取窗：{sum(1 for r in rows if '兜底' in r['notes'])}")
    print(f"  未归族的错误标签：{sorted({r['error_reason'] for r in rows if r['error_family']=='未归族'})}")
    print(f"\n索引已写入：{args.out}")


if __name__ == "__main__":
    main()
