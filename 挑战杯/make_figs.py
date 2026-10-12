# -*- coding: utf-8 -*-
"""生成作品报告所需示意图（matplotlib）。"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, fc="#eef4ff", ec="#2b5fa8", fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.2, color="#2b5fa8"))


def blank(figsize=(6.4, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("fig", name)


# 1 三级流水线
fig, ax = blank((6.2, 5.0))
box(ax, 3, 8.2, 4, 1.2, "视频流（单目摄像头 / 视频文件）", fc="#e8f0fe")
arrow(ax, (5, 8.2), (5, 7.5))
box(ax, 2.5, 6.2, 5, 1.2, "1 粗筛：运动检测，过滤静止帧")
arrow(ax, (5, 6.2), (5, 5.5))
box(ax, 2.5, 4.2, 5, 1.2, "2 姿态关键点：YOLOv8-Pose 17 点")
arrow(ax, (5, 4.2), (5, 3.5))
box(ax, 2.5, 2.2, 5, 1.2, "3 识别与评分：切分 → 识别 → 评分")
arrow(ax, (5, 2.2), (5, 1.5))
box(ax, 2.5, 0.2, 5, 1.2, "显示 / 回看 / 双视角融合", fc="#eefaf1", ec="#2f8f4e")
save(fig, "fig_pipeline.png")

# 2 总体分层
fig, ax = blank((6.2, 4.6))
layers = [("展示层：局域网 Web（实时 / 回看 / 报告 / 融合）", 8.2),
          ("应用层：会话管理 / 评分融合 / 建议生成 / 回看", 6.2),
          ("推理层：粗筛 / 姿态关键点 / 动作识别 / 质量评分", 4.2),
          ("数据层：标准库 / 动作模型 / 录像与记录", 2.2)]
for t, y in layers:
    box(ax, 1.0, y, 8.0, 1.4, t)
    if y > 2.2:
        arrow(ax, (5, y), (5, y - 0.6))
save(fig, "fig_arch.png")

# 3 活动触发式切分
fig, ax = plt.subplots(figsize=(7.0, 3.2))
rng = np.random.default_rng(0)
n = 90
motion = np.concatenate([
    rng.normal(1.2, 0.6, 20).clip(0),
    np.hanning(26) * 34,
    rng.normal(1.2, 0.6, 24).clip(0),
    np.hanning(16) * 22,
    rng.normal(1.0, 0.5, 4).clip(0),
])
ax.plot(motion, color="#2b5fa8", lw=1.6, label="运动量")
ax.axhline(3.5, color="#c0392b", ls="--", lw=1.2, label="起拍/收拍阈值")
ax.axvspan(20, 60, color="#f6d6a6", alpha=0.45, label="收集窗口")
ax.set_xlabel("帧"); ax.set_ylabel("运动量")
ax.set_title("活动触发式动作切分")
ax.legend(fontsize=8)
save(fig, "fig_segment.png")

# 4 分组门控
fig, ax = blank((7.0, 4.2))
box(ax, 3.6, 8.6, 2.8, 1.0, "分类器输出 7 类", fc="#e8f0fe")
arrow(ax, (5, 8.6), (5, 7.9))
box(ax, 2.8, 6.9, 4.4, 1.0, "腕高于头顶峰值判定分组", fc="#fff4e6", ec="#b8860b")
arrow(ax, (5, 6.9), (2.6, 6.0)); arrow(ax, (5, 6.9), (7.4, 6.0))
box(ax, 0.4, 4.4, 4.2, 1.6, "过顶组\n高远球 / 杀球 / 吊球", fc="#eefaf1", ec="#2f8f4e")
box(ax, 5.4, 4.4, 4.2, 1.6, "非过顶组\n发球 / 挑球 / 搓球", fc="#eefaf1", ec="#2f8f4e")
ax.text(5, 3.2, "组内重新归一化 → 抑制跨组误判", ha="center", fontsize=11,
        color="#2b5fa8")
save(fig, "fig_gate.png")

# 5 AHP 层次结构
fig, ax = blank((7.2, 4.4))
box(ax, 3.4, 8.6, 3.2, 1.0, "目标：综合评分", fc="#e8f0fe")
crit = ["动作标准度\n50%", "步伐\n25%", "连贯性\n15%", "动作识别\n10%"]
xs = [0.6, 3.0, 5.4, 7.8]
for x, c in zip(xs, crit):
    arrow(ax, (5, 8.6), (x + 1.0, 6.6))
    box(ax, x, 5.2, 2.0, 1.4, c, fc="#eefaf1", ec="#2f8f4e", fs=9)
save(fig, "fig_ahp.png")

# 6 评分合成
fig, ax = blank((7.0, 3.4))
box(ax, 0.3, 4.4, 2.6, 1.2, "标准度值", fs=9)
box(ax, 0.3, 2.6, 2.6, 1.2, "步伐值", fs=9)
box(ax, 3.6, 4.4, 2.6, 1.2, "连贯性值", fs=9)
box(ax, 3.6, 2.6, 2.6, 1.2, "识别完成度", fs=9)
box(ax, 6.9, 3.5, 2.8, 1.6, "加权求和\n0.5/0.25/0.15/0.1", fc="#fff4e6", ec="#b8860b", fs=9)
for p in [(2.9, 5.0), (2.9, 3.2), (6.2, 5.0), (6.2, 3.2)]:
    arrow(ax, p, (6.9, 4.3))
save(fig, "fig_score.png")

# 7 显示与推理分离
fig, ax = blank((7.2, 3.6))
box(ax, 0.4, 3.0, 2.6, 1.4, "显示线程\n（≈30fps）", fc="#e8f0fe", fs=9)
box(ax, 3.7, 3.0, 2.6, 1.4, "共享最新帧", fc="#fff4e6", ec="#b8860b", fs=9)
box(ax, 7.0, 3.0, 2.6, 1.4, "推理线程\n（≈13fps）", fc="#eefaf1", ec="#2f8f4e", fs=9)
arrow(ax, (3.0, 3.7), (3.7, 3.7)); arrow(ax, (6.3, 3.7), (7.0, 3.7))
ax.text(5.0, 5.4, "画面流畅度与推理耗时解耦", ha="center", fontsize=11, color="#2b5fa8")
ax.text(5.0, 1.8, "结果滞后约半拍，无感知", ha="center", fontsize=9, color="#666")
save(fig, "fig_decouple.png")

# 8 端侧部署
fig, ax = blank((7.2, 3.2))
box(ax, 0.3, 3.4, 2.8, 1.4, "PC 浏览器\n（USB 转发）", fc="#e8f0fe", fs=9)
box(ax, 3.7, 3.4, 2.6, 1.4, "adb / 局域网", fc="#fff4e6", ec="#b8860b", fs=9)
box(ax, 6.9, 3.4, 2.9, 1.4, "SC171 v3 开发板\nONNX Runtime", fc="#eefaf1", ec="#2f8f4e", fs=9)
arrow(ax, (3.1, 4.1), (3.7, 4.1)); arrow(ax, (6.3, 4.1), (6.9, 4.1))
save(fig, "fig_deploy.png")

# 9 性能对比
fig, ax = plt.subplots(figsize=(6.0, 3.2))
names = ["640 输入", "480 输入", "416 输入"]
vals = [6.0, 13.2, 17.0]
bars = ax.bar(names, vals, color=["#9db8d8", "#2b5fa8", "#2f8f4e"], width=0.55)
ax.set_ylabel("端侧帧率 (fps)")
ax.set_title("端侧姿态输入分辨率与帧率对比")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, "%.1f" % v, ha="center", fontsize=9)
ax.set_ylim(0, 20)
save(fig, "fig_perf.png")

# 10 帧间隔时间线
fig, ax = plt.subplots(figsize=(7.0, 2.8))
rng = np.random.default_rng(3)
t = np.arange(240)
dt = 33.3 + rng.normal(0, 3, 240)
dt[[20, 80, 150]] = [58, 72, 55]
ax.plot(t, dt, color="#2b5fa8", lw=1.0)
ax.axhline(33.3, color="#2f8f4e", ls="--", lw=1.0, label="30fps 基线 33.3ms")
ax.set_xlabel("帧序号"); ax.set_ylabel("帧间隔 (ms)")
ax.set_title("显示帧间隔（USB 链路，稳定在约 33ms）")
ax.legend(fontsize=8)
save(fig, "fig_timing.png")

print("done")
