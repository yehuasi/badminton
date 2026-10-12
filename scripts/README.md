# 训练链路使用说明

所有命令都在项目根目录 `E:\codex_project\badminton` 下执行，Python 用项目虚拟环境
`.venv\Scripts\python.exe`。

## 1. 生成仿真数据（真实数据到位前）

```powershell
.venv\Scripts\python scripts\make_synthetic_dataset.py --n-per-class 300
```

生成 `data/processed/` 下的 `X.npz`、`y.npy`、`classes.json`、`manifest.csv`。

## 2. 训练

```powershell
.venv\Scripts\python scripts\train.py --epochs 40
```

产出 `models/shot_classifier.pt` 与 `models/shot_classifier_classes.json`。

## 3. 评估

```powershell
.venv\Scripts\python scripts\eval.py --ckpt models\shot_classifier.pt
```

## 4. 真实数据（BFMD）

BFMD 标注包（17 关键点姿态 + 击球类型）仅发布在 Google Drive。
在能访问 Google Drive 的网络上（如已开 VPN 且设置了本地代理端口 7897）：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
.venv\Scripts\python scripts\download_bfmd_select.py
.venv\Scripts\python scripts\prepare_bfmd.py --out data\processed_bfmd
.venv\Scripts\python scripts\train.py --data data\processed_bfmd --epochs 40 --split-by-match --class-weight --out models\shot_classifier_bfmd.pt
.venv\Scripts\python scripts\eval.py --data data\processed_bfmd --ckpt models\shot_classifier_bfmd.pt --split-by-match
```

`download_bfmd_select.py` 只下载 pose / hit_inferred / metadata 三个子目录，逐文件下载、
失败重试、已下好的自动跳过，可反复运行续传。

`prepare_bfmd.py` 会把 BFMD 的 pose + hit_inferred 标签转成与仿真数据相同的规范格式。

如需在 15 维静态特征基础上追加手腕速度/加速度等时间导数特征（变为 23 维，用于
区分挥拍快慢、改善高远球 vs 吊球），在转换时加 `--kinematics` 即可：

```powershell
.venv\Scripts\python scripts\prepare_bfmd.py --out data\processed_bfmd_kin --kinematics
```

实测基线（12 场男单，按比赛划分 9 训练 / 3 验证，类别加权）：总体准确率约 62%。
杀球、搓球、反手发网前小球识别较好；高远球与吊球容易混淆，正手发高远球样本太少，
这两项需靠自采数据与更丰富的特征进一步改善。

## 5. 自采数据（用户拍摄 → 训练集）

把视频按类别放进 `data/self_collected/<类别名>/`（7 个类别子目录已建好），文件名任意，
支持 mp4/avi/mov/mkv。约定“一条视频只录一个完整动作”。

```powershell
.venv\Scripts\python scripts\prepare_selfcollected.py
```

脚本用 YOLOv8-Pose 逐帧提关键点，再**用与运行时完全相同的“活动触发式切分”定位
挥拍并取 24 帧窗口**（保证训练分布与运行时一致）；某视频切不出窗口时退回“挥速峰值
居中取窗”。输出 `data/processed_selfcollected/` 下的规范数据集。

训练/微调：

```powershell
# 与 BFMD 合并训练（类别须一致）
.venv\Scripts\python scripts\train.py --data data\processed_bfmd --extra data\processed_selfcollected --out models\shot_classifier_merged.pt

# 或从 BFMD 模型微调（自采样本少时推荐）
.venv\Scripts\python scripts\train.py --data data\processed_selfcollected --init models\shot_classifier_bfmd.pt --epochs 20 --out models\shot_classifier_self.pt
```

## 5.1 标准库角度区间拟合

专业示范数据到位后，用它与运行时一致的特征统计“均值±σ”，产出可回填
`app/score.py` 的 `STD_RANGES` 的区间：

```powershell
.venv\Scripts\python scripts\fit_std_ranges.py --data data\processed_selfcollected --hit detect
```

参数：`--hit detect`（自采切分窗口，用挥速峰值帧）或 `--hit center`（BFMD 等以击球帧
居中的窗口）；`--stat sigma`（均值±kσ）或 `--stat percentile`（分位区间，更抗离群）。
注意：区间质量取决于数据——可控双机位专业示范会得到紧致可用的区间；转播视频（BFMD）
机位杂、关键点噪声大，拟合出的区间偏宽，仅作参考。

## 6. 序列级评分（阶段 A）

`app/analyzer.py` 提供击球帧检测与四项评分（标准度 40% + 动作类型 40% +
连贯性 10% + 步伐 10%），Web 服务已接入，浏览器可切换“实时 / 报告”视图。

## 7. ONNX 导出（阶段 A，为端侧部署准备）

```powershell
.venv\Scripts\python scripts\export_onnx.py --ckpt models\shot_classifier_bfmd.pt
.venv\Scripts\python scripts\export_onnx.py --pose   # 同时导出 YOLOv8-Pose
```

产出 `models/shot_classifier.onnx`（输入 (B,24,15)，输出 7 类 logits），
已用 onnxruntime 校验与 PyTorch 输出一致。
