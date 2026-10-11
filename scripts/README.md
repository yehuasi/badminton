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

## 5. 自采数据（后续）

用户自采视频到位后，用 YOLOv8-Pose 提取关键点，再按击球帧切窗口、转 15 维特征，
复用 `app/features.py` 与 `scripts/common.py` 的格式即可接入训练。

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
