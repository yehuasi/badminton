# 羽毛球运动姿态检测与分析系统

面向普通人和初学者的羽毛球动作姿态分析系统：以专业运动员标准为参照，对使用者
的击球动作进行**击球类型识别 + 技术动作规范打分 + 改进建议**，支持回看，并在
广和通 SC171 v3 开发套件上端侧部署。范围限定为“人 + 17 个 COCO 关键点”，不检测羽毛球。

## 核心能力

- 七类击球识别：正手发高远球、反手发网前小球、高远球、杀球、吊球、挑球、搓球
- 四项评分（百分制）：动作标准度 50% / 步伐 25% / 连贯性 15% / 动作识别 10%，识别置信度作门控
- 活动触发式切分：逐帧检测挥拍起止，变长窗口重采样识别，快慢挥拍自适应
- 双视角融合（正面 + 侧面）、局域网 Web 实时/回看/报告
- 端侧 ONNX Runtime 推理（480≈13fps / 416≈17fps，纯 CPU）

## 技术栈

Python + OpenCV + YOLOv8-Pose + 轻量 GRU 时序分类 + FastAPI + ONNX Runtime。

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m app.server                # 合成演示
.venv\Scripts\python -m app.server --source 1     # 摄像头
.venv\Scripts\python -m app.server --backend onnx --source demo
```

浏览器打开 `http://127.0.0.1:8000`。端侧部署与开发板操作见《部署指南.md》。

## 文档索引

- [方案文档.md](方案文档.md) —— 总体方案、架构、决策记录（含全部已确认决策）
- [评分权重依据.md](评分权重依据.md) —— 四项评分权重的 AHP 论证与文献依据
- [评分规则设计.md](评分规则设计.md) —— 评分计算映射与错误清单
- [标准库-技术要领.md](标准库-技术要领.md) —— 七类动作权威定性要领
- [部署指南.md](部署指南.md) —— PC 端与 SC171 v3 开发板完整操作指南
- [自采数据采集要求.docx](自采数据采集要求.docx) —— 自采数据的拍摄与标注规范

## 目录结构

```text
app/        核心代码（姿态、特征、分析、评分、建议、服务）
scripts/    训练 / 数据准备 / 导出 / 离线分析脚本
web/        局域网 Web 前端
tests/      单元测试
models/     分类器与姿态模型（大文件，未纳入版本控制）
data/       数据集（未纳入版本控制）
```

## 测试

```powershell
.venv\Scripts\python -m unittest tests.test_core
```
