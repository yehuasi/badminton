# -*- coding: utf-8 -*-
"""生成挑战杯答辩 PPT（python-pptx）。"""

import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "答辩演示.pptx")

FONT = "微软雅黑"
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GRAY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xEF, 0xF3, 0xF9)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def tb(s, x, y, w, h):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.text_frame.word_wrap = True
    return box, box.text_frame


def para(tf, text, size=18, bold=False, color=BLACK, first=False, space=6):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = FONT
    r.font.color.rgb = color
    return p


def title(s, text, sub=None):
    box, tf = tb(s, 0.7, 0.45, 11.9, 1.0)
    para(tf, text, size=30, bold=True, first=True)
    if sub:
        para(tf, sub, size=15, color=GRAY)


def figure(s, name, x, y, w):
    path = os.path.join(FIG, name)
    if os.path.isfile(path):
        s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def table(s, rows, x, y, w, h, sizes=14):
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    t = shape.table
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(sizes)
            r.font.name = FONT
            r.font.bold = (ri == 0)
            r.font.color.rgb = WHITE if ri == 0 else BLACK
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT if ri == 0 else (LIGHT if ri % 2 == 1 else WHITE)
    return t


# ---- 1 封面 ----
s = slide()
box, tf = tb(s, 1.2, 2.4, 11.0, 3.0)
para(tf, "基于端侧人工智能的", size=40, bold=True, first=True)
para(tf, "羽毛球动作识别与规范评估系统", size=40, bold=True)
para(tf, "科技发明制作 A 类    完成成员：XXXX、XXXX、XXXX", size=16, color=GRAY, space=0)
notes(s, "各位评委老师好。我们的作品是《基于端侧人工智能的羽毛球动作识别与规范评估系统》。"
         "一句话概括：用一台摄像头加一块开发板，在本地离线地看懂一个人的羽毛球动作，"
         "给出识别结果、规范评分和改进建议。下面从背景、方案、关键技术、端侧实现和价值来汇报。")

# ---- 2 背景与痛点 ----
s = slide()
title(s, "背景与痛点", "为什么需要它")
box, tf = tb(s, 0.9, 1.8, 11.5, 4.4)
para(tf, "羽毛球动作在零点几秒内完成，本人无法边发力边观察自己，练错了很难发现。",
     size=20, first=True)
para(tf, "专业教练数量有限、费用高，普通人很难获得“有人看着练”的反馈。", size=20)
para(tf, "现有分析产品多依赖云端、需要网络，且大多只给一个黑盒总分，"
         "用户不知道错在哪、怎么改。", size=20)
notes(s, "先说痛点。第一，羽毛球动作很快，自己看不见自己，容易越练越错。"
         "第二，好教练稀缺而且贵。第三，市面上的分析产品要么依赖云，要么只给一个总分，"
         "不告诉你问题在哪。这三点就是我们要解决的问题。")

# ---- 3 作品定位 ----
s = slide()
title(s, "作品定位", "一句话说清它是什么")
box, tf = tb(s, 0.9, 2.0, 11.5, 4.0)
para(tf, "以单摄像头为输入，只做“人 + 17 个骨架关键点”，不检测羽毛球。",
     size=22, bold=True, first=True)
para(tf, "一套算法、两个入口：PC 端用于开发验证，开发板端离线运行。", size=22, bold=True)
para(tf, "输出：击球类型识别、四项规范评分、可解释建议、训练回看。", size=22, bold=True)
notes(s, "我们的定位是：只用人体的17个关键点，在一台边缘设备上离线完成识别、评分和建议。"
         "特意不检测羽毛球，是为了降低复杂度——球小、快、容易被遮挡，而动作规范看人体骨架就够了。")

# ---- 4 总体方案 ----
s = slide()
title(s, "总体方案", "三级流水线")
figure(s, "fig_pipeline.png", 0.7, 1.6, 5.1)
box, tf = tb(s, 6.2, 1.8, 6.3, 4.4)
para(tf, "1 粗筛：轻量运动判断，过滤静止帧。", size=20, first=True)
para(tf, "2 姿态关键点：YOLOv8-Pose 提取 17 点。", size=20)
para(tf, "3 识别与评分：特征、切分、识别、评分、建议。", size=20)
para(tf, "越靠前的环节越轻，越靠后的越重；用轻环节把无效帧挡在重计算之外。",
     size=17, color=GRAY)
notes(s, "整体是一条三级流水线：粗筛、姿态关键点、识别与评分。设计原则是："
         "轻量的判断放在前面，把重计算留到真正有意义的时候才执行，这样有限算力也能实时。")

# ---- 5 关键技术一 ----
s = slide()
title(s, "关键技术一：活动触发式动作切分", "解决一拍多次判定与快慢难统一")
figure(s, "fig_segment.png", 0.7, 1.8, 6.5)
box, tf = tb(s, 7.5, 1.9, 5.0, 4.2)
para(tf, "逐帧计算运动量，检测挥拍起止。", size=19, first=True)
para(tf, "只在起止之间收集帧，一拍只出一个结果。", size=19)
para(tf, "变长窗口重采样到固定长度，快慢挥拍自适应。", size=19)
para(tf, "分类器只在收拍时运行一次，节省算力。", size=19)
notes(s, "第一个关键技术是动作切分。传统固定滑窗逐帧判定，会导致同一拍被反复判定，"
         "而且固定窗长对快慢挥拍不通用。我们改成活动触发：检测到挥拍开始才收集帧，"
         "检测到结束才切出一拍，再重采样到固定长度。这样一拍一个结果，快慢都能适应，还省算力。")

# ---- 6 关键技术二 ----
s = slide()
title(s, "关键技术二：过顶与非过顶分组门控", "抑制跨组误判")
figure(s, "fig_gate.png", 0.7, 1.9, 6.2)
box, tf = tb(s, 7.4, 2.0, 5.2, 4.0)
para(tf, "七类动作先按击球点高度分成两组。", size=19, first=True)
para(tf, "过顶：高远球、杀球、吊球。", size=19)
para(tf, "非过顶：发球、挑球、搓球。", size=19)
para(tf, "组内重新归一化，把候选从 7 类收窄到 3~4 类，"
         "抑制把低位挑球判成杀球这类错误。", size=19)
notes(s, "第二个关键技术是分组门控。我们发现分类器最容易犯的错是跨组误判，"
         "比如把低位的挑球判成杀球。所以先用击球点高度把动作分成过顶和非过顶两组，"
         "再在组内归一化，把候选从7类收窄到3到4类，跨组误判明显减少。")

# ---- 7 关键技术三 ----
s = slide()
title(s, "关键技术三：可解释的四项评分", "权重经 AHP 论证，识别作门控")
table(s, [["评分维度", "权重", "含义"],
          ["动作标准度", "50%", "技术规范与生物力学质量"],
          ["步伐", "25%", "重心转移、站距、起动"],
          ["连贯性", "15%", "轨迹平滑度、节奏、完整性"],
          ["动作识别", "10%", "识别置信度加权的完成度"]],
      0.7, 1.9, 6.3, 3.0, sizes=14)
box, tf = tb(s, 7.4, 1.9, 5.2, 4.0)
para(tf, "用 AHP 层次分析法确定权重，一致性检验通过（CR 小于 0.1）。", size=18, first=True)
para(tf, "识别是前提与门控，不是质量维度，故降至 10%；置信度低时动作分再减半。", size=18)
para(tf, "标准度由角度区间与错误规则支撑，每条建议都可追溯。", size=18)
notes(s, "第三个关键技术是评分。我们没有拍脑袋定权重，而是用层次分析法做两两比较，"
         "一致性检验通过。更重要的是把识别当成前提和门控：识别错了，后面比对就没意义；"
         "但识别得准不等于做得好。所以识别只占10%，置信度低还会再降权。"
         "标准度由标准角度区间和错误规则支撑，建议都能追溯到具体指标。")

# ---- 8 端侧部署与性能 ----
s = slide()
title(s, "端侧部署与性能", "纯 CPU 实时，无需 GPU 或 NPU")
figure(s, "fig_perf.png", 0.7, 1.8, 5.6)
box, tf = tb(s, 6.7, 1.9, 5.9, 4.2)
para(tf, "平台：广和通 SC171 v3（高通 QCM6490，aarch64 Linux）。", size=18, first=True)
para(tf, "推理：ONNX Runtime 纯 CPU，不依赖 PyTorch。", size=18)
para(tf, "关键优化：降低输入分辨率、固定 4 线程、显示与推理分离。", size=18)
para(tf, "结果：画面约 30fps，动作分析约 13fps。", size=20, bold=True)
notes(s, "端侧部署在广和通的SC171开发板上，纯CPU跑ONNX Runtime，不需要GPU。"
         "做了几项优化：姿态输入分辨率降到480；推理线程固定为4个，因为实测8线程反而更慢；"
         "最关键的是把画面显示和推理拆成两个线程。最后画面约30fps、动作分析约13fps。")

# ---- 9 实测数据 ----
s = slide()
title(s, "实测数据", "在开发板上测得的指标")
table(s, [["指标", "数值"],
          ["显示帧率", "约 30 fps（摄像头原生）"],
          ["分析帧率", "约 13 fps（480 输入）"],
          ["姿态推理单次", "约 67 ms（4 线程）"],
          ["常驻内存", "约 160 MB，长时间稳定"],
          ["识别基线", "公开数据跨场约 62%，待自采数据提升"]],
      0.7, 1.8, 7.2, 3.4, sizes=15)
box, tf = tb(s, 8.3, 1.8, 4.4, 4.2)
para(tf, "切分回归：单拍、慢拍、快拍、两拍、静止，结果数均符合预期。", size=17, first=True)
para(tf, "稳健性：站立或乱动时显示“无有效动作”，不强行归类。", size=17)
notes(s, "这是实测数据：显示约30fps、分析约13fps、内存约160兆且长时间稳定。"
         "识别基线目前是公开数据的62%，这是短板，也是要提升的重点。"
         "切分上我们做了单拍、慢拍、快拍、连续多拍和静止的回归测试，结果都符合预期。")

# ---- 10 系统功能 ----
s = slide()
title(s, "系统功能", "实时、回看、报告、双视角融合")
box, tf = tb(s, 0.9, 1.9, 11.6, 4.2)
para(tf, "实时：画面叠加骨架，显示动作类型、置信度、四项分项、总分与建议。", size=20, first=True)
para(tf, "回看：滑动选帧逐帧回放，并绘制得分趋势曲线。", size=20)
para(tf, "报告：按动作类型汇总次数与平均分。", size=20)
para(tf, "双视角融合：分别记录正面与侧面，自动融合成一套结果。", size=20)
notes(s, "功能上，浏览器打开就能用：实时页看画面和分数建议；回看页滑动选帧、看得分趋势；"
         "报告页按动作类型做统计；还有双视角融合——正面和侧面各记录一次，系统合成一套总评。")

# ---- 11 应用价值 ----
s = slide()
title(s, "应用价值", "谁在用，解决什么")
table(s, [["用户层次", "典型人群", "核心价值"],
          ["教学机构", "体校、俱乐部、培训机构", "把重复的规范检查交给设备"],
          ["学校体育", "中小学、大中专体育课", "可量化的动作评价与训练记录"],
          ["个人用户", "爱好者、初学者", "低成本的随身教练"]],
      0.7, 1.8, 11.9, 2.6, sizes=15)
box, tf = tb(s, 0.9, 4.8, 11.6, 2.0)
para(tf, "硬件成本低（单摄像头加开发板）；端侧离线、视频不出设备，隐私友好；"
         "框架可迁移到其他运动。", size=18, first=True)
notes(s, "价值上，面向三类用户：教学机构可以用它减轻重复的规范检查；"
         "学校可以用它做客观评价和训练记录；个人得到一个低成本的随身教练。"
         "硬件成本低、离线运行、视频不出设备，而且这套框架可以迁移到其他运动。")

# ---- 12 对比 ----
s = slide()
title(s, "与现有方案对比", "差异化的定位")
table(s, [["方案", "推理位置", "评分", "时域切分", "隐私"],
          ["人工录像回看", "无", "依赖人", "无", "好"],
          ["云端分析 App", "云端", "黑盒总分", "固定滑窗", "需上传"],
          ["专业动捕设备", "本地", "指标丰富", "高精度", "好但很贵"],
          ["本作品", "端侧", "可追溯", "活动触发式", "视频不出设备"]],
      0.7, 1.9, 11.9, 2.8, sizes=14)
box, tf = tb(s, 0.9, 5.0, 11.6, 1.8)
para(tf, "本作品落在低成本、端侧、可解释的中间地带，填补云端方案与专业设备之间的空缺。",
     size=18, first=True)
notes(s, "和现有方案比：人工回看没有自动分析；云端App依赖网络而且是黑盒；专业动捕很贵。"
         "我们的位置是中间地带——低成本、端侧、可解释，正好填补这个空缺。")

# ---- 13 结论与展望 ----
s = slide()
title(s, "结论与展望")
box, tf = tb(s, 0.9, 1.8, 11.6, 4.4)
para(tf, "结论：完成了从数据、模型、评分到端侧部署的完整闭环，系统在开发板上实时可运行、可演示。",
     size=20, bold=True, first=True)
para(tf, "四个关键设计：活动触发式切分、分组门控、AHP 可解释评分、显示与推理分离。", size=20)
para(tf, "下一步：以自采数据微调提升识别精度、回填标准库角度区间、条件具备时引入 NPU 加速，"
         "并把框架扩展到其他运动。", size=20)
notes(s, "总结：我们完成了一条从数据、模型、评分到端侧部署的完整闭环，"
         "在真实开发板上实时可运行、可演示。四个关键设计是活动触发式切分、分组门控、"
         "可解释评分、显示与推理分离。下一步用自采数据提升识别精度，回填标准库角度区间，"
         "条件具备时再做NPU加速。谢谢各位老师。")

# ---- 14 问辩准备 ----
s = slide()
title(s, "问辩准备", "评委可能关注的问题")
box, tf = tb(s, 0.9, 1.7, 11.7, 5.2)
para(tf, "识别准确率多少，为什么不更高？公开数据跨场约 62%，受机位差异与类别不均衡限制，"
         "已有自采数据采集规范与微调流程作为提升路径。", size=16, first=True)
para(tf, "为什么不用更强的模型？端侧纯 CPU 算力约束，用特征工程、切分与门控补偿，"
         "兼顾实时与可解释。", size=16)
para(tf, "为什么只做人不做球？球检测成本高、稳定性差，动作规范主要看人体骨架。", size=16)
para(tf, "评分是否主观？权重经 AHP 论证并通过一致性检验，标准度由角度区间与错误规则支撑。",
     size=16)
para(tf, "创新点是什么？活动触发式切分、分组门控、可解释四项评分、显示与推理分离，"
         "以及从训练到部署的完整自研链路。", size=16)
notes(s, "这一页是问辩准备。我们预判评委会问准确率、为什么不用更强模型、为什么不做球、"
         "评分是否主观、创新点在哪，都给出了直接回答，避免临场被动。")

prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides._sldIdLst))
