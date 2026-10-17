# -*- coding: utf-8 -*-
"""用 OfficePlus 蓝色科技风模板生成《轻羽智评》答辩演示。

做法：保留模板的主题、封面/章节页/正文标题样式与结尾页，替换文字内容；
删除模板自带的 AI 主题插图，换成项目内容与 figures/ 里的示意图。
输出：挑战杯/答辩演示.pptx
"""

import os
import struct

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "演示.pptx")
OUT = os.path.join(HERE, "答辩演示.pptx")
FIG = os.path.join(HERE, "figures")

TITLE_MAIN = "轻羽智评"
TITLE_SUB = "端侧人工智能羽毛球动作识别与规范评估系统"
MEMBERS = "成员：XXXX、XXXX、XXXX、XXXX、XXXX"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)   # 深蓝
BODY = RGBColor(0x33, 0x33, 0x33)
FONT = "微软雅黑"


def set_run_font(run, size=None, bold=None, color=None, name=FONT):
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    ea = rpr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rpr, qn("a:ea"))
    ea.set("typeface", name)


def set_ph_text(shape, text):
    """替换占位符第一段文字，保留原字体样式。"""
    tf = shape.text_frame
    p = tf.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.add_run().text = text


def find_placeholder(slide, ph_type):
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.type == ph_type:
            return sh
    return None


def find_textbox(slide, contains):
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        if contains in (sh.text_frame.text or ""):
            return sh
    return None


def strip_slide(slide):
    """删除所有非占位符形状（模板自带的装饰图/组合）。"""
    for sh in list(slide.shapes):
        if not sh.is_placeholder:
            sh._element.getparent().remove(sh._element)


def delete_slides(prs, indexes):
    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)
    for i in sorted(indexes, reverse=True):
        sld = slides[i]
        rid = sld.get(qn("r:id"))
        prs.part.drop_rel(rid)
        xml_slides.remove(sld)


def add_textbox(slide, left, top, width, height, items, size=16, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        p.line_spacing = 1.1
        if isinstance(item, tuple):
            text, bold, color = item
        else:
            text, bold, color = item, False, BODY
        r = p.add_run()
        r.text = text
        set_run_font(r, size=size, bold=bold, color=color)
    return tb


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def jpeg_size(path):
    with open(path, "rb") as f:
        data = f.read(200000)
    i, n = 2, len(data)
    while i < n:
        if data[i] != 0xFF:
            i += 1
            continue
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD9:
            continue
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h = struct.unpack(">H", data[i + 1:i + 3])[0]
            w = struct.unpack(">H", data[i + 3:i + 5])[0]
            return w, h
        seglen = struct.unpack(">H", data[i:i + 2])[0]
        i += seglen
    return None


def img_size(path):
    if path.lower().endswith(".png"):
        return png_size(path)
    if path.lower().endswith((".jpg", ".jpeg")):
        return jpeg_size(path)
    return None


def add_figure(slide, filename, max_w=6.8, max_h=3.25, top=4.05):
    path = os.path.join(FIG, filename)
    if not os.path.isfile(path):
        return
    w_px, h_px = png_size(path)
    aspect = h_px / float(w_px)
    width = max_w
    height = width * aspect
    if height > max_h:
        height = max_h
        width = height / aspect
    left = (13.333 - width) / 2.0
    slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))


def add_picture_fit(slide, rel, box_left, box_top, box_w, box_h):
    """在给定盒内等比缩放并居中插入图片。"""
    path = os.path.join(HERE, rel.replace("/", os.sep))
    if not os.path.isfile(path):
        return
    w_px, h_px = img_size(path) or (1, 1)
    aspect = h_px / float(w_px)
    width = box_w
    height = width * aspect
    if height > box_h:
        height = box_h
        width = height / aspect
    left = box_left + (box_w - width) / 2.0
    top = box_top + (box_h - height) / 2.0
    slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))


def add_image_grid(slide, items):
    """2x2 网格插入截图并加小标题。"""
    cell_w = 5.2
    img_h = cell_w * 0.52
    gap_x = 0.35
    left0 = (13.333 - (cell_w * 2 + gap_x)) / 2.0
    top0 = 1.25
    row_h = img_h + 0.38
    for idx, (rel, cap) in enumerate(items):
        r, c = divmod(idx, 2)
        left = left0 + c * (cell_w + gap_x)
        top = top0 + r * row_h
        add_picture_fit(slide, rel, left, top, cell_w, img_h)
        tb = slide.shapes.add_textbox(Inches(left), Inches(top + img_h + 0.02), Inches(cell_w), Inches(0.32))
        p = tb.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = cap
        set_run_font(run, size=12, color=BODY)


def main():
    prs = Presentation(TEMPLATE)
    slides = list(prs.slides)

    # ---- 封面 ----
    s1 = slides[0]
    ct = find_placeholder(s1, PP_PLACEHOLDER.CENTER_TITLE)
    if ct is not None:
        set_ph_text(ct, TITLE_MAIN)
    st = find_placeholder(s1, PP_PLACEHOLDER.SUBTITLE)
    if st is not None:
        set_ph_text(st, TITLE_SUB)
    for name, text in [
        ("20XX", "2026"),
        ("Presenter name", "成员：XXXX、XXXX、XXXX"),
        ("OfficePLUS.cn", ""),
        ("ARTIFICIAL INTELLIGENCE", "SMART BADMINTON · EDGE AI"),
        ("BLUE TECH STYLE", "端侧 · 实时 · 可解释"),
    ]:
        sh = find_textbox(s1, name)
        if sh is not None:
            sh.text_frame.text = text

    # ---- 目录（第 3 页，Section Header）----
    s3 = slides[2]
    t = find_placeholder(s3, PP_PLACEHOLDER.TITLE)
    if t is not None:
        set_ph_text(t, "目录 CONTENTS")
    body = find_placeholder(s3, PP_PLACEHOLDER.BODY)
    if body is not None:
        body._element.getparent().remove(body._element)
    footer = find_textbox(s3, "ARTIFICIAL INTELLIGENCE GRADUATION DEFENCE")
    if footer is not None:
        footer.text_frame.text = "轻羽智评 · SMART BADMINTON"
    add_textbox(
        s3, 1.0, 4.45, 5.5, 2.6,
        ["01  项目背景与痛点", "02  现有技术对比", "03  系统总体设计", "04  核心技术与创新点"],
        size=18,
    )
    add_textbox(
        s3, 6.9, 4.45, 5.5, 2.6,
        ["05  动作识别与实验数据", "06  四项评分与建议", "07  端侧部署与性能", "08  应用推广与未来规划"],
        size=18,
    )

    # ---- 正文（模板第 4~13 页）----
    content = [
        dict(
            title="项目背景与痛点",
            bullets=[
                "羽毛球普及度高，动作规范却难以自查",
                "专业教练稀缺、费用较高，普通人缺少及时反馈",
                "市面方案多依赖云端，存在隐私与网络依赖",
                "目标：单摄像头加端侧 AI，做随时可用的动作教练",
            ],
            figure=None,
        ),
        dict(
            title="现有技术对比",
            bullets=[
                "人工录像回看：无量化指标，依赖个人经验",
                "云端动作分析：需联网，隐私风险，评分不透明",
                "本系统：端侧本地运行，实时量化，建议可解释",
            ],
            figure=None,
        ),
        dict(
            title="系统总体设计",
            bullets=[
                "三级流水线：视频流粗筛 → 姿态关键点 → 识别与评分",
                "一套算法、两种入口：实时与录像共用",
                "局域网 Web：实时、回看、报告、双视角融合",
            ],
            figure="fig_pipeline.png",
        ),
        dict(
            title="核心技术与创新点",
            bullets=[
                "活动触发式切分：一拍一次，快慢挥拍自适应",
                "识别门控：无动作不强行归类",
                "AHP 可解释评分：权重可追溯，建议可解释",
                "目标域微调：同人同机位识别 87.9%",
                "双视角融合：正面加侧面分时采集",
            ],
            figure=None,
        ),
        dict(
            title="动作识别与实验数据",
            bullets=[
                "七类击球：正反手发球、高远球、杀球、吊球、挑球、搓球",
                "过顶/非过顶分组门控，抑制跨组误判",
                "域适配后同人同机位留出准确率 87.9% ± 3.6%",
            ],
            figure="fig_gate.png",
        ),
        dict(
            title="四项评分与建议",
            bullets=[
                "动作标准度 50%、步伐 25%、连贯性 15%、动作识别 10%",
                "识别置信度低于 0.5 时动作分减半",
                "逐条错误规则加纠正话术，评分可解释",
            ],
            figure="fig_score.png",
        ),
        dict(
            title="端侧部署与性能",
            bullets=[
                "SC171 v3（QCM6490），ONNX Runtime 纯 CPU",
                "显示 30fps，分析帧率 13~17fps",
                "分类器 ONNX 约 0.36MB，姿态模型约 13MB",
                "切分命中率 95.8%",
            ],
            photo="photo/开发板实物照片.jpg",
        ),
        dict(
            title="系统界面",
            images=[
                ("photo/实时.png", "实时识别与建议"),
                ("photo/回看.png", "逐帧回看"),
                ("photo/报告.png", "会话报告"),
                ("photo/融合.png", "双视角融合"),
            ],
        ),
        dict(
            title="性能对比与优势验证",
            bullets=[
                "端侧本地运行，无需联网，保护隐私",
                "量化评分加可解释建议，区别于黑盒打分",
                "实物可在现场演示，非纸面方案",
            ],
            figure="fig_perf.png",
        ),
        dict(
            title="应用场景与推广价值",
            bullets=[
                "体校与俱乐部教学辅助、中小学体育课、居家训练、康复",
                "单摄像头加开发板，硬件成本低",
                "平台解耦：SC171 验证可行性，可迁移到 RK3588 等更低成本平台",
                "同一套框架可迁移到健身、网球、乒乓球",
            ],
            figure=None,
        ),
        dict(
            title="应用推广与未来规划",
            bullets=[
                "已完成数据、训练、评分、部署全链路",
                "未来：NPU 加速、更多运动、专业示范数据集",
                "社会价值：普及科学健身、缓解教练不足",
            ],
            figure=None,
        ),
    ]

    for idx, cfg in enumerate(content):
        slide = slides[3 + idx]
        strip_slide(slide)
        t = find_placeholder(slide, PP_PLACEHOLDER.TITLE)
        if t is not None:
            set_ph_text(t, cfg["title"])
        if "images" in cfg:
            add_image_grid(slide, cfg["images"])
            continue
        if cfg.get("photo"):
            add_textbox(slide, 0.7, 1.4, 7.2, 5.4, cfg["bullets"], size=15)
            add_picture_fit(slide, cfg["photo"], 8.15, 1.45, 4.7, 5.5)
            continue
        has_fig = cfg["figure"] is not None
        add_textbox(
            slide, 0.8, 1.35, 11.7, 2.7 if has_fig else 5.5,
            cfg["bullets"], size=16,
        )
        if has_fig:
            add_figure(slide, cfg["figure"])

    # ---- 结尾 ----
    s15 = slides[14]
    t = find_placeholder(s15, PP_PLACEHOLDER.TITLE)
    if t is not None:
        set_ph_text(t, "恳请各位评委批评指正")
    for name, text in [
        ("Presenter name", "成员：XXXX、XXXX、XXXX"),
        ("OfficePLUS.cn", ""),
        ("ARTIFICIAL INTELLIGENCE", "SMART BADMINTON · EDGE AI"),
    ]:
        sh = find_textbox(s15, name)
        if sh is not None:
            sh.text_frame.text = text

    # ---- 删除第 2 页（装饰页）----
    delete_slides(prs, [1])

    prs.save(OUT)
    print("saved", OUT, "slides", len(prs.slides._sldIdLst))


if __name__ == "__main__":
    main()
