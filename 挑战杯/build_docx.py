# -*- coding: utf-8 -*-
"""把《作品报告.md》转成符合学校格式要求的 Word 文档。

格式：A4；主标题二号华文中宋；章标题三号黑体；节/小节四号黑体；
正文四号仿宋、首行缩进两字符、单倍行距；表格五号；页眉=作品名称（居中）；
页脚=第X页 共Y页（居右），均五号仿宋；封面只含作品名称与成员。
"""

import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "作品报告.md")
OUT = os.path.join(HERE, "作品报告.docx")

TITLE = "基于端侧人工智能的羽毛球动作识别与规范评估系统"
MEMBERS = "XXXX、XXXX、XXXX、XXXX、XXXX"

F_TITLE = ("华文中宋", 22.0)   # 二号
F_H1 = ("黑体", 16.0)          # 三号
F_H2 = ("黑体", 14.0)          # 四号
F_BODY = ("仿宋", 14.0)        # 四号
F_TABLE = ("仿宋", 9.0)        # 表格内文（略小于五号，保证不溢出）
F_MONO = ("Consolas", 9.0)
F_SMALL = ("仿宋", 10.5)       # 五号


def _font(run, name, size, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(attr), name)


def _first_line_chars(par, chars=200):
    ppr = par._p.get_or_add_pPr()
    ind = ppr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        ppr.append(ind)
    ind.set(qn("w:firstLineChars"), str(chars))
    ind.set(qn("w:firstLine"), "0")


def _single_spacing(par):
    pf = par.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)


def _add_field(par, instr, placeholder="1"):
    run = par.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), instr)
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = placeholder
    r.append(t)
    fld.append(r)
    par._p.append(fld)


def _add_toc(doc):
    par = doc.add_paragraph()
    _single_spacing(par)
    _add_field(par, r'TOC \o "1-3" \h \z \u', "（目录：在 Word 中按 F9 更新）")


def _inline(par, text, name, size, bold=False):
    """处理 **加粗** 与 `代码` 行内标记。"""
    for seg in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**"):
            _font(par.add_run(seg[2:-2]), name, size, bold=True)
        elif seg.startswith("`") and seg.endswith("`"):
            _font(par.add_run(seg[1:-1]), "Consolas", size - 1.5)
        else:
            _font(par.add_run(seg), name, size, bold=bold)


def _table(doc, rows):
    ncol = max(len(r) for r in rows)
    tbl = doc.add_table(rows=0, cols=ncol)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        cells = tbl.add_row().cells
        for ci in range(ncol):
            text = row[ci] if ci < len(row) else ""
            p = cells[ci].paragraphs[0]
            _single_spacing(p)
            _inline(p, text, F_TABLE[0], F_TABLE[1], bold=(ri == 0))
    for r in tbl.rows:
        for c in r.cells:
            c.vertical_alignment = 1
    # 表前后留白
    doc.add_paragraph()
    return tbl


def _heading(doc, text, level):
    """用内置标题样式（供目录域收集），但套用学校要求的黑体字号。"""
    par = doc.add_paragraph(style=("Heading 1", "Heading 2", "Heading 3")[level - 1])
    _single_spacing(par)
    if level == 1:
        par.paragraph_format.space_before = Pt(12)
        par.paragraph_format.space_after = Pt(6)
        name, size = F_H1
    else:
        par.paragraph_format.space_before = Pt(8)
        par.paragraph_format.space_after = Pt(2)
        name, size = F_H2
    _inline(par, text, name, size, bold=True)


CODE_FILES = [
    ("app/classes.py", "类别与标签定义"),
    ("app/angles.py", "关节角度计算"),
    ("app/features.py", "关键点特征提取"),
    ("app/analyzer.py", "活动触发式切分与序列评分"),
    ("app/score.py", "单帧标准度与错误规则"),
    ("app/classifier.py", "GRU 分类器（torch）"),
    ("app/classifier_onnx.py", "GRU 分类器（onnx）"),
    ("app/fusion.py", "双视角融合"),
    ("app/store.py", "回看持久化"),
    ("app/advice.py", "建议生成"),
    ("app/server.py", "局域网 Web 服务"),
    ("app/pose.py", "YOLOv8-Pose 封装（torch）"),
    ("app/pose_onnx.py", "YOLOv8-Pose 推理（onnx）"),
    ("app/model.py", "GRU 模型与 checkpoint 读写"),
    ("scripts/prepare_selfcollected.py", "自采视频入库"),
    ("scripts/fit_std_ranges.py", "标准库角度区间拟合"),
    ("scripts/train.py", "识别模型训练脚本"),
]


def _code_run_font(run):
    run.font.size = Pt(F_MONO[1])
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.name = F_MONO[0]
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:ascii"), F_MONO[0])
    rf.set(qn("w:hAnsi"), F_MONO[0])
    rf.set(qn("w:eastAsia"), "宋体")
    rf.set(qn("w:cs"), F_MONO[0])


def _code_appendix(doc):
    repo = os.path.dirname(HERE)
    doc.add_page_break()
    _heading(doc, "附录 J 关键源码清单", 1)
    p = doc.add_paragraph()
    _single_spacing(p)
    _first_line_chars(p, 200)
    _font(p.add_run("以下列出本作品端侧与服务端的关键源码模块，供评审核验实现。"),
          F_BODY[0], F_BODY[1])
    total = 0
    for rel, desc in CODE_FILES:
        path = os.path.join(repo, rel.replace("/", os.sep))
        if not os.path.isfile(path):
            continue
        code = open(path, encoding="utf-8").read().rstrip("\n")
        nlines = code.count("\n") + 1
        total += nlines
        _heading(doc, "%s（%s，共 %d 行）" % (rel, desc, nlines), 2)
        par = doc.add_paragraph()
        _single_spacing(par)
        par.paragraph_format.left_indent = Cm(0.3)
        run = par.add_run()
        _code_run_font(run)
        for k, ln in enumerate(code.split("\n")):
            if k:
                run.add_break()
            run.add_text(ln)
    print("code appendix lines:", total)


def build():
    md = open(SRC, encoding="utf-8").read()
    lines = md.split("\n")
    doc = Document()

    # 页面设置 A4
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

    # 默认样式
    normal = doc.styles["Normal"]
    normal.font.name = F_BODY[0]
    normal.font.size = Pt(F_BODY[1])
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), F_BODY[0])

    # ---- 封面 ----
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run(TITLE), F_TITLE[0], F_TITLE[1])
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run("完成成员：" + MEMBERS), F_BODY[0], F_BODY[1])
    doc.add_page_break()

    # ---- 目录 ----
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run("目  录"), F_H1[0], F_H1[1])
    _add_toc(doc)
    doc.add_page_break()

    # ---- 正文（新节，页码从 1 开始）----
    new_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    new_sec.page_width = Cm(21.0)
    new_sec.page_height = Cm(29.7)
    new_sec.top_margin = Cm(2.54)
    new_sec.bottom_margin = Cm(2.54)
    new_sec.left_margin = Cm(2.5)
    new_sec.right_margin = Cm(2.5)
    new_sec.header.is_linked_to_previous = False
    new_sec.footer.is_linked_to_previous = False
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:start"), "1")
    new_sec._sectPr.append(pg)

    hp = new_sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(hp.add_run(TITLE), F_SMALL[0], F_SMALL[1])
    fp = new_sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _font(fp.add_run("第 "), F_SMALL[0], F_SMALL[1])
    _add_field(fp, "PAGE", "1")
    _font(fp.add_run(" 页　共 "), F_SMALL[0], F_SMALL[1])
    _add_field(fp, "SECTIONPAGES", "1")
    _font(fp.add_run(" 页"), F_SMALL[0], F_SMALL[1])
    for r in fp.runs:
        _font(r, F_SMALL[0], F_SMALL[1])

    # ---- 解析 Markdown 主体 ----
    i = 0
    in_code = False
    code_buf = []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if in_code:
                par = doc.add_paragraph()
                _single_spacing(par)
                par.paragraph_format.left_indent = Cm(0.5)
                _font(par.add_run("\n".join(code_buf)), F_MONO[0], F_MONO[1])
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        s = line.strip()
        if not s:
            i += 1
            continue
        m_img = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", s)
        if m_img:
            cap, rel = m_img.group(1), m_img.group(2)
            img = os.path.join(HERE, rel.replace("/", os.sep))
            if os.path.isfile(img):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.add_run().add_picture(img, width=Cm(13.5))
                cp = doc.add_paragraph()
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _single_spacing(cp)
                cp.paragraph_format.space_after = Pt(6)
                _font(cp.add_run(cap), "黑体", 10.5)
            i += 1
            continue
        if s == "---":
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            _table(doc, rows)
            continue
        if s.startswith("### "):
            _heading(doc, s[4:], 3)
            i += 1; continue
        if s.startswith("## "):
            _heading(doc, s[3:], 2)
            i += 1; continue
        if s.startswith("# "):
            _heading(doc, s[2:], 1)
            i += 1; continue
        if s.startswith(">"):
            par = doc.add_paragraph(); _single_spacing(par)
            par.paragraph_format.left_indent = Cm(0.5)
            _inline(par, s.lstrip("> "), F_SMALL[0], F_SMALL[1])
            i += 1; continue
        if re.match(r"^\d+\.\s", s) or s.startswith("- "):
            body = re.sub(r"^\d+\.\s+", "", s)
            body = body[2:] if body.startswith("- ") else body
            par = doc.add_paragraph(); _single_spacing(par)
            par.paragraph_format.left_indent = Cm(0.74)
            _inline(par, body, F_BODY[0], F_BODY[1])
            i += 1; continue
        # 普通段落
        par = doc.add_paragraph(); _single_spacing(par)
        _first_line_chars(par, 200)
        _inline(par, s, F_BODY[0], F_BODY[1])
        i += 1

    _code_appendix(doc)
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()
