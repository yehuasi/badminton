# -*- coding: utf-8 -*-
"""临时脚本：提取 WIFI 连接指南 PDF 的文本。"""

from pypdf import PdfReader

PDF = r"E:\ruanjian\liaotiaowenjian\xwechat_files\wxid_o4nn126ja44122_1add\msg\file\2026-10\文档：WIFI连接指南.pdf"

r = PdfReader(PDF)
print("pages:", len(r.pages))
for i, p in enumerate(r.pages):
    print(f"--- page {i + 1} ---")
    print(p.extract_text())

