# -*- coding: utf-8 -*-
"""审查用：逐份打印 30 个 OCR txt 的关键字段，供人工核对病理号与器官归属推断依据。
不产出交付物。"""
import re
import pathlib

OCR_DIR = pathlib.Path("/data/lizhe/Medteam_projects/test/OCR文本")


def field(text, name):
    """取某字段的原文（该行），兼容全/半角冒号与无冒号。"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(name):
            return line
    # 字段名可能不在行首，宽松匹配
    m = re.search(r"%s[：:]?\s*([^\n]*)" % re.escape(name), text)
    return m.group(0).strip() if m else ""


ID_RE = re.compile(r"病理号\s*[：:]\s*([A-Za-z]?\s*\d{5,9})")

for p in sorted(OCR_DIR.glob("*.txt")):
    raw = p.read_text(encoding="utf-8", errors="replace")
    m = ID_RE.search(raw)
    pid = re.sub(r"\s+", "", m.group(1)) if m else "(未提取到)"
    print("=" * 90)
    print("文件:", p.name, "| 病理号:", pid)
    for f in ["送检科室", "临床诊断", "送检标本", "送检日期"]:
        print("  %s -> %s" % (f, field(raw, f)))
    # 报告日期可能藏在“注：”行里
    rd = re.search(r"报告日期[：:]?\s*(\d{4}-\d{2}-\d{2})", raw)
    print("  报告日期 -> %s" % (rd.group(1) if rd else "(未提取到)"))
    # 病理诊断首段（供器官判断）
    d = re.search(r"病理诊断[：:]\s*\n?(.{0,120})", raw, re.S)
    print("  病理诊断(截断) -> %s" % (d.group(1).replace("\n", " / ") if d else "(未提取到)"))
    # 肉眼所见/免疫组化里出现的器官词，供交叉参考
    organs = [w for w in ["肺", "肝", "肾", "输尿管", "支气管", "胆"] if w in raw]
    print("  全文出现器官词:", ",".join(organs))
