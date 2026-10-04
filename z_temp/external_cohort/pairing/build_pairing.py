# -*- coding: utf-8 -*-
"""外部数据集 切片<->报告 配对构建。

输入(只读): /data/lizhe/Medteam_projects/test/
  eval_tiff/{liver,lung,kidney}/*.tif   31 张
  OCR文本/*.txt                         30 份
输出: pairing_report.csv, 并打印空白切片表/孤儿报告表/统计数字(供 md 撰写)。
"""
import csv
import os
import pathlib
import re

ROOT = pathlib.Path("/data/lizhe/Medteam_projects/test")
OUT = pathlib.Path("/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init"
                   "/z_temp/external_cohort/pairing")
ORGAN_CN = {"liver": "肝", "lung": "肺", "kidney": "肾"}

ID_RE = re.compile(r"病理号\s*[：:]\s*([A-Za-z]?\s*\d{5,9})")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def read(p):
    return p.read_text(encoding="utf-8", errors="replace")


def get_field_line(text, name, extra_lines=0):
    """返回以 name 开头(或行内出现)的原始行; extra_lines>0 时附带后续若干行。

    病理诊断的阳性结论常被 OCR 折到下一行, 故允许向后取几行。"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(name):
            return " ".join(x.strip() for x in lines[i:i + 1 + extra_lines])
    m = re.search(r"%s[^\n]*" % re.escape(name), text)
    return m.group(0).strip() if m else ""


def organ_of(field_text):
    """从一段原文推断器官, 返回 (器官, 命中关键词) 或 (None, None)。"""
    hits = []
    for kw, org in [("支气管", "肺"), ("肺", "肺"), ("输尿管", "肾"), ("肾", "肾"),
                    ("肝", "肝")]:
        if kw in field_text and org not in [h[1] for h in hits]:
            hits.append((kw, org))
    if len(hits) == 1:
        return hits[0][1], hits[0][0]
    if len(hits) > 1:
        return "多器官冲突", ",".join(h[0] for h in hits)
    return None, None


# ---------- 解析报告 ----------
reports = []
for p in sorted((ROOT / "OCR文本").glob("*.txt")):
    t = read(p)
    m = ID_RE.search(t)
    pid = re.sub(r"\s+", "", m.group(1)) if m else ""
    rec = {
        "ocr": p.name, "pid": pid,
        "送检日期": (DATE_RE.search(get_field_line(t, "送检日期")).group(1)
                  if DATE_RE.search(get_field_line(t, "送检日期")) else ""),
        "报告日期": (DATE_RE.search(re.search(r"报告日期[^\n]*", t).group(0)).group(1)
                  if re.search(r"报告日期[^\n]*", t) and DATE_RE.search(re.search(r"报告日期[^\n]*", t).group(0)) else ""),
    }
    # 器官推断: 按字段优先级, 记录依据原文
    rec["器官"] = "无法判定"
    rec["依据"] = ""
    for fname, extra in [("送检标本", 0), ("送检材料", 0), ("大体所见", 0),
                         ("肉眼所见", 0), ("临床诊断", 0), ("送检科室", 0),
                         ("病理诊断", 4)]:
        raw = get_field_line(t, fname, extra)
        org, kw = organ_of(raw)
        if org:
            rec["器官"] = org
            rec["依据"] = "%s“%s”" % (fname, raw[:60])
            break
    if not rec["依据"]:  # 兜底: 全文找关键诊断词
        for kw, org in [("肺腺癌", "肺"), ("肺非黏液", "肺"), ("肾透明细胞癌", "肾"),
                        ("肝细胞性肝癌", "肝"), ("肝细胞肝癌", "肝")]:
            if kw in t:
                rec["器官"] = org
                rec["依据"] = "全文关键词“%s”" % kw
                break
    reports.append(rec)

# ---------- 解析切片 ----------
tifs = []
for d in ["liver", "lung", "kidney"]:
    for p in sorted((ROOT / "eval_tiff" / d).glob("*.tif")):
        first = p.name.split("_")[0]
        ts = re.search(r"_(\d{14})\.tif$", p.name)
        tifs.append({
            "file": p.name, "organ": ORGAN_CN[d], "dir": d,
            "pid": first if first else "",
            "ts": ("%s-%s-%s %s:%s:%s" % (ts.group(1)[0:4], ts.group(1)[4:6], ts.group(1)[6:8],
                                          ts.group(1)[8:10], ts.group(1)[10:12], ts.group(1)[12:14])) if ts else "",
            "size_gb": round(os.path.getsize(p) / 1024 ** 3, 2),
        })

by_pid = {r["pid"]: r for r in reports}
used = set()
rows = []
for t in tifs:
    r = by_pid.get(t["pid"]) if t["pid"] else None
    if t["pid"] and r:
        used.add(t["pid"])
        cons = "OK" if r["器官"] == t["organ"] else ("无法判定" if r["器官"] in ("无法判定", "多器官冲突") else "不一致")
        rows.append([t["file"], t["organ"], t["pid"], "matched", r["ocr"], r["pid"],
                     r["器官"], cons, "报告器官依据: " + r["依据"]])
    elif not t["pid"]:
        rows.append([t["file"], t["organ"], "", "blank_id", "", "", "",
                     "无法判定", "文件名无病理号; 时间戳 %s; 大小 %.2f GB" % (t["ts"], t["size_gb"])])
    else:
        rows.append([t["file"], t["organ"], t["pid"], "no_report", "", "", "",
                     "无法判定", "切片病理号 %s 在 30 份报告中无对应" % t["pid"]])

orphans = [r for r in reports if r["pid"] not in used]
for r in sorted(orphans, key=lambda x: x["pid"]):
    rows.append(["", "", r["pid"], "orphan_report", r["ocr"], r["pid"], r["器官"],
                 "无法判定",
                 "切片缺失; 送检日期 %s; 报告日期 %s; 报告器官依据: %s" % (r["送检日期"], r["报告日期"], r["依据"])])

header = ["tif文件名", "器官", "病理号", "状态", "匹配报告文件", "报告内病理号",
          "报告器官推断", "器官一致性", "备注"]
with open(OUT / "pairing_report.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

# ---------- 统计 ----------
print("切片总数:", len(tifs), "| 报告总数:", len(reports))
print("matched:", sum(1 for r in rows if r[3] == "matched"),
      "| blank_id:", sum(1 for r in rows if r[3] == "blank_id"),
      "| no_report:", sum(1 for r in rows if r[3] == "no_report"),
      "| orphan_report:", sum(1 for r in rows if r[3] == "orphan_report"))
print("器官一致性分布:", {x: sum(1 for r in rows if r[3] == "matched" and r[7] == x)
                        for x in ("OK", "不一致", "无法判定")})
print("报告器官分布:", {o: sum(1 for r in reports if r["器官"] == o) for o in ("肝", "肺", "肾", "无法判定", "多器官冲突")})
print("无法判定/冲突报告:", [(r["pid"], r["器官"]) for r in reports if r["器官"] in ("无法判定", "多器官冲突")])

print("\n--- 空白号切片表(md) ---")
for t in tifs:
    if not t["pid"]:
        print("| %s | %s | %s | %.2f GB |" % (t["file"], t["organ"], t["ts"], t["size_gb"]))

print("\n--- 无名切片/孤儿报告 数量分布 ---")
for o in ("肝", "肺", "肾"):
    print(o, "blank_id:", sum(1 for t in tifs if not t["pid"] and t["organ"] == o),
          "orphan:", sum(1 for r in orphans if r["器官"] == o))
print("no_report:", [(t["file"], t["organ"]) for t in tifs if t["pid"] and t["pid"] not in used])

print("\n--- 孤儿报告表(md) ---")
for r in sorted(orphans, key=lambda x: x["pid"]):
    print("| %s | %s | %s | %s | %s | %s |" % (r["ocr"], r["pid"], r["器官"], r["送检日期"], r["报告日期"], r["依据"]))
