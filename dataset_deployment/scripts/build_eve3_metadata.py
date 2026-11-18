"""
EVE3 前置：为 TCGA 主线 7 数据集的天然缺失患者补齐生存数据，产出 EVE3 专用
metadata CSV（datasets_csv/eve3_metadata/{study}.csv）。

背景：data_tcgal_stats 记录了 81 名天然缺失患者（缺 P/C/G 模态），其中大多数
因无 WSI（或有 WSI 无特征文件）不满足原 metadata 生成条件（`not slides -> skip`），
从未进入 datasets_csv/metadata/*.csv。EVE3 要对这些患者做真实缺失评估，必须：
  1. 从原始 GDC 临床 JSON 提取其生存数据（口径与 pipeline.extract_survival 完全一致：
     demographic.days_to_death -> censorship=0；否则 diagnoses.days_to_last_follow_up
     -> censorship=1；days/30.44 -> 月）；
  2. 生成含这些患者的 metadata CSV（原 metadata 行保持不动，仅追加缺失患者行）。

输出：
  - datasets_csv/eve3_metadata/{study}.csv（原行 + 追加行，schema 与原 metadata 一致）
  - datasets_csv/eve3_metadata/提取报告.csv（每名缺失患者：是否查到生存、事件/删失、月数）

用法（base env）:
  python dataset_deployment/scripts/build_eve3_metadata.py [--study tcga_coad | --all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from dataset_deployment.registry import get_dataset_config
from dataset_deployment.scripts.pipeline import (
    extract_age,
    extract_is_female,
    extract_oncotree_code,
    extract_survival,
    load_case_lookup,
    load_patient_table,
    pick_primary_diagnosis,
    write_indexed_csv,
)

METADATA_COLUMNS = [
    "case_id", "slide_id", "age", "site", "survival_months",
    "censorship", "is_female", "oncotree_code", "rna_file_name",
]
EVE3_STUDIES = ["tcga_brca", "tcga_coad", "tcga_kirc", "tcga_kirp",
                "tcga_lihc", "tcga_luad", "tcga_lusc"]
OUT_DIR = PROJ_ROOT / "datasets_csv" / "eve3_metadata"


def build_for_study(study: str) -> dict[str, object]:
    config = get_dataset_config(study)
    metadata_csv = Path(config.metadata_csv)
    original = pd.read_csv(metadata_csv)
    original_cases = set(original["case_id"].astype(str))
    display_name = config.display_name  # 如 TCGA-COAD（stats 目录名）

    patient_df = load_patient_table(config)
    stats = pd.read_csv(f"{config.stats_dir}/{display_name}_patients.csv")
    stats.columns = [str(col).strip() for col in stats.columns]
    missing_df = stats[~((stats["has_P"] == 1) & (stats["has_C"] == 1) & (stats["has_G"] == 1))].copy()
    missing_ids = sorted(
        cid for cid in missing_df["submitter_id"].astype(str).unique()
        if cid not in original_cases
    )

    case_lookup = load_case_lookup(config)
    appended_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    for patient_id in missing_ids:
        case = case_lookup.get(patient_id)
        row_base = {
            "study": study,
            "case_id": patient_id,
            "age": extract_age(case) if case else "",
            "site": patient_id.split("-")[1] if "-" in patient_id else patient_id[:4],
            "is_female": extract_is_female(case) if case else "",
            "oncotree_code": extract_oncotree_code(case) if case else "N/A",
            "rna_file_name": "",
        }
        if case is None:
            status = "no_case_in_clinical_jsons"
            report_rows.append({**row_base, "slide_id": "", "status": status,
                                "survival_months": "", "censorship": ""})
            continue
        diagnosis = pick_primary_diagnosis(case)
        survival_months, censorship = extract_survival(case, diagnosis)
        if survival_months is None or censorship is None:
            status = "no_survival_fields"
            report_rows.append({**row_base, "slide_id": "", "status": status,
                                "survival_months": "", "censorship": ""})
            continue
        # 无 WSI 的患者用合成 slide_id（EVE3 前向对缺失模态零填充，slide 名不参与加载）
        slide_id = f"{patient_id}__unavailable"
        appended_rows.append({**row_base, "slide_id": slide_id,
                              "survival_months": survival_months, "censorship": censorship})
        report_rows.append({**row_base, "slide_id": slide_id, "status": "ok",
                            "survival_months": survival_months, "censorship": censorship})

    combined = pd.concat(
        [original, pd.DataFrame(appended_rows, columns=METADATA_COLUMNS)],
        ignore_index=True,
    )
    combined = combined.sort_values("case_id").reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_indexed_csv(OUT_DIR / f"{study}.csv", METADATA_COLUMNS, combined.to_dict("records"))

    n_ok = sum(1 for r in report_rows if r["status"] == "ok")
    n_events = sum(1 for r in report_rows if r["status"] == "ok" and int(r["censorship"]) == 0)
    summary = {
        "study": study,
        "missing_in_stats": len(missing_df),
        "missing_not_in_metadata": len(missing_ids),
        "appended_with_survival": n_ok,
        "appended_events": n_events,
        "no_survival_or_case": len(report_rows) - n_ok,
        "output": str(OUT_DIR / f"{study}.csv"),
    }
    print(summary)
    return {"report_rows": report_rows, **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="EVE3 天然缺失患者生存数据补齐")
    parser.add_argument("--study", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.study:
        studies = [args.study]
    elif args.all:
        studies = EVE3_STUDIES
    else:
        studies = EVE3_STUDIES[:1]

    all_report_rows: list[dict[str, object]] = []
    for study in studies:
        result = build_for_study(study)
        all_report_rows.extend(result.pop("report_rows"))
    report_df = pd.DataFrame(all_report_rows)
    write_indexed_csv(
        OUT_DIR / "提取报告.csv",
        ["study", "case_id", "slide_id", "age", "site", "is_female", "oncotree_code",
         "status", "survival_months", "censorship"],
        report_df.to_dict("records"),
    )
    print(f"[WRITE] {OUT_DIR / '提取报告.csv'}")


if __name__ == "__main__":
    main()
