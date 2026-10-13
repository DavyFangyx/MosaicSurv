#!/usr/bin/env python3
"""CPTAC 协议端数据准备：样本表 / GDC 风格临床 JSON / RNA supplement。

输入（只读）：
  - WSI dcm 树：CPTAC_data/<cohort>/<cohort>/<cohort>/<case>/<study>/<series>/<uuid>.dcm
  - 临床表：cptac_coad/clinical/cbioportal_coad_cptac_2019_clinical_patient.csv
            cptac_brca/clinical/linkedomics_cptac_brca_{survival,meta}.txt
  - RNA 映射：CPTAC_data/gdc_rna_file_case_map.tsv
输出：
  - <cohort>/clinical/gdc_sample_sheet.2026-10-10.tsv（File Name=<uuid>.dcm -> Case ID）
  - <cohort>/clinical/clinical.cart.2026-10-10.json（GDC 风格）
  - dataset_deployment/supplement_files/<study>_rna_file_case_mapping.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

RAW = Path("/data/lizhe/Medteam_projects/CPTAC_data")
REPO = Path(__file__).resolve().parents[1]
H5_DIR = Path("/data/wanshan/workspace/medical_cv/TRIDENT/cptac_processed/20x_256px_0px_overlap/features_uni_v1")
MONTH_TO_DAYS = 30.4375
CPTAC_CASE_RE = re.compile(r"^(\d{2}(?:BR|CO)\d{3})$")

COHORTS = {
    "cptac_coad": {
        "project_id": "CPTAC-2",
        "study": "cptac_coad",
        "display": "CPTAC-COAD",
        "sex_map": {"Male": "male", "Female": "female"},
    },
    "cptac_brca": {
        "project_id": "CPTAC-2",
        "study": "cptac_brca",
        "display": "CPTAC-BRCA",
        "sex_map": {"Male": "male", "Female": "female"},
    },
}


def build_slide_maps() -> dict[str, dict[str, str]]:
    """cohort -> {uuid: case}，遍历 dcm 树（只读）。"""
    maps: dict[str, dict[str, str]] = {}
    for cohort in COHORTS:
        root = RAW / cohort / cohort / cohort
        mapping: dict[str, str] = {}
        for dcm in root.rglob("*.dcm"):
            case = dcm.parent.parent.parent.name
            if not CPTAC_CASE_RE.match(case):
                continue
            mapping[dcm.stem] = case
        maps[cohort] = mapping
        print(f"[slide] {cohort}: {len(mapping)} 个 slide -> case")
    return maps


def load_coad_clinical() -> dict[str, dict]:
    path = RAW / "cptac_coad" / "clinical" / "cbioportal_coad_cptac_2019_clinical_patient.csv"
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            case = (row.get("patientId") or "").strip()
            if not CPTAC_CASE_RE.match(case):
                continue
            out[case] = {
                "os_status": (row.get("OS_STATUS") or "").strip(),
                "os_months": (row.get("OS_MONTHS") or "").strip(),
                "stage": (row.get("STAGE") or "").strip(),
                "age": (row.get("AGE") or "").strip(),
                "sex": (row.get("SEX") or "").strip(),
            }
    return out


def load_brca_clinical() -> dict[str, dict]:
    surv = RAW / "cptac_brca" / "clinical" / "linkedomics_cptac_brca_survival.txt"
    meta = RAW / "cptac_brca" / "clinical" / "linkedomics_cptac_brca_meta.txt"
    info: dict[str, dict] = {}
    with surv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            case = (row.get("case_id") or "").strip()
            if not CPTAC_CASE_RE.match(case):
                continue
            info.setdefault(case, {})["os_days"] = (row.get("OS_days") or "").strip()
            info.setdefault(case, {})["os_event"] = (row.get("OS_event") or "").strip()
    with meta.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            case = (row.get("case_id") or "").strip()
            if not CPTAC_CASE_RE.match(case) or case not in info:
                continue
            info[case]["stage"] = (row.get("Stage") or "").strip()
            info[case]["age"] = (row.get("Age") or "").strip()
            info[case]["sex"] = (row.get("Sex") or "").strip()
    return info


def gdc_case(case: str, *, vital: str, days_to_death, days_to_follow, stage, age, sex, project_id: str) -> dict:
    demographic: dict = {"vital_status": vital}
    if days_to_death is not None:
        demographic["days_to_death"] = days_to_death
    if age is not None:
        demographic["age_at_index"] = int(age)
    if sex:
        demographic["gender"] = sex
    diagnosis: dict = {}
    if stage:
        diagnosis["ajcc_pathologic_stage"] = stage
    if days_to_follow is not None:
        diagnosis["days_to_last_follow_up"] = days_to_follow
    record: dict = {
        "submitter_id": case,
        "case_id": None,
        "project": {"project_id": project_id},
        "demographic": demographic,
        "diagnoses": [diagnosis] if diagnosis else [],
        "follow_ups": [],
    }
    return record


def build_clinical_json(cohort: str, raw_records: dict[str, dict]) -> int:
    records: list[dict] = []
    for case, r in sorted(raw_records.items()):
        vital = "Not Reported"
        dtd = dfu = None
        if cohort == "cptac_coad":
            status, months = r.get("os_status", ""), r.get("os_months", "")
            if months not in ("", "NA", "NaN"):
                days = round(float(months) * MONTH_TO_DAYS, 1)
                if status.startswith("1:"):
                    vital, dtd = "Dead", days
                elif status.startswith("0:"):
                    vital, dfu = "Alive", days
        else:
            days_str, event = r.get("os_days", ""), r.get("os_event", "")
            if days_str not in ("", "NA", "NaN"):
                days = round(float(days_str), 1)
                if event == "1":
                    vital, dtd = "Dead", days
                elif event == "0":
                    vital, dfu = "Alive", days
        stage = r.get("stage", "")
        if stage and stage.lower().startswith("stage "):
            stage = stage[6:]
        age_raw = r.get("age", "")
        try:
            age = int(float(age_raw)) if age_raw not in ("", "NA", "NaN") else None
        except ValueError:
            age = None
        sex_raw = r.get("sex", "")
        sex = COHORTS[cohort]["sex_map"].get(sex_raw.strip(), None) if sex_raw else None
        records.append(
            gdc_case(
                case,
                vital=vital,
                days_to_death=dtd,
                days_to_follow=dfu,
                stage=stage,
                age=age,
                sex=sex,
                project_id=COHORTS[cohort]["project_id"],
            )
        )
    out_path = RAW / cohort / "clinical" / "clinical.cart.2026-10-10.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=1)
    with_surv = sum(1 for r in records if (r["demographic"].get("days_to_death") is not None
                                           or r["diagnoses"] and r["diagnoses"][0].get("days_to_last_follow_up") is not None))
    dead = sum(1 for r in records if r["demographic"].get("vital_status") == "Dead")
    print(f"[clinic] {cohort}: {len(records)} 例，有生存时间 {with_surv}，死亡 {dead} -> {out_path}")
    return len(records)


def write_sample_sheets(slide_maps: dict[str, dict[str, str]]) -> None:
    for cohort, mapping in slide_maps.items():
        out_path = RAW / cohort / "clinical" / "gdc_sample_sheet.2026-10-10.tsv"
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["File Name", "Case ID", "Project ID"])
            for uuid, case in sorted(mapping.items(), key=lambda item: (item[1], item[0])):
                writer.writerow([f"{uuid}.dcm", case, COHORTS[cohort]["project_id"]])
        print(f"[sheet] {cohort}: wrote {len(mapping)} rows -> {out_path}")


def write_rna_supplements() -> None:
    map_path = RAW / "gdc_rna_file_case_map.tsv"
    rows: dict[str, list[list[str]]] = {s: [] for s in ("cptac_coad", "cptac_brca")}
    with map_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            case = (row.get("case_submitter_id") or "").strip()
            file_id = (row.get("file_id") or "").strip()
            if not case or not file_id:
                continue
            match = CPTAC_CASE_RE.match(case)
            if not match:
                continue
            study = "cptac_brca" if case[2:4] == "BR" else "cptac_coad"
            disk_name = f"{file_id}.rna_seq.augmented_star_gene_counts.tsv"
            rows[study].append([disk_name, case, case, case])
    supplement_dir = REPO / "dataset_deployment" / "supplement_files"
    supplement_dir.mkdir(parents=True, exist_ok=True)
    for study, entries in rows.items():
        out_path = supplement_dir / f"{study}_rna_file_case_mapping.csv"
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file_name", "case_id", "submitter_id", "case_uuid"])
            writer.writerows(sorted(entries))
        print(f"[rna-supp] {study}: {len(entries)} rows -> {out_path}")


def check_h5_coverage(slide_maps: dict[str, dict[str, str]]) -> None:
    if not H5_DIR.is_dir():
        print("[h5] WSI 特征目录不存在，跳过覆盖检查")
        return
    total = 0
    covered = 0
    unmatched: list[str] = []
    for path in sorted(H5_DIR.glob("*.h5")):
        total += 1
        uuid = path.stem
        if any(uuid in mapping for mapping in slide_maps.values()):
            covered += 1
        else:
            unmatched.append(uuid)
    print(f"[h5] {covered}/{total} 个 h5 能在 dcm 样本表里找到 case（未匹配 {len(unmatched)}）")
    if unmatched:
        print("     未匹配示例:", unmatched[:5])


def main() -> int:
    slide_maps = build_slide_maps()
    write_sample_sheets(slide_maps)
    build_clinical_json("cptac_coad", load_coad_clinical())
    build_clinical_json("cptac_brca", load_brca_clinical())
    write_rna_supplements()
    check_h5_coverage(slide_maps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
