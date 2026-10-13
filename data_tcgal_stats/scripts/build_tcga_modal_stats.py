#!/usr/bin/env python3
"""Build TCGA patient-level P/C/G modality availability tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_ROOT = Path("/data/lizhe/Medteam_projects")
OUTPUT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = OUTPUT_ROOT.parent / "dataset_deployment" / "mappings"
# 患者 ID 统一模式：TCGA 条码（TCGA-XX-XXXX）或 CPTAC 队列编号（如 01BR001 / 20CO012）。
TCGA_PATIENT_RE = re.compile(
    r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|\d{2}(?:BR|CO)\d{3})", re.IGNORECASE
)

CATEGORIES = [
    ("PCG", "全部完整（PCG）", True, True, True),
    ("CG", "仅缺 P（CG）", False, True, True),
    ("PG", "仅缺 C（PG）", True, False, True),
    ("PC", "仅缺 G（PC）", True, True, False),
    ("G", "缺 PC（仅 G）", False, False, True),
    ("C", "缺 PG（仅 C）", False, True, False),
    ("P", "缺 CG（仅 P）", True, False, False),
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    gene_root: Path
    wsi_root: Path
    clinic_files: tuple[Path, ...]
    case_dirs: tuple[Path, ...] = ()
    project_filter: tuple[str, ...] = ()
    rna_manifest: Path | None = None
    skip_reason: str | None = None
    # WSI 文件约定（TCGA 为 .svs；CPTAC 等 IDC 队列为 .dcm）
    wsi_glob: str = "*.svs"


DATASETS = [
    DatasetConfig(
        name="TCGA_LIHC",
        root=BASE_ROOT / "TCGA_LIHC",
        gene_root=BASE_ROOT / "TCGA_LIHC" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "TCGA_LIHC" / "WSI",
        clinic_files=(
            BASE_ROOT / "TCGA_LIHC" / "clinical" / "clinical.cart.2026-06-01.json",
            BASE_ROOT / "TCGA_LIHC" / "clinical.cart.2026-03-26.json",
        ),
        case_dirs=(BASE_ROOT / "TCGA_LIHC" / "clinical", BASE_ROOT / "TCGA_LIHC"),
        rna_manifest=BASE_ROOT / "TCGA_LIHC" / "gdc_manifest.2026-06-10.160348_rna.txt",
    ),
    DatasetConfig(
        name="TCGA-BRCA",
        root=BASE_ROOT / "TCGA-BRCA",
        gene_root=BASE_ROOT / "TCGA-BRCA" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "TCGA-BRCA" / "WSI",
        clinic_files=(
            BASE_ROOT / "TCGA-BRCA" / "Bulk_RNA_case" / "clinical.cart.2026-06-29.json",
            BASE_ROOT / "TCGA-BRCA" / "WSI_case" / "clinical.cart.2026-06-29.json",
            BASE_ROOT / "TCGA-BRCA" / "Mutation_case" / "clinical.cart.2026-06-29.json",
            BASE_ROOT / "TCGA-BRCA" / "Mythylation_case" / "clinical.cart.2026-06-29.json",
            BASE_ROOT / "TCGA-BRCA" / "RPPA_case" / "clinical.cart.2026-06-29.json",
        ),
        case_dirs=(
            BASE_ROOT / "TCGA-BRCA" / "Bulk_RNA_case",
            BASE_ROOT / "TCGA-BRCA" / "WSI_case",
            BASE_ROOT / "TCGA-BRCA" / "Mutation_case",
            BASE_ROOT / "TCGA-BRCA" / "Mythylation_case",
            BASE_ROOT / "TCGA-BRCA" / "RPPA_case",
        ),
    ),
    DatasetConfig(
        name="TCGA-COAD",
        root=BASE_ROOT / "TCGA-COAD",
        gene_root=BASE_ROOT / "TCGA-COAD" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "TCGA-COAD" / "WSI",
        clinic_files=(BASE_ROOT / "TCGA-COAD" / "Bulk_RNA_cases" / "clinical.cart.2026-06-18.json",),
        case_dirs=(
            BASE_ROOT / "TCGA-COAD" / "Bulk_RNA_cases",
            BASE_ROOT / "TCGA-COAD" / "WSI_cases",
        ),
    ),
    DatasetConfig(
        name="TCGA-PRAD",
        root=BASE_ROOT / "TCGA-PRAD",
        gene_root=BASE_ROOT / "TCGA-PRAD" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "TCGA-PRAD" / "WSI",
        clinic_files=(
            BASE_ROOT / "TCGA-PRAD" / "Bulk_RNA_cases" / "clinical.cart.2026-06-18.json",
            BASE_ROOT / "TCGA-PRAD" / "WSI_cases" / "clinical.cart.2026-06-18.json",
        ),
        case_dirs=(
            BASE_ROOT / "TCGA-PRAD" / "Bulk_RNA_cases",
            BASE_ROOT / "TCGA-PRAD" / "WSI_cases",
        ),
    ),
    DatasetConfig(
        name="TCGA-READ",
        root=BASE_ROOT / "TCGA-READ",
        gene_root=BASE_ROOT / "TCGA-READ" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "TCGA-READ" / "WSI",
        clinic_files=(
            BASE_ROOT / "TCGA-READ" / "Bulk_RNA_cases" / "clinical.cart.2026-06-21.json",
            BASE_ROOT / "TCGA-READ" / "WSI_cases" / "clinical.cart.2026-06-21.json",
        ),
        case_dirs=(
            BASE_ROOT / "TCGA-READ" / "Bulk_RNA_cases",
            BASE_ROOT / "TCGA-READ" / "WSI_cases",
        ),
    ),
    DatasetConfig(
        name="TCGA-STAD",
        root=BASE_ROOT / "TCGA-STAD",
        gene_root=BASE_ROOT / "TCGA-STAD" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "TCGA-STAD" / "WSI",
        clinic_files=(
            BASE_ROOT / "TCGA-STAD" / "Bulk_RNA_cases" / "clinical.cart.2026-06-21.json",
            BASE_ROOT / "TCGA-STAD" / "WSI_cases" / "clinical.cart.2026-06-21.json",
        ),
        case_dirs=(
            BASE_ROOT / "TCGA-STAD" / "Bulk_RNA_cases",
            BASE_ROOT / "TCGA-STAD" / "WSI_cases",
        ),
    ),
    DatasetConfig(
        name="TCGA-KIRC",
        root=BASE_ROOT / "kindey_cancer_TCGA",
        gene_root=BASE_ROOT / "kindey_cancer_TCGA" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "kindey_cancer_TCGA" / "WSI",
        clinic_files=(BASE_ROOT / "kindey_cancer_TCGA" / "clinical" / "clinical.cart.2026-03-17.json",),
        case_dirs=(BASE_ROOT / "kindey_cancer_TCGA" / "clinical",),
        project_filter=("TCGA-KIRC",),
        rna_manifest=BASE_ROOT / "kindey_cancer_TCGA" / "gdc_manifest.2026-03-18.111422_Bulk.txt",
    ),
    DatasetConfig(
        name="TCGA-KIRP",
        root=BASE_ROOT / "kindey_cancer_TCGA",
        gene_root=BASE_ROOT / "kindey_cancer_TCGA" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "kindey_cancer_TCGA" / "WSI",
        clinic_files=(BASE_ROOT / "kindey_cancer_TCGA" / "clinical" / "clinical.cart.2026-03-17.json",),
        case_dirs=(BASE_ROOT / "kindey_cancer_TCGA" / "clinical",),
        project_filter=("TCGA-KIRP",),
        rna_manifest=BASE_ROOT / "kindey_cancer_TCGA" / "gdc_manifest.2026-03-18.111422_Bulk.txt",
    ),
    DatasetConfig(
        name="TCGA-KICH",
        root=BASE_ROOT / "kindey_cancer_TCGA",
        gene_root=BASE_ROOT / "kindey_cancer_TCGA" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "kindey_cancer_TCGA" / "WSI",
        clinic_files=(BASE_ROOT / "kindey_cancer_TCGA" / "clinical" / "clinical.cart.2026-03-17.json",),
        case_dirs=(BASE_ROOT / "kindey_cancer_TCGA" / "clinical",),
        project_filter=("TCGA-KICH",),
        rna_manifest=BASE_ROOT / "kindey_cancer_TCGA" / "gdc_manifest.2026-03-18.111422_Bulk.txt",
    ),
    DatasetConfig(
        name="TCGA-LUAD",
        root=BASE_ROOT / "Lung",
        gene_root=BASE_ROOT / "Lung" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "Lung" / "WSI",
        clinic_files=(BASE_ROOT / "Lung" / "clinical" / "clinical.cart.2026-06-01.json",),
        case_dirs=(BASE_ROOT / "Lung" / "clinical",),
        project_filter=("TCGA-LUAD",),
        rna_manifest=BASE_ROOT / "Lung" / "gdc_manifest.2026-06-10.161746_rna.txt",
    ),
    DatasetConfig(
        name="TCGA-LUSC",
        root=BASE_ROOT / "Lung",
        gene_root=BASE_ROOT / "Lung" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "Lung" / "WSI",
        clinic_files=(BASE_ROOT / "Lung" / "clinical" / "clinical.cart.2026-06-01.json",),
        case_dirs=(BASE_ROOT / "Lung" / "clinical",),
        project_filter=("TCGA-LUSC",),
        rna_manifest=BASE_ROOT / "Lung" / "gdc_manifest.2026-06-10.161746_rna.txt",
    ),
    DatasetConfig(
        name="CPTAC-COAD",
        root=BASE_ROOT / "CPTAC_data" / "cptac_coad",
        gene_root=BASE_ROOT / "CPTAC_data" / "cptac_coad" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "CPTAC_data" / "cptac_coad" / "cptac_coad" / "cptac_coad",
        clinic_files=(BASE_ROOT / "CPTAC_data" / "cptac_coad" / "clinical" / "clinical.cart.2026-10-10.json",),
        case_dirs=(BASE_ROOT / "CPTAC_data" / "cptac_coad" / "clinical",),
        project_filter=("CPTAC-2",),
        wsi_glob="*.dcm",
    ),
    DatasetConfig(
        name="CPTAC-BRCA",
        root=BASE_ROOT / "CPTAC_data" / "cptac_brca",
        gene_root=BASE_ROOT / "CPTAC_data" / "cptac_brca" / "Bulk_RNA",
        wsi_root=BASE_ROOT / "CPTAC_data" / "cptac_brca" / "cptac_brca" / "cptac_brca",
        clinic_files=(BASE_ROOT / "CPTAC_data" / "cptac_brca" / "clinical" / "clinical.cart.2026-10-10.json",),
        case_dirs=(BASE_ROOT / "CPTAC_data" / "cptac_brca" / "clinical",),
        project_filter=("CPTAC-2",),
        wsi_glob="*.dcm",
    ),
]


def normalize_patient_id(value: str | None) -> str | None:
    if not value:
        return None
    match = TCGA_PATIENT_RE.search(value.upper())
    return match.group(1) if match else None


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_case_uuid(case_uuids: dict[str, set[str]], patient_id: str | None, case_uuid: str | None) -> None:
    if patient_id and case_uuid:
        case_uuids.setdefault(patient_id, set()).add(case_uuid)


def load_clinic_index(
    paths: Iterable[Path],
    case_uuids: dict[str, set[str]],
) -> tuple[set[str], dict[str, str]]:
    patients: set[str] = set()
    patient_projects: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        data = read_json(path)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            patient_id = normalize_patient_id(item.get("submitter_id"))
            project_id = ((item.get("project") or {}).get("project_id")) or ""
            if patient_id:
                patients.add(patient_id)
                if project_id:
                    patient_projects[patient_id] = project_id
                add_case_uuid(case_uuids, patient_id, item.get("case_id"))
    return patients, patient_projects


def load_sample_sheet_maps(case_dirs: Iterable[Path]) -> dict[str, tuple[str, str | None]]:
    mapping: dict[str, tuple[str, str | None]] = {}
    for case_dir in case_dirs:
        if not case_dir.exists():
            continue
        for path in case_dir.glob("gdc_sample_sheet*.tsv"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    file_name = (row.get("File Name") or "").strip()
                    patient_id = normalize_patient_id(row.get("Case ID") or row.get("Sample ID"))
                    case_uuid = (row.get("Case UUID") or row.get("Case ID UUID") or "").strip() or None
                    if file_name and patient_id:
                        mapping[file_name] = (patient_id, case_uuid)
    return mapping


def load_metadata_maps(case_dirs: Iterable[Path]) -> dict[str, tuple[str, str | None]]:
    mapping: dict[str, tuple[str, str | None]] = {}
    for case_dir in case_dirs:
        if not case_dir.exists():
            continue
        for path in case_dir.glob("metadata.cart*.json"):
            data = read_json(path)
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                file_name = item.get("file_name")
                if not file_name:
                    continue
                patient_id = normalize_patient_id(item.get("submitter_id") or file_name)
                case_uuid = None
                for entity in item.get("associated_entities") or []:
                    if not isinstance(entity, dict):
                        continue
                    patient_id = patient_id or normalize_patient_id(entity.get("entity_submitter_id"))
                    case_uuid = case_uuid or entity.get("case_id")
                if patient_id:
                    mapping[str(file_name)] = (patient_id, case_uuid)
    return mapping


def iter_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob(pattern) if path.is_file()]


def load_wsi_patients(config: DatasetConfig, case_uuids: dict[str, set[str]]) -> set[str]:
    file_map = load_sample_sheet_maps(config.case_dirs)
    file_map.update(load_metadata_maps(config.case_dirs))
    patients: set[str] = set()
    for path in iter_files(config.wsi_root, config.wsi_glob):
        mapped = file_map.get(path.name)
        patient_id = mapped[0] if mapped else normalize_patient_id(path.name)
        case_uuid = mapped[1] if mapped else None
        if patient_id:
            patients.add(patient_id)
            add_case_uuid(case_uuids, patient_id, case_uuid)
    return patients


def load_manifest_file_ids(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {
            row["filename"]: row["id"]
            for row in reader
            if row.get("filename") and row.get("id")
        }


def study_key_of(dataset_name: str) -> str:
    return dataset_name.lower().replace("-", "_")


def load_cached_file_case_mapping(config: DatasetConfig) -> dict[str, tuple[str, str | None]]:
    """Reuse the RNA file->case mapping cached by dataset_deployment (<study>_rna_file_case_map.csv)."""
    cache_path = CACHE_ROOT / f"{study_key_of(config.name)}_rna_file_case_map.csv"
    if not cache_path.exists():
        return {}
    mapping: dict[str, tuple[str, str | None]] = {}
    with cache_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            file_name = (row.get("file_name") or "").strip()
            patient_id = normalize_patient_id(row.get("case_id") or row.get("submitter_id"))
            case_uuid = (row.get("case_uuid") or "").strip() or None
            if file_name and patient_id:
                mapping[file_name] = (patient_id, case_uuid)
    return mapping


def fetch_gdc_file_mapping(file_ids: list[str]) -> dict[str, tuple[str, str | None]]:
    if not file_ids:
        return {}
    url = "https://api.gdc.cancer.gov/files"
    result: dict[str, tuple[str, str | None]] = {}
    fields = "file_id,file_name,cases.submitter_id,cases.case_id"
    for start in range(0, len(file_ids), 100):
        chunk = file_ids[start : start + 100]
        body = {
            "filters": {"op": "in", "content": {"field": "files.file_id", "value": chunk}},
            "fields": fields,
            "format": "JSON",
            "size": len(chunk),
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        for hit in payload.get("data", {}).get("hits", []):
            file_name = hit.get("file_name")
            cases = hit.get("cases") or []
            case = cases[0] if cases else {}
            patient_id = normalize_patient_id(case.get("submitter_id"))
            if file_name and patient_id:
                result[file_name] = (patient_id, case.get("case_id"))
    return result


def load_gene_patients(
    config: DatasetConfig,
    case_uuids: dict[str, set[str]],
    allow_gdc_api: bool,
) -> tuple[set[str], list[str]]:
    file_map = load_sample_sheet_maps(config.case_dirs)
    file_map.update(load_metadata_maps(config.case_dirs))
    file_map.update(load_cached_file_case_mapping(config))
    existing_files = iter_files(config.gene_root, "*.rna_seq.augmented_star_gene_counts.tsv")
    unmapped = [path.name for path in existing_files if path.name not in file_map]

    if unmapped and allow_gdc_api:
        manifest_map = load_manifest_file_ids(config.rna_manifest)
        file_ids = [manifest_map[name] for name in unmapped if name in manifest_map]
        try:
            file_map.update(fetch_gdc_file_mapping(file_ids))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"GDC API mapping failed for {config.name}: {exc}") from exc

    patients: set[str] = set()
    still_unmapped: list[str] = []
    for path in existing_files:
        mapped = file_map.get(path.name)
        patient_id = mapped[0] if mapped else normalize_patient_id(path.name)
        case_uuid = mapped[1] if mapped else None
        if patient_id:
            patients.add(patient_id)
            add_case_uuid(case_uuids, patient_id, case_uuid)
        else:
            still_unmapped.append(path.name)
    return patients, sorted(still_unmapped)


def classify(has_p: bool, has_c: bool, has_g: bool) -> tuple[str, str]:
    for code, label, need_p, need_c, need_g in CATEGORIES:
        if (has_p, has_c, has_g) == (need_p, need_c, need_g):
            return code, label
    raise ValueError("Unexpected empty modality combination")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_skipped_dataset(config: DatasetConfig) -> None:
    out_dir = OUTPUT_ROOT / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = out_dir / "README.md"
    readme.write_text(
        f"# {config.name}\n\n"
        f"本数据集暂未统计。\n\n"
        f"原因：{config.skip_reason}\n",
        encoding="utf-8",
    )


def process_dataset(config: DatasetConfig, allow_gdc_api: bool) -> list[dict[str, object]]:
    if config.skip_reason:
        write_skipped_dataset(config)
        return []

    out_dir = OUTPUT_ROOT / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_readme = out_dir / "README.md"
    if stale_readme.exists():
        stale_readme.unlink()
    stale_diagnostics = out_dir / f"{config.name}_unmapped_gene_files.txt"
    if stale_diagnostics.exists():
        stale_diagnostics.unlink()
    case_uuids: dict[str, set[str]] = {}

    p_patients = load_wsi_patients(config, case_uuids)
    c_patients, patient_projects = load_clinic_index(config.clinic_files, case_uuids)
    g_patients, unmapped_gene_files = load_gene_patients(config, case_uuids, allow_gdc_api)
    if unmapped_gene_files:
        stale_diagnostics.write_text("\n".join(unmapped_gene_files) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"{config.name} has {len(unmapped_gene_files)} unmapped Gene files. "
            f"See {stale_diagnostics}."
        )

    if config.project_filter:
        allowed_projects = set(config.project_filter)
        p_patients = {pid for pid in p_patients if patient_projects.get(pid) in allowed_projects}
        c_patients = {pid for pid in c_patients if patient_projects.get(pid) in allowed_projects}
        g_patients = {pid for pid in g_patients if patient_projects.get(pid) in allowed_projects}

    all_patients = sorted(p_patients | c_patients | g_patients)
    patient_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {code: [] for code, *_ in CATEGORIES}
    for patient_id in all_patients:
        has_p = patient_id in p_patients
        has_c = patient_id in c_patients
        has_g = patient_id in g_patients
        code, label = classify(has_p, has_c, has_g)
        case_uuid = ";".join(sorted(case_uuids.get(patient_id, set())))
        row = {
            "submitter_id": patient_id,
            "case_uuid": case_uuid,
            "has_P": int(has_p),
            "has_C": int(has_c),
            "has_G": int(has_g),
            "class_code": code,
            "缺失类型": label,
        }
        patient_rows.append(row)
        grouped[code].append({"submitter_id": patient_id, "case_uuid": case_uuid})

    write_csv(
        out_dir / f"{config.name}_patients.csv",
        ["submitter_id", "case_uuid", "has_P", "has_C", "has_G", "class_code", "缺失类型"],
        patient_rows,
    )
    for code, _label, *_ in CATEGORIES:
        write_csv(
            out_dir / f"{config.name}_{code}.csv",
            ["submitter_id", "case_uuid"],
            grouped[code],
        )

    total = len(all_patients)
    summary_rows: list[dict[str, object]] = []
    for code, label, *_ in CATEGORIES:
        count = len(grouped[code])
        ratio = count / total if total else 0.0
        summary_rows.append(
            {
                "dataset": config.name,
                "class_code": code,
                "缺失类型": label,
                "样本数": count,
                "占比": f"{ratio:.2%}",
                "ratio": f"{ratio:.6f}",
                "分母": total,
            }
        )
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-gdc-api",
        action="store_true",
        default=True,
        help="Allow GDC API lookup when local file-to-case metadata is missing.",
    )
    parser.add_argument(
        "--no-gdc-api",
        dest="allow_gdc_api",
        action="store_false",
        help="Disable GDC API lookup.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Only process the named dataset(s), e.g. --dataset TCGA-LUAD. Repeatable, or comma-separated.",
    )
    args = parser.parse_args()

    selected: set[str] = set()
    for item in args.dataset:
        for name in item.split(","):
            name = name.strip()
            if name:
                selected.add(name)

    summary_rows: list[dict[str, object]] = []
    processed_names: set[str] = set()
    for config in DATASETS:
        if selected and config.name not in selected:
            continue
        print(f"[INFO] Processing {config.name}", file=sys.stderr)
        summary_rows.extend(process_dataset(config, allow_gdc_api=args.allow_gdc_api))
        processed_names.add(config.name)

    summary_path = OUTPUT_ROOT / "汇总.csv"
    if selected and summary_path.exists():
        existing_rows: list[dict[str, object]] = []
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dataset") not in processed_names:
                    existing_rows.append(row)
        summary_rows = existing_rows + summary_rows

    write_csv(
        summary_path,
        ["dataset", "class_code", "缺失类型", "样本数", "占比", "ratio", "分母"],
        summary_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
