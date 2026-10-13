from __future__ import annotations

import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from dataset_deployment.registry import (
    DEFAULT_GENE_EXPERIMENT,
    DEFAULT_WSI_EXPERIMENT,
    DatasetConfig,
    get_dataset_config,
    infer_standard_paths,
    list_enabled_studies,
    normalize_patient_id,
)
from dataset_deployment.workspace_features import build_case_feature_index, resolve_case_feature_path


STAT_ROW_PREFIXES = ("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous")
METADATA_COLUMNS = [
    "case_id",
    "slide_id",
    "age",
    "site",
    "survival_months",
    "censorship",
    "is_female",
    "oncotree_code",
    "rna_file_name",
]
CLINICAL_COLUMNS = ["case_id", "stage", "subtype", "grade"]
MANIFEST_COLUMNS = ["case_id", "source_file", "workspace_file", "modality", "embedding"]


def get_rna_case_cache_path(study: str) -> Path:
    cache_dir = PROJ_ROOT / "dataset_deployment" / "mappings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{study}_rna_file_case_map.csv"


def get_rna_case_supplement_path(study: str) -> Path:
    return PROJ_ROOT / "dataset_deployment" / "supplement_files" / f"{study}_rna_file_case_mapping.csv"


def resolve_studies(study: str | None, run_all: bool) -> list[str]:
    if study and run_all:
        raise ValueError("Use either --study or --all, not both.")
    if run_all:
        return list_enabled_studies()
    if study:
        get_dataset_config(study)
        return [study]
    return ["tcga_coad"]


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {
        str(key).replace("\ufeff", "").strip(): "" if value is None else str(value).strip()
        for key, value in row.items()
    }


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_patient_table(config: DatasetConfig) -> pd.DataFrame:
    patient_df = pd.read_csv(config.patient_table_csv)
    patient_df.columns = [str(col).strip() for col in patient_df.columns]
    return patient_df


def iter_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob(pattern) if path.is_file()]


def load_case_lookup(config: DatasetConfig) -> dict[str, dict]:
    case_lookup: dict[str, dict] = {}
    allowed_projects = set(config.raw.project_filter)
    for json_path_str in config.raw.clinical_jsons:
        json_path = Path(json_path_str)
        if not json_path.exists():
            continue
        data = read_json(json_path)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            patient_id = normalize_patient_id(item.get("submitter_id"))
            if not patient_id:
                continue
            project_id = ((item.get("project") or {}).get("project_id")) or ""
            if allowed_projects and project_id not in allowed_projects:
                continue
            case_lookup[patient_id] = item
    return case_lookup


def load_sample_sheet_maps(case_dirs: tuple[str, ...]) -> dict[str, tuple[str, str | None]]:
    mapping: dict[str, tuple[str, str | None]] = {}
    for case_dir_str in case_dirs:
        case_dir = Path(case_dir_str)
        if not case_dir.exists():
            continue
        for path in case_dir.glob("gdc_sample_sheet*.tsv"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    file_name = str(row.get("File Name") or "").strip()
                    patient_id = normalize_patient_id(row.get("Case ID") or row.get("Sample ID"))
                    case_uuid = str(row.get("Case UUID") or row.get("Case ID UUID") or "").strip() or None
                    if file_name and patient_id:
                        mapping[file_name] = (patient_id, case_uuid)
    return mapping


def load_metadata_maps(case_dirs: tuple[str, ...]) -> dict[str, tuple[str, str | None]]:
    mapping: dict[str, tuple[str, str | None]] = {}
    for case_dir_str in case_dirs:
        case_dir = Path(case_dir_str)
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


def load_manifest_file_ids(path_str: str | None) -> dict[str, str]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {
            row["filename"]: row["id"]
            for row in reader
            if row.get("filename") and row.get("id")
        }


def load_cached_file_case_mapping(cache_path: Path) -> dict[str, tuple[str, str | None]]:
    if not cache_path.exists():
        return {}
    df = pd.read_csv(cache_path)
    mapping = {}
    for _, row in df.iterrows():
        file_name = str(row.get("file_name", "")).strip()
        patient_id = normalize_patient_id(row.get("case_id") or row.get("submitter_id"))
        case_uuid = str(row.get("case_uuid", "")).strip() or None
        if file_name and patient_id:
            mapping[file_name] = (patient_id, case_uuid)
    return mapping


def load_supplement_file_case_mapping(study: str) -> dict[str, tuple[str, str | None]]:
    supplement_path = get_rna_case_supplement_path(study)
    if not supplement_path.exists():
        return {}
    df = pd.read_csv(supplement_path)
    mapping: dict[str, tuple[str, str | None]] = {}
    for _, row in df.iterrows():
        file_name = str(row.get("file_name", "")).strip()
        patient_id = normalize_patient_id(row.get("submitter_id") or row.get("case_id"))
        case_uuid = str(row.get("case_id", "")).strip() or None
        if file_name and patient_id:
            mapping[file_name] = (patient_id, case_uuid)
    return mapping


def save_cached_file_case_mapping(cache_path: Path, mapping: dict[str, tuple[str, str | None]]) -> None:
    rows = []
    for file_name, (patient_id, case_uuid) in sorted(mapping.items()):
        rows.append(
            {
                "file_name": file_name,
                "case_id": patient_id,
                "submitter_id": patient_id,
                "case_uuid": case_uuid or "",
            }
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["file_name", "case_id", "submitter_id", "case_uuid"]).to_csv(cache_path, index=False)


def fetch_gdc_file_mapping(file_ids: list[str]) -> dict[str, tuple[str, str | None]]:
    if not file_ids:
        return {}
    url = "https://api.gdc.cancer.gov/files"
    fields = "file_id,file_name,cases.submitter_id,cases.case_id"
    result: dict[str, tuple[str, str | None]] = {}
    for start in range(0, len(file_ids), 100):
        chunk = file_ids[start : start + 100]
        request = urllib.request.Request(
            url,
            data=json.dumps(
                {
                    "filters": {"op": "in", "content": {"field": "files.file_id", "value": chunk}},
                    "fields": fields,
                    "format": "JSON",
                    "size": len(chunk),
                }
            ).encode("utf-8"),
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
            case = (hit.get("cases") or [{}])[0]
            patient_id = normalize_patient_id(case.get("submitter_id"))
            if file_name and patient_id:
                result[file_name] = (patient_id, case.get("case_id"))
    return result


def build_wsi_lookup(config: DatasetConfig) -> dict[str, list[str]]:
    workspace_paths = infer_standard_paths(config.study, ".")
    workspace_wsi_dir = workspace_paths["data_root_dir"]
    if workspace_wsi_dir.exists():
        workspace_pt_files = sorted(workspace_wsi_dir.glob("*.pt"))
        if workspace_pt_files:
            wsi_lookup: dict[str, list[str]] = {}
            for path in workspace_pt_files:
                slide_stub = path.stem
                patient_id = normalize_patient_id(slide_stub)
                if not patient_id:
                    continue
                wsi_lookup.setdefault(patient_id, []).append(f"{slide_stub}{config.raw.wsi_ext}")
            for patient_id in wsi_lookup:
                wsi_lookup[patient_id] = sorted(set(wsi_lookup[patient_id]))
            return wsi_lookup

    file_map = load_sample_sheet_maps(config.raw.case_dirs)
    file_map.update(load_metadata_maps(config.raw.case_dirs))
    allowed_projects = set(config.raw.project_filter)
    case_lookup = load_case_lookup(config)
    wsi_lookup: dict[str, list[str]] = {}
    for path in iter_files(Path(config.raw.wsi_root), config.raw.wsi_glob):
        mapped = file_map.get(path.name)
        patient_id = mapped[0] if mapped else normalize_patient_id(path.name)
        if not patient_id:
            continue
        case = case_lookup.get(patient_id)
        project_id = ((case or {}).get("project") or {}).get("project_id", "")
        if allowed_projects and project_id not in allowed_projects:
            continue
        wsi_lookup.setdefault(patient_id, []).append(path.name)
    for patient_id in wsi_lookup:
        wsi_lookup[patient_id] = sorted(set(wsi_lookup[patient_id]))
    return wsi_lookup


def build_gene_lookup(
    config: DatasetConfig,
    *,
    allow_gdc_api: bool = False,
) -> tuple[dict[str, tuple[str, Path]], list[str]]:
    file_map = load_sample_sheet_maps(config.raw.case_dirs)
    file_map.update(load_metadata_maps(config.raw.case_dirs))
    file_map.update(load_supplement_file_case_mapping(config.study))
    cache_path = get_rna_case_cache_path(config.study)
    file_map.update(load_cached_file_case_mapping(cache_path))
    gene_files = iter_files(Path(config.raw.gene_root), "*.rna_seq.augmented_star_gene_counts.tsv")
    gene_paths = {path.name: path for path in gene_files}
    unmapped = [name for name in gene_paths if name not in file_map]

    if unmapped and allow_gdc_api:
        manifest_map = load_manifest_file_ids(config.raw.rna_manifest)
        file_ids = [manifest_map[name] for name in unmapped if name in manifest_map]
        if file_ids:
            try:
                fetched_mapping = fetch_gdc_file_mapping(file_ids)
                file_map.update(fetched_mapping)
                if fetched_mapping:
                    cached_mapping = load_cached_file_case_mapping(cache_path)
                    cached_mapping.update(fetched_mapping)
                    save_cached_file_case_mapping(cache_path, cached_mapping)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"[WARN] GDC API lookup failed for {config.study}: {exc}")

    case_lookup = load_case_lookup(config)
    allowed_projects = set(config.raw.project_filter)
    gene_lookup: dict[str, tuple[str, Path]] = {}
    unresolved: list[str] = []
    for file_name, file_path in gene_paths.items():
        mapped = file_map.get(file_name)
        patient_id = mapped[0] if mapped else normalize_patient_id(file_name)
        if not patient_id:
            unresolved.append(file_name)
            continue
        case = case_lookup.get(patient_id)
        project_id = ((case or {}).get("project") or {}).get("project_id", "")
        if allowed_projects and project_id not in allowed_projects:
            continue
        gene_lookup[patient_id] = (file_name, file_path)
    return gene_lookup, sorted(unresolved)


def pick_primary_diagnosis(case: dict) -> dict:
    diagnoses = case.get("diagnoses", [])
    for diagnosis in diagnoses:
        if str(diagnosis.get("diagnosis_is_primary_disease", "")).lower() == "true":
            return diagnosis
    return diagnoses[0] if diagnoses else {}


def extract_survival(case: dict, diagnosis: dict) -> tuple[float | None, int | None]:
    demographic = case.get("demographic", {})
    days_to_death = demographic.get("days_to_death")
    if days_to_death not in (None, "", "'--"):
        return float(days_to_death) / 30.44, 0
    last_follow_up = diagnosis.get("days_to_last_follow_up")
    if last_follow_up not in (None, "", "'--"):
        return float(last_follow_up) / 30.44, 1
    for follow_up in reversed(case.get("follow_ups", [])):
        candidate = follow_up.get("days_to_follow_up")
        if candidate not in (None, "", "'--"):
            return float(candidate) / 30.44, 1
    return None, None


def extract_age(case: dict) -> int | None:
    demographic = case.get("demographic", {})
    age = demographic.get("age_at_index")
    if age not in (None, "", "'--"):
        return int(age)
    days_to_birth = demographic.get("days_to_birth")
    if days_to_birth not in (None, "", "'--"):
        return int(abs(float(days_to_birth)) / 365.25)
    return None


def extract_is_female(case: dict) -> int:
    return 1 if str(case.get("demographic", {}).get("gender", "")).lower() == "female" else 0


def extract_oncotree_code(case: dict) -> str:
    project_id = case.get("project", {}).get("project_id", "")
    if project_id.startswith(("TCGA-", "CPTAC-")):
        return project_id.split("-", 1)[1]
    return project_id or "N/A"


def normalize_stage(raw_stage: str | None) -> str:
    if not raw_stage:
        return "N/A"
    cleaned = str(raw_stage).strip()
    if cleaned.lower().startswith("stage "):
        cleaned = cleaned[6:]
    return cleaned or "N/A"


def normalize_grade(raw_grade: str | None) -> str:
    cleaned = "" if raw_grade is None else str(raw_grade).strip()
    return cleaned or "N/A"


def extract_subtype(case: dict) -> str:
    project_id = case.get("project", {}).get("project_id", "")
    if project_id.startswith(("TCGA-", "CPTAC-")):
        return project_id.split("-", 1)[1]
    return project_id or "N/A"


def write_indexed_csv(output_path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + columns)
        for index, row in enumerate(rows):
            writer.writerow([index] + [row.get(column, "") for column in columns])


def load_target_genes(signature_csv: Path) -> list[str]:
    signatures_df = pd.read_csv(signature_csv)
    ordered_genes: list[str] = []
    seen: set[str] = set()
    for column in signatures_df.columns:
        for value in signatures_df[column].dropna():
            gene = str(value).strip()
            if gene and gene not in seen:
                seen.add(gene)
                ordered_genes.append(gene)
    return ordered_genes


def read_tpm_from_tsv(tsv_path: Path) -> pd.Series:
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    df = df[~df["gene_id"].isin(STAT_ROW_PREFIXES)]
    df = df[df["gene_type"] == "protein_coding"]
    return df.groupby("gene_name")["tpm_unstranded"].sum()


def expected_workspace_paths(
    study: str,
    *,
    gene_embedding: str = DEFAULT_GENE_EXPERIMENT,
    wsi_experiment: str = DEFAULT_WSI_EXPERIMENT,
) -> dict[str, Path]:
    return infer_standard_paths(
        study,
        ".",
        gene_experiment=gene_embedding,
        wsi_experiment=wsi_experiment,
    )


def generate_metadata_for_study(
    study: str,
    *,
    dry_run: bool = False,
    validate_only: bool = False,
    allow_gdc_api: bool = False,
) -> dict[str, object]:
    config = get_dataset_config(study)
    output_path = Path(config.metadata_csv)
    if validate_only:
        df = pd.read_csv(output_path)
        missing = [column for column in METADATA_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{output_path} missing columns: {missing}")
        return {"study": study, "rows": len(df), "output": str(output_path), "mode": "validate"}

    patient_df = load_patient_table(config)
    patient_df = patient_df[patient_df["has_P"] == 1].copy()
    case_lookup = load_case_lookup(config)
    wsi_lookup = build_wsi_lookup(config)
    gene_lookup, unresolved_gene_files = build_gene_lookup(config, allow_gdc_api=allow_gdc_api)

    rows: list[dict[str, object]] = []
    for patient_id in sorted(patient_df["submitter_id"].astype(str).unique()):
        case = case_lookup.get(patient_id)
        slides = wsi_lookup.get(patient_id, [])
        if case is None or not slides:
            continue
        diagnosis = pick_primary_diagnosis(case)
        survival_months, censorship = extract_survival(case, diagnosis)
        if survival_months is None or censorship is None:
            continue
        rna_file_name = gene_lookup.get(patient_id, ("", None))[0]
        for slide_id in slides:
            rows.append(
                {
                    "case_id": patient_id,
                    "slide_id": slide_id,
                    "age": extract_age(case),
                    "site": patient_id.split("-")[1] if "-" in patient_id else patient_id[:4],
                    "survival_months": survival_months,
                    "censorship": censorship,
                    "is_female": extract_is_female(case),
                    "oncotree_code": extract_oncotree_code(case),
                    "rna_file_name": rna_file_name,
                }
            )
    rows.sort(key=lambda item: (str(item["case_id"]), str(item["slide_id"])))
    if dry_run:
        return {"study": study, "rows": len(rows), "output": str(output_path), "unresolved_gene_files": unresolved_gene_files}
    write_indexed_csv(output_path, METADATA_COLUMNS, rows)
    return {"study": study, "rows": len(rows), "output": str(output_path), "unresolved_gene_files": unresolved_gene_files}


def generate_clinical_for_study(
    study: str,
    *,
    dry_run: bool = False,
    validate_only: bool = False,
) -> dict[str, object]:
    config = get_dataset_config(study)
    output_path = Path(config.clinical_csv)
    if validate_only:
        df = pd.read_csv(output_path)
        missing = [column for column in CLINICAL_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{output_path} missing columns: {missing}")
        return {"study": study, "rows": len(df), "output": str(output_path), "mode": "validate"}

    patient_df = load_patient_table(config)
    case_lookup = load_case_lookup(config)
    rows: list[dict[str, object]] = []
    for patient_id in sorted(patient_df["submitter_id"].astype(str).unique()):
        case = case_lookup.get(patient_id)
        if case is None:
            continue
        diagnosis = pick_primary_diagnosis(case)
        rows.append(
            {
                "case_id": patient_id,
                "stage": normalize_stage(diagnosis.get("ajcc_pathologic_stage")),
                "subtype": extract_subtype(case),
                "grade": normalize_grade(diagnosis.get("tumor_grade")),
            }
        )
    if dry_run:
        return {"study": study, "rows": len(rows), "output": str(output_path)}
    write_indexed_csv(output_path, CLINICAL_COLUMNS, rows)
    return {"study": study, "rows": len(rows), "output": str(output_path)}


def generate_rna_for_study(
    study: str,
    *,
    dry_run: bool = False,
    validate_only: bool = False,
    allow_gdc_api: bool = False,
    signature_csv: str = "datasets_csv/metadata/combine_signatures.csv",
) -> dict[str, object]:
    config = get_dataset_config(study)
    output_dir = Path(config.rna_dir)
    output_path = output_dir / "rna_clean.csv"
    target_genes = load_target_genes(Path(signature_csv))
    if validate_only:
        df = pd.read_csv(output_path, index_col=0)
        if list(df.columns) != target_genes:
            raise ValueError(f"{output_path} gene columns do not match {signature_csv}")
        return {"study": study, "rows": len(df), "output": str(output_path), "mode": "validate"}

    patient_df = load_patient_table(config)
    patient_df = patient_df[patient_df["has_G"] == 1].copy()
    gene_lookup, unresolved_gene_files = build_gene_lookup(config, allow_gdc_api=allow_gdc_api)

    records: dict[str, pd.Series] = {}
    skipped_cases: list[str] = []
    for patient_id in sorted(patient_df["submitter_id"].astype(str).unique()):
        mapped = gene_lookup.get(patient_id)
        if not mapped:
            skipped_cases.append(patient_id)
            continue
        _source_file, tsv_path = mapped
        records[patient_id] = read_tpm_from_tsv(tsv_path)

    expr_df = pd.DataFrame.from_dict(records, orient="index")
    expr_df = expr_df.reindex(columns=target_genes, fill_value=0.0).fillna(0.0)
    expr_df.index.name = "case_id"

    if dry_run:
        return {
            "study": study,
            "rows": len(expr_df),
            "output": str(output_path),
            "skipped_cases": skipped_cases,
            "unresolved_gene_files": unresolved_gene_files,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    expr_df.to_csv(output_path)
    return {
        "study": study,
        "rows": len(expr_df),
        "output": str(output_path),
        "skipped_cases": skipped_cases,
        "unresolved_gene_files": unresolved_gene_files,
    }


def generate_feature_manifest_for_study(
    study: str,
    *,
    dry_run: bool = False,
    validate_only: bool = False,
    allow_gdc_api: bool = False,
    gene_embedding: str = DEFAULT_GENE_EXPERIMENT,
) -> dict[str, object]:
    config = get_dataset_config(study)
    output_path = Path(config.feature_manifest_csv)
    if validate_only:
        df = pd.read_csv(output_path)
        missing = [column for column in MANIFEST_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{output_path} missing columns: {missing}")
        return {"study": study, "rows": len(df), "output": str(output_path), "mode": "validate"}

    workspace_paths = infer_standard_paths(study, ".", gene_experiment=gene_embedding)
    workspace_index = build_case_feature_index(workspace_paths["gene_dir"])
    gene_lookup, unresolved_gene_files = build_gene_lookup(config, allow_gdc_api=allow_gdc_api)
    rows = []
    for case_id, workspace_file in sorted(workspace_index.items()):
        source_file = ""
        if case_id in gene_lookup:
            source_file = gene_lookup[case_id][0]
        rows.append(
            {
                "case_id": case_id,
                "source_file": source_file,
                "workspace_file": workspace_file.name,
                "modality": "gene",
                "embedding": gene_embedding,
            }
        )

    if rows:
        rows.sort(key=lambda item: str(item["case_id"]))
    else:
        for case_id, (source_file, _path) in sorted(gene_lookup.items()):
            workspace_file = resolve_case_feature_path(
                workspace_paths["gene_dir"],
                case_id,
                case_index=workspace_index,
            )
            if workspace_file is None:
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "source_file": source_file,
                    "workspace_file": workspace_file.name,
                    "modality": "gene",
                    "embedding": gene_embedding,
                }
            )
    if dry_run:
        return {"study": study, "rows": len(rows), "output": str(output_path), "unresolved_gene_files": unresolved_gene_files}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(output_path, index=False)
    return {"study": study, "rows": len(rows), "output": str(output_path), "unresolved_gene_files": unresolved_gene_files}


def generate_splits_for_study(
    study: str,
    *,
    dry_run: bool = False,
    validate_only: bool = False,
    n_splits: int = 5,
    seed: int = 1,
) -> dict[str, object]:
    from dataset_deployment.scripts.generate_5fold_splits import generate_splits_for_study as _generate_splits_for_study

    return _generate_splits_for_study(
        study,
        dry_run=dry_run,
        validate_only=validate_only,
        n_splits=n_splits,
        seed=seed,
    )


def validate_workspace_for_study(
    study: str,
    *,
    gene_embedding: str = DEFAULT_GENE_EXPERIMENT,
    wsi_experiment: str = DEFAULT_WSI_EXPERIMENT,
) -> dict[str, object]:
    config = get_dataset_config(study)
    paths = infer_standard_paths(study, ".", gene_experiment=gene_embedding, wsi_experiment=wsi_experiment)
    metadata_df = pd.read_csv(paths["label_file"])
    manifest_df = pd.read_csv(Path(config.feature_manifest_csv)) if Path(config.feature_manifest_csv).exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    gene_index = build_case_feature_index(paths["gene_dir"])

    missing_wsi: list[str] = []
    for slide_id in metadata_df["slide_id"].dropna().astype(str):
        wsi_ext = get_dataset_config(config.study).raw.wsi_ext
        slide_stub = slide_id[: -len(wsi_ext)] if slide_id.endswith(wsi_ext) else slide_id
        if not (paths["data_root_dir"] / f"{slide_stub}.pt").exists():
            missing_wsi.append(slide_id)

    missing_gene: list[str] = []
    for case_id in manifest_df.get("case_id", pd.Series(dtype=str)).dropna().astype(str):
        if resolve_case_feature_path(paths["gene_dir"], case_id, case_index=gene_index) is None:
            missing_gene.append(case_id)

    if missing_wsi:
        raise FileNotFoundError(f"{study} missing {len(missing_wsi)} WSI feature files, first: {missing_wsi[:5]}")
    if missing_gene:
        raise FileNotFoundError(f"{study} missing {len(missing_gene)} gene feature files, first: {missing_gene[:5]}")

    return {
        "study": study,
        "metadata_rows": len(metadata_df),
        "manifest_rows": len(manifest_df),
        "wsi_dir": str(paths["data_root_dir"]),
        "gene_dir": str(paths["gene_dir"]),
    }


def summarize_workspace_for_study(
    study: str,
    *,
    gene_embedding: str = DEFAULT_GENE_EXPERIMENT,
    wsi_experiment: str = DEFAULT_WSI_EXPERIMENT,
) -> dict[str, object]:
    paths = infer_standard_paths(study, ".", gene_experiment=gene_embedding, wsi_experiment=wsi_experiment)
    return {
        "study": study,
        "wsi_dir": str(paths["data_root_dir"]),
        "clinic_dir": str(paths["clinic_dir"]),
        "gene_dir": str(paths["gene_dir"]),
        "label_file": str(paths["label_file"]),
        "split_dir": str(paths["split_dir"]),
    }


def run_pipeline_for_study(
    study: str,
    *,
    dry_run: bool = False,
    validate_only: bool = False,
    allow_gdc_api: bool = False,
    gene_embedding: str = DEFAULT_GENE_EXPERIMENT,
    wsi_experiment: str = DEFAULT_WSI_EXPERIMENT,
) -> dict[str, object]:
    result = {
        "clinical": generate_clinical_for_study(study, dry_run=dry_run, validate_only=validate_only),
        "metadata": generate_metadata_for_study(study, dry_run=dry_run, validate_only=validate_only, allow_gdc_api=allow_gdc_api),
        "rna": generate_rna_for_study(study, dry_run=dry_run, validate_only=validate_only, allow_gdc_api=allow_gdc_api),
        "feature_manifest": generate_feature_manifest_for_study(
            study,
            dry_run=dry_run,
            validate_only=validate_only,
            allow_gdc_api=allow_gdc_api,
            gene_embedding=gene_embedding,
        ),
        "splits": generate_splits_for_study(study, dry_run=dry_run, validate_only=validate_only),
    }
    result["workspace"] = (
        summarize_workspace_for_study(study, gene_embedding=gene_embedding, wsi_experiment=wsi_experiment)
        if dry_run
        else validate_workspace_for_study(study, gene_embedding=gene_embedding, wsi_experiment=wsi_experiment)
    )
    return result
