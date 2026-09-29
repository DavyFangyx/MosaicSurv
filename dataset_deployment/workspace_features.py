from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TCGA_CASE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def extract_case_id_from_feature_name(file_name: str) -> str | None:
    stem = Path(file_name).stem
    match = TCGA_CASE_RE.search(stem)
    if not match:
        return None
    return match.group(1).upper()


def build_case_feature_index(feature_dir: str | Path) -> dict[str, Path]:
    root = Path(feature_dir)
    index: dict[str, Path] = {}
    if not root.exists():
        return index

    for path in sorted(root.glob("*.pt")):
        stem = path.stem
        if stem.upper().startswith("TCGA-") and stem.count("-") >= 2:
            exact_case = stem[:12].upper()
            index.setdefault(exact_case, path)

        case_id = extract_case_id_from_feature_name(path.name)
        if case_id:
            index.setdefault(case_id, path)

    return index


def resolve_case_feature_path(
    feature_dir: str | Path,
    case_id: str,
    *,
    case_index: dict[str, Path] | None = None,
) -> Path | None:
    root = Path(feature_dir)
    case_id = str(case_id).upper()

    direct = root / f"{case_id}.pt"
    if direct.exists():
        return direct

    index = case_index if case_index is not None else build_case_feature_index(root)
    return index.get(case_id)


def load_patient_allowlist(patient_table_csv: str | Path) -> set[str]:
    path = Path(patient_table_csv)
    if not path.exists():
        raise FileNotFoundError(f'Missing patient table: {path}')
    table = pd.read_csv(path)
    if 'submitter_id' not in table.columns:
        raise ValueError(f'{path} missing submitter_id column')
    return set(table['submitter_id'].dropna().astype(str).str.upper())


def feature_belongs_to_allowlist(file_name: str, allowlist: set[str] | None) -> bool:
    if not allowlist:
        return True
    case_id = extract_case_id_from_feature_name(file_name)
    return bool(case_id and case_id in allowlist)
