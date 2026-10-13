import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from dataset_deployment.registry import get_dataset_config, infer_standard_paths
from dataset_deployment.workspace_features import build_case_feature_index
from dataset_deployment.scripts.pipeline import resolve_studies


def save_split_csv(train_ids: list[str], val_ids: list[str], output_path: Path) -> None:
    max_len = max(len(train_ids), len(val_ids))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "train", "val", "test"])
        for index in range(max_len):
            train_value = train_ids[index] if index < len(train_ids) else ""
            val_value = val_ids[index] if index < len(val_ids) else ""
            writer.writerow([index, train_value, val_value, val_value])


def load_case_ids(metadata_csv: Path) -> list[str]:
    df = pd.read_csv(metadata_csv)
    if "case_id" not in df.columns:
        raise ValueError(f"{metadata_csv} missing case_id column")
    return sorted(df["case_id"].dropna().astype(str).drop_duplicates().tolist())


def build_folds(case_ids: list[str], n_splits: int, seed: int) -> list[list[str]]:
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    folds = [[] for _ in range(n_splits)]
    for index, case_id in enumerate(shuffled):
        folds[index % n_splits].append(case_id)
    return folds


def build_stratified_folds(metadata_csv: Path, n_splits: int, seed: int) -> list[list[str]]:
    df = pd.read_csv(metadata_csv)
    case_df = df.drop_duplicates("case_id").copy()
    if "censorship" not in case_df.columns:
        return build_folds(sorted(case_df["case_id"].astype(str).tolist()), n_splits=n_splits, seed=seed)

    event_cases = sorted(case_df.loc[(1 - case_df["censorship"]).astype(int) == 1, "case_id"].astype(str).tolist())
    censored_cases = sorted(case_df.loc[(1 - case_df["censorship"]).astype(int) == 0, "case_id"].astype(str).tolist())

    rng = random.Random(seed)
    rng.shuffle(event_cases)
    rng.shuffle(censored_cases)

    folds = [[] for _ in range(n_splits)]
    for index, case_id in enumerate(event_cases):
        folds[index % n_splits].append(case_id)
    for index, case_id in enumerate(censored_cases):
        folds[index % n_splits].append(case_id)
    return folds


def load_case_level_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if "case_id" not in df.columns:
        raise ValueError(f"{metadata_csv} missing case_id column")
    return df.drop_duplicates("case_id").copy()


def infer_patient_stats_path(config) -> Path:
    return Path(config.patient_table_csv)


def collect_case_set_from_dirs(root: Path) -> tuple[list[str], dict[str, set[str]], set[str]]:
    if not root.exists():
        return [], {}, set()
    dir_names = sorted([path.name for path in root.iterdir() if path.is_dir()])
    case_sets: dict[str, set[str]] = {}
    for name in dir_names:
        case_sets[name] = set(build_case_feature_index(root / name).keys())
    if not case_sets:
        return dir_names, case_sets, set()
    intersection = set.intersection(*case_sets.values())
    return dir_names, case_sets, intersection


def collect_wsi_ready_cases(metadata_df: pd.DataFrame, data_root_dir: Path) -> set[str]:
    if not data_root_dir.exists():
        return set()
    ready_cases: set[str] = set()
    for case_id, group in metadata_df.groupby("case_id", sort=False):
        all_present = True
        for slide_id in group["slide_id"].dropna().astype(str):
            slide_stub = slide_id[:-4] if slide_id.endswith(".svs") else slide_id
            if not (data_root_dir / f"{slide_stub}.pt").exists():
                all_present = False
                break
        if all_present:
            ready_cases.add(str(case_id))
    return ready_cases


def build_split_eligibility(study: str) -> tuple[pd.DataFrame, dict[str, object]]:
    config = get_dataset_config(study)
    paths = infer_standard_paths(study, PROJ_ROOT)
    patient_stats_path = infer_patient_stats_path(config)
    patient_df = pd.read_csv(patient_stats_path)
    all_cases = sorted(patient_df["submitter_id"].dropna().astype(str).unique().tolist())

    metadata_df = pd.read_csv(paths["label_file"])
    case_metadata_df = load_case_level_metadata(paths["label_file"])
    metadata_cases = set(case_metadata_df["case_id"].astype(str))
    wsi_cases = collect_wsi_ready_cases(metadata_df, paths["data_root_dir"])

    clinic_dir_names, clinic_case_sets, clinic_cases = collect_case_set_from_dirs(paths["workspace_root"] / "C")
    gene_dir_names, gene_case_sets, gene_cases = collect_case_set_from_dirs(paths["workspace_root"] / "G")

    rna_csv_path = paths["omics_dir"] / "rna_clean.csv"
    rna_cases = set()
    if rna_csv_path.exists():
        rna_df = pd.read_csv(rna_csv_path, index_col=0)
        rna_cases = set(rna_df.index.astype(str))

    eligible_cases = set(all_cases) & metadata_cases & wsi_cases & clinic_cases & gene_cases & rna_cases

    censorship_map = {}
    if "censorship" in case_metadata_df.columns:
        censorship_map = dict(zip(case_metadata_df["case_id"].astype(str), case_metadata_df["censorship"]))
    survival_map = {}
    if "survival_months" in case_metadata_df.columns:
        survival_map = dict(zip(case_metadata_df["case_id"].astype(str), case_metadata_df["survival_months"]))

    rows = []
    for case_id in all_cases:
        missing_reasons = []
        if case_id not in metadata_cases:
            missing_reasons.append("missing_metadata")
        if case_id not in wsi_cases:
            missing_reasons.append("missing_wsi")
        if case_id not in clinic_cases:
            missing_reasons.append("missing_clinic")
        if case_id not in gene_cases:
            missing_reasons.append("missing_gene_fm")
        if case_id not in rna_cases:
            missing_reasons.append("missing_rna_csv")

        rows.append(
            {
                "study": study,
                "case_id": case_id,
                "in_patient_stats": True,
                "in_metadata": case_id in metadata_cases,
                "wsi_ready": case_id in wsi_cases,
                "clinic_ready": case_id in clinic_cases,
                "gene_fm_ready": case_id in gene_cases,
                "rna_csv_ready": case_id in rna_cases,
                "eligible_for_split": case_id in eligible_cases,
                "missing_reasons": ";".join(missing_reasons) if missing_reasons else "",
                "censorship": censorship_map.get(case_id, ""),
                "survival_months": survival_map.get(case_id, ""),
            }
        )

    eligibility_df = pd.DataFrame(rows)
    summary = {
        "study": study,
        "patient_total": len(all_cases),
        "metadata_cases": len(metadata_cases),
        "wsi_ready_cases": len(wsi_cases),
        "clinic_ready_cases": len(clinic_cases),
        "gene_fm_ready_cases": len(gene_cases),
        "rna_csv_ready_cases": len(rna_cases),
        "eligible_cases": len(eligible_cases),
        "dropped_cases": len(all_cases) - len(eligible_cases),
        "clinic_dirs": ";".join(clinic_dir_names),
        "gene_dirs": ";".join(gene_dir_names),
        "eligible_event_cases": int(sum((1 - eligibility_df.loc[eligibility_df["eligible_for_split"], "censorship"].astype(float)).astype(int))) if len(eligible_cases) else 0,
        "eligible_censored_cases": int(sum(eligibility_df.loc[eligibility_df["eligible_for_split"], "censorship"].astype(float))) if len(eligible_cases) else 0,
    }
    return eligibility_df, summary


def write_split_summary(summary_rows: list[dict[str, object]], *, merge_existing: bool = False) -> None:
    summary_path = PROJ_ROOT / "splits" / "5foldcv" / "汇总.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if merge_existing and summary_path.exists():
        processed = {str(row.get("study")) for row in summary_rows}
        existing = pd.read_csv(summary_path)
        kept = existing[~existing["study"].astype(str).isin(processed)]
        summary_rows = kept.to_dict("records") + summary_rows
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def cleanup_existing_split_files(output_dir: Path, n_splits: int) -> None:
    for fold in range(n_splits):
        (output_dir / f"splits_{fold}.csv").unlink(missing_ok=True)
    (output_dir / "_tmp_split_metadata.csv").unlink(missing_ok=True)


def generate_splits_for_study(
    study: str,
    *,
    dry_run: bool = False,
    validate_only: bool = False,
    n_splits: int = 5,
    seed: int = 1,
) -> dict[str, object]:
    config = get_dataset_config(study)
    metadata_csv = Path(config.metadata_csv)
    output_dir = Path(config.split_dir)
    eligibility_df, summary = build_split_eligibility(study)
    output_dir.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_dir / "split_eligibility.csv"
    eligibility_df.to_csv(eligibility_path, index=False)
    cohort_metadata_path = output_dir / "split_cohort_metadata.csv"
    status = "ok"

    if dry_run and not metadata_csv.exists():
        return {
            "study": study,
            "folds": n_splits,
            "cases": None,
            "output": str(output_dir),
            "metadata_exists": False,
            "mode": "dry-run",
            "eligibility_csv": str(eligibility_path),
            "status": "missing_metadata",
            **summary,
        }

    if validate_only:
        for fold in range(n_splits):
            split_path = output_dir / f"splits_{fold}.csv"
            df = pd.read_csv(split_path)
            for column in ("train", "val", "test"):
                if column not in df.columns:
                    raise ValueError(f"{split_path} missing column {column}")
        return {"study": study, "folds": n_splits, "output": str(output_dir), "mode": "validate", "eligibility_csv": str(eligibility_path), "status": status, **summary}

    case_ids = sorted(eligibility_df.loc[eligibility_df["eligible_for_split"], "case_id"].astype(str).tolist())
    if len(case_ids) < n_splits:
        cleanup_existing_split_files(output_dir, n_splits)
        if metadata_csv.exists():
            full_metadata_df = pd.read_csv(metadata_csv)
            empty_cohort_df = full_metadata_df.iloc[0:0].copy()
            with cohort_metadata_path.open("w", encoding="utf-8", newline="") as handle:
                empty_cohort_df.to_csv(handle, index=False)
        else:
            with cohort_metadata_path.open("w", encoding="utf-8", newline="") as handle:
                pd.DataFrame(columns=["case_id"]).to_csv(handle, index=False)
        status = "insufficient_eligible_cases"
        return {
            "study": study,
            "folds": n_splits,
            "cases": len(case_ids),
            "output": str(output_dir),
            "eligibility_csv": str(eligibility_path),
            "cohort_metadata_csv": str(cohort_metadata_path),
            "status": status,
            **summary,
        }

    full_metadata_df = pd.read_csv(metadata_csv)
    filtered_metadata_df = full_metadata_df[full_metadata_df["case_id"].isin(case_ids)].reset_index(drop=True)
    filtered_metadata_df.to_csv(cohort_metadata_path, index=False)
    tmp_metadata_csv = output_dir / "_tmp_split_metadata.csv"
    filtered_metadata_df.to_csv(tmp_metadata_csv, index=False)
    folds = build_stratified_folds(tmp_metadata_csv, n_splits=n_splits, seed=seed)
    if dry_run:
        tmp_metadata_csv.unlink(missing_ok=True)
        return {
            "study": study,
            "folds": n_splits,
            "cases": len(case_ids),
            "output": str(output_dir),
            "eligibility_csv": str(eligibility_path),
            "cohort_metadata_csv": str(cohort_metadata_path),
            "status": status,
            **summary,
        }

    for fold, val_ids in enumerate(folds):
        train_ids = [case_id for idx, fold_cases in enumerate(folds) if idx != fold for case_id in fold_cases]
        save_split_csv(train_ids, val_ids, output_dir / f"splits_{fold}.csv")
    cleanup_existing_split_files(output_dir, 0)

    return {
        "study": study,
        "folds": n_splits,
        "cases": len(case_ids),
        "output": str(output_dir),
        "eligibility_csv": str(eligibility_path),
        "cohort_metadata_csv": str(cohort_metadata_path),
        "status": status,
        **summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 5-fold split CSVs for one or more SurvPGC datasets.")
    parser.add_argument("--study", type=str, default=None)
    parser.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for study in resolve_studies(args.study, args.run_all):
        try:
            result = generate_splits_for_study(
                study,
                dry_run=args.dry_run,
                validate_only=args.validate_only,
                n_splits=args.n_splits,
                seed=args.seed,
            )
        except Exception as exc:
            result = {
                "study": study,
                "folds": args.n_splits,
                "cases": None,
                "output": str(Path(get_dataset_config(study).split_dir)),
                "status": f"failed: {exc}",
            }
        summaries.append({key: value for key, value in result.items() if key not in {"eligibility_csv", "cohort_metadata_csv", "output"}})
        print(result)
    if summaries:
        write_split_summary(summaries, merge_existing=bool(args.study))


if __name__ == "__main__":
    main()
