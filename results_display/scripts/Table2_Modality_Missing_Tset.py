"""
Collect Table 2 missing-modality c-index summaries.

Default input:
    results/Table2_Baselines/{study}__L4__gene_raw__uni_v1/{model}/eval_subsets/{subset}/test_result.csv

Default outputs:
    results_display/Table2_Modality_Missing_Tset/{study}_model_summary.csv
    results_display/Table2_Modality_Missing_Tset/summary.csv

Models are discovered by scanning each study directory under
results/Table2_Baselines/, rather than a hardcoded model list.

The summary table is transposed relative to the per-study tables: columns
are studies, and rows are the six missing-type scenarios (C, P, G, PC, CG,
PG; the full-modality PCG subset stays in the per-study tables only) with
one row per scanned model under each scenario. Written as a two-level row
index: first column scenarios, second column models. Within each
dataset-and-subset experiment, the top-3 models are marked as (1), (2),
and (3) on the cell values.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd


GROUP_DIR = "Table2_Baselines"
DISPLAY_DIR = "Table2_Modality_Missing_Tset"
DEFAULT_RUN_SUFFIX = "L4__gene_raw__uni_v1"

STUDY_SPECS = [
    ("BRCA", "tcga_brca"),
    ("COAD", "tcga_coad"),
    ("KIRC", "tcga_kirc"),
    ("KIRP", "tcga_kirp"),
    ("LIHC", "tcga_lihc"),
]

# Preferred listing order when these directories exist. Any other scanned
# model directories are appended alphabetically.
PREFERRED_MODEL_ORDER = [
    "survtri_poe_vae__B",
    "modality_concat__resampler__zero",
    "modality_concat__resampler__mean",
    "mvae_poe",
    "mopoe",
    "hgcn",
]

PREFERRED_MODEL_RANK = {name: index for index, name in enumerate(PREFERRED_MODEL_ORDER)}

SUBSETS = ["P", "C", "G", "PC", "PG", "CG", "PCG"]

# Summary-table big-row order (Clinical first) and coverage: only the six
# missing-type scenarios, PCG is left to the per-study tables.
SUMMARY_SUBSETS = ["C", "P", "G", "PC", "CG", "PG"]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cindex_stats(csv_path: Path) -> tuple[float, float] | None:
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            values: list[float] = []
            for row in reader:
                raw = row.get("test_cindex", "")
                if raw in ("", None):
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
    except Exception as exc:
        print(f"[WARN] Cannot read {csv_path}: {exc}")
        return None

    if not values:
        print(f"[WARN] Empty test_cindex in {csv_path}")
        return None

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return mean, std


def format_mean_std(stats: tuple[float, float] | None) -> str:
    if stats is None:
        return "-"
    mean, std = stats
    return f"{mean:.4f} ± {std:.4f}"


def parse_mean_from_text(value: str) -> float | None:
    text = str(value).strip()
    if text in {"", "-", "nan"}:
        return None
    if "±" in text:
        text = text.split("±", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def format_mean(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def append_rank_suffix(value: str, rank: int) -> str:
    text = str(value).strip()
    if text in {"", "-"}:
        return text
    if text.endswith(")") and "(" in text:
        return text
    return f"{text}({rank})"


def model_label(dir_name: str) -> str:
    return dir_name.replace("__resampler__", "_").replace("__", "_")


def is_model_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "eval_subsets").is_dir() or (path / "test_result.csv").is_file()
    )


def model_sort_key(dir_name: str) -> tuple[int, str]:
    return (PREFERRED_MODEL_RANK.get(dir_name, len(PREFERRED_MODEL_RANK)), dir_name)


def study_run_dir(results_root: Path, group_dir: str, study: str, run_suffix: str) -> Path:
    return results_root / group_dir / f"{study}__{run_suffix}"


def discover_model_dir_names(
    results_root: Path,
    *,
    group_dir: str,
    studies: list[str],
    run_suffix: str,
) -> list[str]:
    found: set[str] = set()
    for study in studies:
        run_dir = study_run_dir(results_root, group_dir, study, run_suffix)
        if not run_dir.is_dir():
            continue
        for child in run_dir.iterdir():
            if is_model_dir(child):
                found.add(child.name)
    return sorted(found, key=model_sort_key)


def row_mean(row: dict[str, str]) -> str:
    values = [parsed for subset in SUBSETS if (parsed := parse_mean_from_text(row.get(subset, "-"))) is not None]
    return format_mean(sum(values) / len(values) if values else None)


def subset_csv_path(model_dir: Path, subset: str) -> Path | None:
    eval_path = model_dir / "eval_subsets" / subset / "test_result.csv"
    if eval_path.is_file():
        return eval_path
    if subset == "PCG":
        top_path = model_dir / "test_result.csv"
        if top_path.is_file():
            return top_path
    return None


def collect_study_table(
    results_root: Path,
    *,
    group_dir: str,
    study: str,
    run_suffix: str,
    model_dir_names: list[str],
) -> pd.DataFrame:
    run_dir = study_run_dir(results_root, group_dir, study, run_suffix)
    rows: list[dict[str, str]] = []

    if not run_dir.is_dir():
        print(f"[WARN] Missing study directory: {run_dir}")
        for dir_name in model_dir_names:
            row = {"model": model_label(dir_name)}
            row.update({subset: "-" for subset in SUBSETS})
            row["mean"] = "-"
            rows.append(row)
        return pd.DataFrame(rows, columns=["model", *SUBSETS, "mean"])

    print(f"[STUDY] {study} | {run_dir.name}")
    for model_dir_name in model_dir_names:
        model_dir = run_dir / model_dir_name
        label = model_label(model_dir_name)
        row: dict[str, str] = {"model": label}
        if not model_dir.is_dir():
            print(f"[WARN] Missing model directory: {model_dir}")
            row.update({subset: "-" for subset in SUBSETS})
            row["mean"] = "-"
            rows.append(row)
            continue

        for subset in SUBSETS:
            csv_path = subset_csv_path(model_dir, subset)
            if csv_path is None:
                print(f"[WARN] Missing {study} {label} {subset}")
                row[subset] = "-"
                continue
            row[subset] = format_mean_std(load_cindex_stats(csv_path))
        row["mean"] = row_mean(row)
        rows.append(row)

    return pd.DataFrame(rows, columns=["model", *SUBSETS, "mean"])


def write_study_table(df: pd.DataFrame, output_root: Path, study: str) -> Path:
    out_dir = output_root / DISPLAY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{study}_model_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path.relative_to(output_root)}")
    return out_path


def annotate_subset_top_ranks(
    subset_rows: list[list[str]],
    model_labels: list[str],
    top_k: int = 3,
) -> list[list[str]]:
    if not subset_rows:
        return subset_rows

    annotated = [list(row) for row in subset_rows]
    n_studies = len(subset_rows[0])
    for col in range(n_studies):
        ranked_values: list[tuple[float, str, int]] = []
        for row_idx, row in enumerate(subset_rows):
            mean = parse_mean_from_text(row[col])
            if mean is None:
                continue
            ranked_values.append((mean, model_labels[row_idx], row_idx))

        ranked_values.sort(key=lambda item: (-item[0], item[1]))
        for rank, (_, _, row_idx) in enumerate(ranked_values, start=1):
            if rank > top_k:
                break
            annotated[row_idx][col] = append_rank_suffix(annotated[row_idx][col], rank)
    return annotated


def build_summary_table(
    study_dfs: dict[str, pd.DataFrame],
    model_labels: list[str],
) -> pd.DataFrame:
    studies = [study for _, study in STUDY_SPECS]
    index = pd.MultiIndex.from_tuples(
        [(subset, model) for subset in SUMMARY_SUBSETS for model in model_labels],
        names=["subset", "model"],
    )
    rows: list[list[str]] = []
    study_model_rows = {
        study: {str(row["model"]): row for _, row in study_dfs[study].iterrows()}
        for study in studies
    }
    for subset in SUMMARY_SUBSETS:
        subset_rows: list[list[str]] = []
        for model in model_labels:
            row = [
                str(study_model_rows.get(study, {}).get(model, {}).get(subset, "-"))
                for study in studies
            ]
            subset_rows.append(row)
        rows.extend(annotate_subset_top_ranks(subset_rows, model_labels))
    return pd.DataFrame(rows, index=index, columns=studies)


def write_summary_table(summary_df: pd.DataFrame, output_root: Path) -> Path:
    out_dir = output_root / DISPLAY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.csv"
    summary_df.to_csv(out_path, index=True, header=True)
    print(f"[WRITE] {out_path.relative_to(output_root)}")
    return out_path


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Collect Table 2 missing-modality c-index summaries")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=project_root / "results",
        help="Root results directory",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=project_root / "results_display",
        help="Root output directory",
    )
    parser.add_argument(
        "--group-dir",
        default=GROUP_DIR,
        help="Grouped results directory name",
    )
    parser.add_argument(
        "--run-suffix",
        default=DEFAULT_RUN_SUFFIX,
        help="Official Table2 run-name suffix after study, e.g. L4__gene_raw__uni_v1",
    )
    args = parser.parse_args()

    studies = [study for _, study in STUDY_SPECS]
    model_dir_names = discover_model_dir_names(
        args.results_root,
        group_dir=args.group_dir,
        studies=studies,
        run_suffix=args.run_suffix,
    )
    if not model_dir_names:
        print(
            f"[WARN] No model directories found under "
            f"{args.results_root / args.group_dir}"
        )
    else:
        print("[MODELS] " + ", ".join(model_dir_names))

    study_dfs: dict[str, pd.DataFrame] = {}
    for _, study in STUDY_SPECS:
        study_df = collect_study_table(
            args.results_root,
            group_dir=args.group_dir,
            study=study,
            run_suffix=args.run_suffix,
            model_dir_names=model_dir_names,
        )
        study_dfs[study] = study_df
        write_study_table(study_df, args.output_root, study)

    summary_df = build_summary_table(
        study_dfs,
        [model_label(dir_name) for dir_name in model_dir_names],
    )
    write_summary_table(summary_df, args.output_root)

    print("Done.")


if __name__ == "__main__":
    main()
