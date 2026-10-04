"""
Collect Table 4 ablation c-index comparison tables.

The three ablation tests are defined in z_temp/Table4/ablation_checklist.md:

    A_readout   readout layer x training scheme (mosaic_surv_single,
                mosaic_surv_single_enum, mosaic_surv_noenum, mosaic_surv,
                mosaic_surv_multi)
    B_training  training paradigm (mosaic_surv_twostage, mosaic_surv_frozen)
    C_loss      joint-loss terms (mosaic_surv_kl, mosaic_surv_nojeffreys,
                mosaic_surv_detached; 原 c_film_kl / c_film_beta0 / c_film_surv0)

Every test is compared against the main model `mosaic_surv` (first row of
each table). The ablation batch was trained with three hyperparameter
settings — the rows with `T4` enabled in the manifest
configs/z_exp_gen/Cfilm_Hparam_Eval/cfilm_table1_hparams.csv — so one
output folder is written per run_id and each folder holds one c-index table
per test.

Default input:
    results/Table4_Abaltion_Test/{study}__*__{run_id}/{model}/test_result.csv

Default outputs:
    results_display/Table4_Abaltion_Test/{run_id}/summary_A_readout_5datasets.csv
    results_display/Table4_Abaltion_Test/{run_id}/summary_B_training_5datasets.csv
    results_display/Table4_Abaltion_Test/{run_id}/summary_C_loss_5datasets.csv

Every table keeps the classic summary_5datasets.csv row/column layout:
rows are models, columns are the five datasets plus a mean column. Missing
experiments are written as `-`, so the script can be re-run while training
results are still arriving.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import pandas as pd

from model_names import display_name


GROUP_DIR = "Table4_Abaltion_Test"

MAIN_MODEL = "mosaic_surv"
MAIN_MODEL_TYPE = "Main"

GROUP_SPECS = {
    "A_readout": [
        "mosaic_surv_single",
        "mosaic_surv_single_enum",
        "mosaic_surv_noenum",
        "mosaic_surv_multi",
    ],
    "B_training": [
        "mosaic_surv_twostage",
        "mosaic_surv_frozen",
    ],
    "C_loss": [
        "mosaic_surv_kl",
        "mosaic_surv_nojeffreys",
        "mosaic_surv_detached",
    ],
}

STUDIES = ["tcga_brca", "tcga_coad", "tcga_kirc", "tcga_kirp", "tcga_lihc"]

# 行标签用集中映射的论文正式名（Table4 汇总 CSV）
MODEL_LABELS = {
    model: display_name(model)
    for model in {
        MAIN_MODEL,
        *(model for models in GROUP_SPECS.values() for model in models),
    }
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def default_hparams_path() -> Path:
    return (
        project_root_from_script()
        / "configs/z_exp_gen/Cfilm_Hparam_Eval/cfilm_table1_hparams.csv"
    )


def read_t4_run_ids(path: Path) -> list[str]:
    """run_ids of the rows with `T4` enabled in the hparam manifest."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def is_enabled(value: str) -> bool:
        return (value or "").strip().lower() in {"t4", "1", "true", "yes", "on"}

    run_ids = [row.get("run_id", "").strip() for row in rows if is_enabled(row.get("T4") or "")]
    if not run_ids:
        raise SystemExit(f"No T4-enabled rows in {path}")
    return run_ids


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


def resolve_model_dir(run_dir: Path, model: str) -> Path | None:
    candidate = run_dir / model
    if (candidate / "test_result.csv").is_file():
        return candidate
    if candidate.is_dir():
        return candidate
    return None


def resolve_result_csv(run_dir: Path, model: str) -> Path | None:
    """Locate test_result.csv for one study/model in the Table4 batch dir."""
    model_dir = resolve_model_dir(run_dir, model)
    if model_dir is None:
        return None
    csv_path = model_dir / "test_result.csv"
    return csv_path if csv_path.is_file() else None


def collect_run(
    results_root: Path,
    *,
    group_dir: str,
    run_id: str,
    models: set[str],
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Collect `mean ± std` text per study/model for one run_id folder."""
    results_dir = results_root / group_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    study_to_model_text: dict[str, dict[str, str]] = {study: {} for study in STUDIES}
    missing: dict[str, list[str]] = {}

    for study in STUDIES:
        run_dirs = sorted(results_dir.glob(f"{study}__*__{run_id}"))
        if not run_dirs:
            for model in models:
                missing.setdefault(model, []).append(study)
            continue
        if len(run_dirs) > 1:
            print(
                f"[WARN] Multiple run dirs for {study}/{run_id}: "
                f"{[path.name for path in run_dirs]}; using {run_dirs[0].name}"
            )
        run_dir = run_dirs[0]
        for model in models:
            result_csv = resolve_result_csv(run_dir, model)
            stats = load_cindex_stats(result_csv) if result_csv is not None else None
            if stats is None:
                missing.setdefault(model, []).append(study)
                continue
            mean, std = stats
            study_to_model_text[study][model] = f"{mean:.4f} ± {std:.4f}"

    return study_to_model_text, missing


def build_table_frame(
    study_to_model_text: dict[str, dict[str, str]],
    group_key: str,
) -> pd.DataFrame:
    specs = [(MAIN_MODEL_TYPE, MAIN_MODEL)] + [
        (group_key, model) for model in GROUP_SPECS[group_key]
    ]
    rows: list[dict[str, object]] = []
    for model_type, model in specs:
        row: dict[str, object] = {"model_type": model_type, "model": MODEL_LABELS.get(model, model)}
        values: list[float] = []
        for study in STUDIES:
            text = study_to_model_text.get(study, {}).get(model, "-")
            row[study] = text
            parsed = parse_mean_from_text(text)
            if parsed is not None:
                values.append(parsed)
        row["mean"] = format_mean(sum(values) / len(values) if values else None)
        rows.append(row)
    return pd.DataFrame(rows, columns=["model_type", "model", *STUDIES, "mean"])


def write_table(frame: pd.DataFrame, out_dir: Path, group_key: str) -> None:
    out_path = out_dir / f"summary_{group_key}_5datasets.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path.relative_to(project_root_from_script())}")


def clean_legacy_outputs(out_root: Path) -> None:
    """Remove the old flat outputs (single summary.csv/summary_5datasets.csv
    plus per-study summary/), superseded by the per-run_id folder layout.
    Only called when --clean-legacy is passed explicitly."""
    for name in ("summary.csv", "summary_5datasets.csv"):
        path = out_root / name
        if path.exists():
            path.unlink()
            print(f"[CLEAN] {path.relative_to(project_root_from_script())}")
    legacy_dir = out_root / "summary"
    if legacy_dir.is_dir():
        shutil.rmtree(legacy_dir)
        print(f"[CLEAN] {legacy_dir.relative_to(project_root_from_script())}")


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(
        description="Collect Table 4 ablation c-index comparison tables"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "results",
        help="Root results directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "results_display",
        help="Root output directory",
    )
    parser.add_argument(
        "--group-dir",
        default=GROUP_DIR,
        help="Grouped results/output directory name",
    )
    parser.add_argument(
        "--hparams",
        type=Path,
        default=default_hparams_path(),
        help=(
            "hparam manifest CSV; T4-enabled rows select the run_id folders "
            "(default: configs/z_exp_gen/Cfilm_Hparam_Eval/cfilm_table1_hparams.csv)"
        ),
    )
    parser.add_argument(
        "--run-ids",
        dest="run_ids",
        action="append",
        help=(
            "run_id folder(s) to collect; repeat this option to select "
            "multiple run_ids (default: T4-enabled rows of --hparams)"
        ),
    )
    parser.add_argument(
        "--clean-legacy",
        action="store_true",
        help=(
            "Delete the legacy flat outputs (summary.csv, "
            "summary_5datasets.csv, summary/) before writing"
        ),
    )
    args = parser.parse_args()

    run_ids = args.run_ids or read_t4_run_ids(args.hparams)

    out_root = args.output_root / args.group_dir
    if args.clean_legacy:
        clean_legacy_outputs(out_root)

    all_models = {MAIN_MODEL} | {
        model for models in GROUP_SPECS.values() for model in models
    }

    for run_id in run_ids:
        study_to_model_text, missing = collect_run(
            args.results_root,
            group_dir=args.group_dir,
            run_id=run_id,
            models=all_models,
        )
        for group_key in GROUP_SPECS:
            frame = build_table_frame(study_to_model_text, group_key)
            write_table(frame, out_root / run_id, group_key)

        total_cells = len(STUDIES) * len(all_models)
        complete_cells = total_cells - sum(len(studies) for studies in missing.values())
        print(f"[RUN] {run_id}: {complete_cells}/{total_cells} cells complete")
        for model, studies in sorted(missing.items()):
            print(f"[MISSING] {run_id}/{model}: {', '.join(studies)}")

    print("Done.")


if __name__ == "__main__":
    main()
