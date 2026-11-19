"""
Collect Table 4（_Cindex_Main 子实验）ablation c-index comparison tables.

The ablation tables are defined in z_temp/Table4/ablation_checklist.md:

    A1_head          head structure (mosaic_surv_single_enum,
                     mosaic_surv_multi), sharing mosaic_surv as reference.
    A2_augmentation  training augmentation (mosaic_surv_noenum,
                     mosaic_surv_nodropout), sharing mosaic_surv as reference.
    B_training  training paradigm (mosaic_surv_twostage, mosaic_surv_frozen)
    C_loss      joint-loss terms (mosaic_surv_kl, mosaic_surv_nojeffreys,
                mosaic_surv_detached)

Every test is compared against the main model `mosaic_surv` (first row of
each table). The whole batch uses the single unified MosaicSurv
hyperparameters from configs/z_exp_gen/mosaic_hparams.sh（由
configs/z_exp_gen/gen_Table4_Abaltion_Test_Cindex_Main.sh 生成，结果目录
不带 run_id 后缀）。A1/A2/B/C 四张单因素对照表都进行。

Default input:
    results/Table4_Abaltion_Test/_Cindex_Main/{study}__* /{model}/test_result.csv
    （每个 study 一个运行目录，如 tcga_brca__cell_norm__uni_v1）

Default outputs:
    results_display/Table4_Abaltion_Test/_Cindex_Main/summary_A1_head_7datasets.csv
    results_display/Table4_Abaltion_Test/_Cindex_Main/summary_A2_augmentation_7datasets.csv
    results_display/Table4_Abaltion_Test/_Cindex_Main/summary_B_training_7datasets.csv
    results_display/Table4_Abaltion_Test/_Cindex_Main/summary_C_loss_7datasets.csv

L1 并行批次：训练结果在 Table4_Abaltion_Test_L1/_Cindex_Main，展示仍镜像
L0 目录结构，写入 results_display_l1/Table4_Abaltion_Test/_Cindex_Main：
    python results_display/scripts/Table4_Abaltion_Test_Cindex_Main.py \
      --group-dir Table4_Abaltion_Test_L1/_Cindex_Main \
      --display-dir Table4_Abaltion_Test/_Cindex_Main \
      --output-root results_display_l1

Every table keeps the classic summary_7datasets.csv row/column layout:
rows are models, columns are the seven mainline datasets
(BRCA/COAD/KIRC/KIRP/LIHC/LUAD/LUSC) plus a mean column. Missing
experiments are written as `-`, so the script can be re-run while training
results are still arriving.

兄弟实验（_Modality_Missing 子实验，仅 A 组、Table2 式缺失模态评测）见
Table4_Abaltion_Test_Modality_Missing.py。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

from model_names import display_name


GROUP_DIR = "Table4_Abaltion_Test/_Cindex_Main"
# 展示目录固定镜像 L0 的 results_display/Table4_Abaltion_Test/_Cindex_Main，
# 与训练结果目录（L1 批次可不同）解耦。
DISPLAY_DIR = "Table4_Abaltion_Test/_Cindex_Main"

MAIN_MODEL = "mosaic_surv"

GROUP_SPECS = {
    "A1_head": [
        "mosaic_surv_single_enum",
        "mosaic_surv_multi",
    ],
    "A2_augmentation": [
        "mosaic_surv_noenum",
        "mosaic_surv_nodropout",
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

# 主线 7 数据集（与 Table1/Table2 主线 cohort 一致）：原 5 数据集 + LUAD/LUSC。
STUDIES = [
    "tcga_brca",
    "tcga_coad",
    "tcga_kirc",
    "tcga_kirp",
    "tcga_lihc",
    "tcga_luad",
    "tcga_lusc",
]

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
    models: set[str],
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Collect `mean ± std` text per study/model for the unified single batch."""
    results_dir = results_root / group_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    study_to_model_text: dict[str, dict[str, str]] = {study: {} for study in STUDIES}
    missing: dict[str, list[str]] = {}

    for study in STUDIES:
        run_dirs = sorted(results_dir.glob(f"{study}__*"))
        if not run_dirs:
            for model in models:
                missing.setdefault(model, []).append(study)
            continue
        if len(run_dirs) > 1:
            print(
                f"[WARN] Multiple run dirs for {study}: "
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
    specs = [MAIN_MODEL] + [model for model in GROUP_SPECS[group_key]]
    rows: list[dict[str, object]] = []
    for model in specs:
        row: dict[str, object] = {"model": MODEL_LABELS.get(model, model)}
        values: list[float] = []
        for study in STUDIES:
            text = study_to_model_text.get(study, {}).get(model, "-")
            row[study] = text
            parsed = parse_mean_from_text(text)
            if parsed is not None:
                values.append(parsed)
        row["mean"] = format_mean(sum(values) / len(values) if values else None)
        rows.append(row)
    return pd.DataFrame(rows, columns=["model", *STUDIES, "mean"])


def write_table(frame: pd.DataFrame, out_dir: Path, group_key: str) -> None:
    out_path = out_dir / f"summary_{group_key}_7datasets.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path.relative_to(project_root_from_script())}")


def _unique_target(target: Path, occupied: set[Path]) -> Path:
    """Return a non-conflicting archive target, suffixes .1/.2/... on collision."""
    candidate = target
    index = 0
    while candidate.exists() or candidate in occupied:
        index += 1
        candidate = target.with_name(f"{target.name}.{index}")
    occupied.add(candidate)
    return candidate


def archive_paths(paths: list[Path], assume_yes: bool = False) -> list[Path]:
    """Archive display outputs to AnysoleWorkspace/_trash/<date>/.
    Print the full plan first; move only after --yes or an interactive y/N."""
    if not paths:
        return []
    project_root = project_root_from_script()
    trash_root = (
        project_root / "AnysoleWorkspace" / "_trash" / datetime.date.today().isoformat()
    )
    occupied: set[Path] = set()
    plan: list[tuple[Path, Path]] = []
    for path in paths:
        try:
            relative = path.relative_to(project_root)
        except ValueError:
            relative = Path(path.name)
        target = _unique_target(trash_root / relative, occupied)
        plan.append((path, target))

    print("[ARCHIVE-PLAN]")
    for source, target in plan:
        print(f"  {source} -> {target}")
    if not assume_yes:
        if not sys.stdin.isatty():
            print("[ARCHIVE-SKIP] non-interactive; pass --yes to archive")
            return []
        answer = input("Archive the paths above? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("[ARCHIVE-SKIP] cancelled")
            return []

    moved: list[Path] = []
    for source, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        print(f"[ARCHIVE] {source} -> {target}")
        moved.append(target)
    return moved


def clean_legacy_outputs(out_root: Path, assume_yes: bool = False) -> None:
    """Archive legacy display outputs at the pre-reorganization locations:
    the flat files and per-run_id folders under the old
    results_display/Table4_Abaltion_Test/ root (superseded by the
    _Cindex_Main / _Modality_Missing sub-experiment layout).
    这里的文件名按历史原名逐字匹配（含 `summary_5datasets.csv`），
    不随主线数据集范围扩大而改名。
    Only called when --clean-legacy is passed explicitly."""
    old_root = out_root.parent
    paths: list[Path] = []
    for name in ("summary.csv", "summary_5datasets.csv"):
        path = old_root / name
        if path.exists():
            paths.append(path)
    for name in ("summary", "t025_alpha_learn", "t028_orig", "t028_alpha_learn"):
        legacy_dir = old_root / name
        if legacy_dir.is_dir():
            paths.append(legacy_dir)
    # 旧版把 A1/A2 合并成一张 A_readout 表；已拆成两张单因素表后归档该派生文件。
    legacy_a = out_root / "summary_A_readout_7datasets.csv"
    if legacy_a.exists():
        paths.append(legacy_a)
    archive_paths(paths, assume_yes=assume_yes)


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
        help="Grouped results directory name (training tree)",
    )
    parser.add_argument(
        "--display-dir",
        default=DISPLAY_DIR,
        help=(
            "Display subdirectory under --output-root; default mirrors the L0 "
            "layout Table4_Abaltion_Test/_Cindex_Main"
        ),
    )
    parser.add_argument(
        "--clean-legacy",
        action="store_true",
        help=(
            "Archive the legacy display outputs (flat summary files and the "
            "per-run_id folders of the superseded three-batch layout) to "
            "AnysoleWorkspace/_trash/<date>/ before writing"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the --clean-legacy archive plan without an interactive prompt",
    )
    args = parser.parse_args()

    out_root = args.output_root / args.display_dir
    if args.clean_legacy:
        clean_legacy_outputs(out_root, assume_yes=args.yes)

    all_models = {MAIN_MODEL} | {
        model for models in GROUP_SPECS.values() for model in models
    }

    study_to_model_text, missing = collect_run(
        args.results_root,
        group_dir=args.group_dir,
        models=all_models,
    )
    for group_key in GROUP_SPECS:
        frame = build_table_frame(study_to_model_text, group_key)
        write_table(frame, out_root, group_key)

    total_cells = len(STUDIES) * len(all_models)
    complete_cells = total_cells - sum(len(studies) for studies in missing.values())
    print(f"[RUN] {complete_cells}/{total_cells} cells complete")
    for model, studies in sorted(missing.items()):
        print(f"[MISSING] {model}: {', '.join(studies)}")

    print("Done.")


if __name__ == "__main__":
    main()
