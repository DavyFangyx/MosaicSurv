"""
Table1 统一生成入口：运行一次，产出全部 Table1 展示文件。

Pipeline（实验目录 results/Table1_Cindex_Main/<test-dir>/，
默认 L0Test；--test-dir 指定其他批次目录）：

  1. Cox Breslow 前向缓存：为本目录的 cox 类模型（clinic_cox 基线 +
     mosaic_surv 主模型，以及走第三方训练器的 hgcn——Table1_HGCN_Forward）
     补算 fold 级 IBS/AUC（需要 GPU，--skip-cox 可跳过），
     写入本目录专属缓存
     results_display/.cache_table1/cox_ibs_auc_folds__<test_dir>.csv，
     幂等：已缓存的 (study, model, fold) 不会重算。

  2. 论文五表（Table1a_Cindex_Matrix / Table1b_MultiMetric_Summary /
     AppendixS1_IBS_Matrix / AppendixS2_AUC2y_Matrix / AppendixS3_AUC5y_Matrix）：
     官方口径，5 cohort；Ours 行与本目录树内的官方 mosaic_surv 结果同源。
     hgcn 的 IBS/AUC 单元格由第 1 步的前向缓存补齐（其 runner 不落盘
     标准 split_N_results.pkl，离线 nll 链路读不到，故走 Breslow 前向）。

  3. summary 家族（summary.csv / summary_5datasets.csv / baselines+ours
     的 per-study summary）：同一批结果、同一次运行，8 cohort 全视图，
     供 FigC/FigD 与 hparam-eval 显示沿用。

用法（项目根目录，SurvPGC conda env）：
  python results_display/scripts/Table1_Generate_All.py                # 默认 --test-dir L0Test
  python results_display/scripts/Table1_Generate_All.py --test-dir L0Test
  python results_display/scripts/Table1_Generate_All.py --test-dir L0Test --skip-cox
  python results_display/scripts/Table1_Generate_All.py --test-dir L0Test --gpu 4
  旧批次目录（L0Test_BeforeTune 等）如需重算，显式 --test-dir 指定即可。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from Table1_Cindex_Main import (  # noqa: E402
    GROUP_CONFIG,
    STUDY_SPECS,
    collect_experiment as collect_summary_family,
    infer_layer_dir,
    project_root_from_script,
    resolve_group_study_dir,
)
from Table1_CoxBreslow_Forward import (  # noqa: E402
    load_effective_config,
    process_model_dir,
)
from Table1_Paper_Tables import build_tables  # noqa: E402

LEGACY_COX_CACHE_CSV = ".cache_table1/cox_ibs_auc_folds.csv"
COX_BASELINE_MODEL_DIRS = ("clinic_cox",)
OURS_MODEL_DIRS = ("mosaic_surv",)
HGCN_MODEL_DIRS = ("hgcn",)
ALL_FOLDS = (0, 1, 2, 3, 4)
# 旧批次目录已不参与默认口径；如需对旧批次目录重算，请显式 --test-dir 指定。
DEFAULT_LEGACY_COX_DIRS = ""


def cox_cache_path(project_root: Path, test_dir: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", test_dir)
    return (
        project_root
        / "results_display"
        / ".cache_table1"
        / f"cox_ibs_auc_folds__{safe}.csv"
    )


def load_cache_coverage(*csv_paths: Path) -> dict[tuple[str, str], set[int]]:
    """(study, model) -> 已缓存的 fold 集合。"""
    coverage: dict[tuple[str, str], set[int]] = {}
    for path in csv_paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (str(row.get("study", "")), str(row.get("model", "")))
                try:
                    fold = int(float(row.get("fold", "")))
                except (TypeError, ValueError):
                    continue
                coverage.setdefault(key, set()).add(fold)
    return coverage


def find_cox_model_dirs(
    results_root: Path,
    group_dir: str,
    layer_dir: str,
    test_dir: str,
) -> list[Path]:
    """本实验目录内的 cox 类模型目录：clinic_cox / hgcn 基线 + mosaic_surv 主模型。"""
    model_dirs: list[Path] = []
    for study_token, _study in STUDY_SPECS:
        full_dir = resolve_group_study_dir(
            results_root,
            group_dir,
            layer_dir,
            test_dir,
            study_token,
            GROUP_CONFIG["baselines"]["results_suffix"],
        )
        if full_dir is not None:
            for name in (*COX_BASELINE_MODEL_DIRS, *HGCN_MODEL_DIRS):
                model_dirs.extend(sorted(full_dir.glob(f"**/{name}")))
        poe_dir = resolve_group_study_dir(
            results_root,
            group_dir,
            layer_dir,
            test_dir,
            study_token,
            GROUP_CONFIG["ours"]["results_suffix"],
        )
        if poe_dir is not None:
            for name in OURS_MODEL_DIRS:
                model_dirs.extend(sorted(poe_dir.glob(f"**/{name}")))
    return model_dirs


def ensure_cox_cache(
    project_root: Path,
    results_root: Path,
    *,
    group_dir: str,
    test_dir: str,
    layer_dir: str,
    gpu: int,
    folds: list[int],
) -> Path:
    """为缺缓存的 cox 模型目录补算 Breslow 前向 IBS/AUC（幂等）。"""
    cache_path = cox_cache_path(project_root, test_dir)
    shared_path = project_root / "results_display" / LEGACY_COX_CACHE_CSV
    coverage = load_cache_coverage(cache_path, shared_path)

    model_dirs = find_cox_model_dirs(results_root, group_dir, layer_dir, test_dir)
    if not model_dirs:
        print("[COX] no cox model dirs found under this experiment")
        return cache_path

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for model_dir in model_dirs:
        try:
            config = load_effective_config(model_dir)
        except FileNotFoundError:
            print(f"[COX] skip {model_dir} (missing effective_config.txt)")
            continue
        study = config.get("STUDY", "unknown")
        model = model_dir.name
        missing = [
            fold for fold in folds if fold not in coverage.get((study, model), set())
        ]
        if not missing:
            print(f"[COX] cached: {study}/{model}")
            continue
        if model == "hgcn":
            from Table1_HGCN_Forward import process_hgcn_model_dir

            print(f"[COX] hgcn forward: {study}/{model} folds={missing}")
            process_hgcn_model_dir(model_dir, missing, cache_path)
        else:
            print(f"[COX] forward: {study}/{model} folds={missing}")
            process_model_dir(model_dir, missing, cache_path)
        coverage.setdefault((study, model), set()).update(missing)
    return cache_path


def run_experiment(args: argparse.Namespace, project_root: Path, test_dir: str) -> None:
    results_root = args.results_root
    output_root = args.output_root
    group_dir = args.group_dir
    experiment_dir = results_root / group_dir / test_dir
    layer_dir = args.layer_dir or infer_layer_dir(test_dir, experiment_dir)
    if not layer_dir:
        print(f"[WARN] Skip {test_dir}: cannot infer layer prefix")
        return
    print(f"[EXP] {test_dir} (layer={layer_dir})")

    folds = [int(part) for part in args.folds.split(",") if part.strip()]

    # ---- 1. cox Breslow 前向缓存
    if not args.skip_cox:
        ensure_cox_cache(
            project_root,
            results_root,
            group_dir=group_dir,
            test_dir=test_dir,
            layer_dir=layer_dir,
            gpu=args.gpu,
            folds=folds,
        )

    # ---- 2. 论文五表（官方口径；Ours 来自本目录树内同批次的 mosaic_surv）
    build_tables(
        results_root,
        output_root,
        project_root,
        group_dir=group_dir,
        test_dir=test_dir,
        layer_dir=layer_dir,
        with_cox_cache=True,
        cox_cache_csv=cox_cache_path(project_root, test_dir),
        cox_shared_fallback=test_dir in args.cox_cache_dirs,
    )

    # ---- 3. summary 家族（8 cohort 全视图）
    collect_summary_family(
        results_root,
        output_root,
        group_dir=group_dir,
        test_dir=test_dir,
        layer_dir=layer_dir,
    )
    print(f"[DONE] {test_dir}")


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(
        description="Table1 统一生成入口（论文五表 + summary 家族 + cox 缓存）"
    )
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
    parser.add_argument("--group-dir", default="Table1_Cindex_Main")
    parser.add_argument("--layer-dir", default=None)
    parser.add_argument(
        "--test-dir",
        default="L0Test",
        help="Experiment folder under the grouped results directory; default L0Test",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=4,
        help="GPU id for the cox Breslow forward pass",
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--skip-cox",
        action="store_true",
        help="Skip the cox Breslow forward pass (Ours IBS/AUC cells show '-')",
    )
    parser.add_argument(
        "--cox-cache-dirs",
        default=DEFAULT_LEGACY_COX_DIRS,
        help=(
            "Dirs whose clinic_cox runs may fall back to the legacy shared "
            "Breslow cache (default none; 新批次用专属缓存)"
        ),
    )
    args = parser.parse_args()

    args.cox_cache_dirs = {
        part.strip() for part in args.cox_cache_dirs.split(",") if part.strip()
    }

    run_experiment(args, project_root, args.test_dir)
    print("Done.")


if __name__ == "__main__":
    main()
