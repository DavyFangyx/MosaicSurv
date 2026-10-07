"""
Collect Table 4（_Modality_Missing 子实验）A_readout 组缺失模态消融汇总。

A_readout 组（读出层变体：mosaic_surv / _single / _single_enum / _noenum /
_multi）的设计动机是缺失模态推理，因此本子实验采用 Table2 式的缺失模态
评测（EVAL_MODALITIES=all + MISSING_MODE=model_gen：每个模型只训一次
三模态，推理期生成缺失模态并跑 7 个子集 eval）。协议与 Table2 对齐
（L0 / gene_raw / uni_v1），主模型同批次重跑作参照。

Default input:
    results/Table4_Abaltion_Test/_Modality_Missing/{study}__L0__gene_raw__uni_v1/
        {model}/eval_subsets/{P,C,G,PC,PG,CG,PCG}/test_result.csv
        （PCG 回退 {model}/test_result.csv）

Default outputs:
    results_display/Table4_Abaltion_Test/_Modality_Missing/summary_A_readout_missing.csv
        （Table2 summary 式双层行：(subset, model) x 5 数据集 + mean）
    results_display/Table4_Abaltion_Test/_Modality_Missing/baseline_ranking.png
        （5 数据集行 x 8 列（7 子集 + MEAN）的排名图，与 Table2 构图一致）

兄弟实验（_Cindex_Main 子实验，A/B/C 三组完整模态 C-index）见
Table4_Abaltion_Test_Cindex_Main.py。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from model_names import display_name

GROUP_DIR = "Table4_Abaltion_Test/_Modality_Missing"
DISPLAY_DIR = "Table4_Abaltion_Test/_Modality_Missing"

STUDY_SPECS = [
    ("BRCA", "tcga_brca"),
    ("COAD", "tcga_coad"),
    ("KIRC", "tcga_kirc"),
    ("KIRP", "tcga_kirp"),
    ("LIHC", "tcga_lihc"),
]

# A_readout 组（读出层变体）+ 主模型参照；顺序即表格行序
A_GROUP_MODELS = [
    "mosaic_surv",
    "mosaic_surv_single",
    "mosaic_surv_single_enum",
    "mosaic_surv_noenum",
    "mosaic_surv_nodropout",
    "mosaic_surv_multi",
]

SUBSETS = ["P", "C", "G", "PC", "PG", "CG", "PCG"]

MODEL_COLORS = {
    "Mosaic-Surv (Ours)": "#d62728",
    "Mosaic-Surv w/ Single Head": "#1f77b4",
    "Mosaic-Surv w/ Single Head + Enumeration": "#2ca02c",
    "Mosaic-Surv w/o Pattern Enumeration": "#ff7f0e",
    "Mosaic-Surv w/o Train-Time Modality Dropout": "#8c564b",
    "Mosaic-Surv w/ Multi-Head": "#9467bd",
}
MODEL_COLOR_FALLBACK = "#17becf"


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def mean_std(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


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
    return mean_std(values)


def format_mean_std(stats: tuple[float, float] | None) -> str:
    if stats is None:
        return "-"
    mean, std = stats
    return f"{mean:.4f} ± {std:.4f}"


def parse_mean_std_from_text(value: str) -> tuple[float, float] | None:
    """解析 "0.6595 ± 0.0606" 形式的单元格，返回 (mean, std)；缺失返回 None。"""
    text = str(value).strip()
    if text in {"", "-", "nan"} or "±" not in text:
        return None
    mean_s, rest = text.split("±", 1)
    rest = re.sub(r"\(\d+\)$", "", rest.strip()).strip()
    try:
        return float(mean_s.strip()), float(rest)
    except ValueError:
        return None


def subset_csv_path(model_dir: Path, subset: str) -> Path | None:
    eval_path = model_dir / "eval_subsets" / subset / "test_result.csv"
    if eval_path.is_file():
        return eval_path
    if subset == "PCG":
        top_path = model_dir / "test_result.csv"
        if top_path.is_file():
            return top_path
    return None


def collect_group(results_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    """study -> subset -> model -> 'mean ± std' 文本。"""
    results_dir = results_root / GROUP_DIR
    if not results_dir.is_dir():
        print(f"[WARN] Missing results directory: {results_dir} (training not started?)")
        return {study: {subset: {display_name(model): "-" for model in A_GROUP_MODELS} for subset in SUBSETS} for _, study in STUDY_SPECS}

    study_to_tables: dict[str, dict[str, dict[str, str]]] = {}
    for _, study in STUDY_SPECS:
        run_dirs = sorted(results_dir.glob(f"{study}__*"))
        table: dict[str, dict[str, str]] = {}
        if not run_dirs:
            print(f"[WARN] Missing run dir for {study}")
            study_to_tables[study] = table
            continue
        if len(run_dirs) > 1:
            print(
                f"[WARN] Multiple run dirs for {study}: "
                f"{[path.name for path in run_dirs]}; using {run_dirs[0].name}"
            )
        run_dir = run_dirs[0]
        for model in A_GROUP_MODELS:
            model_dir = run_dir / model
            if not model_dir.is_dir():
                print(f"[WARN] Missing model directory: {model_dir}")
                for subset in SUBSETS:
                    table.setdefault(subset, {})[display_name(model)] = "-"
                continue
            for subset in SUBSETS:
                csv_path = subset_csv_path(model_dir, subset)
                if csv_path is None:
                    print(f"[WARN] Missing {study} {model} {subset}")
                    table.setdefault(subset, {})[display_name(model)] = "-"
                    continue
                table.setdefault(subset, {})[display_name(model)] = format_mean_std(
                    load_cindex_stats(csv_path)
                )
        study_to_tables[study] = table
    return study_to_tables


def write_summary_table(
    study_to_tables: dict[str, dict[str, dict[str, str]]], output_root: Path
) -> Path:
    studies = [study for _, study in STUDY_SPECS]
    model_labels = [display_name(model) for model in A_GROUP_MODELS]
    rows: list[list[str]] = []
    for subset in SUBSETS:
        for model in model_labels:
            row = [subset, model]
            means: list[float] = []
            for study in studies:
                text = study_to_tables.get(study, {}).get(subset, {}).get(model, "-")
                parsed = parse_mean_std_from_text(text)
                if parsed is not None:
                    means.append(parsed[0])
                row.append(text)
            row.append(
                f"{sum(means) / len(means):.4f}" if means else "-"
            )
            rows.append(row)

    out_dir = output_root / DISPLAY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_A_readout_missing.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "model", *studies, "mean"])
        writer.writerows(rows)
    print(f"[WRITE] {out_path.relative_to(output_root)}")
    return out_path


def write_ranking_figure(
    study_to_tables: dict[str, dict[str, dict[str, str]]], output_root: Path
) -> Path:
    """baseline_ranking.png：与 Table2 构图一致——5 行数据集 x 8 列
    （7 个测试子集 + MEAN），每个面板是 A 组模型的 C-index 竖条
    （固定顺序）+ 排名数字；主模型第一红框、胜过其余变体均值时蓝框。"""
    models = [display_name(model) for model in A_GROUP_MODELS]
    figure_subsets = [*SUBSETS, "MEAN"]
    fig, axes = plt.subplots(
        5, 8, figsize=(21, 13), sharey=True, constrained_layout=True
    )
    for i, (_, study) in enumerate(STUDY_SPECS):
        table = study_to_tables.get(study, {})
        for j, subset in enumerate(figure_subsets):
            ax = axes[i, j]
            if subset == "MEAN":
                mean_values = {}
                for model in models:
                    stats_list = [
                        parse_mean_std_from_text(table.get(s, {}).get(model, "-"))
                        for s in SUBSETS
                    ]
                    stats_list = [stats for stats in stats_list if stats is not None]
                    if stats_list:
                        mean_values[model] = (
                            sum(mean for mean, _ in stats_list) / len(stats_list),
                            sum(std for _, std in stats_list) / len(stats_list),
                        )
                panel_values = mean_values
            else:
                panel_values = {}
                for model in models:
                    parsed = parse_mean_std_from_text(
                        table.get(subset, {}).get(model, "-")
                    )
                    if parsed is not None:
                        panel_values[model] = parsed
            available = [
                (model, stats)
                for model, stats in panel_values.items()
                if stats is not None
            ]
            ranked = sorted(available, key=lambda item: (-item[1][0], item[0]))
            ranks = {model: rank for rank, (model, _) in enumerate(ranked, start=1)}

            ours_stats = panel_values.get("Mosaic-Surv (Ours)")
            variant_means = [
                stats[0]
                for model, stats in panel_values.items()
                if model != "Mosaic-Surv (Ours)"
            ]
            frame_color = None
            if ours_stats is not None and ranks.get("Mosaic-Surv (Ours)") == 1:
                frame_color = "#d62728"
            elif (
                ours_stats is not None
                and variant_means
                and ours_stats[0] > max(variant_means)
            ):
                frame_color = "#1f77b4"
            if frame_color is not None:
                for spine in ax.spines.values():
                    spine.set_color(frame_color)
                    spine.set_linewidth(2.0)

            for k, model in enumerate(models):
                stats = panel_values.get(model)
                if stats is None:
                    continue
                ax.bar(
                    k,
                    stats[0],
                    yerr=stats[1],
                    capsize=2,
                    color=MODEL_COLORS.get(model, MODEL_COLOR_FALLBACK),
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.annotate(
                    str(ranks[model]),
                    xy=(k, stats[0] + stats[1]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )
            ax.set_title(f"{study.replace('tcga_', '').upper()} / {subset}", fontsize=8)
            ax.set_xticks(range(len(models)), models, rotation=55, ha="right", fontsize=6)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
            if j == 0:
                ax.set_ylabel("C-index")

    out_dir = output_root / DISPLAY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_ranking.png"
    fig.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out_path.relative_to(output_root)}")
    return out_path


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(
        description="Collect Table 4 (_Modality_Missing) A-group missing-modality summaries"
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
    args = parser.parse_args()

    study_to_tables = collect_group(args.results_root)
    write_summary_table(study_to_tables, args.output_root)
    write_ranking_figure(study_to_tables, args.output_root)
    print("Done.")


if __name__ == "__main__":
    main()
