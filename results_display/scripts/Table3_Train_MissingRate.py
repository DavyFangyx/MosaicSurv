"""
Collect Table 3 train-missing-rate c-index summaries and line plots.

Default input:
    results/Table3_MissingRate/{study}__{clinic}__{gene_tag}__{wsi}__pW_G_C/
        {model}/eval_subsets/summary_long.csv

Default outputs:
    results_display/Table3_Train_MissingRate/{P,C,G,PG,PC,CG,PCG}/
        {P,C,G,PG,PC,CG,PCG}.csv
        stability_summary.csv
        train_missing_rate.png
    results_display/Table3_Train_MissingRate/train_missing_rate.png

Each CSV is one test subset. Rows are dataset+model. The figure is a 3-column
grid of the seven subsets, overlaying models by marker/linestyle and datasets by
color. Cells are 5-fold c-index mean ± std; plotted lines use means only.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.style.use("seaborn-v0_8-whitegrid")

GROUP_DIR = "Table3_MissingRate"
OUTPUT_DIR = "Table3_Train_MissingRate"

STUDY_SPECS = [
    ("BRCA", "tcga_brca", "#1b9e77"),
    ("COAD", "tcga_coad", "#d95f02"),
    ("KIRC", "tcga_kirc", "#7570b3"),
    ("KIRP", "tcga_kirp", "#e7298a"),
    ("LIHC", "tcga_lihc", "#66a61e"),
]
TEST_SUBSETS = ["P", "C", "G", "PG", "PC", "CG", "PCG"]
MISSING_PATTERNS = [
    "p60_0_0",
    "p0_60_0",
    "p0_0_60",
    "p30_30_0",
    "p30_0_30",
    "p0_30_30",
    "p20_20_20",
]
PATTERN_LABELS = {
    "p60_0_0": "60/0/0",
    "p0_60_0": "0/60/0",
    "p0_0_60": "0/0/60",
    "p30_30_0": "30/30/0",
    "p30_0_30": "30/0/30",
    "p0_30_30": "0/30/30",
    "p20_20_20": "20/20/20",
}
PREFERRED_MODEL_ORDER = [
    "survtri_poe_vae_c_film",
    "hgcn",
]
MODEL_DISPLAY_NAMES = {
    "survtri_poe_vae_c_film": "Cfilm",
    "hgcn": "HCGN",
}
MODEL_STYLES = {
    "survtri_poe_vae_c_film": ("o", "-"),
    "hgcn": ("^", "--"),
}
EXTRA_MODEL_MARKERS = ["s", "D", "v", "P", "X"]
EXTRA_MODEL_LINESTYLES = ["-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
Stats = tuple[float, float]
SubsetTables = dict[str, dict[str, dict[str, dict[str, Stats]]]]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def mean_std(values: list[float]) -> Stats | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def load_cindex_values(csv_path: Path, column: str = "test_cindex") -> list[float]:
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get(column, "")
            if raw in ("", None):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return values


def parse_run_dir(run_dir: Path) -> tuple[str, str] | None:
    parts = run_dir.name.split("__")
    if len(parts) < 2:
        return None
    study = parts[0]
    pattern = parts[-1]
    if not pattern.startswith("p"):
        return None
    return study, pattern


def display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def model_sort_key(model_name: str) -> tuple[int, str]:
    if model_name in PREFERRED_MODEL_ORDER:
        return PREFERRED_MODEL_ORDER.index(model_name), model_name
    return len(PREFERRED_MODEL_ORDER), model_name


def style_for_models(model_names: Iterable[str]) -> dict[str, tuple[str, object]]:
    styles: dict[str, tuple[str, object]] = {}
    extra_index = 0
    for model_name in model_names:
        if model_name in MODEL_STYLES:
            styles[model_name] = MODEL_STYLES[model_name]
            continue
        marker = EXTRA_MODEL_MARKERS[extra_index % len(EXTRA_MODEL_MARKERS)]
        linestyle = EXTRA_MODEL_LINESTYLES[extra_index % len(EXTRA_MODEL_LINESTYLES)]
        styles[model_name] = (marker, linestyle)
        extra_index += 1
    return styles


def is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "eval_subsets").is_dir()


def discover_model_dirs(run_dir: Path, requested: set[str] | None) -> list[Path]:
    model_dirs: list[Path] = []
    for child in run_dir.iterdir():
        if not is_model_dir(child):
            continue
        if requested is not None and child.name not in requested:
            continue
        model_dirs.append(child)
    return sorted(model_dirs, key=lambda path: model_sort_key(path.name))


def ordered_models(tables: SubsetTables) -> list[str]:
    names: set[str] = set()
    for study_map in tables.values():
        for model_map in study_map.values():
            names.update(model_map)
    return sorted(names, key=model_sort_key)


def load_subset_stats(model_dir: Path) -> dict[str, Stats]:
    subset_to_stats: dict[str, Stats] = {}
    summary_path = model_dir / "eval_subsets" / "summary_long.csv"
    if summary_path.is_file():
        try:
            subset_values: dict[str, list[float]] = {subset: [] for subset in TEST_SUBSETS}
            with summary_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    subset = str(row.get("subset", "")).strip()
                    if subset not in subset_values:
                        continue
                    raw = row.get("test_cindex", "")
                    if raw in ("", None):
                        continue
                    try:
                        subset_values[subset].append(float(raw))
                    except ValueError:
                        continue
            for subset, values in subset_values.items():
                stats = mean_std(values)
                if stats is None:
                    continue
                subset_to_stats[subset] = stats
            if subset_to_stats:
                return subset_to_stats
        except Exception as exc:
            print(f"[WARN] Cannot read {summary_path}: {exc}")

    for subset in TEST_SUBSETS:
        csv_path = model_dir / "eval_subsets" / subset / "test_result.csv"
        if not csv_path.is_file():
            continue
        try:
            stats = mean_std(load_cindex_values(csv_path))
        except Exception as exec_exc:
            print(f"[WARN] Cannot read {csv_path}: {exec_exc}")
            continue
        if stats is None:
            print(f"[WARN] Empty test_cindex in {csv_path}")
            continue
        subset_to_stats[subset] = stats
    return subset_to_stats


def collect_group(
    results_root: Path,
    *,
    group_dir: str,
    model_names: list[str] | None,
) -> SubsetTables:
    results_dir = results_root / group_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    requested = set(model_names) if model_names else None
    valid_studies = {study for _, study, _ in STUDY_SPECS}
    tables: SubsetTables = {
        subset: {study: {} for _, study, _ in STUDY_SPECS} for subset in TEST_SUBSETS
    }

    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        parsed = parse_run_dir(run_dir)
        if parsed is None:
            continue
        study, pattern = parsed
        if study not in valid_studies or pattern not in MISSING_PATTERNS:
            continue

        model_dirs = discover_model_dirs(run_dir, requested)
        if not model_dirs:
            if requested is None:
                print(f"[WARN] No model directories in {run_dir}")
            else:
                missing = ", ".join(sorted(requested))
                print(f"[WARN] Missing requested model directories in {run_dir}: {missing}")
            continue

        for model_dir in model_dirs:
            subset_to_stats = load_subset_stats(model_dir)
            if not subset_to_stats:
                print(f"[WARN] No subset metrics in {model_dir}")
                continue

            print(f"[LOAD] {run_dir.name}/{model_dir.name}")
            for subset, stats in subset_to_stats.items():
                tables[subset][study].setdefault(model_dir.name, {})[pattern] = stats

    return tables


def write_subset_table(
    subset: str,
    study_to_model_stats: dict[str, dict[str, dict[str, Stats]]],
    model_names: list[str],
    output_dir: Path,
) -> Path:
    rows: list[dict[str, str]] = []
    for _, study, _ in STUDY_SPECS:
        model_map = study_to_model_stats.get(study, {})
        for model_name in model_names:
            pattern_map = model_map.get(model_name, {})
            if not pattern_map:
                continue
            row = {"dataset": study, "model": display_model_name(model_name)}
            for pattern in MISSING_PATTERNS:
                stats = pattern_map.get(pattern)
                row[pattern] = format_mean_std(*stats) if stats is not None else "-"
            rows.append(row)

    out_path = output_dir / f"{subset}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "model", *MISSING_PATTERNS])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_stability_summary(
    tables: SubsetTables,
    subset: str,
    model_names: list[str],
    output_dir: Path,
) -> Path:
    metric_names = [
        "curve_std",
        "curve_range",
        "adjacent_total_variation",
        "max_adjacent_change",
    ]
    dataset_labels = [label for label, _, _ in STUDY_SPECS]
    model_values: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in model_names:
        dataset_values: dict[str, dict[str, float]] = {}
        for label, study, _ in STUDY_SPECS:
            pattern_map = tables[subset].get(study, {}).get(model_name, {})
            if not all(pattern in pattern_map for pattern in MISSING_PATTERNS):
                continue
            means = [pattern_map[pattern][0] for pattern in MISSING_PATTERNS]
            curve_stats = mean_std(means)
            if curve_stats is None:
                continue
            dataset_values[label] = {
                "curve_std": curve_stats[1],
                "curve_range": max(means) - min(means),
                "adjacent_total_variation": sum(
                    abs(right - left) for left, right in zip(means, means[1:])
                ),
                "max_adjacent_change": max(
                    abs(right - left) for left, right in zip(means, means[1:])
                ),
            }
        if dataset_values:
            dataset_values["Mean"] = {
                name: sum(dataset[name] for dataset in dataset_values.values())
                / len(dataset_values)
                for name in metric_names
            }
        model_values[model_name] = dataset_values

    column_values = {
        MODEL_DISPLAY_NAMES.get(name, name): model_values[name] for name in model_names
    }
    cfilm_values = column_values.get("Cfilm", {})
    hgcn_values = column_values.get("HCGN", {})

    rows: list[dict[str, str]] = []
    for metric_name in metric_names:
        for label in [*dataset_labels, "Mean"]:
            cfilm = cfilm_values.get(label, {}).get(metric_name)
            hgcn = hgcn_values.get(label, {}).get(metric_name)
            if cfilm is None and hgcn is None:
                continue
            delta = hgcn - cfilm if cfilm is not None and hgcn is not None else None
            rows.append(
                {
                    "metric": metric_name,
                    "dataset": label,
                    "Cfilm": f"{cfilm:.4f}" if cfilm is not None else "",
                    "HCGN": f"{hgcn:.4f}" if hgcn is not None else "",
                    "Δ=HCGN-Cfilm": f"{delta:.4f}" if delta is not None else "",
                }
            )

    out_path = output_dir / "stability_summary.csv"
    fieldnames = ["metric", "dataset", "Cfilm", "HCGN", "Δ=HCGN-Cfilm"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_subset_figure(
    subset: str,
    tables: SubsetTables,
    model_names: list[str],
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    styles = style_for_models(model_names)
    draw_subset_panel(ax, subset, tables[subset], model_names, styles, show_ylabel=True)
    dataset_handles, model_handles = split_legend_handles(tables, model_names, styles)
    fig.legend(
        handles=dataset_handles + model_handles,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fontsize=8,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def infer_ylim(study_to_model_stats: dict[str, dict[str, dict[str, Stats]]]) -> tuple[float, float]:
    values: list[float] = []
    for model_map in study_to_model_stats.values():
        for pattern_map in model_map.values():
            for mean, _std in pattern_map.values():
                values.append(mean)
    if not values:
        return 0.40, 0.80
    ymin = math.floor(min(values) / 0.05) * 0.05
    ymax = max(values) + 0.03
    return ymin, ymax


def draw_subset_panel(
    ax,
    subset: str,
    study_to_model_stats: dict[str, dict[str, dict[str, Stats]]],
    model_names: list[str],
    styles: dict[str, tuple[str, object]],
    *,
    show_ylabel: bool,
) -> None:
    xs = list(range(len(MISSING_PATTERNS)))
    ymin, ymax = infer_ylim(study_to_model_stats)
    for _, study, color in STUDY_SPECS:
        model_map = study_to_model_stats.get(study, {})
        for model_name in model_names:
            pattern_map = model_map.get(model_name, {})
            means: list[float] = []
            valid_xs: list[int] = []
            for idx, pattern in enumerate(MISSING_PATTERNS):
                stats = pattern_map.get(pattern)
                if stats is None:
                    continue
                valid_xs.append(idx)
                means.append(stats[0])
            if not valid_xs:
                continue
            marker, linestyle = styles[model_name]
            ax.plot(
                valid_xs,
                means,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=6.0,
                linewidth=1.7,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.5,
                zorder=3,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([PATTERN_LABELS[pattern] for pattern in MISSING_PATTERNS], fontsize=8)
    ax.margins(x=0.05)
    ax.set_xlim(-0.35, len(MISSING_PATTERNS) - 0.65)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter("{x:.2f}")
    ax.set_title(f"Test subset {subset}", fontsize=11, pad=6)
    if show_ylabel:
        ax.set_ylabel("C-index", fontsize=10, labelpad=4)
    ax.grid(True, linestyle="--", linewidth=0.7, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8, pad=1.5, length=3.0, width=0.8, direction="out", labelbottom=True)
    ax.tick_params(axis="y", labelsize=8, pad=2, length=3.0, width=0.8, direction="out")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def split_legend_handles(
    tables: SubsetTables,
    model_names: list[str],
    styles: dict[str, tuple[str, object]],
) -> tuple[list[Line2D], list[Line2D]]:
    dataset_handles = []
    for label, study, color in STUDY_SPECS:
        if any(tables[subset].get(study, {}) for subset in TEST_SUBSETS):
            dataset_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=8,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    label=label,
                )
            )
    model_handles = []
    for model_name in model_names:
        if any(
            model_name in tables[subset].get(study, {})
            for subset in TEST_SUBSETS
            for _, study, _ in STUDY_SPECS
        ):
            marker, linestyle = styles[model_name]
            model_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.2",
                    marker=marker,
                    linestyle=linestyle,
                    markersize=7,
                    linewidth=1.7,
                    markerfacecolor="0.2",
                    markeredgecolor="0.2",
                    label=display_model_name(model_name),
                )
            )
    return dataset_handles, model_handles


def write_combined_figure(
    tables: SubsetTables,
    model_names: list[str],
    figure_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(12.2, 9.0))
    gs = fig.add_gridspec(
        3,
        3,
        left=0.06,
        right=0.985,
        top=0.93,
        bottom=0.06,
        wspace=0.18,
        hspace=0.28,
    )
    styles = style_for_models(model_names)
    axes = []
    for idx, subset in enumerate(TEST_SUBSETS):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        draw_subset_panel(
            ax,
            subset,
            tables[subset],
            model_names,
            styles,
            show_ylabel=col == 0,
        )
        axes.append(ax)

    axes[-1].set_xlabel("Train missing rate (P/G/C)", fontsize=10, labelpad=3)

    dataset_handles, model_handles = split_legend_handles(tables, model_names, styles)
    spacer = Line2D([], [], linestyle="None", marker="None", color="none", label=" ")
    handles = dataset_handles + [spacer] + model_handles
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=len(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        fontsize=10,
        columnspacing=1.2,
        handletextpad=0.35,
        handlelength=2.2,
        borderaxespad=0.0,
    )

    out_path = figure_dir / "train_missing_rate.png"
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Collect Table 3 train-missing-rate c-index summaries")
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
        "--output-dir",
        default=OUTPUT_DIR,
        help="Grouped output directory name",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=None,
        help="Model directory name to collect. Repeat to keep a subset; default discovers all.",
    )
    args = parser.parse_args()

    tables = collect_group(
        args.results_root,
        group_dir=args.group_dir,
        model_names=args.model_name,
    )
    model_names = ordered_models(tables)

    output_dir = args.output_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in output_dir.glob("*.csv"):
        csv_path.unlink()
    for png_path in output_dir.glob("*.png"):
        png_path.unlink()
    legacy_figure_dir = output_dir / "figures"
    for png_path in legacy_figure_dir.glob("*.png"):
        png_path.unlink()

    if not model_names:
        print("[WARN] No model metrics collected.")

    for subset in TEST_SUBSETS:
        subset_dir = output_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_subset_table(subset, tables[subset], model_names, subset_dir)
        print(f"[WRITE] {out_path.relative_to(args.output_root)}")
        stability_path = write_stability_summary(tables, subset, model_names, subset_dir)
        print(f"[WRITE] {stability_path.relative_to(args.output_root)}")
        subset_fig_path = write_subset_figure(
            subset, tables, model_names, subset_dir / "train_missing_rate.png"
        )
        print(f"[WRITE] {subset_fig_path.relative_to(args.output_root)}")
    fig_path = write_combined_figure(tables, model_names, output_dir)
    print(f"[WRITE] {fig_path.relative_to(args.output_root)}")
    print("Done.")


if __name__ == "__main__":
    main()
