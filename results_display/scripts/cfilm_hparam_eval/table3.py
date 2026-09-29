from __future__ import annotations

import math
from pathlib import Path

import matplotlib

from common import (
    CFILM_MODEL,
    MISSING_PATTERNS,
    STUDIES,
    STUDY_LABELS,
    SUBSETS,
    Stats,
    assign_ranks,
    hparam_columns,
    load_subset_stats,
    mean_std,
    parse_io_args,
    read_hparams,
    table_enabled,
    with_hparams,
    write_csv,
)

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


RUN_SUFFIX = "L0__gene_raw__uni_v1"
STUDY_COLORS = {
    "tcga_brca": "#1b9e77",
    "tcga_coad": "#d95f02",
    "tcga_kirc": "#7570b3",
    "tcga_kirp": "#e7298a",
    "tcga_lihc": "#66a61e",
}
PATTERN_LABELS = [pattern.removeprefix("p").replace("_", "/") for pattern in MISSING_PATTERNS]
MODEL_NAMES = [CFILM_MODEL, "hgcn"]
MODEL_DISPLAY_NAMES = {CFILM_MODEL: "Cfilm", "hgcn": "HCGN"}
MODEL_STYLES = {CFILM_MODEL: ("o", "-"), "hgcn": ("^", "--")}
# Panel order of the classic combined Table3 figure.
FIGURE_SUBSETS = ["P", "C", "G", "PG", "PC", "CG", "PCG"]
STABILITY_METRICS = [
    "curve_std",
    "curve_range",
    "adjacent_total_variation",
    "max_adjacent_change",
]


def _model_dir(
    results_root: Path,
    run_id: str,
    study: str,
    pattern: str,
) -> Path:
    """Resolve both the canonical and legacy Table3 Cfilm layouts."""
    run_name = f"{study}__{RUN_SUFFIX}__{pattern}"
    canonical = (
        results_root / "Cfilm_Hparam_Eval" / "Table3" / run_id
        / run_name / CFILM_MODEL
    )
    if canonical.is_dir():
        return canonical
    legacy = (
        results_root / "Cfilm_Hparam_Eval" / "Table3_MissingRate" / run_id
        / run_name / CFILM_MODEL
    )
    return legacy


def _hgcn_dir(results_root: Path, study: str, pattern: str) -> Path:
    run_name = f"{study}__{RUN_SUFFIX}__{pattern}"
    return results_root / "Table3_MissingRate" / run_name / "hgcn"


def _format_mean_std(stats: Stats | None) -> str:
    if stats is None:
        return "-"
    return f"{stats.mean:.4f} ± {stats.std:.4f}"


def _panel_tables(
    subset: str,
    cfilm: dict[tuple[str, str, str], Stats],
    hgcn: dict[tuple[str, str, str], Stats],
) -> dict[str, dict[str, dict[str, Stats]]]:
    """Build study -> model -> pattern stats for one test subset."""
    tables: dict[str, dict[str, dict[str, Stats]]] = {study: {} for study in STUDIES}
    for study in STUDIES:
        for model_name, values in ((CFILM_MODEL, cfilm), ("hgcn", hgcn)):
            tables[study][model_name] = {
                pattern: values[(study, pattern, subset)]
                for pattern in MISSING_PATTERNS
                if (study, pattern, subset) in values
            }
    return tables


def _split_legend_handles(
    cfilm: dict[tuple[str, str, str], Stats],
    hgcn: dict[tuple[str, str, str], Stats],
) -> tuple[list[Line2D], list[Line2D]]:
    dataset_handles = []
    for study in STUDIES:
        present = any(
            (study, pattern, subset) in cfilm or (study, pattern, subset) in hgcn
            for pattern in MISSING_PATTERNS
            for subset in SUBSETS
        )
        if present:
            dataset_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=STUDY_COLORS[study],
                    marker="o",
                    linestyle="None",
                    markersize=8,
                    markerfacecolor=STUDY_COLORS[study],
                    markeredgecolor=STUDY_COLORS[study],
                    label=STUDY_LABELS[study],
                )
            )
    model_handles = []
    for model_name in MODEL_NAMES:
        values = cfilm if model_name == CFILM_MODEL else hgcn
        if values:
            marker, linestyle = MODEL_STYLES[model_name]
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
                    label=MODEL_DISPLAY_NAMES[model_name],
                )
            )
    return dataset_handles, model_handles


def _draw_subset_panel(
    ax,
    subset: str,
    tables: dict[str, dict[str, dict[str, Stats]]],
    *,
    show_ylabel: bool,
) -> None:
    xs = list(range(len(MISSING_PATTERNS)))
    means = [
        stats.mean
        for model_map in tables.values()
        for pattern_map in model_map.values()
        for stats in pattern_map.values()
    ]
    if not means:
        return
    ymin = math.floor(min(means) / 0.05) * 0.05
    ymax = max(means) + 0.03
    for study in STUDIES:
        model_map = tables.get(study, {})
        for model_name in MODEL_NAMES:
            pattern_map = model_map.get(model_name, {})
            valid_xs: list[int] = []
            ys: list[float] = []
            for index, pattern in enumerate(MISSING_PATTERNS):
                stats = pattern_map.get(pattern)
                if stats is None:
                    continue
                valid_xs.append(index)
                ys.append(stats.mean)
            if not valid_xs:
                continue
            marker, linestyle = MODEL_STYLES[model_name]
            ax.plot(
                valid_xs,
                ys,
                color=STUDY_COLORS[study],
                marker=marker,
                linestyle=linestyle,
                markersize=6.0,
                linewidth=1.7,
                markerfacecolor=STUDY_COLORS[study],
                markeredgecolor=STUDY_COLORS[study],
                markeredgewidth=0.5,
                zorder=3,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(PATTERN_LABELS, fontsize=8)
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


def _write_subset_figure(
    path: Path,
    subset: str,
    cfilm: dict[tuple[str, str, str], Stats],
    hgcn: dict[tuple[str, str, str], Stats],
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    _draw_subset_panel(ax, subset, _panel_tables(subset, cfilm, hgcn), show_ylabel=True)
    dataset_handles, model_handles = _split_legend_handles(cfilm, hgcn)
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


def _write_figure(
    path: Path,
    cfilm: dict[tuple[str, str, str], Stats],
    hgcn: dict[tuple[str, str, str], Stats],
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
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
    axes = []
    for index, subset in enumerate(FIGURE_SUBSETS):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(gs[row, col])
        _draw_subset_panel(
            ax,
            subset,
            _panel_tables(subset, cfilm, hgcn),
            show_ylabel=(col == 0),
        )
        axes.append(ax)

    axes[-1].set_xlabel("Train missing rate (P/G/C)", fontsize=10, labelpad=3)

    dataset_handles, model_handles = _split_legend_handles(cfilm, hgcn)
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

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def _model_stability_values(
    subset: str,
    values: dict[tuple[str, str, str], Stats],
) -> dict[str, dict[str, float]]:
    """Dataset label -> metric value for datasets with complete pattern curves, plus Mean."""
    dataset_values: dict[str, dict[str, float]] = {}
    for study in STUDIES:
        means = [
            values[(study, pattern, subset)].mean
            for pattern in MISSING_PATTERNS
            if (study, pattern, subset) in values
        ]
        if len(means) != len(MISSING_PATTERNS):
            continue
        curve_stats = mean_std(means)
        if curve_stats is None:
            continue
        dataset_values[STUDY_LABELS[study]] = {
            "curve_std": curve_stats.std,
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
            for name in STABILITY_METRICS
        }
    return dataset_values


def _write_stability_summary(
    path: Path,
    subset: str,
    cfilm: dict[tuple[str, str, str], Stats],
    hgcn: dict[tuple[str, str, str], Stats],
) -> None:
    dataset_labels = [STUDY_LABELS[study] for study in STUDIES]
    cfilm_values = _model_stability_values(subset, cfilm)
    hgcn_values = _model_stability_values(subset, hgcn)
    rows: list[dict[str, object]] = []
    for metric_name in STABILITY_METRICS:
        for label in [*dataset_labels, "Mean"]:
            cfilm_value = cfilm_values.get(label, {}).get(metric_name)
            hgcn_value = hgcn_values.get(label, {}).get(metric_name)
            if cfilm_value is None and hgcn_value is None:
                continue
            delta = (
                hgcn_value - cfilm_value
                if cfilm_value is not None and hgcn_value is not None
                else None
            )
            rows.append(
                {
                    "metric": metric_name,
                    "dataset": label,
                    "Cfilm": f"{cfilm_value:.4f}" if cfilm_value is not None else "",
                    "HCGN": f"{hgcn_value:.4f}" if hgcn_value is not None else "",
                    "Δ=HCGN-Cfilm": f"{delta:.4f}" if delta is not None else "",
                }
            )
    write_csv(path, rows, ["metric", "dataset", "Cfilm", "HCGN", "Δ=HCGN-Cfilm"])


def collect(results_root: Path, output_root: Path, hparams_path: Path) -> list[dict[str, object]]:
    all_hparams_rows = read_hparams(hparams_path)
    hparams_rows = [row for row in all_hparams_rows if table_enabled(row, "T3")]
    aggregate: list[dict[str, object]] = []
    hgcn: dict[tuple[str, str, str], Stats] = {}
    for study in STUDIES:
        for pattern in MISSING_PATTERNS:
            model_dir = _hgcn_dir(results_root, study, pattern)
            for subset in SUBSETS:
                stats = load_subset_stats(model_dir, subset)
                if stats is not None:
                    hgcn[(study, pattern, subset)] = stats

    for hparams in hparams_rows:
        run_id = hparams["run_id"].strip()
        cfilm: dict[tuple[str, str, str], Stats] = {}
        deltas: list[float] = []
        fold_stds: list[float] = []
        wins = 0
        ties = 0
        losses = 0
        completed = 0

        for study in STUDIES:
            for pattern in MISSING_PATTERNS:
                model_dir = _model_dir(results_root, run_id, study, pattern)
                for subset in SUBSETS:
                    key = (study, pattern, subset)
                    cfilm_stats = load_subset_stats(model_dir, subset)
                    hgcn_stats = hgcn.get(key)
                    if cfilm_stats is not None:
                        cfilm[key] = cfilm_stats
                    comparable = cfilm_stats is not None and hgcn_stats is not None
                    delta = cfilm_stats.mean - hgcn_stats.mean if comparable else None
                    if delta is not None:
                        completed += 1
                        deltas.append(delta)
                        fold_stds.append(cfilm_stats.std)
                        if delta > 1e-12:
                            wins += 1
                        elif delta < -1e-12:
                            losses += 1
                        else:
                            ties += 1

        run_dir = output_root / "Table3" / run_id
        # Drop artifacts of the previous per-run layout (comparison_summary.csv,
        # root-level subset CSVs, and the old figures/ directory).
        for stale_path in run_dir.glob("*.csv"):
            stale_path.unlink()
        figures_dir = run_dir / "figures"
        for stale_path in figures_dir.glob("*.png"):
            stale_path.unlink()
        try:
            figures_dir.rmdir()
        except OSError:
            pass

        for subset in SUBSETS:
            subset_dir = run_dir / subset
            rows: list[dict[str, object]] = []
            for study in STUDIES:
                for model_name, values in (("Cfilm", cfilm), ("HCGN", hgcn)):
                    row: dict[str, object] = {"dataset": study, "model": model_name}
                    for pattern in MISSING_PATTERNS:
                        row[pattern] = _format_mean_std(values.get((study, pattern, subset)))
                    rows.append(row)
            write_csv(
                subset_dir / f"{subset}.csv",
                rows,
                ["dataset", "model", *MISSING_PATTERNS],
            )
            _write_stability_summary(subset_dir / "stability_summary.csv", subset, cfilm, hgcn)
            _write_subset_figure(subset_dir / "train_missing_rate.png", subset, cfilm, hgcn)

        if cfilm:
            _write_figure(run_dir / "train_missing_rate.png", cfilm, hgcn)

        curve_stds: list[float] = []
        for study in STUDIES:
            for subset in SUBSETS:
                curve = [
                    cfilm[(study, pattern, subset)].mean
                    for pattern in MISSING_PATTERNS
                    if (study, pattern, subset) in cfilm
                ]
                if len(curve) == len(MISSING_PATTERNS):
                    curve_stats = mean_std(curve)
                    if curve_stats is not None:
                        curve_stds.append(curve_stats.std)

        expected = len(STUDIES) * len(MISSING_PATTERNS) * len(SUBSETS)
        delta_stats = mean_std(deltas)
        stability = mean_std(curve_stds)
        aggregate.append(
            with_hparams(
                {
                    "completed_cells": completed,
                    "expected_cells": expected,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": wins / completed if completed else None,
                    "mean_delta_hgcn": delta_stats.mean if delta_stats else None,
                    "worst_delta_hgcn": min(deltas) if deltas else None,
                    "mean_curve_std": stability.mean if stability else None,
                    "worst_curve_std": max(curve_stds) if curve_stds else None,
                    "mean_fold_std": sum(fold_stds) / len(fold_stds) if fold_stds else None,
                    "status": "complete" if completed == expected else "incomplete",
                },
                hparams,
            )
        )

    assign_ranks(
        aggregate,
        eligible=lambda row: row["status"] == "complete",
        sort_key=lambda row: (
            -int(row["wins"]),
            float(row["mean_curve_std"]),
            -float(row["mean_delta_hgcn"]),
        ),
    )
    fields = hparam_columns(hparams_rows) + [
        "completed_cells", "expected_cells", "wins", "ties", "losses", "win_rate",
        "mean_delta_hgcn", "worst_delta_hgcn", "mean_curve_std", "worst_curve_std",
        "mean_fold_std", "status", "rank",
    ]
    ranked = sorted(aggregate, key=lambda row: (row["rank"] == "", row["rank"] or 10**9, row["run_id"]))
    write_csv(output_root / "Table3" / "parameter_ranking.csv", ranked, fields)
    return aggregate


def main() -> None:
    args = parse_io_args("Collect Table3 Cfilm hyperparameter results")
    collect(args.results_root, args.output_root, args.hparams)
    print(f"Collected Table3 results into {args.output_root / 'Table3'}")


if __name__ == "__main__":
    main()
