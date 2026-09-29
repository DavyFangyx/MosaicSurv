from __future__ import annotations

from pathlib import Path

from common import (
    CFILM_MODEL,
    MISSING_SUBSETS,
    STUDIES,
    SUBSETS,
    assign_ranks,
    hparam_columns,
    load_subset_stats,
    mean_std,
    number,
    parse_io_args,
    read_hparams,
    table_enabled,
    with_hparams,
    write_csv,
)


BASELINES = {
    "zero": "modality_concat__resampler__zero",
    "mean": "modality_concat__resampler__mean",
    "hgcn": "hgcn",
}
RUN_SUFFIX = "L4__gene_raw__uni_v1"
MODEL_COLORS = {"cfilm": "#d62728", "zero": "#1f77b4", "mean": "#2ca02c", "hgcn": "#7f7f7f"}


def cfilm_model_dir(results_root: Path, run_id: str, study: str) -> Path:
    """Resolve both the canonical and legacy Table2 Cfilm layouts."""
    run_name = f"{study}__{RUN_SUFFIX}"
    canonical = (
        results_root / "Cfilm_Hparam_Eval" / "Table2" / run_id
        / run_name / CFILM_MODEL
    )
    if canonical.is_dir():
        return canonical
    legacy = (
        results_root / "Cfilm_Hparam_Eval" / "Table2_Baselines" / run_id
        / run_name / CFILM_MODEL
    )
    return legacy


def write_ranking_figure(path: Path, values: dict[tuple[str, str, str], object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["cfilm", "zero", "mean", "hgcn"]
    figure_subsets = [*SUBSETS, "MEAN"]
    fig, axes = plt.subplots(5, 8, figsize=(21, 13), sharey=True, constrained_layout=True)
    for i, study in enumerate(STUDIES):
        for j, subset in enumerate(figure_subsets):
            ax = axes[i, j]
            if subset == "MEAN":
                mean_values = {}
                for model in models:
                    stats_list = [values.get((study, s, model)) for s in SUBSETS]
                    stats_list = [stats for stats in stats_list if stats is not None]
                    if stats_list:
                        from common import Stats
                        mean_values[model] = Stats(
                            mean=sum(stats.mean for stats in stats_list) / len(stats_list),
                            std=sum(stats.std for stats in stats_list) / len(stats_list),
                            count=len(stats_list),
                        )
                panel_values = mean_values
            else:
                panel_values = {model: values.get((study, subset, model)) for model in models}
            available = [
                (model, panel_values.get(model))
                for model in models
                if panel_values.get(model) is not None
            ]
            ranked = sorted(available, key=lambda item: (-item[1].mean, item[0]))
            ranks = {model: rank for rank, (model, _) in enumerate(ranked, start=1)}
            cfilm_stats = panel_values.get("cfilm")
            hgcn_stats = panel_values.get("hgcn")
            frame_color = None
            if cfilm_stats is not None and ranks.get("cfilm") == 1:
                frame_color = "#d62728"
            elif cfilm_stats is not None and hgcn_stats is not None and cfilm_stats.mean > hgcn_stats.mean:
                frame_color = "#1f77b4"
            if frame_color is not None:
                for spine in ax.spines.values():
                    spine.set_color(frame_color)
                    spine.set_linewidth(2.0)
            for k, model in enumerate(models):
                stats = panel_values.get(model)
                if stats is not None:
                    ax.bar(k, stats.mean, yerr=stats.std, capsize=2, color=MODEL_COLORS[model], edgecolor="black", linewidth=0.3)
                    ax.annotate(
                        str(ranks[model]),
                        xy=(k, stats.mean + stats.std),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )
            ax.set_title(f"{study.removeprefix('tcga_').upper()} / {subset}", fontsize=8)
            ax.set_xticks(range(4), ["Cfilm", "zero", "mean", "HCGN"], rotation=55, ha="right", fontsize=6)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
            if j == 0:
                ax.set_ylabel("C-index")
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def collect(results_root: Path, output_root: Path, hparams_path: Path) -> list[dict[str, object]]:
    all_hparams_rows = read_hparams(hparams_path)
    hparams_rows = [row for row in all_hparams_rows if table_enabled(row, "T2")]
    aggregate: list[dict[str, object]] = []

    for hparams in hparams_rows:
        run_id = hparams["run_id"].strip()
        summary_rows: list[dict[str, object]] = []
        delta_rows: list[dict[str, object]] = []
        ranking_rows: list[dict[str, object]] = []
        figure_values: dict[tuple[str, str, str], object] = {}
        cfilm_values: list[float] = []
        best_deltas: list[float] = []
        named_deltas = {name: [] for name in BASELINES}
        wins = {name: 0 for name in BASELINES}
        wins_all = 0
        completed = 0

        for study in STUDIES:
            run_name = f"{study}__{RUN_SUFFIX}"
            cfilm_dir = cfilm_model_dir(results_root, run_id, study)
            baseline_root = results_root / "Table2_Baselines" / run_name
            for subset in SUBSETS:
                model_stats = {"cfilm": load_subset_stats(cfilm_dir, subset)}
                model_stats.update(
                    {
                        name: load_subset_stats(baseline_root / model_dir, subset)
                        for name, model_dir in BASELINES.items()
                    }
                )
                # Rank the Cfilm model and the three fixed baselines under
                # the same study/subset conditions. Missing values are
                # excluded, so a rank is emitted only for comparable cells.
                ranked_models = sorted(
                    (
                        (model, stats.mean)
                        for model, stats in model_stats.items()
                        if stats is not None
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                ranks = {model: rank for rank, (model, _) in enumerate(ranked_models, start=1)}
                hgcn_stats = model_stats.get("hgcn")
                for model, stats in model_stats.items():
                    if stats is not None:
                        figure_values[(study, subset, model)] = stats
                    value = stats.mean if stats is not None else None
                    delta_hgcn = value - hgcn_stats.mean if stats is not None and hgcn_stats is not None else None
                    ranking_rows.append(
                        {
                            "run_id": run_id,
                            "study": study,
                            "subset": subset,
                            "model": model,
                            "cindex_mean": number(value),
                            "rank_1_to_4": ranks.get(model, ""),
                            "delta_vs_hgcn": number(delta_hgcn),
                            "vs_hgcn": (
                                "↑" if delta_hgcn is not None and delta_hgcn > 1e-12
                                else "↓" if delta_hgcn is not None and delta_hgcn < -1e-12
                                else ""
                            ),
                            "rank_display": (
                                f"{ranks[model]}{'↑' if delta_hgcn is not None and delta_hgcn > 1e-12 else '↓' if delta_hgcn is not None and delta_hgcn < -1e-12 else ''}"
                                if model in ranks else ""
                            ),
                            "status": "complete" if value is not None and hgcn_stats is not None else "incomplete",
                        }
                    )
                for model, stats in model_stats.items():
                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "study": study,
                            "subset": subset,
                            "model": model,
                            "cindex_mean": number(stats.mean if stats else None),
                            "cindex_std": number(stats.std if stats else None),
                            "folds": stats.count if stats else 0,
                            "status": "complete" if stats else "missing",
                        }
                    )

                if subset not in MISSING_SUBSETS:
                    continue
                cfilm = model_stats["cfilm"]
                baseline_stats = {name: model_stats[name] for name in BASELINES}
                comparable = cfilm is not None and all(value is not None for value in baseline_stats.values())
                row: dict[str, object] = {
                    "run_id": run_id,
                    "study": study,
                    "subset": subset,
                    "cfilm_mean": number(cfilm.mean if cfilm else None),
                    "status": "complete" if comparable else "incomplete",
                }
                for name, stats in baseline_stats.items():
                    delta = cfilm.mean - stats.mean if cfilm and stats else None
                    row[f"{name}_mean"] = number(stats.mean if stats else None)
                    row[f"delta_vs_{name}"] = number(delta)
                    row[f"rank_{name}"] = ranks.get(name, "")
                    row[f"vs_hgcn_{name}"] = (
                        "↑" if stats is not None and hgcn_stats is not None and stats.mean > hgcn_stats.mean + 1e-12
                        else "↓" if stats is not None and hgcn_stats is not None and stats.mean < hgcn_stats.mean - 1e-12
                        else ""
                    )
                    if comparable and delta is not None:
                        named_deltas[name].append(delta)
                        if delta > 0:
                            wins[name] += 1
                if comparable and cfilm is not None:
                    best_name, best_stats = max(
                        baseline_stats.items(), key=lambda item: item[1].mean  # type: ignore[union-attr]
                    )
                    best_delta = cfilm.mean - best_stats.mean  # type: ignore[union-attr]
                    row["best_baseline"] = best_name
                    row["best_baseline_mean"] = number(best_stats.mean)  # type: ignore[union-attr]
                    row["delta_vs_best"] = number(best_delta)
                    cfilm_values.append(cfilm.mean)
                    best_deltas.append(best_delta)
                    completed += 1
                    if best_delta > 0:
                        wins_all += 1
                else:
                    row.update({"best_baseline": "", "best_baseline_mean": "", "delta_vs_best": ""})
                row["rank_cfilm"] = ranks.get("cfilm", "")
                row["vs_hgcn_cfilm"] = (
                    "↑" if cfilm is not None and hgcn_stats is not None and cfilm.mean > hgcn_stats.mean + 1e-12
                    else "↓" if cfilm is not None and hgcn_stats is not None and cfilm.mean < hgcn_stats.mean - 1e-12
                    else ""
                )
                delta_rows.append(row)

        run_dir = output_root / "Table2" / run_id
        write_csv(
            run_dir / "summary.csv",
            summary_rows,
            ["run_id", "study", "subset", "model", "cindex_mean", "cindex_std", "folds", "status"],
        )
        delta_fields = ["run_id", "study", "subset", "cfilm_mean"]
        for name in BASELINES:
            delta_fields.extend([f"{name}_mean", f"delta_vs_{name}", f"rank_{name}", f"vs_hgcn_{name}"])
        delta_fields.extend(["rank_cfilm", "vs_hgcn_cfilm"])
        delta_fields.extend(["best_baseline", "best_baseline_mean", "delta_vs_best", "status"])
        write_csv(
            run_dir / "baseline_ranking.csv",
            ranking_rows,
            ["run_id", "study", "subset", "model", "cindex_mean", "rank_1_to_4", "rank_display", "delta_vs_hgcn", "vs_hgcn", "status"],
        )
        write_ranking_figure(run_dir / "baseline_ranking.png", figure_values)

        expected = len(STUDIES) * len(MISSING_SUBSETS)
        cfilm_spread = mean_std(cfilm_values)
        aggregate.append(
            with_hparams(
                {
                    "completed_cells": completed,
                    "expected_cells": expected,
                    "mean_cfilm": cfilm_spread.mean if cfilm_spread else None,
                    "across_cell_std": cfilm_spread.std if cfilm_spread else None,
                    "mean_delta_zero": mean_std(named_deltas["zero"]).mean if named_deltas["zero"] else None,
                    "mean_delta_mean": mean_std(named_deltas["mean"]).mean if named_deltas["mean"] else None,
                    "mean_delta_hgcn": mean_std(named_deltas["hgcn"]).mean if named_deltas["hgcn"] else None,
                    "mean_delta_best": mean_std(best_deltas).mean if best_deltas else None,
                    "worst_delta_best": min(best_deltas) if best_deltas else None,
                    "wins_vs_zero": wins["zero"],
                    "wins_vs_mean": wins["mean"],
                    "wins_vs_hgcn": wins["hgcn"],
                    "wins_vs_all": wins_all,
                    "status": "complete" if completed == expected else "incomplete",
                },
                hparams,
            )
        )

    assign_ranks(
        aggregate,
        eligible=lambda row: row["status"] == "complete",
        sort_key=lambda row: (-float(row["mean_delta_best"]), -int(row["wins_vs_all"]), float(row["across_cell_std"])),
    )
    fields = hparam_columns(hparams_rows) + [
        "completed_cells", "expected_cells", "mean_cfilm", "across_cell_std",
        "mean_delta_zero", "mean_delta_mean", "mean_delta_hgcn", "mean_delta_best",
        "worst_delta_best", "wins_vs_zero", "wins_vs_mean", "wins_vs_hgcn",
        "wins_vs_all", "status", "rank",
    ]
    ranked = sorted(aggregate, key=lambda row: (row["rank"] == "", row["rank"] or 10**9, row["run_id"]))
    write_csv(output_root / "Table2" / "parameter_ranking.csv", ranked, fields)
    return aggregate


def main() -> None:
    args = parse_io_args("Collect Table2 Cfilm hyperparameter results")
    collect(args.results_root, args.output_root, args.hparams)
    print(f"Collected Table2 results into {args.output_root / 'Table2'}")


if __name__ == "__main__":
    main()
