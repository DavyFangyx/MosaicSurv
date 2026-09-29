"""Summarize Optuna C-film / POE tune logs without reading the whole file by hand.

Accepts either the full training log from bg_tune.sh, or the compact analysis log
written to results/optuna/<study_name>.log.

Example:
    python results_display/scripts/summarize_optuna_tune_log.py gpu1_optunaCfilm.log
    python results_display/scripts/summarize_optuna_tune_log.py results/optuna/<study_name>.log
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


TRIAL_START_RE = re.compile(
    r"^\[optuna\] trial (?P<trial>\d+) studies=(?P<studies>\[[^\]]*\]) "
    r"batch_size=(?P<batch_size>\S+) batch_size_stage1=(?P<batch_size_stage1>\S+) "
    r"sampled=(?P<sampled>\{.*\})\s*$"
)
STUDY_MEAN_RE = re.compile(
    r"^\[optuna\] trial (?P<trial>\d+) study (?P<study>\S+) "
    r"mean_val_cindex=(?P<mean>[0-9.]+) folds=(?P<folds>\[[^\]]*\])\s*$"
)
PRUNE_RE = re.compile(
    r"^\[I .*\] Trial (?P<trial>\d+) pruned\. Pruned after (?P<study>\S+): "
    r"mean val_cindex=(?P<mean>[0-9.]+) < (?P<threshold>[0-9.]+)\s*$"
)
FINISH_RE = re.compile(
    r"^\[I .*\] Trial (?P<trial>\d+) finished with value: (?P<value>[-0-9.]+) .*"
)
SKIP_RE = re.compile(r"multi_pattern_surv_step skipped all 7 patterns")
ANALYSIS_RE = re.compile(r"^\[analysis\] (?P<body>.*)\s*$")


def parse_literal(text: str):
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def parse_analysis_fields(body: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    for token in body.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = parse_literal(value)
    return fields


def flatten_sampled(sampled) -> dict[str, object]:
    if not isinstance(sampled, dict):
        return {"sampled_raw": sampled}
    return dict(sampled)


def parse_log(path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    trials: dict[int, dict[str, object]] = {}
    counts = {
        "trial_starts": 0,
        "study_means": 0,
        "pruned": 0,
        "finished": 0,
        "cox_skip_warnings": 0,
        "epochs": 0,
        "fold_bests": 0,
    }

    def trial_row(trial: int) -> dict[str, object]:
        row = trials.get(trial)
        if row is None:
            row = {"trial": trial, "status": "running"}
            trials[trial] = row
        return row

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            analysis = ANALYSIS_RE.match(line)
            if analysis:
                fields = parse_analysis_fields(analysis.group("body"))
                event = fields.get("event")
                trial = fields.get("trial")
                if event == "cox_skip":
                    counts["cox_skip_warnings"] += 1
                    continue
                if trial is None:
                    continue
                row = trial_row(int(trial))
                if event == "trial_start":
                    counts["trial_starts"] += 1
                    row["studies"] = fields.get("studies")
                    if "batch_size" in fields:
                        row["batch_size"] = int(fields["batch_size"])
                    if "batch_size_stage1" in fields:
                        row["batch_size_stage1"] = int(fields["batch_size_stage1"])
                    sampled = fields.get("sampled")
                    if sampled is not None:
                        row.update(flatten_sampled(sampled))
                elif event == "study_mean":
                    counts["study_means"] += 1
                    study = fields.get("study")
                    if study is not None:
                        row[f"{study}_mean_val_cindex"] = float(fields.get("mean_val_cindex", 0.0))
                        row[f"{study}_folds"] = fields.get("folds")
                elif event == "fold_best":
                    counts["fold_bests"] += 1
                    study = fields.get("study")
                    fold = fields.get("fold")
                    if study is not None and fold is not None:
                        row[f"{study}_fold{fold}_best"] = fields.get("val_cindex")
                elif event == "epoch":
                    counts["epochs"] += 1
                    skip_count = fields.get("cox_skip_count") or 0
                    try:
                        counts["cox_skip_warnings"] += int(skip_count)
                    except (TypeError, ValueError):
                        pass
                    row["cox_skip_count"] = int(row.get("cox_skip_count") or 0) + int(skip_count or 0)
                    best = row.get("best_selection_metric")
                    selection = fields.get("selection_metric")
                    if selection is not None and (best is None or float(selection) >= float(best)):
                        row["best_selection_metric"] = selection
                        row["best_epoch"] = fields.get("epoch")
                        row["best_val_cindex"] = fields.get("val_cindex")
                        row["best_train_cindex"] = fields.get("train_cindex")
                        row["best_batch_size"] = fields.get("batch_size")
                elif event == "trial_pruned":
                    counts["pruned"] += 1
                    row["status"] = "pruned"
                    row["pruned_study"] = fields.get("study")
                    row["pruned_mean_val_cindex"] = fields.get("mean_val_cindex")
                    row["pruned_threshold"] = fields.get("min_cindex")
                elif event == "fold_test":
                    study = fields.get("study")
                    fold = fields.get("fold")
                    if study is not None and fold is not None:
                        row[f"{study}_fold{fold}_test_cindex"] = fields.get("test_cindex")
                        row[f"{study}_fold{fold}_test_cindex_weighted"] = fields.get("test_cindex_weighted")
                        row[f"{study}_fold{fold}_test_cindex_PGC"] = fields.get("test_cindex_PGC")
                        row[f"{study}_fold{fold}_val_selection"] = fields.get("val_selection")
                elif event == "trial_complete":
                    counts["finished"] += 1
                    row["status"] = "complete"
                    row["objective"] = fields.get("objective")
                continue

            if SKIP_RE.search(line):
                counts["cox_skip_warnings"] += 1
                continue

            start = TRIAL_START_RE.match(line)
            if start:
                counts["trial_starts"] += 1
                trial = int(start.group("trial"))
                row = trial_row(trial)
                row["studies"] = start.group("studies")
                row["batch_size"] = int(start.group("batch_size"))
                row["batch_size_stage1"] = int(start.group("batch_size_stage1"))
                sampled = parse_literal(start.group("sampled"))
                row.update(flatten_sampled(sampled))
                continue

            mean = STUDY_MEAN_RE.match(line)
            if mean:
                counts["study_means"] += 1
                trial = int(mean.group("trial"))
                row = trial_row(trial)
                study = mean.group("study")
                folds = parse_literal(mean.group("folds"))
                row[f"{study}_mean_val_cindex"] = float(mean.group("mean"))
                row[f"{study}_folds"] = folds
                continue

            pruned = PRUNE_RE.match(line)
            if pruned:
                counts["pruned"] += 1
                trial = int(pruned.group("trial"))
                row = trial_row(trial)
                row["status"] = "pruned"
                row["pruned_study"] = pruned.group("study")
                row["pruned_mean_val_cindex"] = float(pruned.group("mean"))
                row["pruned_threshold"] = float(pruned.group("threshold"))
                continue

            finished = FINISH_RE.match(line)
            if finished:
                counts["finished"] += 1
                trial = int(finished.group("trial"))
                row = trial_row(trial)
                row["status"] = "complete"
                row["objective"] = float(finished.group("value"))

    rows = [trials[key] for key in sorted(trials)]
    return rows, counts


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames: list[str] = []
    seen = set()
    preferred = [
        "trial",
        "status",
        "batch_size",
        "batch_size_stage1",
        "lr",
        "reg",
        "poe_surv_lambda",
        "poe_beta_target",
        "poe_modality_dropout",
        "poe_mmhid",
        "alphafix",
        "alphapgc",
        "tcga_kirp_mean_val_cindex",
        "tcga_kirp_folds",
        "best_selection_metric",
        "best_epoch",
        "best_val_cindex",
        "best_train_cindex",
        "pruned_study",
        "pruned_mean_val_cindex",
        "pruned_threshold",
        "objective",
        "studies",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]], counts: dict[str, int], top_n: int) -> None:
    kirp_rows = [row for row in rows if "tcga_kirp_mean_val_cindex" in row]
    kirp_vals = [float(row["tcga_kirp_mean_val_cindex"]) for row in kirp_rows]
    print(f"trials started: {counts['trial_starts']}")
    print(f"trials with KIRP mean: {len(kirp_rows)}")
    print(f"pruned: {counts['pruned']}")
    print(f"finished: {counts['finished']}")
    print(f"cox skip warnings: {counts['cox_skip_warnings']}")
    print(f"epoch rows: {counts.get('epochs', 0)}")
    print(f"fold bests: {counts.get('fold_bests', 0)}")
    if not kirp_vals:
        return
    print(
        "KIRP mean_val_cindex: "
        f"min={min(kirp_vals):.4f} max={max(kirp_vals):.4f} "
        f"avg={sum(kirp_vals) / len(kirp_vals):.4f}"
    )
    print(
        "KIRP thresholds: "
        f">=0.65 {sum(v >= 0.65 for v in kirp_vals)} | "
        f">=0.70 {sum(v >= 0.70 for v in kirp_vals)} | "
        f">=0.72 {sum(v >= 0.72 for v in kirp_vals)}"
    )
    print(f"top {top_n} KIRP trials:")
    ranked = sorted(kirp_rows, key=lambda row: float(row["tcga_kirp_mean_val_cindex"]), reverse=True)
    for row in ranked[:top_n]:
        print(
            f"  trial {row['trial']:>3}  "
            f"kirp={float(row['tcga_kirp_mean_val_cindex']):.4f}  "
            f"lr={row.get('lr')}  reg={row.get('reg')}  "
            f"lambda={row.get('poe_surv_lambda')}  beta={row.get('poe_beta_target')}  "
            f"drop={row.get('poe_modality_dropout')}  mmhid={row.get('poe_mmhid')}  "
            f"alpha={row.get('alphapgc')}  "
            f"bs={row.get('batch_size')}/{row.get('batch_size_stage1')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Optuna tune logs")
    parser.add_argument("log_file", type=Path, help="Optuna log file, e.g. gpu1_optunaCfilm.log")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional CSV path. Default: results_display/optuna_tune_logs/<log_stem>_summary.csv",
    )
    parser.add_argument("--top_n", type=int, default=10, help="How many top KIRP trials to print")
    args = parser.parse_args()

    log_file = args.log_file
    if not log_file.exists():
        raise SystemExit(f"log not found: {log_file}")

    rows, counts = parse_log(log_file)
    print_summary(rows, counts, args.top_n)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = Path("results_display/optuna_tune_logs") / f"{log_file.stem}_summary.csv"
    write_csv(rows, output_csv)
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
