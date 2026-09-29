#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import table1
import table2
import table3
from common import (
    assign_ranks,
    default_hparams_path,
    hparam_columns,
    project_root,
    read_hparams,
    table_enabled,
    write_csv,
)


def build_final_ranking(
    output_root: Path,
    hparams_path: Path,
    table1_rows: list[dict[str, object]],
    table2_rows: list[dict[str, object]],
    table3_rows: list[dict[str, object]],
) -> None:
    hparams_rows = read_hparams(hparams_path)
    maps = [
        {str(row["run_id"]): row for row in rows}
        for rows in (table1_rows, table2_rows, table3_rows)
    ]
    final_rows: list[dict[str, object]] = []

    for hparams in hparams_rows:
        run_id = hparams["run_id"].strip()
        first, second, third = (mapping.get(run_id) for mapping in maps)
        selected = [table_enabled(hparams, column) for column in ("T1", "T2", "T3")]
        complete = all(
            is_selected and row is not None and row["status"] == "complete"
            for is_selected, row in zip(selected, (first, second, third))
        )
        table1_pass = first is not None and first["all_pass"] == "yes"
        if not all(selected):
            status = "not_selected"
        elif not complete:
            status = "incomplete"
        elif not table1_pass:
            status = "table1_failed"
        else:
            status = "eligible"
        final_rows.append(
            {
                **hparams,
                "table1_all_pass": first["all_pass"] if first else "",
                "table1_passed_studies": first["passed_studies"] if first else "",
                "table1_mean_cindex": first["mean_cindex"] if first else "",
                "table1_min_margin": first["min_margin"] if first else "",
                "table2_mean_delta_best": second["mean_delta_best"] if second else "",
                "table2_wins_vs_all": second["wins_vs_all"] if second else "",
                "table3_wins": third["wins"] if third else "",
                "table3_win_rate": third["win_rate"] if third else "",
                "table3_mean_delta_hgcn": third["mean_delta_hgcn"] if third else "",
                "table3_mean_curve_std": third["mean_curve_std"] if third else "",
                "table1_status": first["status"] if first else "not_selected",
                "table2_status": second["status"] if second else "not_selected",
                "table3_status": third["status"] if third else "not_selected",
                "status": status,
            }
        )

    assign_ranks(
        final_rows,
        eligible=lambda row: row["status"] == "eligible",
        sort_key=lambda row: (
            -int(row["table3_wins"]),
            -float(row["table2_mean_delta_best"]),
            float(row["table3_mean_curve_std"]),
            -float(row["table1_mean_cindex"]),
        ),
    )
    fields = hparam_columns(hparams_rows) + [
        "table1_all_pass", "table1_passed_studies", "table1_mean_cindex",
        "table1_min_margin", "table2_mean_delta_best", "table2_wins_vs_all",
        "table3_wins", "table3_win_rate", "table3_mean_delta_hgcn",
        "table3_mean_curve_std", "table1_status", "table2_status", "table3_status",
        "status", "rank",
    ]
    final_rows.sort(key=lambda row: (row["rank"] == "", row["rank"] or 10**9, row["run_id"]))
    write_csv(output_root / "final_parameter_ranking.csv", final_rows, fields)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Collect and evaluate all Cfilm hyperparameter results")
    parser.add_argument("--results-root", type=Path, default=root / "results")
    parser.add_argument("--output-root", type=Path, default=root / "results_display/Cfilm_Hparam_Eval")
    parser.add_argument("--hparams", type=Path, default=default_hparams_path())
    args = parser.parse_args()

    table1_rows = table1.collect(args.results_root, args.output_root, args.hparams)
    table2_rows = table2.collect(args.results_root, args.output_root, args.hparams)
    table3_rows = table3.collect(args.results_root, args.output_root, args.hparams)
    build_final_ranking(
        args.output_root,
        args.hparams,
        table1_rows,
        table2_rows,
        table3_rows,
    )
    print(f"Collected Cfilm hyperparameter results into {args.output_root}")


if __name__ == "__main__":
    main()
