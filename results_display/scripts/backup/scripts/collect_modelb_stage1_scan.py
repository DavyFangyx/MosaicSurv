"""
Collect and visualize Model-B stage1-scan results.

Inputs:
    Scan:
        results/poe_vae_test/model_B_STAGE1test/survtri_poe_vae__B/stage2_scan/fold_*/scan_summary.csv
    Non-scan baseline:
        results/poe_vae_test/model_B/survtri_poe_vae/val_result_fold*.csv

Outputs:
    results_display/ModelB_Stage1_scan/summary/modelb_stage1_scan_by_fold.csv
    results_display/ModelB_Stage1_scan/summary/modelb_stage1_scan_mean_std.csv
    results_display/ModelB_Stage1_scan/figures/modelb_stage1_scan_val_metrics.png
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_val_metrics(csv_path: Path) -> tuple[float, float]:
    df = pd.read_csv(csv_path, index_col=0)
    return float(df["val_cindex"].iloc[0]), float(df["val_cindex_ipcw"].iloc[0])


def collect_scan_rows(scan_root: Path, folds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in folds:
        scan_csv = scan_root / f"fold_{fold}" / "scan_summary.csv"
        if not scan_csv.exists():
            print(f"[WARN] Missing scan summary: {scan_csv}")
            continue

        df = pd.read_csv(scan_csv)
        if df.empty:
            continue

        for _, row in df.iterrows():
            results_dir = Path(row["results_dir"])
            val_csv = results_dir / f"val_result_fold{fold}.csv"
            if not val_csv.exists():
                print(f"[WARN] Missing val_result file: {val_csv}")
                continue

            val_cindex, val_cindex_ipcw = load_val_metrics(val_csv)
            rows.append(
                {
                    "fold": fold,
                    "group": "scan",
                    "stage1_epoch": int(row["stage1_epoch"]),
                    "val_cindex": val_cindex,
                    "val_cindex_ipcw": val_cindex_ipcw,
                }
            )

    return pd.DataFrame(rows)


def collect_non_scan_rows(non_scan_root: Path, folds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in folds:
        val_csv = non_scan_root / f"val_result_fold{fold}.csv"
        if not val_csv.exists():
            print(f"[WARN] Missing non-scan val file: {val_csv}")
            continue

        val_cindex, val_cindex_ipcw = load_val_metrics(val_csv)
        rows.append(
            {
                "fold": fold,
                "group": "non_scan",
                "stage1_epoch": pd.NA,
                "val_cindex": val_cindex,
                "val_cindex_ipcw": val_cindex_ipcw,
            }
        )

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["group", "stage1_epoch"], dropna=False, as_index=False)
        .agg(
            val_cindex_mean=("val_cindex", "mean"),
            val_cindex_std=("val_cindex", lambda x: float(np.std(x.to_numpy(dtype=float), ddof=0))),
            val_cindex_ipcw_mean=("val_cindex_ipcw", "mean"),
            val_cindex_ipcw_std=("val_cindex_ipcw", lambda x: float(np.std(x.to_numpy(dtype=float), ddof=0))),
            n_folds=("fold", "nunique"),
        )
    )

    summary["stage1_epoch"] = summary["stage1_epoch"].astype("Int64")
    summary["stage1_epoch_label"] = summary["stage1_epoch"].astype(str).replace("<NA>", "NULL")
    summary["sort_key"] = summary["stage1_epoch"].fillna(-1)
    summary = summary.sort_values(by=["group", "sort_key"], kind="stable").drop(columns=["sort_key"]).reset_index(drop=True)
    return summary


def _category_order(summary_df: pd.DataFrame) -> list[tuple[str, str]]:
    categories: list[tuple[str, str]] = [("non_scan", "NULL")]
    scan_epochs = (
        summary_df.loc[summary_df["group"] == "scan", "stage1_epoch"]
        .dropna()
        .astype(int)
        .sort_values()
        .tolist()
    )
    categories.extend(("scan", str(epoch)) for epoch in scan_epochs)
    return categories


def plot_metrics(by_fold_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    categories = _category_order(summary_df)
    x = np.arange(len(categories))
    colors = {"non_scan": "#4C78A8", "scan": "#D1495B"}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharex=True)
    metric_specs = [
        ("val_cindex", "val_cindex_mean", "val_cindex_std", "Validation C-index"),
        ("val_cindex_ipcw", "val_cindex_ipcw_mean", "val_cindex_ipcw_std", "Validation C-index IPCW"),
    ]

    for ax, (metric, mean_col, std_col, title) in zip(axes, metric_specs):
        means = []
        stds = []
        labels = []
        bar_colors = []
        for group, epoch_label in categories:
            labels.append("non_scan" if group == "non_scan" else f"stage1_{epoch_label}")
            bar_colors.append(colors[group])
            row = summary_df[
                (summary_df["group"] == group)
                & (summary_df["stage1_epoch_label"] == epoch_label)
            ]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row[mean_col].iloc[0]))
                stds.append(float(row[std_col].iloc[0]))

        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.82, edgecolor="black", linewidth=0.8)

        for idx, (group, epoch_label) in enumerate(categories):
            subset = by_fold_df[
                (by_fold_df["group"] == group)
                & (by_fold_df["stage1_epoch"].astype("string").fillna("NULL") == epoch_label)
            ]
            if subset.empty:
                continue
            values = subset[metric].to_numpy(dtype=float)
            jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), x[idx]) + jitter, values, color="black", s=24, zorder=3, alpha=0.7)

        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Model-B Stage1 Scan vs Non-scan Baseline (fold 0-3)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Model-B stage1 scan summaries")
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=Path("results/poe_vae_test/model_B_STAGE1test/survtri_poe_vae__B/stage2_scan"),
    )
    parser.add_argument(
        "--non-scan-root",
        type=Path,
        default=Path("results/poe_vae_test/model_B/survtri_poe_vae"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results_display/ModelB_Stage1_scan"),
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="fold ids used in the scan comparison",
    )
    args = parser.parse_args()

    project_root = project_root_from_script()
    scan_root = (project_root / args.scan_root).resolve()
    non_scan_root = (project_root / args.non_scan_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    summary_dir = output_root / "summary"
    figure_dir = output_root / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    scan_df = collect_scan_rows(scan_root, args.folds)
    non_scan_df = collect_non_scan_rows(non_scan_root, args.folds)
    by_fold_df = pd.concat([non_scan_df, scan_df], ignore_index=True)
    by_fold_df["stage1_epoch"] = by_fold_df["stage1_epoch"].astype("Int64")
    by_fold_df["stage1_epoch_label"] = by_fold_df["stage1_epoch"].astype(str).replace("<NA>", "NULL")
    by_fold_df = by_fold_df.sort_values(by=["group", "stage1_epoch_label", "fold"], kind="stable").reset_index(drop=True)

    summary_df = build_summary(by_fold_df)

    by_fold_path = summary_dir / "modelb_stage1_scan_by_fold.csv"
    summary_path = summary_dir / "modelb_stage1_scan_mean_std.csv"
    figure_path = figure_dir / "modelb_stage1_scan_val_metrics.png"

    by_fold_df.to_csv(by_fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_metrics(by_fold_df, summary_df, figure_path)

    print(f"[OK] by-fold table: {by_fold_path}")
    print(f"[OK] mean/std table: {summary_path}")
    print(f"[OK] figure: {figure_path}")


if __name__ == "__main__":
    main()
