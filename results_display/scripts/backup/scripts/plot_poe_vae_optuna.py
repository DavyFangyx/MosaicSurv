"""
Summarize and plot LIHC POE-VAE Optuna trials.

Input structure:
    results/poe_vae_optuna_optuna/
        lihc_poeA_trial_0000/survtri_poe_vae/val_result_fold0.csv
        lihc_poeB_trial_0000/survtri_poe_vae/val_result_fold0.csv
        lihc_poeC_trial_0000/survtri_poe_vae/val_result_fold0.csv

Outputs:
    results_display/poe_vae_optuna_optuna/summary/poe_vae_optuna_trial_cindex.csv
    results_display/poe_vae_optuna_optuna/summary/poe_vae_optuna_model_summary.csv
    results_display/poe_vae_optuna_optuna/figures/model_A_cindex_trials.png
    results_display/poe_vae_optuna_optuna/figures/model_B_cindex_trials.png
    results_display/poe_vae_optuna_optuna/figures/model_C_cindex_trials.png
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRIAL_PATTERN = re.compile(r"lihc_poe(?P<model>[ABC])_trial_(?P<trial>\d{4})$")
MODELS = ("A", "B", "C")
MODEL_COLORS = {
    "A": "#1768AC",
    "B": "#D1495B",
    "C": "#5B8E7D",
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_trial_values(trial_dir: Path) -> tuple[float, int, str] | None:
    result_dir = trial_dir / "survtri_poe_vae"
    csv_paths = sorted(result_dir.glob("val_result_fold*.csv"))
    values: list[float] = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception as exc:
            print(f"[WARN] Cannot read {csv_path}: {exc}")
            continue

        if "val_cindex" not in df.columns:
            print(f"[WARN] Missing val_cindex in {csv_path}")
            continue

        current = pd.to_numeric(df["val_cindex"], errors="coerce").dropna().tolist()
        values.extend(float(value) for value in current)

    if not values:
        return None

    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), int(array.size), "val_result_fold*.csv"


def collect_trial_rows(results_dir: Path, max_trials: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    found_trials: set[tuple[str, int]] = set()

    for trial_dir in sorted(results_dir.iterdir()):
        if not trial_dir.is_dir():
            continue

        match = TRIAL_PATTERN.fullmatch(trial_dir.name)
        if match is None:
            continue

        model = match.group("model")
        trial_index = int(match.group("trial"))
        if trial_index >= max_trials:
            print(f"[SKIP] {trial_dir.name}: outside Trial 1-{max_trials}")
            continue

        found_trials.add((model, trial_index))
        values = read_trial_values(trial_dir)
        if values is None:
            print(f"[WARN] Missing val_cindex: {trial_dir}")
            rows.append(
                {
                    "model": model,
                    "trial_index": trial_index,
                    "trial": trial_index + 1,
                    "trial_dir": trial_dir.name,
                    "cindex": np.nan,
                    "value_count": 0,
                    "source": "",
                    "status": "missing",
                }
            )
            continue

        mean, value_count, source = values
        rows.append(
            {
                "model": model,
                "trial_index": trial_index,
                "trial": trial_index + 1,
                "trial_dir": trial_dir.name,
                "cindex": mean,
                "value_count": value_count,
                "source": source,
                "status": "ok",
            }
        )

    for model in MODELS:
        for trial_index in range(max_trials):
            if (model, trial_index) in found_trials:
                continue
            rows.append(
                {
                    "model": model,
                    "trial_index": trial_index,
                    "trial": trial_index + 1,
                    "trial_dir": f"lihc_poe{model}_trial_{trial_index:04d}",
                    "cindex": np.nan,
                    "value_count": 0,
                    "source": "",
                    "status": "missing_directory",
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(by=["model", "trial_index"], kind="stable").reset_index(drop=True)


def add_model_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    stats = (
        result.groupby("model", as_index=False)
        .agg(
            available_trials=("cindex", lambda values: int(values.notna().sum())),
            mean_cindex=("cindex", "mean"),
            best_cindex=("cindex", "max"),
        )
    )

    best_trial = (
        result.dropna(subset=["cindex"])
        .sort_values(by=["model", "cindex", "trial"], ascending=[True, False, True], kind="stable")
        .groupby("model", as_index=False)
        .first()[["model", "trial", "trial_dir"]]
        .rename(columns={"trial": "best_trial", "trial_dir": "best_trial_dir"})
    )
    stats = stats.merge(best_trial, on="model", how="left")

    result = result.merge(
        stats[
            [
                "model",
                "mean_cindex",
                "available_trials",
            ]
        ],
        on="model",
        how="left",
    )
    return result, stats


def save_plot(model_df: pd.DataFrame, stats_row: pd.Series, output_path: Path, max_trials: int) -> None:
    x = model_df["trial"].to_numpy(dtype=float)
    y = model_df["cindex"].to_numpy(dtype=float)
    available = np.isfinite(y)
    color = MODEL_COLORS.get(str(stats_row["model"]), "#355C7D")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        x[available],
        y[available],
        color=color,
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Trial c-index",
    )

    for trial, value in zip(x[available], y[available]):
        ax.text(trial, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_title(
        f"LIHC | Model {stats_row['model']} | Optuna Trial C-index",
        fontsize=14,
        loc="left",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation C-index")
    ax.set_xlim(0.5, max_trials + 0.5)
    ax.set_xticks(np.arange(1, max_trials + 1))
    ax.set_ylim(
        max(0.0, np.nanmin(y[available]) - 0.04),
        min(1.0, np.nanmax(y[available]) + 0.06),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Plot POE-VAE Optuna trial c-index curves")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=project_root / "results/poe_vae_optuna_optuna",
        help="POE-VAE Optuna result directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "results_display/poe_vae_optuna_optuna",
        help="Output directory",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=10,
        help="Number of trials to plot, starting from trial_0000",
    )
    args = parser.parse_args()

    summary_dir = args.output_dir / "summary"
    figure_dir = args.output_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = collect_trial_rows(args.results_dir, args.max_trials)
    if df.empty:
        raise SystemExit(f"No trial results found under {args.results_dir}")

    result_df, stats_df = add_model_statistics(df)
    result_df.to_csv(summary_dir / "poe_vae_optuna_trial_cindex.csv", index=False)
    stats_df.to_csv(summary_dir / "poe_vae_optuna_model_summary.csv", index=False)

    for model in MODELS:
        model_df = result_df[result_df["model"] == model].sort_values("trial_index")
        stats_row = stats_df[stats_df["model"] == model]
        if model_df.empty or stats_row.empty or stats_row["available_trials"].iloc[0] == 0:
            print(f"[SKIP] Model {model}: no usable cindex values")
            continue

        output_path = figure_dir / f"model_{model}_cindex_trials.png"
        save_plot(model_df, stats_row.iloc[0], output_path, args.max_trials)
        print(f"[WRITE] {output_path.relative_to(project_root)}")

    print(f"[WRITE] {summary_dir.relative_to(project_root)}")
    print("Done.")


if __name__ == "__main__":
    main()
