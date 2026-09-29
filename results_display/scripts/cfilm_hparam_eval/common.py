from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STUDIES = ["tcga_brca", "tcga_coad", "tcga_kirc", "tcga_kirp", "tcga_lihc"]
STUDY_LABELS = {
    "tcga_brca": "BRCA",
    "tcga_coad": "COAD",
    "tcga_kirc": "KIRC",
    "tcga_kirp": "KIRP",
    "tcga_lihc": "LIHC",
}
SUBSETS = ["P", "C", "G", "PC", "PG", "CG", "PCG"]
MISSING_SUBSETS = ["C", "P", "G", "PC", "CG", "PG"]
MISSING_PATTERNS = [
    "p60_0_0",
    "p0_60_0",
    "p0_0_60",
    "p30_30_0",
    "p30_0_30",
    "p0_30_30",
    "p20_20_20",
]
CFILM_MODEL = "survtri_poe_vae_c_film"


@dataclass(frozen=True)
class Stats:
    mean: float
    std: float
    count: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_hparams_path() -> Path:
    return (
        project_root()
        / "configs/z_exp_gen/Cfilm_Hparam_Eval/cfilm_table1_hparams.csv"
    )


def default_table1_summary_path() -> Path:
    return (
        project_root()
        / "results_display/Table1_Cindex_Main/L0Test After fixed modal missing/summary_5datasets.csv"
    )


def read_hparams(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No hyperparameter rows found in {path}")
    run_ids = [row.get("run_id", "").strip() for row in rows]
    if any(not value for value in run_ids):
        raise ValueError(f"Every hyperparameter row must have a run_id in {path}")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"run_id values must be unique in {path}")
    return rows


def table_enabled(row: dict[str, str], column: str) -> bool:
    value = row.get(column, "").strip().lower()
    if not value:
        return False
    if value in {"1", "true", "yes", "on", column.lower()}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {column} switch {value!r} for run_id={row.get('run_id', '')}"
    )


def mean_std(values: Iterable[float]) -> Stats | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return Stats(mean=mean, std=math.sqrt(variance), count=len(clean))


def load_cindex_stats(path: Path, subset: str | None = None) -> Stats | None:
    if not path.is_file():
        return None
    values: list[float] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if subset is not None and str(row.get("subset", "")).strip() != subset:
                    continue
                raw = row.get("test_cindex", "")
                if raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
    except (OSError, csv.Error):
        return None
    return mean_std(values)


def load_subset_stats(model_dir: Path, subset: str) -> Stats | None:
    summary = model_dir / "eval_subsets" / "summary_long.csv"
    stats = load_cindex_stats(summary, subset=subset)
    if stats is not None:
        return stats
    stats = load_cindex_stats(model_dir / "eval_subsets" / subset / "test_result.csv")
    if stats is not None:
        return stats
    if subset == "PCG":
        return load_cindex_stats(model_dir / "test_result.csv")
    return None


def number(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def display_stats(stats: Stats | None) -> str:
    if stats is None:
        return "-"
    return f"{stats.mean:.4f} +/- {stats.std:.4f}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def hparam_columns(rows: list[dict[str, str]]) -> list[str]:
    return list(rows[0]) if rows else ["run_id"]


def with_hparams(summary: dict[str, object], hparams: dict[str, str]) -> dict[str, object]:
    return {**hparams, **summary}


def assign_ranks(
    rows: list[dict[str, object]],
    eligible,
    sort_key,
) -> None:
    ranked = sorted((row for row in rows if eligible(row)), key=sort_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    for row in rows:
        row.setdefault("rank", "")


def parse_io_args(description: str):
    root = project_root()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results-root", type=Path, default=root / "results")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "results_display/Cfilm_Hparam_Eval",
    )
    parser.add_argument("--hparams", type=Path, default=default_hparams_path())
    parser.add_argument(
        "--classic-summary",
        type=Path,
        default=default_table1_summary_path(),
        help="Classic five-dataset Table1 CSV used for Cfilm ranks",
    )
    return parser.parse_args()
