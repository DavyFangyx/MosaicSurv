from __future__ import annotations

from pathlib import Path


def build_results_root(args, model_folder: str) -> Path:
    base = Path(getattr(args, "results_dir", "./results"))
    exp_group = getattr(args, "exp_group", "default")
    run_name = getattr(args, "run_name", "default")
    root = base / exp_group / run_name / model_folder
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_seed_dir(root: Path, seed: int | None = None) -> Path:
    if seed is None:
        return root
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def build_fold_dir(root: Path, fold: int | None = None) -> Path:
    if fold is None:
        return root
    fold_dir = root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def write_experiment_file(root: Path, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "experiment.txt").open("w", encoding="utf-8") as handle:
        for key, value in payload.items():
            handle.write(f"{key}: {value}\n")
