"""
Offline forward-only Breslow IBS/AUC for cox_surv models (clinic_cox,
mosaic_surv, 原名 Cfilm)。

No training, no gradients: loads the saved checkpoint and runs ONE eval
forward pass over the TRAIN split to collect train risks, then Breslow
baseline H0 -> per-test-patient survival functions -> IBS / AUC@24 / AUC@60
with the shared conventions in utils/survival_metrics.py.

The exact training args are reconstructed from the `[CMD]` line that run.sh
wrote into the model dir's run.log, so model construction matches the
original run bit-for-bit.

Usage (SurvPGC env, GPU):
  python results_display/scripts/Table1_CoxBreslow_Forward.py \
      --model-dir <model dir with effective_config.txt + run.log + checkpoints> \
      --out <output csv (appended)> \
      [--gpu 4] [--folds 0,1,2,3,4]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # dataset code reads ./datasets_csv/... relative to cwd

from sksurv.util import Surv

from utils.survival_metrics import (
    AUC_LANDMARK_MONTHS,
    IBS_GRID_MONTHS,
    MIN_EVENTS_FOR_AUC,
    breslow_survival,
    compute_ibs,
    compute_landmark_aucs,
)


def load_effective_config(model_dir: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    path = model_dir / "effective_config.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        config[key.strip()] = value.strip().strip('"').strip("'")
    return config


def build_args_from_cmd_line(model_dir: Path):
    """Reconstruct the original training args from run.log's [CMD] line."""
    run_log = model_dir / "run.log"
    if not run_log.is_file():
        raise FileNotFoundError(f"missing {run_log}")
    cmd_line = None
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[CMD] "):
            cmd_line = line[len("[CMD] "):].strip()
            break
    if not cmd_line:
        raise ValueError(f"no [CMD] line in {run_log}")

    argv = [
        token for token in shlex.split(cmd_line)
        if token not in ("python", "-u", "main.py")
    ]
    sys.argv = [str(Path(__file__).name)] + argv
    from utils.process_args import _process_args

    args = _process_args()
    args.results_dir = str(model_dir.parents[1])  # exp dir, not the run dir
    return args


def load_test_pkl(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pickle

    with pkl_path.open("rb") as handle:
        patients = pickle.load(handle)
    patient_ids = list(patients)
    times = np.array([patients[i]["time"] for i in patient_ids], dtype=np.float64)
    censorships = np.array(
        [patients[i]["censorship"] for i in patient_ids], dtype=np.float64
    )
    risks = np.array([patients[i]["risk"] for i in patient_ids], dtype=np.float64)
    return times, censorships, risks


def load_csv_cindex_by_fold(csv_path: Path) -> dict[int, float]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fold_to_cindex: dict[int, float] = {}
    for row in rows:
        raw = row.get("test_cindex", "")
        if raw in ("", None):
            continue
        try:
            fold = int(float(row[None] if None in row else row.get("", 0)))
            fold_to_cindex[fold] = float(raw)
        except (TypeError, ValueError):
            continue
    return fold_to_cindex


def process_model_dir(model_dir: Path, folds: list[int], out_csv: Path) -> None:
    from utils.core_utils import (
        _build_survival_dataset_factory,
        _collect_train_survival_risks,
        _get_split_loader,
        _init_loss_function,
        _init_model,
        _load_model_checkpoint,
    )

    config = load_effective_config(model_dir)
    study = config.get("STUDY", "unknown")
    args = build_args_from_cmd_line(model_dir)
    if args.bag_loss != "cox_surv":
        print(f"[SKIP] {model_dir}: bag_loss={args.bag_loss} (not cox)")
        return

    print(f"[FORWARD] {study} folds={folds}")
    dataset_factory = _build_survival_dataset_factory(args)
    args.dataset_factory = dataset_factory  # used by return_splits

    # censoring cohort: full dataset (train+val+test), project convention
    patients = dataset_factory.patients_df
    time_arr = patients[args.label_col].to_numpy()
    cens_arr = patients[dataset_factory.censorship_var].to_numpy()
    keep = time_arr >= 0.0
    cohort_surv = Surv.from_arrays(
        event=(1.0 - cens_arr[keep]) > 0.5, time=time_arr[keep]
    )

    csv_path = model_dir / "test_result.csv"
    fold_to_cindex = load_csv_cindex_by_fold(csv_path) if csv_path.is_file() else {}

    rows: list[dict[str, object]] = []
    for fold in folds:
        ckpt = model_dir / f"s_{fold}_checkpoint.pt"
        if not ckpt.is_file():
            print(f"[WARN] missing {ckpt}")
            continue
        pkl = model_dir / f"split_{fold}_results.pkl"
        if not pkl.is_file():
            print(f"[WARN] missing {pkl}")
            continue

        datasets = dataset_factory.return_splits(
            args,
            csv_path=os.path.join(args.split_dir, f"splits_{fold}.csv"),
            fold=fold,
        )
        train_split, _, test_split = datasets
        loss_fn = _init_loss_function(args)
        model = _init_model(args, current_fold=fold)
        model = _load_model_checkpoint(model, ckpt, args)
        train_loader = _get_split_loader(
            args, train_split, training=False, testing=False,
            weighted=False, batch_size=1, disable_cox_batch_override=True,
        )
        train_times, train_events, train_risks = _collect_train_survival_risks(
            args, model, train_loader, loss_fn
        )

        # Test risks: prefer the saved pkl; some runs (clinic_cox) saved only
        # one patient per fold, in which case run a test forward pass instead.
        pkl_times, pkl_cens, pkl_risks = load_test_pkl(pkl)
        if len(pkl_times) >= 2:
            test_times, test_cens, test_risks = pkl_times, pkl_cens, pkl_risks
        else:
            print(f"[FOLD {fold}] test pkl incomplete ({len(pkl_times)} rows) -> test forward")
            test_loader = _get_split_loader(
                args, test_split, training=False, testing=False,
                weighted=False, batch_size=1, disable_cox_batch_override=True,
            )
            test_times, test_events_unused, test_risks = _collect_train_survival_risks(
                args, model, test_loader, loss_fn
            )
            test_cens = (~test_events_unused).astype(np.float64)
        del model
        valid = test_times >= 0.0
        test_times, test_cens, test_risks = (
            test_times[valid], test_cens[valid], test_risks[valid]
        )
        test_surv = Surv.from_arrays(
            event=(1.0 - test_cens) > 0.5, time=test_times
        )
        test_min, test_max = float(test_times.min()), float(test_times.max())

        row: dict[str, object] = {
            "study": study,
            "model": model_dir.name,
            "fold": fold,
        }
        recorded = fold_to_cindex.get(fold)
        if recorded is not None:
            from sksurv.metrics import concordance_index_censored

            recomputed = float(
                concordance_index_censored(
                    (1.0 - test_cens) > 0.5, test_times, test_risks,
                    tied_tol=1e-08,
                )[0]
            )
            row["cindex_abs_diff"] = abs(recomputed - recorded)
        else:
            row["cindex_abs_diff"] = math.nan

        estimate_grid = breslow_survival(
            train_times, train_events, train_risks, test_risks, IBS_GRID_MONTHS
        )
        estimate_landmarks = 1.0 - breslow_survival(
            train_times, train_events, train_risks, test_risks,
            np.asarray(AUC_LANDMARK_MONTHS, dtype=np.float64),
        )

        if test_min <= 1.0 and test_max >= 60.0:
            try:
                ibs = compute_ibs(cohort_surv, test_surv, estimate_grid)
                row["ibs"] = ibs if math.isfinite(ibs) else math.nan
            except Exception as exc:
                print(f"[WARN] IBS failed fold {fold}: {exc}")
                row["ibs"] = math.nan
        else:
            row["ibs"] = math.nan

        available = []
        for landmark in AUC_LANDMARK_MONTHS:
            n_events = int(
                ((test_cens < 1.0) & (test_times <= landmark)).sum()
            )
            row[f"n_events_{int(landmark)}"] = n_events
            if (
                test_min <= landmark <= test_max
                and n_events >= MIN_EVENTS_FOR_AUC.get(landmark, 5)
            ):
                available.append(landmark)
        for landmark in AUC_LANDMARK_MONTHS:
            row[f"auc{int(landmark)}"] = math.nan
        if available:
            try:
                idx = [AUC_LANDMARK_MONTHS.index(lm) for lm in available]
                aucs = compute_landmark_aucs(
                    cohort_surv, test_surv, estimate_landmarks[:, idx], available
                )
                for landmark, value in zip(available, aucs):
                    row[f"auc{int(landmark)}"] = (
                        value if math.isfinite(value) else math.nan
                    )
            except Exception as exc:
                print(f"[WARN] AUC failed fold {fold}: {exc}")

        rows.append(row)
        print(
            f"[FOLD {fold}] ibs={row.get('ibs')} auc24={row.get('auc24')} "
            f"auc60={row.get('auc60')} cindex_diff={row.get('cindex_abs_diff')}"
        )

    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "study", "model", "fold", "ibs", "auc24", "auc60",
            "n_events_24", "n_events_60", "cindex_abs_diff",
        ]
        write_header = not out_csv.is_file()
        with out_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[WRITE] {out_csv} (+{len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward-only Breslow IBS/AUC for cox models"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    folds = [int(part) for part in args.folds.split(",") if part.strip()]
    process_model_dir(args.model_dir, folds, args.out)


if __name__ == "__main__":
    main()
