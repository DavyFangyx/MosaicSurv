"""
Offline fold-level IBS / AUC@24 / AUC@60 computation for the Table 1 tables.

Pure computation module, no output files: the paper-table script
(results_display/scripts/Table1_Paper_Tables.py) calls collect_study_model_metrics
and folds the results directly into Table1a/1b and Appendix S1-S3.

Covers the nll_surv baselines whose hazard logits are saved in
split_N_results.pkl; cox models (clinic_cox, C_film) are skipped here - their
IBS/AUC come from the forward-only Breslow pass (Table1_CoxBreslow_Forward.py).

Self-check: the c-index recomputed from the saved per-patient risk must match
test_result.csv's test_cindex fold by fold (abs diff < 1e-6), otherwise the
offline chain is off and the row is reported with a warning.

Requires the SurvPGC conda env.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from Table1_Cindex_Main import (  # noqa: E402
    GROUP_CONFIG,
    STUDY_SPECS,
    extract_model_name,
    resolve_group_study_dir,
)

from table1_ibs_auc_utils import (  # noqa: E402
    AUC_LANDMARK_MONTHS,
    IBS_GRID_MONTHS,
    build_surv_object,
    cindex_from_risk,
    compute_ibs,
    compute_landmark_aucs,
    interpolate_survival,
    survival_columns_from_logits,
)

COX_BASELINE_MODELS = {"clinic_cox"}  # scalar Cox risk, no survival curve saved

# Folds with fewer events by the landmark are dropped from AUC (degenerate
# IPCW / 0.5-by-construction artifacts, e.g. READ folds at 60mo).
MIN_EVENTS_FOR_AUC = {24.0: 5, 60.0: 6}

CONFIG_KEY_RE = re.compile(r"^([A-Z0-9_]+)\s*=\s*(.*)$")
MAX_SELFCHECK_TOLERANCE = 1e-6


def parse_effective_config(model_dir: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    config_path = model_dir / "effective_config.txt"
    if not config_path.is_file():
        return config
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = CONFIG_KEY_RE.match(line.strip())
        if match:
            config[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return config


class BinProvider:
    """Per-study dataset bins + censoring cohort, built with the project's factory.

    The censoring cohort follows the training-time convention of
    utils/core_utils.py::_extract_survival_metadata: train+val+test combined,
    i.e. the full study dataset (patients_df).
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[np.ndarray | None, object | None]] = {}

    def get_for_model_dir(
        self, study: str, model_dir: Path
    ) -> tuple[np.ndarray | None, object | None]:
        """Return (bins, cohort Surv object) for the study/dataset config."""
        config = parse_effective_config(model_dir)
        key = (
            study,
            config.get("WSI_EXPERIMENT", "uni_v1"),
            config.get("TYPE_OF_PATH", "combine"),
        )
        if key in self._cache:
            return self._cache[key]

        try:
            project_root = str(Path(__file__).resolve().parents[2])
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # SurvivalDatasetFactory reads ./datasets_csv/... relative to cwd
            cwd = os.getcwd()
            os.chdir(project_root)
            try:
                from dataset_deployment.registry import infer_standard_paths
                from datasets.dataset_survival import SurvivalDatasetFactory

                paths = infer_standard_paths(
                    study,
                    project_root,
                    wsi_experiment=key[1],
                    type_of_path=key[2],
                )
                factory = SurvivalDatasetFactory(
                    study=study,
                    label_file=str(paths["label_file"]),
                    omics_dir=str(paths["omics_dir"]),
                    data_dir=str(paths["data_root_dir"]),
                    clinical_file=str(paths["clinical_file"]),
                    seed=1,
                    print_info=False,
                    n_bins=4,
                    label_col="survival_months",
                    num_patches=4096,
                )
            finally:
                os.chdir(cwd)
            bins = np.asarray(factory.bins, dtype=np.float64)
            patients = factory.patients_df
            time_arr = patients["survival_months"].to_numpy()
            cens_arr = patients["censorship"].to_numpy()
            keep = time_arr >= 0.0  # a few BRCA rows carry negative months (data quirk)
            if not keep.all():
                print(
                    f"[WARN] {study}: dropping {int((~keep).sum())} cohort row(s) "
                    "with negative survival months"
                )
            cohort = build_surv_object(time_arr[keep], cens_arr[keep])
        except Exception as exc:  # never hard-block the whole table on bins
            print(f"[WARN] Cannot build dataset bins for {study} ({key}): {exc}")
            bins, cohort = None, None
        self._cache[key] = (bins, cohort)
        return bins, cohort


def load_fold_pkl(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, censorships, risks, logits) aligned per patient."""
    import pickle

    with pkl_path.open("rb") as handle:
        patients = pickle.load(handle)
    patient_ids = list(patients)
    times = np.array([patients[i]["time"] for i in patient_ids], dtype=np.float64)
    censorships = np.array(
        [patients[i]["censorship"] for i in patient_ids], dtype=np.float64
    )
    risks = np.array([patients[i]["risk"] for i in patient_ids], dtype=np.float64)
    logits = np.stack(
        [np.asarray(patients[i]["logits"], dtype=np.float64) for i in patient_ids]
    )
    return times, censorships, risks, logits


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


def compute_model_metrics(
    model_dir: Path,
    csv_path: Path,
    edges: np.ndarray | None,
    cohort_surv,
) -> tuple[dict[int, dict[str, float]], float | None]:
    """Per-fold {ibs, auc24, auc60, cindex_diff} plus the max self-check diff.

    sksurv requires evaluation times to lie within the test fold's follow-up
    range, so a fold only contributes to a metric if its test data can support
    the horizon (IBS grid [1..60], AUC landmarks 24/60); otherwise nan.
    """
    fold_to_csv_cindex = load_csv_cindex_by_fold(csv_path)
    fold_metrics: dict[int, dict[str, float]] = {}
    max_diff: float | None = None

    for test_pkl in sorted(model_dir.glob("split_*_results.pkl")):
        match = re.search(r"split_(\d+)_results\.pkl$", test_pkl.name)
        if not match:
            continue
        fold = int(match.group(1))

        try:
            test_times, test_cens, test_risks, test_logits = load_fold_pkl(test_pkl)
        except Exception as exc:
            print(f"[WARN] Cannot load pkl for {test_pkl}: {exc}")
            continue

        metrics: dict[str, float] = {}

        recorded = fold_to_csv_cindex.get(fold)
        if recorded is not None:
            diff = abs(cindex_from_risk(test_times, test_cens, test_risks) - recorded)
            metrics["cindex_diff"] = diff
            max_diff = diff if max_diff is None else max(max_diff, diff)

        if edges is None or cohort_surv is None:
            fold_metrics[fold] = metrics
            continue

        valid = test_times >= 0.0  # negative months (data quirk) cannot enter KM weights
        survival_columns = survival_columns_from_logits(test_logits[valid])
        test_surv = build_surv_object(test_times[valid], test_cens[valid])
        test_min, test_max = (
            float(test_times[valid].min()),
            float(test_times[valid].max()),
        )

        if test_min <= 1.0 and test_max >= 60.0:
            estimate_grid = interpolate_survival(
                survival_columns, edges[1:], IBS_GRID_MONTHS
            )
            try:
                metrics["ibs"] = compute_ibs(
                    cohort_surv, test_surv, estimate_grid
                )
                if not math.isfinite(metrics["ibs"]):
                    metrics["ibs"] = math.nan
            except Exception as exc:
                print(f"[WARN] IBS failed for {test_pkl}: {exc}")
                metrics["ibs"] = math.nan
        else:
            metrics["ibs"] = math.nan

        landmark_times: list[float] = []
        for landmark in AUC_LANDMARK_MONTHS:
            n_events = int(
                ((test_cens[valid] < 1.0) & (test_times[valid] <= landmark)).sum()
            )
            metrics[f"n_events_{int(landmark)}"] = n_events
            if (
                test_min <= landmark <= test_max
                and n_events >= MIN_EVENTS_FOR_AUC[landmark]
            ):
                landmark_times.append(landmark)
        for landmark in AUC_LANDMARK_MONTHS:
            metrics[f"auc{int(landmark)}"] = math.nan
        if landmark_times:
            # cumulative_dynamic_auc ranks by risk (higher = worse) -> pass 1 - S
            estimate_landmarks = 1.0 - interpolate_survival(
                survival_columns,
                edges[1:],
                np.array(landmark_times, dtype=np.float64),
            )
            try:
                aucs = compute_landmark_aucs(
                    cohort_surv, test_surv, estimate_landmarks, landmark_times
                )
            except Exception as exc:
                print(f"[WARN] AUC failed for {test_pkl}: {exc}")
                aucs = [math.nan] * len(landmark_times)
            for landmark, value in zip(landmark_times, aucs):
                metrics[f"auc{int(landmark)}"] = (
                    value if math.isfinite(value) else math.nan
                )

        fold_metrics[fold] = metrics

    return fold_metrics, max_diff


def collect_study_model_metrics(
    results_root: Path,
    group_dir: str,
    layer_dir: str,
    test_dir: str | None,
    study_token: str,
    study: str,
    results_suffix: str,
    bin_provider: BinProvider,
) -> dict[str, dict[int, dict[str, float]]]:
    """model -> {fold -> metric dict}; scans test_result.csv files like the display script."""
    study_dir = resolve_group_study_dir(
        results_root, group_dir, layer_dir, test_dir, study_token, results_suffix
    )
    if study_dir is None:
        print(f"[WARN] Missing study directory: {layer_dir}_{study_token}_{results_suffix}")
        return {}

    model_to_folds: dict[str, dict[int, dict[str, float]]] = {}
    selfcheck_issues: list[str] = []
    for csv_path in sorted(study_dir.rglob("test_result.csv")):
        model_dir = csv_path.parent
        model = extract_model_name("baselines", csv_path)
        if model in COX_BASELINE_MODELS:
            continue  # scalar Cox risk; covered by the forward Breslow pass
        edges, cohort_surv = bin_provider.get_for_model_dir(study, model_dir)
        fold_metrics, max_diff = compute_model_metrics(
            model_dir, csv_path, edges, cohort_surv
        )
        if max_diff is not None and max_diff > MAX_SELFCHECK_TOLERANCE:
            selfcheck_issues.append(f"{model}: max cindex diff {max_diff:.2e}")
        model_to_folds[model] = fold_metrics

    if selfcheck_issues:
        print(f"[SELFCHECK FAIL] {study}: {len(selfcheck_issues)} model(s)")
        for issue in selfcheck_issues:
            print(f"  - {issue}")
    return model_to_folds
