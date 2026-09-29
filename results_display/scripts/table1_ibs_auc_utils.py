"""
Offline helpers for the Table 1 IBS / AUC recompute.

Thin re-export of the shared metric conventions in utils/survival_metrics.py
(same conventions as the training pipeline) plus small offline extras:
c-index self-check and fold aggregation.

Runs standalone from anywhere: the project root is added to sys.path so
utils.survival_metrics is importable (requires the SurvPGC conda env).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.survival_metrics import (  # noqa: E402
    AUC_LANDMARK_MONTHS,
    IBS_GRID_MONTHS,
    compute_ibs,
    compute_landmark_aucs,
    interpolate_survival,
    survival_columns_from_logits,
)

__all__ = [
    "AUC_LANDMARK_MONTHS",
    "IBS_GRID_MONTHS",
    "build_surv_object",
    "cindex_from_risk",
    "compute_ibs",
    "compute_landmark_aucs",
    "format_mean_std",
    "interpolate_survival",
    "mean_std",
    "survival_columns_from_logits",
]


def build_surv_object(times, censorships) -> Surv:
    """Surv array with event = 1 - censorship, matching _calculate_metrics."""
    return Surv.from_arrays(
        event=(1.0 - np.asarray(censorships, dtype=np.float64)) > 0.5,
        time=np.asarray(times, dtype=np.float64),
    )


def cindex_from_risk(
    times, censorships, risks, tied_tol: float = 1e-08
) -> float:
    """Same call as utils/core_utils.py::_calculate_metrics (self-check)."""
    return float(
        concordance_index_censored(
            (1.0 - np.asarray(censorships, dtype=np.float64)) > 0.5,
            np.asarray(times, dtype=np.float64),
            np.asarray(risks, dtype=np.float64),
            tied_tol=tied_tol,
        )[0]
    )


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Population mean/std over folds, same convention as load_cindex_stats.

    Non-finite values (nan/inf, e.g. folds whose follow-up cannot support the
    metric horizon) are dropped; if none remain, both are None.
    """
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    return f"{mean:.4f} ± {std:.4f}"
