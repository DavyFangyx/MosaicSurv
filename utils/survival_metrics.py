"""
Shared survival metric conventions: IBS (0-60 months) and landmark time-dependent AUC.

Single source of truth used by both:
  - the training pipeline (utils/core_utils.py::_calculate_metrics), and
  - the offline Table 1 recompute (results_display/scripts/table1_ibs_auc_utils.py).

Discrete survival models (nll_surv) output one hazard logit per time bin:
    logits -> hazards = sigmoid(logits)
           -> survival columns S(t_j) = cumprod(1 - hazards) at the dataset
              bin edges t_j = dataset_factory.bins[1:] = [q1, q2, q3, tmax+eps].
    S(t) at arbitrary t is linear interpolation between the bin edges,
    anchored at S(0) = 1 and extrapolated to 0 beyond the last edge.

Cox models (cox_surv) output a scalar log relative risk; their survival
function is obtained with the Breslow baseline hazard estimated on the
training fold: S(t|x) = exp(-H0(t) * exp(risk)).

For cumulative_dynamic_auc the estimate must be a risk score (higher =
worse), so callers pass 1 - S(t) (same direction as the original code).

Availability guards (sksurv requires evaluation times within the test
follow-up; degenerate folds are dropped instead of fabricating values):
  - IBS grid [1..60]: fold needs test_min <= 1 and test_max >= 60
  - AUC@24 / AUC@60: fold needs test_min <= landmark <= test_max and at
    least MIN_EVENTS_FOR_AUC[landmark] events by the landmark
"""

from __future__ import annotations

import numpy as np
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

IBS_GRID_MONTHS = np.arange(1, 61, dtype=np.float64)  # [1..60], monthly
AUC_LANDMARK_MONTHS = [24.0, 60.0]

# Folds with fewer events by the landmark produce degenerate IPCW AUCs
# (0.5-by-construction, e.g. READ folds at 60mo with 5 events).
MIN_EVENTS_FOR_AUC = {24.0: 5, 60.0: 6}


def survival_columns_from_logits(logits: np.ndarray) -> np.ndarray:
    """(n, n_bins) hazard logits -> (n, n_bins) survival columns S(edge_j)."""
    logits = np.asarray(logits, dtype=np.float64)
    hazards = 1.0 / (1.0 + np.exp(-logits))
    return np.cumprod(1.0 - hazards, axis=1)


def interpolate_survival(
    survival_columns: np.ndarray,
    edges: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """S(t) by linear interpolation between bin edges; S(0)=1, 0 past the last edge."""
    edges = np.asarray(edges, dtype=np.float64)
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))
    xp = np.concatenate(([0.0], edges))
    out = np.empty((survival_columns.shape[0], times.shape[0]), dtype=np.float64)
    for row in range(survival_columns.shape[0]):
        fp = np.concatenate(([1.0], survival_columns[row]))
        out[row] = np.interp(times, xp, fp, left=1.0, right=0.0)
    return out


def breslow_survival(
    train_times: np.ndarray,
    train_events: np.ndarray,
    train_risks: np.ndarray,
    risks: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Breslow baseline hazard -> per-patient survival S(t|x) at `times`.

    train_events: bool array (True = event); risks: log relative hazards.
    H0(t) = sum over event times t_i <= t of d_i / sum_{j in R(t_i)} exp(risk_j)
    (no tie correction, standard Breslow).
    """
    train_times = np.asarray(train_times, dtype=np.float64)
    train_events = np.asarray(train_events, dtype=bool)
    train_risks = np.asarray(train_risks, dtype=np.float64)
    risks = np.asarray(risks, dtype=np.float64).reshape(-1, 1)
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))

    exp_train_risk = np.exp(train_risks)  # (n_train,), for the H0 denominator
    event_times = np.sort(np.unique(train_times[train_events]))
    h0 = np.empty(event_times.shape[0], dtype=np.float64)
    for idx, t in enumerate(event_times):
        at_risk = train_times >= t
        events_at_t = train_events & (train_times == t)
        denom = exp_train_risk[at_risk].sum()
        h0[idx] = events_at_t.sum() / denom if denom > 0 else 0.0
    h0 = np.cumsum(h0)

    if event_times.shape[0] == 0:
        return np.ones((risks.shape[0], times.shape[0]), dtype=np.float64)
    h0_at_times = np.interp(times, event_times, h0, left=0.0, right=h0[-1])
    return np.exp(-h0_at_times * np.exp(risks))


def compute_ibs(
    cohort_surv: Surv,
    test_surv: Surv,
    estimate_grid: np.ndarray,
    times_grid: np.ndarray = IBS_GRID_MONTHS,
) -> float:
    return float(
        integrated_brier_score(cohort_surv, test_surv, estimate_grid, times_grid)
    )


def compute_landmark_aucs(
    cohort_surv: Surv,
    test_surv: Surv,
    estimate_landmarks: np.ndarray,
    landmark_times: list[float] = AUC_LANDMARK_MONTHS,
) -> list[float]:
    """Cumulative/dynamic AUC at each landmark time (estimate = risk, higher = worse)."""
    auc_list, _ = cumulative_dynamic_auc(
        cohort_surv, test_surv, estimate_landmarks, landmark_times
    )
    return [float(value) for value in auc_list]
