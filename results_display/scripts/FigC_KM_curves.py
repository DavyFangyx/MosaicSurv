"""Plot Table 1 Kaplan-Meier curves from saved OOF predictions.

分组方式：折内中位数二分（fold-median split）。cox_surv 的线性预测子跨折不可比
（偏似然对平移不变），全局切分会把 HR 向 1 衰减，因此每折在各自中位数处二分后合并。

Example:
python results_display/scripts/FigC_KM_curves.py \
    --test-dir "L0Test After fixed modal missing" \
    --dataset BRCA \
    --time-unit month \
    --annotate-hr

python results_display/scripts/FigC_KM_curves.py \
    --test-dir "L0Test After fixed modal missing" \
    --dataset all \
    --time-unit year \
    --annotate-hr
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-figc")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP_DIR = "Table1_Cindex_Main"
OUTPUT_SERIES = "FigC_KM_curves"
LAYER_PREFIX_RE = re.compile(r"^(L\d+)")
OURS_MODEL_TOKEN_RE = re.compile(r"survtri_poe_vae(?:__|_)([A-Za-z][A-Za-z0-9_]*)", re.IGNORECASE)
STUDY_FOLDER_RE = re.compile(r"^(L\d+)_([A-Za-z0-9]+)_(full_model_val|poe_model_val)$")

# Cfilm 主模型的 OOF 预测统一使用 hparam 扫描选出的参数点（t018_alpha_beta_learn）
CFILM_HPARAM_RUN_ID = "t018_alpha_beta_learn"
CFILM_HPARAM_ROOT = PROJECT_ROOT / "results" / "Cfilm_Hparam_Eval" / "Table1" / CFILM_HPARAM_RUN_ID

STUDY_SPECS = [
    ("BRCA", "tcga_brca"),
    ("COAD", "tcga_coad"),
    ("KICH", "tcga_kich"),
    ("KIRC", "tcga_kirc"),
    ("KIRP", "tcga_kirp"),
    ("LIHC", "tcga_lihc"),
    ("PRAD", "tcga_prad"),
    ("READ", "tcga_read"),
]
STUDY_TOKEN_TO_NAME = {token: study for token, study in STUDY_SPECS}
STUDY_NAME_TO_TOKEN = {study: token for token, study in STUDY_SPECS}
STUDY_ALIASES = {
    **{token.lower(): (token, study) for token, study in STUDY_SPECS},
    **{study.lower(): (token, study) for token, study in STUDY_SPECS},
    **{study.replace("tcga_", "").lower(): (token, study) for token, study in STUDY_SPECS},
}

BASELINE_MODEL_SPECS = [
    ("P", "abmil_wsi"),
    ("P", "mlp_wsi"),
    ("P", "transmil_wsi"),
    ("C", "clinic_cox"),
    ("C", "mlp_clinic_mean"),
    ("C", "mlp_clinic_flatten"),
    ("C", "snn_clinic_mean"),
    ("C", "snn_clinic_flatten"),
    ("G", "mlp_gene"),
    ("G", "snn_gene"),
    ("G", "mlp_gene_f"),
    ("G", "snn_gene_f"),
    ("P+C", "survpc_f"),
    ("P+G", "porpoise"),
    ("P+G", "survpath"),
    ("P+G", "mcat"),
    ("C+G", "survgc_f"),
    ("P+C+G", "survpgc_f"),
]
BASELINE_TYPE = {model: model_type for model_type, model in BASELINE_MODEL_SPECS}
TYPE_ORDER = {"P": 0, "C": 1, "G": 2, "P+C": 3, "P+G": 4, "C+G": 5, "P+C+G": 6, "Ours": 7, "Other": 99}

LOW_COLOR = "#2ca02c"
HIGH_COLOR = "#d62728"
LOW_LABEL = "Low risk"
HIGH_LABEL = "High risk"

TIME_UNIT_LABELS = {"month": "Time (months)", "year": "Time (years)"}
MONTHS_PER_YEAR = 12.0


@dataclass
class ModelRun:
    model_type: str
    model: str
    kind: str
    csv_path: Path
    model_dir: Path


@dataclass
class KMResult:
    model_type: str
    model: str
    times: np.ndarray
    events: np.ndarray
    groups: np.ndarray
    n: int
    n_low: int
    n_high: int
    events_low: int
    events_high: int
    median_risk: float
    logrank_p: float
    hr: float
    hr_ci_lo: float
    hr_ci_hi: float
    hr_wald_p: float
    km_low: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    km_high: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    censor_low: tuple[np.ndarray, np.ndarray]
    censor_high: tuple[np.ndarray, np.ndarray]


def normalize_ours_model_name(token: str) -> str:
    parts = [part for part in str(token).split("_") if part]
    if not parts:
        return str(token)
    parts[0] = parts[0].upper()
    parts[1:] = [part.lower() for part in parts[1:]]
    return "_".join(parts)


def extract_model_name(kind: str, csv_path: Path) -> str:
    model_dir = csv_path.parent.name
    if kind != "ours":
        return model_dir
    match = OURS_MODEL_TOKEN_RE.search(model_dir)
    if match:
        return normalize_ours_model_name(match.group(1))
    if "__" in model_dir:
        return normalize_ours_model_name(model_dir.rsplit("__", 1)[-1])
    return model_dir


def infer_layer_dir(test_dir: str, experiment_dir: Path | None = None) -> str | None:
    if experiment_dir is not None and experiment_dir.is_dir():
        for child in sorted(experiment_dir.iterdir()):
            match = LAYER_PREFIX_RE.match(child.name)
            if match:
                return match.group(1)
    match = LAYER_PREFIX_RE.match(test_dir)
    if match:
        return match.group(1)
    return None


def list_experiment_dirs(results_root: Path, group_dir: str) -> list[str]:
    group_root = results_root / group_dir
    if not group_root.is_dir():
        return []
    return sorted(path.name for path in group_root.iterdir() if path.is_dir())


def resolve_study_spec(dataset: str) -> tuple[str, str]:
    key = str(dataset).strip().lower()
    if key in STUDY_ALIASES:
        return STUDY_ALIASES[key]
    raise SystemExit(
        f"Unknown dataset {dataset!r}. Available: all, "
        + ", ".join(f"{token}/{study}" for token, study in STUDY_SPECS)
    )


def ordered_available_study_tokens(experiment_dir: Path) -> list[str]:
    available = set(list_available_study_tokens(experiment_dir))
    known = [token for token, _ in STUDY_SPECS if token in available]
    extras = sorted(token for token in available if token not in {item for item, _ in STUDY_SPECS})
    return known + extras


def resolve_requested_study_tokens(dataset: str, experiment_dir: Path) -> list[str]:
    available_tokens = ordered_available_study_tokens(experiment_dir)
    if not available_tokens:
        raise SystemExit(f"No dataset folders found under {experiment_dir}")

    key = str(dataset).strip().lower()
    if key in {"all", "*"}:
        return available_tokens

    study_token, _ = resolve_study_spec(dataset)
    if study_token not in available_tokens:
        available = ", ".join(available_tokens)
        raise SystemExit(
            f"Dataset {dataset!r} not found under {experiment_dir.name}. Available: all, {available}"
        )
    return [study_token]


def list_available_study_tokens(experiment_dir: Path) -> list[str]:
    tokens: set[str] = set()
    if not experiment_dir.is_dir():
        return []
    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        match = STUDY_FOLDER_RE.match(child.name)
        if match:
            tokens.add(match.group(2))
    return sorted(tokens)


def model_sort_key(model_type: str, model: str) -> tuple[int, str, str]:
    type_rank = TYPE_ORDER.get(model_type, 99)
    if model_type == "Ours":
        family, _, variant = model.partition("_")
        return (type_rank, family.upper(), variant.lower())
    return (type_rank, model.lower(), "")


def discover_models(study_dir: Path, kind: str) -> list[ModelRun]:
    runs: list[ModelRun] = []
    seen: set[str] = set()
    for csv_path in sorted(study_dir.rglob("test_result.csv")):
        model = extract_model_name(kind, csv_path)
        if not model or model in seen:
            continue
        seen.add(model)
        model_type = "Ours" if kind == "ours" else BASELINE_TYPE.get(model, "Other")
        runs.append(
            ModelRun(
                model_type=model_type,
                model=model,
                kind=kind,
                csv_path=csv_path,
                model_dir=csv_path.parent,
            )
        )
    return runs


def load_oof_predictions(model_dir: Path) -> pd.DataFrame | None:
    rows: list[dict[str, object]] = []
    found = False
    for fold in range(5):
        pkl_path = model_dir / f"split_{fold}_results.pkl"
        if not pkl_path.is_file():
            continue
        found = True
        try:
            with pkl_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            print(f"[WARN] Cannot read {pkl_path}: {exc}")
            continue
        if not isinstance(payload, dict) or not payload:
            print(f"[WARN] Empty prediction dict: {pkl_path}")
            continue
        for case_id, rec in payload.items():
            if not isinstance(rec, dict):
                continue
            try:
                time = float(np.asarray(rec.get("time")).reshape(-1)[0])
                risk = float(np.asarray(rec.get("risk")).reshape(-1)[0])
                censorship = float(np.asarray(rec.get("censorship")).reshape(-1)[0])
            except Exception:
                continue
            rows.append(
                {
                    "case_id": str(case_id),
                    "fold": fold,
                    "time": time,
                    "risk": risk,
                    "censorship": censorship,
                }
            )
    if not found:
        return None
    if not rows:
        return pd.DataFrame(columns=["case_id", "fold", "time", "risk", "censorship", "event"])
    df = pd.DataFrame(rows)
    dup_mask = df.duplicated(subset=["case_id"], keep="first")
    if dup_mask.any():
        print(f"[WARN] Dropping {int(dup_mask.sum())} duplicated case_ids under {model_dir}")
        df = df.loc[~dup_mask].copy()
    df["event"] = (1.0 - df["censorship"]).astype(int)
    return df


def kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if times.size == 0:
        zeros = np.array([0.0])
        ones = np.array([1.0])
        return zeros, ones, ones, ones
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]
    km_t = [0.0]
    km_s = [1.0]
    km_lo = [1.0]
    km_hi = [1.0]
    survival = 1.0
    greenwood = 0.0
    n_at_risk = times.size
    idx = 0
    n = times.size
    z95 = float(stats.norm.ppf(0.975))
    for t in np.unique(times):
        n_event = 0
        n_censor = 0
        while idx < n and times[idx] == t:
            if events[idx]:
                n_event += 1
            else:
                n_censor += 1
            idx += 1
        if n_event > 0 and n_at_risk > 0:
            survival *= 1.0 - (n_event / n_at_risk)
            denom = n_at_risk * (n_at_risk - n_event)
            if denom > 0:
                greenwood += n_event / denom
            se = survival * math.sqrt(greenwood) if greenwood > 0 else 0.0
            lower = max(0.0, survival - z95 * se)
            upper = min(1.0, survival + z95 * se)
            if not np.isfinite(lower):
                lower = survival
            if not np.isfinite(upper):
                upper = survival
            km_t.append(float(t))
            km_s.append(float(survival))
            km_lo.append(float(lower))
            km_hi.append(float(upper))
        n_at_risk -= n_event + n_censor
    max_t = float(times.max())
    if km_t[-1] < max_t:
        km_t.append(max_t)
        km_s.append(km_s[-1])
        km_lo.append(km_lo[-1])
        km_hi.append(km_hi[-1])
    return (
        np.asarray(km_t, dtype=float),
        np.asarray(km_s, dtype=float),
        np.asarray(km_lo, dtype=float),
        np.asarray(km_hi, dtype=float),
    )

def survival_at(query_times: np.ndarray, km_t: np.ndarray, km_s: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(km_t, query_times, side="right") - 1
    idx = np.clip(idx, 0, len(km_s) - 1)
    return km_s[idx]


def censor_points(times: np.ndarray, events: np.ndarray, km_t: np.ndarray, km_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    censor_t = times[~events]
    if censor_t.size == 0:
        return np.array([]), np.array([])
    return censor_t, survival_at(censor_t, km_t, km_s)


def logrank_p(times: np.ndarray, events: np.ndarray, groups: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    groups = np.asarray(groups, dtype=int)
    event_times = np.unique(times[events])
    o_minus_e = 0.0
    variance = 0.0
    for t in event_times:
        at_risk = times >= t
        n0 = int(np.sum(at_risk & (groups == 0)))
        n1 = int(np.sum(at_risk & (groups == 1)))
        n_tot = n0 + n1
        if n_tot < 2:
            continue
        died = (times == t) & events
        d0 = int(np.sum(died & (groups == 0)))
        d1 = int(np.sum(died & (groups == 1)))
        d_tot = d0 + d1
        if d_tot == 0:
            continue
        expected1 = d_tot * n1 / n_tot
        o_minus_e += d1 - expected1
        variance += (n0 * n1 * d_tot * (n_tot - d_tot)) / (n_tot * n_tot * (n_tot - 1))
    if variance <= 0 or not np.isfinite(variance):
        return float("nan")
    chi2 = (o_minus_e * o_minus_e) / variance
    return float(stats.chi2.sf(chi2, df=1))


def p_text(p_value: float) -> str:
    """纯数字形式的 p 值文本（无 "p = " 前缀），用于表格单元格。"""
    if p_value is None or not np.isfinite(p_value):
        return "NA"
    if p_value < 1e-16:
        return "<1e-16"
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.3f}"


def format_p(p_value: float) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "p = NA"
    if p_value < 1e-16:
        return "p < 1e-16"
    if p_value < 0.001:
        return f"p = {p_value:.2e}"
    return f"p = {p_value:.3f}"


def coxph_efron(
    times: np.ndarray,
    events: np.ndarray,
    z: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> dict[str, float]:
    """Univariate Cox PH via Newton-Raphson on the partial likelihood, Efron tie handling.

    Efron 结处理与 R coxph / lifelines 的默认 ties="efron" 一致；已用精确条件 logistic
    （ties="exact"）和 log-rank chi2（beta=0 处 score test，相对差 <= 0.11%）双重校验。
    Returns beta/se/wald_p/hr/ci_lo/ci_hi/score_chi2; nan on failure.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    z = np.asarray(z, dtype=float)
    event_times = np.unique(times[events])

    def score_info(beta: float) -> tuple[float, float]:
        score = 0.0
        info = 0.0
        for t in event_times:
            at_risk = times >= t
            zr = z[at_risk]
            ez = np.exp(beta * zr)
            ez_sum = float(ez.sum())
            died = (times == t) & events
            d = int(died.sum())
            zd = z[died]
            ezd = np.exp(beta * zd)
            ezd_sum = float(ezd.sum())
            score += float(zd.sum())
            for k in range(d):
                denom = ez_sum - (k / d) * ezd_sum
                b_k = float((ez * zr).sum()) - (k / d) * float((ezd * zd).sum())
                c_k = float((ez * zr * zr).sum()) - (k / d) * float((ezd * zd * zd).sum())
                wbar = b_k / denom
                score -= wbar
                info += c_k / denom - wbar * wbar
        return score, info

    score0, info0 = score_info(0.0)
    beta = 0.0
    converged = False
    for _ in range(max_iter):
        score, info = score_info(beta)
        if info <= 0 or not np.isfinite(info):
            break
        step = score / info
        beta += step
        if abs(step) < tol:
            converged = True
            break
    score, info = score_info(beta)
    if not converged or info <= 0 or not np.isfinite(beta):
        return {"beta": np.nan, "se": np.nan, "wald_p": np.nan, "hr": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "score_chi2": np.nan}
    se = 1.0 / np.sqrt(info)
    z_stat = beta / se
    return {
        "beta": float(beta),
        "se": float(se),
        "wald_p": float(2.0 * stats.norm.sf(abs(z_stat))),
        "hr": float(np.exp(beta)),
        "ci_lo": float(np.exp(beta - 1.959963984540054 * se)),
        "ci_hi": float(np.exp(beta + 1.959963984540054 * se)),
        "score_chi2": float(score0 * score0 / info0) if info0 > 0 else np.nan,
    }


def format_hr_ci(hr: float, ci_lo: float, ci_hi: float) -> str | None:
    if not all(np.isfinite(v) for v in (hr, ci_lo, ci_hi)):
        return None
    return f"HR {hr:.2f} ({ci_lo:.2f}-{ci_hi:.2f})"


def time_axis_scale(time_unit: str) -> float:
    if time_unit == "year":
        return 1.0 / MONTHS_PER_YEAR
    return 1.0


def time_axis_label(time_unit: str) -> str:
    return TIME_UNIT_LABELS[time_unit]


def split_risk_groups(df: pd.DataFrame, split_mode: str = "foldmed") -> tuple[pd.DataFrame, float] | None:
    """按 risk 把患者分成 High/Low 两组。

    split_mode:
      - "foldmed" (默认): 折内中位数二分。cox_surv 的线性预测子跨折不可比
        （偏似然对平移不变，每折 offset 任意），全局切分会混入折间尺度噪声、把 HR 向 1
        衰减，因此每折在各自中位数处二分后再合并。
      - "pooled": 所有折拼接后取全局中位数（旧行为，仅供对照）。
    """
    valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "risk", "event"]).copy()
    valid = valid.loc[valid["time"] > 0].copy()
    if len(valid) < 20:
        return None
    risks = valid["risk"].to_numpy(dtype=float)
    if np.unique(np.round(risks, 12)).size < 2:
        return None
    if split_mode == "pooled":
        median_risk = float(np.median(risks))
        valid["group"] = np.where(risks >= median_risk, 1, 0)
    else:
        fold_medians = valid.groupby("fold")["risk"].transform("median").to_numpy(dtype=float)
        valid["group"] = np.where(risks >= fold_medians, 1, 0)
        median_risk = float(np.median(risks))  # 仅作参考值保留
    n_low = int((valid["group"] == 0).sum())
    n_high = int((valid["group"] == 1).sum())
    events_low = int(valid.loc[valid["group"] == 0, "event"].sum())
    events_high = int(valid.loc[valid["group"] == 1, "event"].sum())
    if n_low == 0 or n_high == 0:
        return None
    if events_low + events_high == 0:
        return None
    return valid, median_risk


def build_km_result(run: ModelRun, time_unit: str = "month") -> KMResult | None:
    df = load_oof_predictions(run.model_dir)
    if df is None:
        print(f"[WARN] Missing split_*_results.pkl for {run.model}: {run.model_dir}")
        return None
    if df.empty:
        print(f"[WARN] Empty OOF predictions for {run.model}: {run.model_dir}")
        return None
    split = split_risk_groups(df)
    if split is None:
        print(f"[WARN] Skip {run.model}: degenerate High/Low split")
        return None
    valid, median_risk = split
    times = valid["time"].to_numpy(dtype=float) * time_axis_scale(time_unit)
    events = valid["event"].to_numpy(dtype=int).astype(bool)
    groups = valid["group"].to_numpy(dtype=int)
    low_mask = groups == 0
    high_mask = groups == 1
    km_low = kaplan_meier(times[low_mask], events[low_mask])
    km_high = kaplan_meier(times[high_mask], events[high_mask])
    cox_fit = coxph_efron(times, events, groups.astype(float))
    return KMResult(
        model_type=run.model_type,
        model=run.model,
        times=times,
        events=events,
        groups=groups,
        n=int(len(valid)),
        n_low=int(low_mask.sum()),
        n_high=int(high_mask.sum()),
        events_low=int(events[low_mask].sum()),
        events_high=int(events[high_mask].sum()),
        median_risk=median_risk,
        logrank_p=logrank_p(times, events, groups),
        hr=cox_fit["hr"],
        hr_ci_lo=cox_fit["ci_lo"],
        hr_ci_hi=cox_fit["ci_hi"],
        hr_wald_p=cox_fit["wald_p"],
        km_low=km_low,
        km_high=km_high,
        censor_low=censor_points(times[low_mask], events[low_mask], km_low[0], km_low[1]),
        censor_high=censor_points(times[high_mask], events[high_mask], km_high[0], km_high[1]),
    )


def style_km_axis(
    ax: plt.Axes,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    time_unit: str = "month",
) -> None:
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    if show_xlabel:
        ax.set_xlabel(time_axis_label(time_unit))
    if show_ylabel:
        ax.set_ylabel("Survival probability")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fill_km_ci(ax: plt.Axes, times: np.ndarray, lower: np.ndarray, upper: np.ndarray, color: str) -> None:
    ax.fill_between(times, lower, upper, step="post", color=color, alpha=0.16, linewidth=0, zorder=1)


def draw_km(ax: plt.Axes, result: KMResult, *, with_legend: bool, annotate_hr: bool = False) -> None:
    low_t, low_s, low_lo, low_hi = result.km_low
    high_t, high_s, high_lo, high_hi = result.km_high
    fill_km_ci(ax, low_t, low_lo, low_hi, LOW_COLOR)
    fill_km_ci(ax, high_t, high_lo, high_hi, HIGH_COLOR)
    ax.step(low_t, low_s, where="post", color=LOW_COLOR, lw=2.0, label=f"{LOW_LABEL} (n={result.n_low})", zorder=2)
    ax.step(high_t, high_s, where="post", color=HIGH_COLOR, lw=2.0, label=f"{HIGH_LABEL} (n={result.n_high})", zorder=2)
    low_ct, low_cs = result.censor_low
    high_ct, high_cs = result.censor_high
    if low_ct.size:
        ax.scatter(low_ct, low_cs, marker="|", s=28, color=LOW_COLOR, linewidths=1.0, zorder=3)
    if high_ct.size:
        ax.scatter(high_ct, high_cs, marker="|", s=28, color=HIGH_COLOR, linewidths=1.0, zorder=3)
    if with_legend:
        ax.legend(loc="upper left", frameon=False, fontsize=8, borderaxespad=0.2)
    if annotate_hr:
        hr_line = format_hr_ci(result.hr, result.hr_ci_lo, result.hr_ci_hi)
        text = f"{hr_line}\n{format_p(result.logrank_p)}" if hr_line else format_p(result.logrank_p)
    else:
        text = format_p(result.logrank_p)
    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8 if not with_legend else 9,
    )

def risk_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=LOW_COLOR, lw=2.0, label=LOW_LABEL),
        Line2D([0], [0], color=HIGH_COLOR, lw=2.0, label=HIGH_LABEL),
    ]


def savefig(fig: plt.Figure, path: Path, *, pdf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_per_model(result: KMResult, out_dir: Path, time_unit: str = "month", annotate_hr: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.set_title(f"{result.model} ({result.model_type})", fontsize=11, loc="left")
    draw_km(ax, result, with_legend=True, annotate_hr=annotate_hr)
    style_km_axis(ax, time_unit=time_unit)
    savefig(fig, out_dir / result.model)


def grid_shape(n: int) -> tuple[int, int]:
    if n <= 8:
        ncols = 4
    elif n <= 18:
        ncols = 5
    else:
        ncols = 6
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def plot_all_models(
    results: list[KMResult],
    out_path: Path,
    *,
    title: str,
    time_unit: str = "month",
    annotate_hr: bool = False,
) -> None:
    n = len(results)
    nrows, ncols = grid_shape(n)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.15, nrows * 2.55 + 0.55),
        sharex=False,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, result in enumerate(results):
        ax = axes_arr[idx // ncols, idx % ncols]
        ax.set_title(f"{result.model} ({result.model_type})", fontsize=8, pad=3)
        draw_km(ax, result, with_legend=False, annotate_hr=annotate_hr)
        style_km_axis(ax, show_xlabel=idx // ncols == nrows - 1, show_ylabel=idx % ncols == 0, time_unit=time_unit)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    for idx in range(n, nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    fig.legend(
        handles=risk_legend_handles(),
        loc="upper left",
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.012, 1.0),
    )
    fig.suptitle(title, fontsize=13, y=0.995, x=0.52)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    savefig(fig, out_path, pdf=True)


def write_stats(results: list[KMResult], out_path: Path) -> None:
    rows = [
        {
            "model_type": item.model_type,
            "model": item.model,
            "n": item.n,
            "n_low": item.n_low,
            "n_high": item.n_high,
            "events_low": item.events_low,
            "events_high": item.events_high,
            "median_risk": item.median_risk,
            "hr": item.hr,
            "ci_lo": item.hr_ci_lo,
            "ci_hi": item.hr_ci_hi,
            "cox_wald_p": item.hr_wald_p,
            "logrank_p": item.logrank_p,
        }
        for item in results
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_experiment_matrix(test_dir: str, output_root: Path, study_tokens: list[str]) -> None:
    """实验目录级别的 HR 矩阵：行 = 模型（前两列 Model Type / Model，Ours 最后），
    列 = 数据集 + Mean HR；单元格 = "HR (95% CI) | p=..."，缺失写 "-"。"""
    per_token: dict[str, pd.DataFrame] = {}
    for token in study_tokens:
        path = output_root / test_dir / token / "km_stats.csv"
        if not path.is_file():
            continue
        per_token[token] = pd.read_csv(path)
    if not per_token:
        return
    datasets = [t for t in study_tokens if t in per_token]
    first = next(iter(per_token.values()))
    baseline_order = list(BASELINE_MODEL_SPECS)
    ours_order = [(r.model_type, r.model) for r in first.itertuples() if r.model_type == "Ours"]
    model_order = baseline_order + ours_order
    out_path = output_root / test_dir / "km_hr_matrix.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model Type", "Model"] + datasets + ["Mean HR"])
        for model_type, model in model_order:
            cells = []
            hrs = []
            for token in datasets:
                frame = per_token[token]
                sub = frame[(frame["model_type"] == model_type) & (frame["model"] == model)]
                rec = sub.iloc[0] if not sub.empty else None
                if rec is None or not np.isfinite(rec["hr"]):
                    cells.append("-")
                else:
                    hrs.append(float(rec["hr"]))
                    cells.append(
                        f"{rec['hr']:.2f} ({rec['ci_lo']:.2f}-{rec['ci_hi']:.2f}) | p={p_text(rec['logrank_p'])}"
                    )
            mean_hr = f"{np.mean(hrs):.2f}" if hrs else "-"
            writer.writerow([model_type, model] + cells + [mean_hr])
    print(f"[OK] experiment matrix: {out_path}")


def remap_cfilm_hparam(runs: list[ModelRun], study_token: str) -> None:
    """Cfilm 的预测目录替换为 hparam 参数点 t018_alpha_beta_learn 的结果目录。

    找不到对应文件夹时保留原目录并警告（例如 hparam 扫描未覆盖的数据集）。
    """
    study = STUDY_TOKEN_TO_NAME.get(study_token)
    if study is None:
        return
    for run in runs:
        if run.model != "C_film":
            continue
        candidates = [
            child / "survtri_poe_vae_c_film"
            for child in sorted(CFILM_HPARAM_ROOT.glob(f"{study}__*"))
        ]
        model_dir = next((c for c in candidates if c.is_dir()), None)
        if model_dir is None:
            print(f"[WARN] {CFILM_HPARAM_RUN_ID} Cfilm missing for {study_token}; keep original {run.model_dir}")
            continue
        run.model_dir = model_dir
        run.csv_path = model_dir / "test_result.csv"
        print(f"[OVERRIDE] C_film -> {CFILM_HPARAM_RUN_ID}: {model_dir}")


def collect_runs(experiment_dir: Path, layer_dir: str, study_token: str) -> list[ModelRun]:
    runs: list[ModelRun] = []
    mapping = [
        ("baselines", f"{layer_dir}_{study_token}_full_model_val"),
        ("ours", f"{layer_dir}_{study_token}_poe_model_val"),
    ]
    for kind, folder_name in mapping:
        study_dir = experiment_dir / folder_name
        if not study_dir.is_dir():
            print(f"[WARN] Missing study directory: {study_dir}")
            continue
        found = discover_models(study_dir, kind)
        print(f"[KIND] {kind} | {folder_name} | models={len(found)}")
        runs.extend(found)
    remap_cfilm_hparam(runs, study_token)
    runs.sort(key=lambda item: model_sort_key(item.model_type, item.model))
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Table 1 Kaplan-Meier curves from saved OOF predictions.")
    parser.add_argument("--test-dir", required=True, help="Experiment folder under Table1_Cindex_Main")
    parser.add_argument("--dataset", required=True, help="Dataset token, study name, or all")
    parser.add_argument(
        "--time-unit",
        choices=("month", "year"),
        default="month",
        help="X-axis time unit. Stored times are months; year divides by 12.",
    )
    parser.add_argument(
        "--annotate-hr",
        action="store_true",
        help="在图内标注 HR + 95%% CI（Efron Cox，High vs Low）+ log-rank p；不传则只标 p。",
    )
    parser.add_argument("--group-dir", default=GROUP_DIR, help="Grouped results directory name")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results", help="Root results directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results_display" / OUTPUT_SERIES,
        help="Output directory under results_display",
    )
    return parser


def process_dataset(
    *,
    test_dir: str,
    experiment_dir: Path,
    layer_dir: str,
    study_token: str,
    output_root: Path,
    time_unit: str,
    annotate_hr: bool,
) -> bool:
    print(f"[DATASET] {study_token}")
    runs = collect_runs(experiment_dir, layer_dir, study_token)
    if not runs:
        print(f"[WARN] No models found for {test_dir} / {study_token}")
        return False

    results: list[KMResult] = []
    for run in runs:
        print(f"[MODEL] {run.model_type} | {run.model}")
        result = build_km_result(run, time_unit=time_unit)
        if result is not None:
            results.append(result)

    if not results:
        print(f"[WARN] No KM curves could be generated for {test_dir} / {study_token}")
        return False

    out_dir = output_root / test_dir / study_token
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    for old_png in per_model_dir.glob("*.png"):
        old_png.unlink()
    for result in results:
        plot_per_model(result, per_model_dir, time_unit=time_unit, annotate_hr=annotate_hr)

    grid_title = f"{test_dir} / {study_token}"
    plot_all_models(results, out_dir / "all_models", title=grid_title, time_unit=time_unit, annotate_hr=annotate_hr)
    stats_path = out_dir / "km_stats.csv"
    write_stats(results, stats_path)

    print(f"[OK] {study_token} models plotted: {len(results)}/{len(runs)}")
    print(f"[OK] {study_token} per-model figures: {per_model_dir}")
    print(f"[OK] {study_token} grid: {out_dir / 'all_models.png'}")
    print(f"[OK] {study_token} stats: {stats_path}")
    return True


def main() -> None:
    args = build_parser().parse_args()
    experiment_dirs = list_experiment_dirs(args.results_root, args.group_dir)
    if args.test_dir not in experiment_dirs:
        available = ", ".join(experiment_dirs) if experiment_dirs else "(none)"
        raise SystemExit(f"Unknown --test-dir {args.test_dir!r}. Available: {available}")

    experiment_dir = args.results_root / args.group_dir / args.test_dir
    layer_dir = infer_layer_dir(args.test_dir, experiment_dir)
    if not layer_dir:
        raise SystemExit(f"Cannot infer layer prefix from {experiment_dir}")

    study_tokens = resolve_requested_study_tokens(args.dataset, experiment_dir)
    print(f"[PLAN] datasets: {', '.join(study_tokens)}")

    ok_tokens: list[str] = []
    failed_tokens: list[str] = []
    for study_token in study_tokens:
        try:
            ok = process_dataset(
                test_dir=args.test_dir,
                experiment_dir=experiment_dir,
                layer_dir=layer_dir,
                study_token=study_token,
                output_root=args.output_root,
                time_unit=args.time_unit,
                annotate_hr=args.annotate_hr,
            )
        except Exception as exc:  # 单数据集失败不中断整轮，保证矩阵与汇总仍会写出
            import traceback
            traceback.print_exc()
            print(f"[ERROR] dataset {study_token} failed: {exc}")
            ok = False
        (ok_tokens if ok else failed_tokens).append(study_token)

    if not ok_tokens:
        raise SystemExit(f"No KM curves could be generated for {args.test_dir}.")
    if failed_tokens:
        print(f"[WARN] skipped datasets: {', '.join(failed_tokens)}")
    write_experiment_matrix(args.test_dir, args.output_root, ok_tokens)
    print(f"[DONE] {args.test_dir}: {len(ok_tokens)}/{len(study_tokens)} datasets")


if __name__ == "__main__":
    main()
