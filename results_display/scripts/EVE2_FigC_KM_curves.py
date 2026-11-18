"""
EVE2 风险分层（外部预定义阈值 + 多数票）。

与 FigC 的折内中位数二分不同，EVE2 的阈值完全不使用外部队列信息：
  - 对每个折模型 k，τ_k = 第 k 折 TCGA-COAD 训练集风险分数的中位数
    （results/EVE2_FigC_KM_curves/train_forward/{model}/split_{k}_results_train.pkl）；
  - 用第 k 折模型预测每个 CPTAC-COAD 患者的风险（results/EVE1_Table1_Cindex_Main/
    {model}/split_{k}_results.pkl），risk_k > τ_k 记 1 票 High；
  - 每患者 5 票（奇数，无平票），≥3 票定组 High。

统计与绘图复用 FigC_KM_curves：手写 Mantel-Haenszel log-rank + Efron Cox HR
（coxph_efron，与 R coxph ties="efron" 一致）、KM 阶梯图与 CI。

输出（results_display/EVE2_FigC_KM_curves/）：
  - all_models.png|pdf（全部模型网格）
  - per_model/{model}.png
  - km_stats.csv（行=模型；含 n/事件数/HR/CI/log-rank p/τ_0..τ_4）

用法（SurvPGC env）:
python results_display/scripts/EVE2_FigC_KM_curves.py

# 图内标注两行 — 第一行 HR (95% CI)，第二行 log-rank p
python results_display/scripts/EVE2_FigC_KM_curves.py [--annotate-hr]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-eve2")

from EVE_External_Forward import EXTERNAL_COHORTS, discover_table1_model_dirs
from FigC_KM_curves import (
    BASELINE_TYPE,
    KMResult,
    censor_points,
    coxph_efron,
    kaplan_meier,
    logrank_p,
    model_sort_key,
    plot_all_models,
    plot_per_model,
)
from model_names import display_name as model_display_name

EVE1_ROOT = Path("results/EVE1_Table1_Cindex_Main")
TRAIN_FORWARD_ROOT = Path("results/EVE2_FigC_KM_curves/train_forward")
OUTPUT_ROOT = Path("results_display/EVE2_FigC_KM_curves")
MAJORITY_THRESHOLD = 3  # 5 票中 ≥3 票 High 定组（奇数无平票）


def load_pkl_risks(pkl_path: Path) -> dict[str, float]:
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        str(case_id): float(np.asarray(rec["risk"]).reshape(-1)[0])
        for case_id, rec in payload.items()
    }


def load_train_thresholds(model_dir: Path) -> dict[int, float]:
    """第 k 折 TCGA-COAD 训练集风险中位数 τ_k（阈值完全来自源队列）。"""
    thresholds: dict[int, float] = {}
    for fold in range(5):
        pkl = model_dir / f"split_{fold}_results_train.pkl"
        if not pkl.is_file():
            continue
        risks = np.asarray(list(load_pkl_risks(pkl).values()))
        thresholds[fold] = float(np.median(risks))
    return thresholds


def load_patient_records(model_dir: Path) -> tuple[dict[str, float], dict[str, int], dict[int, dict[str, float]]]:
    """CPTAC 患者的 time/event（取 fold0 记录）与各折风险。"""
    times: dict[str, float] = {}
    events: dict[str, int] = {}
    fold_risks: dict[int, dict[str, float]] = {}
    for fold in range(5):
        pkl = model_dir / f"split_{fold}_results.pkl"
        if not pkl.is_file():
            continue
        fold_risks[fold] = load_pkl_risks(pkl)
        if fold == 0:
            with pkl.open("rb") as handle:
                payload = pickle.load(handle)
            for case_id, rec in payload.items():
                times[str(case_id)] = float(np.asarray(rec["time"]).reshape(-1)[0])
                events[str(case_id)] = int((1.0 - float(np.asarray(rec["censorship"]).reshape(-1)[0])) > 0.5)
    return times, events, fold_risks


def build_majority_result(model: str, thresholds: dict[int, float], target: str) -> KMResult | None:
    model_dir = EVE1_ROOT / target / model
    times, events, fold_risks = load_patient_records(model_dir)
    if len(thresholds) != 5 or not times:
        print(f"[WARN] {model}: thresholds={sorted(thresholds)} patients={len(times)} -> skip")
        return None

    case_ids = sorted(times)
    t_arr = np.asarray([times[c] for c in case_ids], dtype=float)
    e_arr = np.asarray([events[c] for c in case_ids], dtype=bool)
    votes = np.zeros(len(case_ids), dtype=int)
    for fold, tau in thresholds.items():
        risks = fold_risks.get(fold, {})
        for idx, case_id in enumerate(case_ids):
            if case_id in risks and risks[case_id] > tau:
                votes[idx] += 1
    groups = (votes >= MAJORITY_THRESHOLD).astype(int)

    low_mask = groups == 0
    high_mask = groups == 1
    if low_mask.sum() == 0 or high_mask.sum() == 0:
        print(f"[WARN] {model}: single-group after majority vote -> skip")
        return None

    km_low = kaplan_meier(t_arr[low_mask], e_arr[low_mask])
    km_high = kaplan_meier(t_arr[high_mask], e_arr[high_mask])
    cox = coxph_efron(t_arr, e_arr, groups.astype(float))
    model_type = "Ours" if model == "mosaic_surv" else BASELINE_TYPE.get(model, "Other")
    return KMResult(
        model_type=model_type,
        model=model_display_name(model),
        times=t_arr,
        events=e_arr,
        groups=groups,
        n=len(case_ids),
        n_low=int(low_mask.sum()),
        n_high=int(high_mask.sum()),
        events_low=int(e_arr[low_mask].sum()),
        events_high=int(e_arr[high_mask].sum()),
        median_risk=float(np.mean(list(thresholds.values()))),
        logrank_p=float(logrank_p(t_arr, e_arr, groups)),
        hr=float(cox["hr"]),
        hr_ci_lo=float(cox["ci_lo"]),
        hr_ci_hi=float(cox["ci_hi"]),
        hr_wald_p=float(cox["wald_p"]),
        km_low=km_low,
        km_high=km_high,
        censor_low=censor_points(t_arr[low_mask], e_arr[low_mask], *km_low[:2]),
        censor_high=censor_points(t_arr[high_mask], e_arr[high_mask], *km_high[:2]),
    )


def write_stats_with_thresholds(results: list[KMResult], thresholds_by_model: dict[str, dict[int, float]], target: str = "cptac_coad") -> None:
    rows = []
    for model, thresholds in thresholds_by_model.items():
        match = next((r for r in results if r.model == model_display_name(model)), None)
        if match is None:
            continue
        rows.append({
            "model_type": match.model_type,
            "model": match.model,
            "n": match.n,
            "n_low": match.n_low,
            "n_high": match.n_high,
            "events_low": match.events_low,
            "events_high": match.events_high,
            "hr": match.hr,
            "ci_lo": match.hr_ci_lo,
            "ci_hi": match.hr_ci_hi,
            "cox_wald_p": match.hr_wald_p,
            "logrank_p": match.logrank_p,
            **{f"tau_fold{k}": thresholds[k] for k in sorted(thresholds)},
        })
    (OUTPUT_ROOT / target).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / target / "km_stats.csv", index=False)
    print(f"[WRITE] {OUTPUT_ROOT / target / 'km_stats.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EVE2 外部风险分层 KM（阈值来自 TCGA 训练集 + 5 折多数票）")
    parser.add_argument("--annotate-hr", action="store_true", help="图内标注 HR (CI) + p")
    parser.add_argument("--time-unit", default="month", choices=["month", "year"])
    args = parser.parse_args()

    for target, source in EXTERNAL_COHORTS.items():
        print(f"\n===== EVE2 cohort: {target} (source {source}) =====")
        model_dirs = discover_table1_model_dirs(source)
        if (EVE1_ROOT / target / "hgcn").is_dir():
            model_dirs["hgcn"] = EVE1_ROOT / target / "hgcn"
        models = sorted(model_dirs, key=lambda m: model_sort_key(
            "Ours" if m == "mosaic_surv" else BASELINE_TYPE.get(m, "Other"), m))

        results: list[KMResult] = []
        thresholds_by_model: dict[str, dict[int, float]] = {}
        for model in models:
            train_dir = TRAIN_FORWARD_ROOT / source / model
            thresholds = load_train_thresholds(train_dir)
            thresholds_by_model[model] = thresholds
            result = build_majority_result(model, thresholds, target)
            if result is None:
                continue
            results.append(result)
            plot_per_model(result, OUTPUT_ROOT / target / "per_model",
                           time_unit=args.time_unit, annotate_hr=args.annotate_hr)
            print(f"[KM] {model}: n={result.n} low={result.n_low} high={result.n_high} "
                  f"HR={result.hr:.2f} ({result.hr_ci_lo:.2f}-{result.hr_ci_hi:.2f}) p={result.logrank_p:.4f}")

        if results:
            plot_all_models(
                results,
                OUTPUT_ROOT / target / "all_models",
                title=f"EVE2: {target} risk stratification (threshold = {source} train median, 5-fold majority vote)",
                time_unit=args.time_unit,
                annotate_hr=args.annotate_hr,
            )
        write_stats_with_thresholds(results, thresholds_by_model, target)
        print(f"[WRITE] {OUTPUT_ROOT / target / 'all_models.png'}")


if __name__ == "__main__":
    main()
