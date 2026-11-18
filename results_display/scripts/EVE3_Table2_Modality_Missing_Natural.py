"""
EVE3 真实缺失评估汇总表（参照 Table2 口径）。

混合队列 C-index：
  - 对每个数据集 d、每一折 k：评估集 S = 第 k 折测试集完整患者（PCG）∪ 该数据集
    全部天然缺失患者 M_d（有生存数据者）。M_d 从未参与任何折训练。
  - 枚举 S 内可比较对 (i,j)：T_i < T_j 且 i 发生事件，且至少一方 ∈ M_d
    （"缺失-完整"对 + "缺失-缺失"对）；risk_i > risk_j 记一致、相等记 0.5。
  - 跨数据集按折合并：C_k = Σ_d conc(d,k) / Σ_d comp(d,k)，只在数据集内部配对；
    5 折得到 5 个 C_k，报告 mean±std。
  - 按缺失类型拆分：仅统计"至少一方属于该缺失类型"的对
    （缺P=CG、缺G=PC、仅G、仅P，来自 data_tcgal_stats class_code）。
  - PCG 参照列：同一折模型在完整测试集上的 C-index，直接复用
    Table2 树顶层 test_result.csv（PCG 子集口径）与 Table1 L0Test poe 树
    （mosaic_surv）；hgcn 缺 pack 无法预测缺失患者，仅出参照列并表注。

输入：
  - results/EVE3_Table2_Modality_Missing_Natural/{study}/{model}/split_{k}_results.pkl
  - results/Table2_Baselines/{study}__L0__gene_raw__uni_v1/{model}/test_result.csv
  - data_tcgal_stats/{DISPLAY}/{DISPLAY}_patients.csv（has_* 与 class_code）

输出（results_display/EVE3_Table2_Modality_Missing_Natural/）：
  - EVE3_Mixed_Cindex_Matrix.csv（行=模型；列=Mixed(全部)/缺P/缺G/仅G/仅P/PCG参照，值 mean±std）
  - EVE3_missing_stats.csv（表注：每数据集各缺失类型的缺失人数/事件数/可比对数）

用法（SurvPGC_results_display env）:
  python results_display/scripts/EVE3_Table2_Modality_Missing_Natural.py
"""

from __future__ import annotations

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

try:
    from model_names import display_name as model_display_name
except Exception:  # 名称映射不可用时退回原始键
    model_display_name = lambda name: name  # noqa: E731

EVE3_ROOT = Path("results/EVE3_Table2_Modality_Missing_Natural")
TABLE2_ROOT = Path("results/Table2_Baselines")
TABLE1_ROOT = Path("results/Table1_Cindex_Main/L0Test")
OUTPUT_ROOT = Path("results_display/EVE3_Table2_Modality_Missing_Natural")

EVE3_STUDIES = ["tcga_brca", "tcga_coad", "tcga_kirc", "tcga_kirp",
                "tcga_lihc", "tcga_luad", "tcga_lusc"]
EVE3_MODELS = ["modality_concat_zero", "modality_concat_mean", "mvae_poe",
               "mopoe", "mosaic_surv", "hgcn"]
TABLE2_DIR_NAMES = {
    "modality_concat_zero": "modality_concat__resampler__zero",
    "modality_concat_mean": "modality_concat__resampler__mean",
    "mvae_poe": "mvae_poe",
    "mopoe": "mopoe",
    "hgcn": "hgcn",
}
# 六种缺失类型（data_tcgal_stats class_code 全集）：仅缺P/仅缺C/仅缺G/缺PC(仅G)/缺PG(仅C)/缺CG(仅P)
ALL_MISSING_TYPES = ["仅缺P (CG)", "仅缺C (PG)", "仅缺G (PC)", "缺PC (仅G)", "缺PG (仅C)", "缺CG (仅P)"]
CLASS_TO_TYPE = {
    "CG": "仅缺P (CG)", "PG": "仅缺C (PG)", "PC": "仅缺G (PC)",
    "G": "缺PC (仅G)", "C": "缺PG (仅C)", "P": "缺CG (仅P)",
}


def load_study_missing(study: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (全部缺失患者统计行, 有生存数据的缺失患者行)。"""
    from dataset_deployment.registry import get_dataset_config

    config = get_dataset_config(study)
    stats = pd.read_csv(f"{config.stats_dir}/{config.display_name}_patients.csv")
    stats.columns = [str(col).strip() for col in stats.columns]
    missing = stats[~((stats["has_P"] == 1) & (stats["has_C"] == 1) & (stats["has_G"] == 1))]
    eve3_meta = pd.read_csv(f"datasets_csv/eve3_metadata/{study}.csv", index_col=0)
    meta_cases = set(eve3_meta["case_id"].astype(str))
    usable = missing[missing["submitter_id"].astype(str).isin(meta_cases)]
    return missing, usable


def load_pkl(pkl_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    case_ids = [str(cid) for cid in payload]
    times = np.asarray([float(np.asarray(payload[cid]["time"]).reshape(-1)[0]) for cid in case_ids])
    risks = np.asarray([float(np.asarray(payload[cid]["risk"]).reshape(-1)[0]) for cid in case_ids])
    cens = np.asarray([float(np.asarray(payload[cid]["censorship"]).reshape(-1)[0]) for cid in case_ids])
    return case_ids, times, risks, cens


def pair_cindex(times, events, risks, in_group: np.ndarray) -> tuple[float, int]:
    """可比较对中至少一方属于指定组（如 M_d / 某缺失类型）的 C-index。

    可比较对 (i,j)：T_i < T_j 且 i 发生事件；只保留 i、j 至少一方在组内的对
    （缺失-完整对 + 缺失-缺失对）。risk_i > risk_j 记一致，相等记 0.5。
    """
    conc = 0.0
    comp = 0
    n = len(times)
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if j == i or times[j] <= times[i]:
                continue
            if not (in_group[i] or in_group[j]):
                continue
            comp += 1
            if risks[i] > risks[j]:
                conc += 1.0
            elif risks[i] == risks[j]:
                conc += 0.5
    return conc, comp


def read_pcg_reference(study: str, model: str) -> list[float]:
    """PCG 参照：同一模型在第 k 折完整测试集上的 C-index（复用既有 test_result.csv）。"""
    if model == "mosaic_surv":
        candidates = sorted(TABLE1_ROOT.glob(f"L0_*_poe_model_val/{study}__L0__cell_norm__uni_v1__mosaic_surv/mosaic_surv"))
        csv_path = candidates[0] / "test_result.csv" if candidates else None
    elif model == "hgcn":
        csv_path = TABLE2_ROOT / f"{study}__L0__gene_raw__uni_v1" / "hgcn" / "test_result.csv"
    else:
        csv_path = (TABLE2_ROOT / f"{study}__L0__gene_raw__uni_v1"
                    / TABLE2_DIR_NAMES[model] / "test_result.csv")
    if csv_path is None or not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    col = "test_cindex" if "test_cindex" in df.columns else None
    if col is None:
        # hgcn 的 fold_result.csv 可能是其他列名，做一次宽松匹配
        for candidate in ("cindex", "C_index", "test_c_index"):
            if candidate in df.columns:
                col = candidate
                break
    if col is None:
        return []
    values = df[col].dropna().astype(float)
    return values.tolist()


def main() -> None:
    # per (model, type) 的跨数据集合并缓存：fold -> (conc, comp)
    merged: dict[tuple[str, str | None], dict[int, tuple[float, int]]] = {}
    missing_stats_rows: list[dict[str, object]] = []
    pcg_reference: dict[str, list[float]] = {}

    for study in EVE3_STUDIES:
        all_missing, usable_missing = load_study_missing(study)
        missing_set = set(usable_missing["submitter_id"].astype(str))
        class_map = dict(zip(usable_missing["submitter_id"].astype(str),
                             usable_missing["class_code"].astype(str)))
        event_map = None  # 事件数来自 eve3 metadata
        eve3_meta = pd.read_csv(f"datasets_csv/eve3_metadata/{study}.csv", index_col=0)
        meta_map = dict(zip(eve3_meta["case_id"].astype(str),
                            eve3_meta[["survival_months", "censorship"]].to_dict("records")))

        for model in EVE3_MODELS:
            if model == "hgcn":
                pcg_reference.setdefault(model, []).extend(read_pcg_reference(study, model))
                continue
            model_dir = EVE3_ROOT / study / model
            if not model_dir.is_dir():
                continue
            for fold in range(5):
                pkl = model_dir / f"split_{fold}_results.pkl"
                if not pkl.is_file():
                    continue
                case_ids, times, risks, cens = load_pkl(pkl)
                events = (1.0 - cens) > 0.5
                in_missing = np.asarray([cid in missing_set for cid in case_ids])
                # 全部缺失类型
                conc, comp = pair_cindex(times, events, risks, in_missing)
                key = (model, None)
                merged.setdefault(key, {}).setdefault(fold, [0.0, 0])
                merged[key][fold][0] += conc
                merged[key][fold][1] += comp
                # 按缺失类型拆分（六种全算，空组后续记 "-"）
                for cid_group in ALL_MISSING_TYPES:
                    in_type = np.asarray([class_map.get(cid) and CLASS_TO_TYPE[class_map[cid]] == cid_group
                                          for cid in case_ids])
                    conc_t, comp_t = pair_cindex(times, events, risks, in_type)
                    key_t = (model, cid_group)
                    merged.setdefault(key_t, {}).setdefault(fold, [0.0, 0])
                    merged[key_t][fold][0] += conc_t
                    merged[key_t][fold][1] += comp_t
            pcg_reference.setdefault(model, []).extend(read_pcg_reference(study, model))

        # 表注统计：每数据集每缺失类型的缺失人数/事件数
        for type_col, class_codes in (
                ("仅缺P (CG)", ("CG",)), ("仅缺C (PG)", ("PG",)), ("仅缺G (PC)", ("PC",)),
                ("缺PC (仅G)", ("G",)), ("缺PG (仅C)", ("C",)), ("缺CG (仅P)", ("P",))):
            group = all_missing[all_missing["class_code"].isin(class_codes)]
            n_usable = sum(1 for cid in group["submitter_id"] if cid in missing_set)
            n_events = sum(1 for cid in group["submitter_id"]
                           if cid in missing_set and int(meta_map.get(cid, {}).get("censorship", 1)) == 0)
            missing_stats_rows.append({
                "study": study, "missing_type": type_col,
                "n_missing_total": len(group), "n_missing_with_survival": n_usable,
                "n_events": n_events,
            })

    # 汇总矩阵：先算六类全值
    rows = []
    for model in EVE3_MODELS:
        row: dict[str, object] = {"Model": model_display_name(model)}
        for type_col in ([None] + ALL_MISSING_TYPES):
            key = (model, type_col)
            folds = merged.get(key, {})
            if not folds:
                row[type_col or "Mixed C-index（全部）"] = "-"
                continue
            c_k = [conc / comp for conc, comp in folds.values() if comp > 0]
            if not c_k:
                row[type_col or "Mixed C-index（全部）"] = "-"
                continue
            mean = float(np.mean(c_k))
            std = float(np.std(c_k, ddof=1)) if len(c_k) > 1 else 0.0
            row[type_col or "Mixed C-index（全部）"] = f"{mean:.3f}±{std:.3f}"
        ref = pcg_reference.get(model, [])
        row["PCG 参照（完整-完整）"] = (
            f"{np.mean(ref):.3f}±{np.std(ref, ddof=1):.3f}" if len(ref) > 1
            else (f"{ref[0]:.3f}" if ref else "-"))
        rows.append(row)

    df_full = pd.DataFrame(rows, columns=[
        "Model", "Mixed C-index（全部）", *ALL_MISSING_TYPES, "PCG 参照（完整-完整）"])
    # 精简表：只保留非空（非全 "-"）的缺失类型列
    present_types = [t for t in ALL_MISSING_TYPES
                     if any(str(v) != "-" for v in df_full[t])]
    df_present = df_full[["Model", "Mixed C-index（全部）", *present_types,
                          "PCG 参照（完整-完整）"]]
    stats_df = pd.DataFrame(missing_stats_rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(OUTPUT_ROOT / "EVE3_Mixed_Cindex_Matrix_full.csv", index=False)
    df_present.to_csv(OUTPUT_ROOT / "EVE3_Mixed_Cindex_Matrix.csv", index=False)
    stats_df.to_csv(OUTPUT_ROOT / "EVE3_missing_stats.csv", index=False)
    print("== 完整表（6 种缺失类型全列）==")
    print(df_full.to_string(index=False))
    print("\n== 精简表（只写有的缺失类型）==")
    print(df_present.to_string(index=False))
    print(f"\n[WRITE] {OUTPUT_ROOT / 'EVE3_Mixed_Cindex_Matrix_full.csv'}")
    print(f"[WRITE] {OUTPUT_ROOT / 'EVE3_Mixed_Cindex_Matrix.csv'}")
    print(f"[WRITE] {OUTPUT_ROOT / 'EVE3_missing_stats.csv'}")
    print("\n口径：混合 C-index 仅在数据集内部配对，跨数据集按折合并（Σ conc / Σ comp），5 折 mean±std；")
    print("      PCG 参照 = 同折模型完整测试集 C-index（复用 Table2 顶层 / Table1 L0Test poe 树 test_result.csv）；")
    print("      hgcn 缺图 pack 无法预测缺失患者，仅出参照列；")
    print("      完整表 6 种缺失类型全列（无患者的记 '-'）；精简表只保留有患者的类型；")
    print("      各类型缺失人数/事件数表注见 EVE3_missing_stats.csv。")


if __name__ == "__main__":
    main()
