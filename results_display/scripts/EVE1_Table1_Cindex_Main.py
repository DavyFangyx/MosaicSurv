"""
EVE1 直接迁移评估汇总表。

用源 TCGA 队列（COAD/BRCA）训练好的 Table1 官方模型，零微调直接预测对应 CPTAC
外部队列全部符合条件患者，报告 C-index（5 折 mean±std），与 Table1/Table1b 口径一致。

输入（EVE_External_Forward.py / EVE_HGCN_Forward.py 产出）：
  - results/EVE1_Table1_Cindex_Main/{cptac_coad,cptac_brca}/{model}/test_result.csv
    （per-fold C-index，_summary 同口径）+ split_{k}_results.pkl（表注：可比对数/事件数）
  - 内部参照列：Table1 L0Test 源队列各模型目录的 test_result.csv

输出：
  - results_display/EVE1_Table1_Cindex_Main/EVE1_Cindex_Matrix.csv
    （行=模型；列=每队列 external mean±std + internal mean±std + 可比对数/事件数）

用法（SurvPGC_results_display env 即可）:
  python results_display/scripts/EVE1_Table1_Cindex_Main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from EVE_External_Forward import EXTERNAL_COHORTS, discover_table1_model_dirs
from FigC_KM_curves import BASELINE_TYPE, TYPE_ORDER, model_sort_key
from model_names import display_name as model_display_name

EVE1_RESULTS_ROOT = Path("results/EVE1_Table1_Cindex_Main")
TABLE1_ROOT = Path("results/Table1_Cindex_Main/L0Test")
OUTPUT_ROOT = Path("results_display/EVE1_Table1_Cindex_Main")
# 行顺序：以 coad 树为基准的官方 11 模型 + hgcn
MODEL_ORDER = sorted(
    list(discover_table1_model_dirs("tcga_coad")) + ["hgcn"],
    key=lambda m: model_sort_key("Ours" if m == "mosaic_surv" else BASELINE_TYPE.get(m, "Other"), m),
)
COHORT_TOKENS = {"cptac_coad": "CPTAC-COAD", "cptac_brca": "CPTAC-BRCA"}
SOURCE_TOKENS = {"tcga_coad": "TCGA-COAD", "tcga_brca": "TCGA-BRCA"}


def load_pkl_stats(model_dir: Path) -> tuple[int, int]:
    """跨 5 折统计可比对数与事件数（C-index 的分母信息量，表注用）。"""
    n_pairs_total = 0
    n_events_total = 0
    found = 0
    for fold in range(5):
        pkl = model_dir / f"split_{fold}_results.pkl"
        if not pkl.is_file():
            continue
        import pickle
        with pkl.open("rb") as handle:
            payload = pickle.load(handle)
        times = np.asarray([float(np.asarray(rec["time"]).reshape(-1)[0]) for rec in payload.values()])
        events = np.asarray([(1.0 - float(np.asarray(rec["censorship"]).reshape(-1)[0])) > 0.5 for rec in payload.values()])
        n_events = int(events.sum())
        # 可比对：i 发生事件且 T_i < T_j
        n_pairs = 0
        for i in range(len(times)):
            if not events[i]:
                continue
            n_pairs += int((times > times[i]).sum())
        n_pairs_total += n_pairs
        n_events_total += n_events
        found += 1
    return (n_pairs_total // max(found, 1)), (n_events_total // max(found, 1))


def read_cindex_cols(csv_path: Path) -> dict[str, float] | None:
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if "test_cindex" not in df.columns:
        return None
    values = df["test_cindex"].dropna().astype(float)
    if values.empty:
        return None
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def main() -> None:
    rows = []
    for model in MODEL_ORDER:
        row: dict[str, object] = {"Model": model_display_name(model)}
        for target, source in EXTERNAL_COHORTS.items():
            if model == "hgcn":
                eve_dir = EVE1_RESULTS_ROOT / target / "hgcn"
                ext = read_cindex_cols(eve_dir / "test_result.csv")
            else:
                eve_dir = EVE1_RESULTS_ROOT / target / model
                ext = read_cindex_cols(eve_dir / "test_result.csv") if eve_dir.is_dir() else None
            if model == "hgcn":
                from dataset_deployment.registry import get_dataset_config
                token = get_dataset_config(source).display_name.split("-")[1]
                source_dir = TABLE1_ROOT / f"L0_{token}_full_model_val" / f"{source}__L0__cell_norm__uni_v1" / "hgcn"
            else:
                source_dir = discover_table1_model_dirs(source).get(model)
            internal = read_cindex_cols(source_dir / "test_result.csv") if source_dir else None
            n_pairs, n_events = load_pkl_stats(eve_dir) if eve_dir.is_dir() else (0, 0)
            token = COHORT_TOKENS[target]
            row[f"{token} external"] = f"{ext['mean']:.3f}±{ext['std']:.3f}" if ext else "-"
            row[f"{SOURCE_TOKENS[source]} internal"] = (
                f"{internal['mean']:.3f}±{internal['std']:.3f}" if internal else "-")
            row[f"{token} comp_pairs"] = n_pairs
            row[f"{token} n_events"] = n_events
        rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_ROOT / "EVE1_Cindex_Matrix.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n[WRITE] {OUTPUT_ROOT / 'EVE1_Cindex_Matrix.csv'}")
    print("口径：external = 源 TCGA 5 折模型零微调直接预测外部队列全部符合条件患者；")
    print("      internal = 同一模型目录的 Table1 L0Test 官方批次；可比对数/事件数为 5 折均值；")
    print("      hgcn 走独立 HGCN 前向（EVE_HGCN_Forward.py），未产出前记 '-'。")


if __name__ == "__main__":
    main()
