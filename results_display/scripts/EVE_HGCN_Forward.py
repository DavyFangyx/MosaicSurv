"""
EVE1/EVE2 的 hgcn 前向（hgcn 为第三方管线，不走 EVE_External_Forward 引擎）。

- EVE1：加载 TCGA-COAD hgcn 各折 checkpoint（seed_*/fold_{k+1}/*.pth），对
  CPTAC-COAD 的 hgcn pack 全部患者做无梯度前向，输出与引擎同格式的
  results/EVE1_Table1_Cindex_Main/hgcn/split_{k}_results.pkl
  + test_result.csv（per-fold C-index，sksurv tied_tol 与内部同口径）。
- EVE2：对 TCGA-COAD hgcn pack 的第 k 折训练集患者前向，输出
  results/EVE2_FigC_KM_curves/train_forward/hgcn/split_{k}_results_train.pkl
  （τ_k = 训练集风险中位数，供 EVE2 display 多数票投票）。

前置：CPTAC-COAD hgcn pack 已构建
（SurvPGC_Workspace/hgcn data/cptac_coad/pack/kimianet__msigdb_gsea_families__L0）。

用法（SurvPGC env，GPU）:
  python results_display/scripts/EVE_HGCN_Forward.py --gpu 4 [--folds 0,1,2,3,4]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from Table1_HGCN_Forward import (
    HGCN_DROPOUT,
    HGCN_MIX,
    HGCN_N_HIDDEN,
    HGCN_OUT_CLASSES,
    HGCN_TRAIN_USE_TYPE,
    load_hgcn_modules,
)

EVE1_ROOT = Path("results/EVE1_Table1_Cindex_Main")
EVE2_TRAIN_ROOT = Path("results/EVE2_FigC_KM_curves/train_forward")


def hgcn_model_dir(source: str) -> Path:
    """Table1 L0Test 树里源 study 的 hgcn 模型目录。"""
    from dataset_deployment.registry import get_dataset_config

    token = get_dataset_config(source).display_name.split("-")[1]
    return Path("results/Table1_Cindex_Main/L0Test") / f"L0_{token}_full_model_val" \
        / f"{source}__L0__cell_norm__uni_v1" / "hgcn"
HGCN_WSI_EXPERIMENT = "kimianet"
HGCN_GENE_EXPERIMENT = "msigdb_gsea_families"
HGCN_CLINIC_SCHEME = "L0"


def load_pack(study: str, build) -> tuple[list[str], dict, dict]:
    pack_dir = build.hgcn_pack_dir(
        study,
        wsi_experiment=HGCN_WSI_EXPERIMENT,
        gene_experiment=HGCN_GENE_EXPERIMENT,
        clinic_scheme=HGCN_CLINIC_SCHEME,
        repo_root=Path(_PROJECT_ROOT),
    )
    if not pack_dir.is_dir():
        raise FileNotFoundError(f"missing hgcn pack: {pack_dir}")
    patients = [str(p) for p in joblib.load(pack_dir / "patients.pkl")]
    sur_and_time = joblib.load(pack_dir / "sur_and_time.pkl")
    all_data = joblib.load(pack_dir / "all_data.pkl")
    patients = [p for p in patients if p in all_data]
    return patients, sur_and_time, all_data


def build_model(train, all_data):
    import torch

    return train.fusion_model_mae_2(
        img_in_feats=train._infer_hgcn_in_feats(all_data, "x_img", 1024),
        rna_in_feats=train._infer_hgcn_in_feats(all_data, "x_rna", 1024),
        cli_in_feats=train._infer_hgcn_in_feats(all_data, "x_cli", 1024),
        n_hidden=HGCN_N_HIDDEN,
        out_classes=HGCN_OUT_CLASSES,
        dropout=HGCN_DROPOUT,
        train_type_num=len(HGCN_TRAIN_USE_TYPE),
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def forward_risks(train, model, all_data, patient_ids) -> dict[str, float]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    risks: dict[str, float] = {}
    with torch.no_grad():
        for pid in patient_ids:
            graph = all_data[pid].to(device)
            out_pre, _, _, _ = model(graph, HGCN_TRAIN_USE_TYPE, HGCN_TRAIN_USE_TYPE, mix=HGCN_MIX)
            risks[pid] = float(out_pre[0].detach().cpu().numpy()[0])
    return risks


def load_clinical_lookup(target: str) -> dict[str, tuple]:
    import pandas as pd

    clinical = pd.read_csv(f"datasets_csv/clinical_data/{target}_clinical.csv", index_col=0)
    default_subtype = target.replace("cptac_", "").upper()
    lookup: dict[str, tuple] = {}
    for case_id, row in clinical.iterrows():
        stage = str(row.get("stage", "N/A")) if pd.notna(row.get("stage", "N/A")) else "N/A"
        subtype = str(row.get("subtype", default_subtype)) if pd.notna(row.get("subtype", default_subtype)) else default_subtype
        lookup[str(case_id)] = (stage, subtype, target)
    return lookup


def save_pkl(out_path: Path, patient_ids, risks, times, censorships, clinical_lookup) -> None:
    from utils.file_utils import _save_pkl

    payload = {}
    for pid in patient_ids:
        payload[pid] = {
            "time": np.asarray(times[pid], dtype=np.float64),
            "risk": np.asarray(risks[pid], dtype=np.float64),
            "censorship": np.asarray(censorships[pid], dtype=np.float64),
            "clinical": [clinical_lookup.get(pid, ("N/A", "COAD", "cptac_coad"))],
            "logits": np.asarray([risks[pid]], dtype=np.float64),
        }
    _save_pkl(str(out_path), payload)


def run_cohort(source: str, target: str, folds: list[int]) -> None:
    import torch
    from sksurv.metrics import concordance_index_censored

    train, build = load_hgcn_modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_patients, target_sur, target_data = load_pack(target, build)
    source_patients, source_sur, source_data = load_pack(source, build)
    print(f"[HGCN] target pack {target} n={len(target_patients)} source pack {source} n={len(source_patients)}")

    clinical_lookup = load_clinical_lookup(target)
    target_times, target_events, target_cens = {}, {}, {}
    for pid in target_patients:
        target_times[pid] = float(target_sur[pid][-1])
        target_events[pid] = float(target_sur[pid][0]) > 0.5
        target_cens[pid] = 1.0 - float(target_sur[pid][0])

    model_dir = hgcn_model_dir(source)
    split_dir = Path(f"splits/5foldcv/{source}")
    eve1_out = EVE1_ROOT / target / "hgcn"
    eve2_out = EVE2_TRAIN_ROOT / source / "hgcn"
    eve1_out.mkdir(parents=True, exist_ok=True)
    eve2_out.mkdir(parents=True, exist_ok=True)
    cindex_rows = []
    for fold in folds:
        ckpt_paths = sorted(model_dir.glob(f"seed_*/fold_{fold + 1}/*.pth"))
        if not ckpt_paths:
            print(f"[HGCN] fold {fold}: missing checkpoint -> skip")
            continue
        model = build_model(train, target_data)
        model.load_state_dict(torch.load(str(ckpt_paths[0]), map_location=device))
        model.eval()

        # EVE1：目标队列全部患者
        risks = forward_risks(train, model, target_data, target_patients)
        save_pkl(eve1_out / f"split_{fold}_results.pkl", target_patients, risks,
                 target_times, target_cens, clinical_lookup)
        times_arr = np.asarray([target_times[p] for p in target_patients])
        event_arr = np.asarray([target_events[p] for p in target_patients])
        risk_arr = np.asarray([risks[p] for p in target_patients])
        cindex = float(concordance_index_censored(event_arr, times_arr, risk_arr, tied_tol=1e-08)[0])
        cindex_rows.append({
            "test_cindex": cindex, "test_cindex_ipcw": 0.0, "test_IBS": 0.0,
            "test_iauc": 0.0, "test_iauc_list": 0.0, "test_loss": 0.0, "test_BS": 0.0,
        })
        print(f"[HGCN EVE1 {target}] fold {fold}: cindex={cindex:.4f} n={len(target_patients)}")

        # EVE2：源队列第 k 折训练集
        import pandas as pd

        split_df = pd.read_csv(split_dir / f"splits_{fold}.csv")
        train_ids = [str(p) for p in split_df["train"].dropna().tolist() if str(p) in source_data]
        train_risks = forward_risks(train, model, source_data, train_ids)
        source_times, source_cens = {}, {}
        for pid in train_ids:
            source_times[pid] = float(source_sur[pid][-1])
            source_cens[pid] = 1.0 - float(source_sur[pid][0])
        save_pkl(eve2_out / f"split_{fold}_results_train.pkl", train_ids, train_risks,
                 source_times, source_cens, {})
        print(f"[HGCN EVE2 {source}] fold {fold}: train forward n={len(train_ids)}")
        del model

    if cindex_rows:
        import pandas as pd
        pd.DataFrame(cindex_rows).to_csv(eve1_out / "test_result.csv", index=False)
        print(f"[WRITE] {eve1_out / 'test_result.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EVE1/EVE2 hgcn 前向（按队列）")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--cohort", default="coad,brca", help="逗号分隔：coad/brca")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    folds = [int(part) for part in args.folds.split(",") if part.strip()]
    cohort_map = {"coad": ("tcga_coad", "cptac_coad"), "brca": ("tcga_brca", "cptac_brca")}
    for name in [item.strip() for item in args.cohort.split(",") if item.strip()]:
        source, target = cohort_map[name]
        run_cohort(source, target, folds)


if __name__ == "__main__":
    main()
