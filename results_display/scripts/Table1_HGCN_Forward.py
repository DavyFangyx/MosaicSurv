"""
hgcn（third-party HGCN，fusion_model_mae_2）前向 Breslow IBS/AUC。

hgcn 训练器只把测试集 risk 存进 all_gnn_time.pkl（test_result.csv 里的
test_IBS/test_iauc 列是 0.0 占位）；本脚本按 Table1_CoxBreslow_Forward 的
同一口径补齐 fold 级 IBS / AUC@24 / AUC@60：

  - 重建第三方图数据（load_hgcn_graphs_from_dirs，特征目录取自本 run 的
    effective_config；重建后顺手 dump 成 study pack 缓存，下次直接读）；
  - 每个 fold：加载该 fold 的 .pth checkpoint，对 train split 做一次无梯度
    前向收集 train risks；
  - Breslow 基线 H0 → 测试集生存函数 → IBS / AUC@24 / AUC@60（grid、
    min-events 护栏与 cox 前向完全一致，复用 utils/survival_metrics）；
  - 测试 risks 直接复用训练器保存的 all_gnn_time.pkl，并做 c-index 自检
    （重算 vs test_result.csv 该 fold 的 test_cindex，记入 cindex_abs_diff）。

输出：追加行到统一缓存 CSV（study, model=hgcn, fold, ibs, auc24, auc60,
n_events_24, n_events_60, cindex_abs_diff），与 cox 前向共用一份 schema，
Table1_Paper_Tables 直接读取。

用法（SurvPGC env，GPU）：
  python results_display/scripts/Table1_HGCN_Forward.py \
      --model-dir <hgcn run dir> --out <cache csv> [--gpu 1] [--folds 0,1,2,3,4]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_HGCN_CODE_DIR = (
    Path(_PROJECT_ROOT)
    / "models"
    / "missing_modality_baselines"
    / "third_party"
    / "HGCN"
    / "HGCN_code"
)

HGCN_TRAIN_USE_TYPE = ["img", "rna", "cli"]
HGCN_N_HIDDEN = 512
HGCN_OUT_CLASSES = 512
HGCN_DROPOUT = 0.5
HGCN_MIX = True


def _load_local(module_name: str, file_path: Path, base_dir: Path | None = None):
    if base_dir is not None and str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hgcn_modules():
    """导入第三方 hgcn train 模块与项目侧 graph build 模块。"""
    train = _load_local(
        "survpgc_hgcn_train_forward", _HGCN_CODE_DIR / "train.py", _HGCN_CODE_DIR
    )
    build = _load_local(
        "survpgc_hgcn_graph_build_forward",
        Path(_PROJECT_ROOT) / "models" / "missing_modality_baselines" / "hgcn_graph_build.py",
    )
    return train, build


def _load_effective_config(model_dir: Path) -> dict[str, str]:
    from Table1_CoxBreslow_Forward import load_effective_config

    return load_effective_config(model_dir)


def _load_graphs(train, build, config: dict[str, str], project_root: Path):
    """重建（或读取 pack 缓存的）hgcn 图数据，口径与训练器一致。"""
    study = config["STUDY"]  # tcga_brca
    wsi_exp = config.get("WSI_EXPERIMENT", "uni_v1")
    gene_exp = config.get("GENE_EXPERIMENT", "cell_norm")
    clinic_exp = config.get("CLINIC_EXPERIMENT", "L0")

    pack_dir = build.hgcn_pack_dir(
        study,
        wsi_experiment=wsi_exp,
        gene_experiment=gene_exp,
        clinic_scheme=clinic_exp,
        repo_root=project_root,
    )
    if pack_dir.is_dir() and train.hgcn_pack_complete(pack_dir):
        print(f"[HGCN] load pack {pack_dir}")
        patients = joblib.load(pack_dir / "patients.pkl")
        sur_and_time = joblib.load(pack_dir / "sur_and_time.pkl")
        all_data = joblib.load(pack_dir / "all_data.pkl")
        return patients, sur_and_time, all_data

    # effective_config 里的路径带 bash 转义（如 "hgcn\ data"），先还原再重映射
    def _unescape(value: str) -> str:
        return value.replace("\\ ", " ").replace('\\"', '"')

    data_root_dir = train.remap_workspace_path_to_hgcn_data(
        _unescape(config["DATA_ROOT_DIR"]), project_root
    )
    gene_dir = train.remap_workspace_path_to_hgcn_data(
        _unescape(config["GENE_DIR"]), project_root
    )
    clinic_dir = train.remap_workspace_path_to_hgcn_data(
        _unescape(config["CLINIC_DIR"]), project_root
    )
    print(f"[HGCN] build graphs: {study} from {data_root_dir}")
    patients, sur_and_time, all_data = train.load_hgcn_graphs_from_dirs(
        study,
        data_root_dir=data_root_dir,
        gene_dir=gene_dir,
        clinic_dir=clinic_dir,
        repo_root=project_root,
    )
    print(f"[HGCN] dump pack {pack_dir}")
    build.dump_study_pack(
        study,
        patients,
        sur_and_time,
        all_data,
        wsi_experiment=wsi_exp,
        gene_experiment=gene_exp,
        clinic_scheme=clinic_exp,
        repo_root=project_root,
    )
    return patients, sur_and_time, all_data


def process_hgcn_model_dir(model_dir: Path, folds: list[int], out_csv: Path) -> None:
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    from Table1_CoxBreslow_Forward import (
        load_csv_cindex_by_fold,
    )
    from utils.survival_metrics import (
        AUC_LANDMARK_MONTHS,
        IBS_GRID_MONTHS,
        MIN_EVENTS_FOR_AUC,
        breslow_survival,
        compute_ibs,
        compute_landmark_aucs,
    )

    config = _load_effective_config(model_dir)
    study = config.get("STUDY", "unknown")
    project_root = Path(_PROJECT_ROOT)
    train, build = load_hgcn_modules()

    patients_raw, sur_and_time, all_data = _load_graphs(
        train, build, config, project_root
    )
    patients = [str(case_id) for case_id in patients_raw if str(case_id) in all_data]
    if not patients:
        raise ValueError("hgcn pack has no assembled graphs")
    patient_sur_type, patient_and_time, _ = train.get_patients_information(
        patients, sur_and_time
    )

    # censoring cohort：全数据集（train+val+test 并集），与 cox 前向同口径
    cohort_times = np.array([patient_and_time[p] for p in patients], dtype=np.float64)
    cohort_cens = np.array(
        [1.0 - float(patient_sur_type[p]) for p in patients], dtype=np.float64
    )
    keep = cohort_times >= 0.0
    cohort_surv = Surv.from_arrays(
        event=(1.0 - cohort_cens[keep]) > 0.5, time=cohort_times[keep]
    )

    gnn_seeds = joblib.load(model_dir / "all_gnn_time.pkl")
    if not gnn_seeds:
        raise FileNotFoundError(f"{model_dir / 'all_gnn_time.pkl'} has no seed dicts")
    seed_gnn = gnn_seeds[0]

    split_dir = config.get("SPLIT_DIR") or str(
        project_root / "splits" / "5foldcv" / study
    )
    csv_path = model_dir / "test_result.csv"
    fold_to_cindex = load_csv_cindex_by_fold(csv_path) if csv_path.is_file() else {}

    device = train.device

    for fold in folds:
        split_csv_path = Path(split_dir) / f"splits_{fold}.csv"
        train_split, _, test_split = train._load_split_csv(split_csv_path)
        train_ids = train._intersect_split_ids(
            train_split, patients, "train", split_csv_path
        )
        test_ids = train._intersect_split_ids(
            test_split, patients, "test", split_csv_path
        )
        test_ids = [pid for pid in test_ids if pid in seed_gnn]
        if not test_ids:
            print(f"[HGCN] fold {fold}: no test predictions in all_gnn_time.pkl")
            continue
        ckpt_paths = sorted(Path(model_dir).glob(f"seed_*/fold_{fold + 1}/*.pth"))
        if not ckpt_paths:
            print(f"[HGCN] fold {fold}: missing checkpoint under seed_*/fold_{fold + 1}/")
            continue

        model = train.fusion_model_mae_2(
            img_in_feats=train._infer_hgcn_in_feats(all_data, "x_img", 1024),
            rna_in_feats=train._infer_hgcn_in_feats(all_data, "x_rna", 1024),
            cli_in_feats=train._infer_hgcn_in_feats(all_data, "x_cli", 1024),
            n_hidden=HGCN_N_HIDDEN,
            out_classes=HGCN_OUT_CLASSES,
            dropout=HGCN_DROPOUT,
            train_type_num=len(HGCN_TRAIN_USE_TYPE),
        ).to(device)
        import torch

        model.load_state_dict(torch.load(str(ckpt_paths[0]), map_location=device))
        model.eval()

        print(f"[HGCN] fold {fold}: forward train risks ({len(train_ids)} patients)")
        train_risks: list[float] = []
        with torch.no_grad():
            for pid in train_ids:
                graph = all_data[pid].to(device)
                out_pre, _, _, _ = model(
                    graph, HGCN_TRAIN_USE_TYPE, HGCN_TRAIN_USE_TYPE, mix=HGCN_MIX
                )
                train_risks.append(float(out_pre[0].detach().cpu().numpy()[0]))
        del model

        train_times = np.array(
            [patient_and_time[pid] for pid in train_ids], dtype=np.float64
        )
        train_events = np.array(
            [float(patient_sur_type[pid]) > 0.5 for pid in train_ids], dtype=bool
        )
        train_risks_arr = np.asarray(train_risks, dtype=np.float64)

        test_times = np.array(
            [patient_and_time[pid] for pid in test_ids], dtype=np.float64
        )
        test_cens = np.array(
            [1.0 - float(patient_sur_type[pid]) for pid in test_ids], dtype=np.float64
        )
        test_risks = np.array(
            [float(seed_gnn[pid]) for pid in test_ids], dtype=np.float64
        )
        valid = test_times >= 0.0
        test_times, test_cens, test_risks = (
            test_times[valid],
            test_cens[valid],
            test_risks[valid],
        )
        test_surv = Surv.from_arrays(
            event=(1.0 - test_cens) > 0.5, time=test_times
        )
        test_min, test_max = float(test_times.min()), float(test_times.max())

        row: dict[str, object] = {
            "study": study,
            "model": "hgcn",
            "fold": fold,
        }
        recorded = fold_to_cindex.get(fold)
        if recorded is not None:
            recomputed = float(
                concordance_index_censored(
                    (1.0 - test_cens) > 0.5, test_times, test_risks, tied_tol=1e-08
                )[0]
            )
            row["cindex_abs_diff"] = abs(recomputed - recorded)
        else:
            row["cindex_abs_diff"] = math.nan

        estimate_grid = breslow_survival(
            train_times, train_events, train_risks_arr, test_risks, IBS_GRID_MONTHS
        )
        estimate_landmarks = 1.0 - breslow_survival(
            train_times,
            train_events,
            train_risks_arr,
            test_risks,
            np.asarray(AUC_LANDMARK_MONTHS, dtype=np.float64),
        )

        if test_min <= 1.0 and test_max >= 60.0:
            try:
                ibs = compute_ibs(cohort_surv, test_surv, estimate_grid)
                row["ibs"] = ibs if math.isfinite(ibs) else math.nan
            except Exception as exc:
                print(f"[HGCN] fold {fold}: IBS failed: {exc}")
                row["ibs"] = math.nan
        else:
            row["ibs"] = math.nan

        available = []
        for landmark in AUC_LANDMARK_MONTHS:
            n_events = int(((test_cens < 1.0) & (test_times <= landmark)).sum())
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
                print(f"[HGCN] fold {fold}: AUC failed: {exc}")

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
            writer.writerow(row)
        print(
            f"[HGCN] fold {fold}: ibs={row.get('ibs')} auc24={row.get('auc24')} "
            f"auc60={row.get('auc60')} cindex_diff={row.get('cindex_abs_diff')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hgcn forward-only Breslow IBS/AUC"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    folds = [int(part) for part in args.folds.split(",") if part.strip()]
    process_hgcn_model_dir(args.model_dir, folds, args.out)


if __name__ == "__main__":
    main()
