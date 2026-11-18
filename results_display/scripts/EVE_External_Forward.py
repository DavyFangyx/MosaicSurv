"""
外部验证前向引擎（EVE1/EVE2 共用，零训练、零梯度）。

用源数据集（TCGA-COAD）训练好的 checkpoint，对目标队列做纯前向，产出与训练
管线同格式的 `split_{k}_results.pkl`（{case_id: {time, risk, censorship,
clinical, logits}}），供 EVE1/EVE2 的 display 脚本消费。

- 模型与训练 args 从模型目录 run.log 的 [CMD] 行精确重建（与
  Table1_CoxBreslow_Forward.py 同机制），保证模型构造与原训练逐位一致；
- 目标为外部队列（--target-study cptac_coad）时：
    * 路径按目标 study 重映射（label/omics/WSI/clinical/gene/splits）；
    * 注入源工厂的冻结 bins（TCGA-COAD 分箱边界，仅首尾按 CPTAC 时间范围放宽）；
    * 每折使用源数据集第 k 折训练集的 MinMaxScaler（与内部验证口径一致）；
- 目标为源队列训练折（--target-split train）时：直接前向源训练折，
  产出 EVE2 阈值 τ_k 所需的训练集风险分数。

用法（SurvPGC env, GPU）:
  # CPTAC-COAD 外部前向（EVE1/EVE2 共享一次）
  python results_display/scripts/EVE_External_Forward.py --all-models \
      --target-study cptac_coad --out-root results/EVE1_Table1_Cindex_Main \
      --gpu 2 [--folds 0,1,2,3,4]

  # TCGA-COAD 训练折前向（EVE2 的 τ_k）
  python results_display/scripts/EVE_External_Forward.py --all-models \
      --target-split train --out-root results/EVE2_FigC_KM_curves/train_forward \
      --gpu 2 [--folds 0,1,2,3,4]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # 数据集代码按 cwd 读 ./datasets_csv/...

from utils.file_utils import _save_pkl

# Table1 官方模型清单（coad 树内的 11 个非 hgcn 模型；hgcn 走独立前向管线）
TABLE1_COAD_MODELS = {
    "mlp_clinic_flatten": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__L0/mlp_clinic_flatten",
    "snn_clinic_flatten": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__L0/snn_clinic_flatten",
    "mlp_gene_f": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__scFoundation_embedding_cell_norm/mlp_gene_f",
    "snn_gene_f": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__scFoundation_embedding_cell_norm/snn_gene_f",
    "survpc_f": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__L0__cell_norm__uni_v1/survpc_f",
    "porpoise": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__L0__cell_norm__uni_v1/porpoise",
    "survpath": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__L0__cell_norm__uni_v1/survpath",
    "mcat": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__L0__cell_norm__uni_v1/mcat",
    "survgc_f": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__L0__cell_norm__uni_v1/survgc_f",
    "survpgc_f": "results/Table1_Cindex_Main/L0Test/L0_COAD_full_model_val/tcga_coad__L0__cell_norm__uni_v1/survpgc_f",
    "mosaic_surv": "results/Table1_Cindex_Main/L0Test/L0_COAD_poe_model_val/tcga_coad__L0__cell_norm__uni_v1__mosaic_surv/mosaic_surv",
}

# ── EVE3（真实缺失评估）──────────────────────────────────────────────
EVE3_STUDIES = ["tcga_brca", "tcga_coad", "tcga_kirc", "tcga_kirp",
                "tcga_lihc", "tcga_luad", "tcga_lusc"]
# Table2 树的模型目录名（与用户口径名不同处做映射；hgcn 走标注不入前向）
EVE3_MISSING_MODELS = {
    "modality_concat_zero": "modality_concat__resampler__zero",
    "modality_concat_mean": "modality_concat__resampler__mean",
    "mvae_poe": "mvae_poe",
    "mopoe": "mopoe",
}


def load_missing_case_avail(study: str) -> dict[str, dict[str, bool]]:
    """天然缺失患者（有生存数据者）的自然模态可用性，来自 data_tcgal_stats。"""
    from dataset_deployment.registry import get_dataset_config

    config = get_dataset_config(study)
    stats = pd.read_csv(f"{config.stats_dir}/{config.display_name}_patients.csv")
    stats.columns = [str(col).strip() for col in stats.columns]
    missing = stats[~((stats["has_P"] == 1) & (stats["has_C"] == 1) & (stats["has_G"] == 1))]
    eve3_meta = pd.read_csv(f"datasets_csv/eve3_metadata/{study}.csv", index_col=0)
    meta_cases = set(eve3_meta["case_id"].astype(str))
    avail: dict[str, dict[str, bool]] = {}
    for _, row in missing.iterrows():
        cid = str(row["submitter_id"]).strip()
        if cid not in meta_cases:
            continue  # 无生存数据，无法入 C-index 计算
        avail[cid] = {
            "wsi": int(row["has_P"]) == 1,
            "gene": int(row["has_G"]) == 1,
            "clinic": int(row["has_C"]) == 1,
        }
    return avail


def discover_eve3_model_dirs(study: str) -> dict[str, Path]:
    """EVE3 六模型目录：5 个来自 Table2 树，mosaic_surv 来自 Table1 L0Test poe 树。"""
    dirs: dict[str, Path] = {}
    base = Path("results/Table2_Baselines")
    for model_key, dir_name in EVE3_MISSING_MODELS.items():
        candidate = base / f"{study}__L0__gene_raw__uni_v1" / dir_name
        if candidate.is_dir():
            dirs[model_key] = candidate
    for poe_root in sorted(Path("results/Table1_Cindex_Main/L0Test").glob("L0_*_poe_model_val")):
        candidate = poe_root / f"{study}__L0__cell_norm__uni_v1__mosaic_surv" / "mosaic_surv"
        if candidate.is_dir():
            dirs["mosaic_surv"] = candidate
    return dirs


def write_natural_mask_csv(args, fold: int, study: str, eval_cases: list[str],
                           missing_avail: dict[str, dict[str, bool]]) -> Path:
    """把混合队列每个患者的自然 avail 写成 unified_mask_csv 格式的 mask CSV。"""
    from utils.missing_mask_protocol import unified_mask_csv_path

    mask_path = unified_mask_csv_path(args, fold, study=study)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in eval_cases:
        avail = missing_avail.get(cid, {"wsi": True, "gene": True, "clinic": True})
        rows.append({
            "case_id": cid,
            "split": "test" if cid not in missing_avail else "missing",
            "wsi": int(avail["wsi"]),
            "gene": int(avail["gene"]),
            "clinic": int(avail["clinic"]),
        })
    import csv as csv_mod
    with mask_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_mod.DictWriter(handle, fieldnames=["case_id", "split", "wsi", "gene", "clinic"])
        writer.writeheader()
        writer.writerows(rows)
    return mask_path


def process_eve3_model(model_dir: Path, model_key: str, study: str, out_root: Path, folds: list[int]):
    """EVE3 前向：第 k 折测试集完整患者 ∪ 全部天然缺失患者，按自然 avail 输入。"""
    from utils.core_utils import _build_survival_dataset_factory

    from Table1_CoxBreslow_Forward import build_args_from_cmd_line

    args_src = build_args_from_cmd_line(model_dir)
    out_dir = out_root / study / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dst = copy.deepcopy(args_src)
    args_dst.label_file = str(Path(_PROJECT_ROOT) / "datasets_csv" / "eve3_metadata" / f"{study}.csv")
    args_dst.allow_missing_modality_files = True
    args_dst.missing_mode = "unified_mask_csv"
    args_dst.missing_pattern = "100,100,100"
    args_dst.missing_seed = 1
    args_dst.results_dir = str(out_dir)
    args_dst.exp_group = out_root.name
    factory_dst = _build_survival_dataset_factory(args_dst)
    args_dst.dataset_factory = factory_dst
    args_dst.omic_sizes = factory_dst.omic_sizes
    missing_avail = load_missing_case_avail(study)
    print(f"[EVE3] study={study} model={model_key} missing_patients={len(missing_avail)} folds={folds}")

    metrics: dict[str, list] = {key: [] for key in (
        "test_cindex", "test_cindex_ipcw", "test_IBS", "test_iauc",
        "test_iauc_list", "test_loss", "test_BS")}
    for fold in folds:
        split_csv = pd.read_csv(os.path.join(args_dst.split_dir, f"splits_{fold}.csv"))
        test_cases = [str(c) for c in split_csv["test"].dropna().tolist()]
        eval_cases = test_cases + [c for c in sorted(missing_avail) if c not in test_cases]
        write_natural_mask_csv(args_dst, fold, study, eval_cases, missing_avail)
        _, scaler = factory_dst._get_split_from_df(
            args_dst, split_csv, split_key="train", fold=fold, scaler=None)
        eval_df = pd.DataFrame({"test": eval_cases})
        split_dataset = factory_dst._get_split_from_df(
            args_dst, eval_df, split_key="test", fold=fold, scaler=scaler)
        out_pkl = out_dir / f"split_{fold}_results.pkl"
        forward_fold(args_dst, model_dir, fold, split_dataset, out_pkl, metrics)

    if metrics["test_cindex"]:
        pd.DataFrame(metrics).to_csv(out_dir / "test_result.csv", index=False)
        print(f"[WRITE] {out_dir / 'test_result.csv'}")


# 源 study -> 目标 study 的路径重映射规则（锚定已知 token，避免误替换）
_PATH_REMAP_RULES = {
    "cptac_coad": [
        ("datasets_csv/metadata/tcga_coad.csv", "datasets_csv/metadata/cptac_coad.csv"),
        ("datasets_csv/clinical_data/tcga_coad_clinical.csv", "datasets_csv/clinical_data/cptac_coad_clinical.csv"),
        ("datasets_csv/raw_rna_data/combine/coad", "datasets_csv/raw_rna_data/combine/cptac_coad"),
        ("SurvPGC_Workspace/tcga_coad", "SurvPGC_Workspace/cptac_coad"),
        ("splits/5foldcv/tcga_coad", "splits/5foldcv/cptac_coad"),
    ],
    "cptac_brca": [
        ("datasets_csv/metadata/tcga_brca.csv", "datasets_csv/metadata/cptac_brca.csv"),
        ("datasets_csv/clinical_data/tcga_brca_clinical.csv", "datasets_csv/clinical_data/cptac_brca_clinical.csv"),
        ("datasets_csv/raw_rna_data/combine/brca", "datasets_csv/raw_rna_data/combine/cptac_brca"),
        ("SurvPGC_Workspace/tcga_brca", "SurvPGC_Workspace/cptac_brca"),
        ("splits/5foldcv/tcga_brca", "splits/5foldcv/cptac_brca"),
    ],
}

# 外部验证队列：目标 study -> 源 study
EXTERNAL_COHORTS = {"cptac_coad": "tcga_coad", "cptac_brca": "tcga_brca"}


def discover_table1_model_dirs(study: str) -> dict[str, Path]:
    """按源 study 发现 Table1 官方 11 个非 hgcn 模型目录（目录树结构逐 study 相同）。"""
    from dataset_deployment.registry import get_dataset_config

    token = get_dataset_config(study).display_name.split("-")[1]
    base = Path("results/Table1_Cindex_Main/L0Test")
    dirs: dict[str, Path] = {}
    for model in ("mlp_clinic_flatten", "snn_clinic_flatten"):
        d = base / f"L0_{token}_full_model_val" / f"{study}__L0" / model
        if d.is_dir():
            dirs[model] = d
    for model in ("mlp_gene_f", "snn_gene_f"):
        d = base / f"L0_{token}_full_model_val" / f"{study}__scFoundation_embedding_cell_norm" / model
        if d.is_dir():
            dirs[model] = d
    for model in ("survpc_f", "porpoise", "survpath", "mcat", "survgc_f", "survpgc_f"):
        d = base / f"L0_{token}_full_model_val" / f"{study}__L0__cell_norm__uni_v1" / model
        if d.is_dir():
            dirs[model] = d
    d = base / f"L0_{token}_poe_model_val" / f"{study}__L0__cell_norm__uni_v1__mosaic_surv" / "mosaic_surv"
    if d.is_dir():
        dirs["mosaic_surv"] = d
    return dirs


def remap_args_for_target(args, target_study: str):
    """复制源 args 并把数据路径重映射到目标 study（模型相关字段不变）。"""
    dst = copy.deepcopy(args)
    dst.study = target_study
    rules = _PATH_REMAP_RULES[target_study]
    for field in ("label_file", "clinical_file", "omics_dir", "data_root_dir",
                  "clinic_dir", "gene_dir", "split_dir"):
        value = getattr(dst, field, None)
        if not value:
            continue
        for old, new in rules:
            value = value.replace(old, new)
        setattr(dst, field, value)
    return dst


def forward_fold(
    args,
    model_dir: Path,
    fold: int,
    split_dataset,
    out_pkl: Path,
    out_metrics: dict[str, list],
):
    """对给定 split 数据集做一次前向并落盘 pkl（与训练管线同格式）。"""
    from utils.core_utils import (
        _init_loss_function,
        _init_model,
        _load_model_checkpoint,
        _summary,
    )
    from utils.general_utils import _get_split_loader

    ckpt = model_dir / f"s_{fold}_checkpoint.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")

    model = _init_model(args, current_fold=fold)
    model = _load_model_checkpoint(model, ckpt, args)
    loss_fn = _init_loss_function(args)

    loader = _get_split_loader(
        args, split_dataset, training=False, testing=False,
        weighted=False, batch_size=1, disable_cox_batch_override=True,
    )
    results, cindex, cindex_ipcw, bs, ibs, iauc, iauc_list, total_loss, _ = _summary(
        args, args.dataset_factory, model, args.modality, loader, loss_fn, None
    )
    _save_pkl(str(out_pkl), results)
    out_metrics["test_cindex"].append(cindex)
    out_metrics["test_cindex_ipcw"].append(cindex_ipcw)
    out_metrics["test_IBS"].append(ibs)
    out_metrics["test_iauc"].append(iauc)
    out_metrics["test_iauc_list"].append(iauc_list)
    out_metrics["test_loss"].append(total_loss)
    out_metrics["test_BS"].append(bs)
    print(f"[FOLD {fold}] cindex={cindex:.4f} n={len(results)} -> {out_pkl}")


def process_model_dir(
    model_dir: Path,
    target_study: str | None,
    target_split: str,
    out_root: Path,
    folds: list[int],
):
    from utils.core_utils import _build_survival_dataset_factory

    from Table1_CoxBreslow_Forward import build_args_from_cmd_line

    args_src = build_args_from_cmd_line(model_dir)
    model_name = model_dir.name
    cohort = target_study if (target_split == "test" and target_study != args_src.study) else args_src.study
    out_dir = out_root / cohort / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    args_src.results_dir = str(out_dir)  # model.txt 等运行产物不污染官方树

    # 源工厂（coad）：提供冻结 bins 与每折训练集 scaler
    factory_src = _build_survival_dataset_factory(args_src)
    args_src.dataset_factory = factory_src
    args_src.omic_sizes = factory_src.omic_sizes  # return_splits 同款副作用
    source_study = args_src.study

    if target_split == "train":
        args_dst, factory_dst, target_study_used = args_src, factory_src, source_study
    elif target_study is None or target_study == source_study:
        # 目标即源队列（内部回归验证）：路径与 bins 均不改动
        args_dst, factory_dst, target_study_used = args_src, factory_src, source_study
    else:
        args_dst = remap_args_for_target(args_src, target_study)
        args_dst.results_dir = str(out_dir)
        args_dst.bins_override = factory_src.bins  # 冻结 TCGA-COAD 分箱
        factory_dst = _build_survival_dataset_factory(args_dst)
        args_dst.dataset_factory = factory_dst
        args_dst.omic_sizes = factory_dst.omic_sizes
        target_study_used = target_study

    meta = {
        "source_model_dir": str(model_dir),
        "source_study": source_study,
        "target_study": target_study_used,
        "target_split": target_split,
        "modality": args_src.modality,
        "frozen_bins_from": source_study if target_split == "test" else None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[FORWARD] model={model_name} modality={args_src.modality} "
          f"target={target_study_used}/{target_split} folds={folds}")

    metrics: dict[str, list] = {key: [] for key in (
        "test_cindex", "test_cindex_ipcw", "test_IBS", "test_iauc",
        "test_iauc_list", "test_loss", "test_BS")}
    for fold in folds:
        src_csv = pd.read_csv(os.path.join(args_src.split_dir, f"splits_{fold}.csv"))
        if target_split == "train":
            split_dataset, _ = factory_src._get_split_from_df(
                args_src, src_csv, split_key="train", fold=fold, scaler=None)
            out_pkl = out_dir / f"split_{fold}_results_train.pkl"
        else:
            # 第 k 折源训练集的 scaler（与内部验证对 test 折的归一化口径一致）
            _, scaler = factory_src._get_split_from_df(
                args_src, src_csv, split_key="train", fold=fold, scaler=None)
            dst_csv = pd.read_csv(os.path.join(args_dst.split_dir, f"splits_{fold}.csv"))
            split_dataset = factory_dst._get_split_from_df(
                args_dst, dst_csv, split_key="test", fold=fold, scaler=scaler)
            out_pkl = out_dir / f"split_{fold}_results.pkl"
        forward_fold(args_dst, model_dir, fold, split_dataset, out_pkl, metrics)

    if target_split != "train" and metrics["test_cindex"]:
        pd.DataFrame(metrics).to_csv(out_dir / "test_result.csv", index=False)
        print(f"[WRITE] {out_dir / 'test_result.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EVE 外部验证前向引擎")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--all-models", action="store_true",
                        help="遍历 Table1 官方 11 个非 hgcn 模型目录")
    parser.add_argument("--source-study", type=str, default="tcga_coad",
                        help="源训练 study（--all-models 时用于发现模型目录）")
    parser.add_argument("--target-study", type=str, default=None,
                        help="外部目标 study（如 cptac_coad）；不传则保持源 study")
    parser.add_argument("--target-split", type=str, default="test",
                        choices=["test", "train"],
                        help="test=外部/内部测试折前向；train=源队列训练折前向（EVE2 τ_k）")
    parser.add_argument("--eve3-study", type=str, default=None,
                        help="EVE3 真实缺失评估：对指定 TCGA study 跑 5 模型混合队列前向")
    parser.add_argument("--eve3-all", action="store_true",
                        help="EVE3 全部 7 个主线 TCGA study")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="EVE1/EVE2 前向必填；EVE3 默认 results/EVE3_Table2_Modality_Missing_Natural")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    folds = [int(part) for part in args.folds.split(",") if part.strip()]

    if args.eve3_study or args.eve3_all:
        studies = EVE3_STUDIES if args.eve3_all else [args.eve3_study]
        out_root = args.out_root or Path("results/EVE3_Table2_Modality_Missing_Natural")
        for study in studies:
            if args.model_dir is not None:
                model_dirs = {args.model_dir.name: args.model_dir}
            else:
                model_dirs = discover_eve3_model_dirs(study)
                if not model_dirs:
                    print(f"[SKIP] study={study}: no EVE3 model dirs found")
                    continue
            for model_key, model_dir in model_dirs.items():
                try:
                    process_eve3_model(model_dir, model_key, study, out_root, folds)
                except Exception as exc:
                    import traceback
                    print(f"[FAIL] study={study} model={model_key}: {exc}")
                    traceback.print_exc()
        return

    if args.out_root is None:
        raise SystemExit("--out-root is required for non-EVE3 forward")
    if args.model_dir is not None:
        model_dirs = [args.model_dir]
    elif args.all_models:
        model_dirs = [Path(p) for p in discover_table1_model_dirs(args.source_study).values()]
        if not model_dirs:
            raise SystemExit(f"no Table1 model dirs found for source study {args.source_study}")
    else:
        raise SystemExit("use --model-dir or --all-models")

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            print(f"[SKIP] missing model dir: {model_dir}")
            continue
        try:
            process_model_dir(
                model_dir,
                target_study=args.target_study,
                target_split=args.target_split,
                out_root=args.out_root,
                folds=folds,
            )
        except Exception as exc:  # 单个模型失败不阻断其余模型
            import traceback
            print(f"[FAIL] model_dir={model_dir}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
