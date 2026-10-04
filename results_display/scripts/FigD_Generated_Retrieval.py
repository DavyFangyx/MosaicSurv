"""FigD retrieval metrics for PoE-reconstructed gene/clinic features.

For each eval subset, reconstruct the missing modality, then retrieve G'_i or C'_i
against the gallery of all real G or C vectors. Report Top-1 / Top-5 accuracy,
mean reciprocal rank, and percentile rank. Chance is 1/N.

Results go under results_display/FigD_Generated_Heatmaps/<dataset>/fold<k>/<poe_model>/retrieval/...

Example:
conda activate SurvPGC
python results_display/scripts/FigD_Generated_Retrieval.py \
    --dataset LIHC \
    --fold 0 \
    --poe-model mosaic_surv \
    --eval-subset P,PC,PG,C,G \
    --split test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-figd")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_figd")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _dir in (_SCRIPT_DIR, str(PROJECT_ROOT)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from dataset_deployment.registry import DATASET_CONFIGS, infer_standard_paths
from dataset_deployment.workspace_features import resolve_case_feature_path
from models.missing_modality_baselines.common import apply_eval_subset
from model_names import MAIN_MODEL, canonical_model_key  # noqa: E402


GROUP_DIR = "Table1_Cindex_Main"
TEST_DIR_TEMPLATE = "L0Test"
# 变体模型（mosaic_surv_single / _multi / _twostage / ...）的 checkpoint 在
# Table4 消融批次里（新单组方案：results/Table4_Abaltion_Test/{study}__cell_norm__uni_v1/）。
VARIANT_GROUP_DIR = "Table4_Abaltion_Test"
OUTPUT_SERIES = "FigD_Generated_Heatmaps"
OUTPUT_EXPERIMENT = "retrieval"
ALLOWED_EVAL_SUBSETS = ("P", "PC", "PG", "C", "G")
SUBSET_RETRIEVAL = {
    "P": ("G", "C"),
    "PC": ("G",),
    "PG": ("C",),
    "C": ("G",),
    "G": ("C",),
}
STUDY_SPECS = {
    "BRCA": "tcga_brca",
    "COAD": "tcga_coad",
    "KICH": "tcga_kich",
    "KIRC": "tcga_kirc",
    "KIRP": "tcga_kirp",
    "LIHC": "tcga_lihc",
    "LUAD": "tcga_luad",
    "LUSC": "tcga_lusc",
    "PRAD": "tcga_prad",
    "STAD": "tcga_stad",
}


def parse_args():
    parser = argparse.ArgumentParser(description="FigD retrieval of generated G'/C' against real galleries")
    parser.add_argument("--dataset", type=str, default="LIHC")
    parser.add_argument("--study", type=str, default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--poe-model", type=str, default="mosaic_surv")
    parser.add_argument(
        "--eval-subset",
        type=str,
        default="P,PC,PG",
        help="Observed modalities. Accepts P,PC,PG,C,G or a brace list such as {P,PC,PG}",
    )
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--max-cases", type=int, default=0, help="0 means use the full split as queries")
    parser.add_argument("--cases", type=str, default="", help="Comma-separated query case ids; default is fold split")
    parser.add_argument("--gallery", type=str, default="all", choices=["all", "split"], help="Retrieve against all real cases or only the query split")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--clinic-experiment", type=str, default="L0")
    parser.add_argument("--wsi-experiment", type=str, default="uni_v1")
    parser.add_argument("--gene-experiment", type=str, default="scFoundation_embedding_cell_norm")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--group-dir", type=str, default=GROUP_DIR)
    parser.add_argument("--test-dir", type=str, default=TEST_DIR_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--poe-ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def parse_eval_subsets(raw):
    text = str(raw).strip().replace("{", "").replace("}", "").replace("，", ",")
    parts = [token.strip().upper() for token in re.split(r"[,\s]+", text) if token.strip()]
    if not parts:
        raise ValueError("No eval subsets provided")
    subsets = []
    seen = set()
    for subset in parts:
        if subset not in ALLOWED_EVAL_SUBSETS:
            raise ValueError(f"Unsupported eval subset {subset}. Allowed: {', '.join(ALLOWED_EVAL_SUBSETS)}")
        if subset not in seen:
            subsets.append(subset)
            seen.add(subset)
    return subsets


def resolve_study(args):
    if args.study:
        study = args.study if args.study.startswith("tcga_") else f"tcga_{args.study.lower()}"
        display = args.dataset or study.replace("tcga_", "").upper()
        return study, display
    key = str(args.dataset).strip().upper()
    if key in STUDY_SPECS:
        return STUDY_SPECS[key], key
    study = key.lower()
    if not study.startswith("tcga_"):
        study = f"tcga_{study}"
    return study, key


def normalize_poe_model(name):
    """旧拼写/正式名 → 注册键（model_names.py 集中映射）；映射外小写原样。"""
    return canonical_model_key(name)


def choose_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    obj = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        raise ValueError(f"{path} is a state-dict checkpoint; expected a full model object.")
    if isinstance(obj, torch.nn.Module):
        return obj.to(device).eval()
    raise TypeError(f"Unsupported checkpoint type {type(obj)} in {path}")


def find_poe_ckpt(args, study, poe_model):
    if args.poe_ckpt:
        path = Path(args.poe_ckpt)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    study_tag = study.replace("tcga_", "").upper()
    if poe_model == MAIN_MODEL:
        # 主模型 checkpoint 在 Table1 本批次的 ours 树里
        poe_root = (
            args.results_root
            / args.group_dir
            / args.test_dir
            / f"L0_{study_tag}_poe_model_val"
        )
    else:
        # 变体模型 checkpoint 在 Table4 消融批次里（单组方案，目录无 run_id 后缀）
        poe_root = (
            args.results_root
            / VARIANT_GROUP_DIR
            / f"{study}__cell_norm__{args.wsi_experiment}"
        )
    matches = sorted(poe_root.glob(f"**/{poe_model}/s_{args.fold}_checkpoint.pt"))
    matches = [p for p in matches if "stage1" not in p.name]
    if not matches:
        matches = sorted(poe_root.glob(f"**/*{poe_model}*/s_{args.fold}_checkpoint.pt"))
        matches = [p for p in matches if "stage1" not in p.name]
    if not matches:
        raise FileNotFoundError(f"PoE checkpoint not found under {poe_root} for {poe_model}")
    preferred = [p for p in matches if args.wsi_experiment in str(p) and "cell_norm" in str(p)]
    return (preferred or matches)[0]


def load_split_ids(study, fold, split):
    split_csv = PROJECT_ROOT / "splits" / "5foldcv" / study / f"splits_{fold}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)
    table = pd.read_csv(split_csv)
    if split == "all":
        ids = []
        for col in ("train", "val", "test"):
            if col in table.columns:
                ids.extend([str(x).strip().upper()[:12] for x in table[col].dropna().tolist()])
        seen = set()
        uniq = []
        for case_id in ids:
            if case_id and case_id not in seen:
                uniq.append(case_id)
                seen.add(case_id)
        return uniq
    if split not in table.columns:
        raise KeyError(f"{split} not in {split_csv}")
    return [str(x).strip().upper()[:12] for x in table[split].dropna().tolist()]


def load_query_ids(args, study, metadata):
    available = set(metadata["case_id"].astype(str).str.upper().str[:12])
    if args.cases:
        ids = [token.strip().upper()[:12] for token in args.cases.split(",") if token.strip()]
    else:
        ids = load_split_ids(study, args.fold, args.split)
    ids = [case_id for case_id in ids if case_id in available]
    if args.max_cases > 0:
        ids = ids[: args.max_cases]
    return ids


def load_tensor(path):
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        obj = obj.get("features", next(iter(obj.values())))
    return obj.float()


def first_slide_id(metadata, case_id):
    rows = metadata[metadata["case_id"].astype(str).str.upper().str[:12] == case_id]
    if rows.empty:
        raise KeyError(case_id)
    return str(rows.iloc[0]["slide_id"])


def flatten_feature(tensor):
    arr = np.asarray(tensor.detach().cpu(), dtype=np.float32).reshape(-1)
    return arr


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return matrix / norms


def make_avail(subset, device):
    dummy = {
        "wsi": torch.ones(1, dtype=torch.bool, device=device),
        "gene": torch.ones(1, dtype=torch.bool, device=device),
        "clinic": torch.ones(1, dtype=torch.bool, device=device),
    }
    return apply_eval_subset(dummy, subset)


def encode_case(poe_model, x_path, x_omic, x_clinic):
    poe_model.eval()
    if hasattr(poe_model, "set_training_stage"):
        poe_model.set_training_stage("stage2")
    with torch.no_grad():
        x_omic_tok = poe_model._reshape_gene(x_omic.float())
        x_clinic_tok = poe_model._reshape_clinic(x_clinic.float())
        wsi_tokens = poe_model.wsi_resampler(x_path.float(), padding_mask=None)
        mu_list, logvar_list = poe_model._encode_tokens(wsi_tokens, x_omic_tok, x_clinic_tok)
        return {
            "x_omic_tok": x_omic_tok.detach(),
            "x_clinic_tok": x_clinic_tok.detach(),
            "mu_list": [mu.detach() for mu in mu_list],
            "logvar_list": [lv.detach() for lv in logvar_list],
        }


def reconstruct_from_encoding(poe_model, encoded, avail):
    poe_model.eval()
    with torch.no_grad():
        x_omic_tok = encoded["x_omic_tok"]
        x_clinic_tok = encoded["x_clinic_tok"]
        available_mask = poe_model._normalize_avail(avail, x_omic_tok.device)
        mu_joint, logvar_joint, poe_weights = poe_model.poe(
            mus=encoded["mu_list"],
            logvars=encoded["logvar_list"],
            available_mask=available_mask,
        )
        recon_gene, recon_clinic = poe_model._decode(mu_joint)[1:]
        recon_gene = recon_gene.reshape_as(x_omic_tok)
        recon_clinic = recon_clinic.reshape_as(x_clinic_tok)
        gene_keep = available_mask[:, 1].view(-1, *([1] * (x_omic_tok.ndim - 1)))
        clinic_keep = available_mask[:, 2].view(-1, *([1] * (x_clinic_tok.ndim - 1)))
        gene_out = torch.where(gene_keep, x_omic_tok, recon_gene)
        clinic_out = torch.where(clinic_keep, x_clinic_tok, recon_clinic)
        return gene_out, clinic_out


def collect_query_cases(case_ids, metadata, paths):
    records = []
    skipped = []
    for case_id in case_ids:
        try:
            slide_id = first_slide_id(metadata, case_id)
            slide_stem = str(slide_id).replace(".svs", "")
            wsi_path = Path(paths["data_root_dir"]) / f"{slide_stem}.pt"
            gene_path = resolve_case_feature_path(paths["gene_dir"], case_id)
            clinic_path = resolve_case_feature_path(paths["clinic_dir"], case_id)
            if gene_path is None or clinic_path is None or not Path(gene_path).exists() or not Path(clinic_path).exists() or not wsi_path.exists():
                skipped.append(case_id)
                continue
            records.append(
                {
                    "case_id": case_id,
                    "slide_id": slide_id,
                    "wsi_path": wsi_path,
                    "gene_path": Path(gene_path),
                    "clinic_path": Path(clinic_path),
                }
            )
        except Exception:
            skipped.append(case_id)
    return records, skipped


def load_modality_gallery(feature_dir, keep_ids=None):
    ids = []
    vecs = []
    keep = None if keep_ids is None else set(keep_ids)
    for path in sorted(Path(feature_dir).glob("*.pt")):
        case_id = path.stem.upper()[:12]
        if keep is not None and case_id not in keep:
            continue
        try:
            vecs.append(flatten_feature(load_tensor(path)))
            ids.append(case_id)
        except Exception:
            continue
    matrix = np.stack(vecs, axis=0) if vecs else np.zeros((0, 1), dtype=np.float32)
    return {"ids": ids, "matrix": matrix}


def rank_against_gallery(query_vec, gallery_ids, gallery_matrix, true_id, metric):
    if true_id not in gallery_ids:
        return None
    true_idx = gallery_ids.index(true_id)
    if metric == "cosine":
        query = l2_normalize(query_vec.reshape(1, -1))[0]
        scores = gallery_matrix @ query
        order = np.argsort(-scores)
        score = float(scores[true_idx])
    else:
        diffs = gallery_matrix - query_vec.reshape(1, -1)
        dist = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dist)
        score = float(-dist[true_idx])
    rank = int(np.where(order == true_idx)[0][0]) + 1
    n = len(gallery_ids)
    return {
        "rank": rank,
        "n_gallery": n,
        "reciprocal_rank": 1.0 / rank,
        "percentile_rank": 100.0 * (n - rank + 1) / n,
        "score": score,
        "top1": int(rank == 1),
        "top5": int(rank <= 5),
    }


def plot_retrieval_table(summary_df, out_path, title):
    if summary_df.empty:
        return
    display_df = summary_df.copy()
    display_df["pair"] = display_df["eval_subset"] + " " + display_df["query_modality"] + "->" + display_df["gallery_modality"]
    cols = ["pair", "n_query", "n_gallery", "chance_top1", "top1", "top5", "mrr", "mean_rank", "mean_percentile_rank"]
    table = display_df[cols]
    fig_h = max(2.8, 0.55 * (len(table) + 2))
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    cell_text = []
    for _, row in table.iterrows():
        cell_text.append(
            [
                row["pair"],
                f"{int(row['n_query'])}",
                f"{int(row['n_gallery'])}",
                f"{row['chance_top1']:.4f}" if pd.notna(row["chance_top1"]) else "-",
                f"{row['top1']:.3f}" if pd.notna(row["top1"]) else "-",
                f"{row['top5']:.3f}" if pd.notna(row["top5"]) else "-",
                f"{row['mrr']:.3f}" if pd.notna(row["mrr"]) else "-",
                f"{row['mean_rank']:.1f}" if pd.notna(row["mean_rank"]) else "-",
                f"{row['mean_percentile_rank']:.1f}" if pd.notna(row["mean_percentile_rank"]) else "-",
            ]
        )
    headers = ["Query", "Nq", "Ng", "Chance", "Top-1", "Top-5", "MRR", "Mean rank", "Pct rank"]
    tbl = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_ranks(rows):
    if not rows:
        return {
            "n_query": 0,
            "n_gallery": 0,
            "chance_top1": None,
            "top1": None,
            "top5": None,
            "mrr": None,
            "mean_rank": None,
            "median_rank": None,
            "mean_percentile_rank": None,
        }
    ranks = np.asarray([row["rank"] for row in rows], dtype=np.float32)
    n_gallery = int(rows[0]["n_gallery"])
    return {
        "n_query": int(len(rows)),
        "n_gallery": n_gallery,
        "chance_top1": 1.0 / n_gallery if n_gallery else None,
        "top1": float(np.mean([row["top1"] for row in rows])),
        "top5": float(np.mean([row["top5"] for row in rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in rows])),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_percentile_rank": float(np.mean([row["percentile_rank"] for row in rows])),
    }


def load_case_inputs(rec, device):
    x_path = load_tensor(rec["wsi_path"]).unsqueeze(0).to(device)
    x_omic = load_tensor(rec["gene_path"]).unsqueeze(0).to(device)
    x_clinic = load_tensor(rec["clinic_path"]).unsqueeze(0).to(device)
    return x_path, x_omic, x_clinic


def main():
    args = parse_args()
    study, display = resolve_study(args)
    poe_model_name = normalize_poe_model(args.poe_model)
    subsets = parse_eval_subsets(args.eval_subset)
    device = choose_device(args.device)
    paths = infer_standard_paths(
        study,
        repo_root=PROJECT_ROOT,
        wsi_experiment=args.wsi_experiment,
        clinic_experiment=args.clinic_experiment,
        gene_experiment=args.gene_experiment,
    )
    config = DATASET_CONFIGS[study]
    metadata = pd.read_csv(PROJECT_ROOT / config.metadata_csv)
    output_root = args.output_dir or (
        PROJECT_ROOT / "results_display" / OUTPUT_SERIES / display / f"fold{args.fold}" / poe_model_name / OUTPUT_EXPERIMENT
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    poe_ckpt = find_poe_ckpt(args, study, poe_model_name)
    print(f"[info] device={device}")
    print(f"[info] PoE ckpt={poe_ckpt}")
    print(f"[info] output_root={output_root}")
    print(f"[info] eval_subsets={subsets}")
    print(f"[info] query_split={args.split} gallery={args.gallery} metric={args.metric}")

    query_ids = load_query_ids(args, study, metadata)
    if args.gallery == "all":
        gallery_ids = load_split_ids(study, args.fold, "all")
        available = set(metadata["case_id"].astype(str).str.upper().str[:12])
        gallery_ids = [case_id for case_id in gallery_ids if case_id in available]
    else:
        gallery_ids = list(query_ids)
    print(f"[info] n_query_requested={len(query_ids)} n_gallery_requested={len(gallery_ids)}")

    poe_model = load_checkpoint(poe_ckpt, device)
    keep_ids = None if args.gallery == "all" else set(gallery_ids)
    galleries = {
        "G": load_modality_gallery(paths["gene_dir"], keep_ids),
        "C": load_modality_gallery(paths["clinic_dir"], keep_ids),
    }
    if args.metric == "cosine":
        for modality in galleries:
            if len(galleries[modality]["ids"]):
                galleries[modality]["matrix"] = l2_normalize(galleries[modality]["matrix"])
    query_records, query_skipped = collect_query_cases(query_ids, metadata, paths)
    print(f"[info] n_query_loaded={len(query_records)} n_gallery_G={len(galleries['G']['ids'])} n_gallery_C={len(galleries['C']['ids'])}")
    if query_skipped:
        print(f"[info] skipped query cases={len(query_skipped)}")

    subset_avails = {subset: make_avail(subset, device) for subset in subsets}
    encoded_queries = []
    for rec in query_records:
        x_path, x_omic, x_clinic = load_case_inputs(rec, device)
        encoded = encode_case(poe_model, x_path, x_omic, x_clinic)
        encoded_queries.append((rec, encoded))
        del x_path, x_omic, x_clinic
        print(f"[encode] {rec['case_id']}", flush=True)
    summary_rows = []
    all_query_rows = []
    for subset in subsets:
        modalities = SUBSET_RETRIEVAL[subset]
        avail = subset_avails[subset]
        per_mod_rows = {mod: [] for mod in modalities}
        for rec, encoded in encoded_queries:
            gene_gen, clinic_gen = reconstruct_from_encoding(poe_model, encoded, avail)
            generated = {
                "G": flatten_feature(gene_gen),
                "C": flatten_feature(clinic_gen),
            }
            del gene_gen, clinic_gen
            for modality in modalities:
                gallery = galleries[modality]
                result = rank_against_gallery(
                    generated[modality],
                    gallery["ids"],
                    gallery["matrix"],
                    rec["case_id"],
                    args.metric,
                )
                if result is None:
                    continue
                row = {
                    "case_id": rec["case_id"],
                    "slide_id": rec["slide_id"],
                    "eval_subset": subset,
                    "query_modality": f"{modality}'",
                    "gallery_modality": modality,
                    **result,
                }
                per_mod_rows[modality].append(row)
                all_query_rows.append(row)
        subset_dir = output_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        for modality, rows in per_mod_rows.items():
            stats = summarize_ranks(rows)
            summary = {
                "dataset": display,
                "study": study,
                "fold": args.fold,
                "poe_model": poe_model_name,
                "eval_subset": subset,
                "query_split": args.split,
                "gallery": args.gallery,
                "metric": args.metric,
                "query_modality": f"{modality}'",
                "gallery_modality": modality,
                **stats,
            }
            summary_rows.append(summary)
            detail_path = subset_dir / f"{modality}_retrieval_ranks.csv"
            pd.DataFrame(rows).to_csv(detail_path, index=False)
            print(
                f"[metric] {subset} {modality}' -> {modality} | "
                f"Nq={stats['n_query']} Ng={stats['n_gallery']} chance={stats['chance_top1']}\n"
                f"         top1={stats['top1']} top5={stats['top5']} mrr={stats['mrr']} "
                f"mean_rank={stats['mean_rank']} pct={stats['mean_percentile_rank']}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "retrieval_summary.csv"
    details_path = output_root / "retrieval_ranks.csv"
    meta_path = output_root / "retrieval_meta.json"
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(all_query_rows).to_csv(details_path, index=False)
    figure_path = output_root / "retrieval_summary.png"
    plot_retrieval_table(
        summary_df,
        figure_path,
        f"{display} | {poe_model_name} | fold{args.fold} | {args.split} vs {args.gallery} gallery",
    )
    meta = {
        "dataset": display,
        "study": study,
        "fold": args.fold,
        "poe_model": poe_model_name,
        "poe_ckpt": str(poe_ckpt),
        "eval_subsets": subsets,
        "query_split": args.split,
        "gallery": args.gallery,
        "metric": args.metric,
        "n_query_requested": len(query_ids),
        "n_query_loaded": len(query_records),
        "n_gallery_G": len(galleries["G"]["ids"]),
        "n_gallery_C": len(galleries["C"]["ids"]),
        "skipped_query": query_skipped,
        "outputs": {
            "summary": str(summary_path),
            "ranks": str(details_path),
            "figure": str(figure_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] summary={summary_path}")
    print(f"[done] ranks={details_path}")
    print(f"[done] figure={figure_path}")


if __name__ == "__main__":
    main()
