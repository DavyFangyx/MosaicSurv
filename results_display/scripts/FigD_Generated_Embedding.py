"""FigD UMAP/t-SNE overlay of real vs PoE-reconstructed gene/clinic features.

Real G/C and completed G'/C' are reduced together and plotted on the same axes.
This is a qualitative check: completed features should overlap the real cloud
instead of collapsing to a single mean patient.

The reducer follows results_display/scripts/backup/scripts/clip_alignment_vis.py.
Results go under results_display/FigD_Generated_Heatmaps/<dataset>/fold<k>/<poe_model>/embedding/...

Example:
conda activate SurvPGC
python results_display/scripts/FigD_Generated_Embedding.py \
    --dataset LIHC \
    --fold 0 \
    --poe-model mosaic_surv \
    --eval-subset P,PC,PG,C,G \
    --method umap \
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
OUTPUT_EXPERIMENT = "embedding"
ALLOWED_EVAL_SUBSETS = ("P", "PC", "PG", "C", "G")
SUBSET_EMBED = {
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
REAL_COLOR = "#2A9D8F"
GEN_COLOR = "#D1495B"


def parse_args():
    parser = argparse.ArgumentParser(description="FigD UMAP/t-SNE overlay of real vs generated G/C features")
    parser.add_argument("--dataset", type=str, default="LIHC")
    parser.add_argument("--study", type=str, default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--poe-model", type=str, default="mosaic_surv_single")
    parser.add_argument(
        "--eval-subset",
        type=str,
        default="P,PC,PG",
        help="Observed modalities. Accepts P,PC,PG,C,G or a brace list such as {P,PC,PG}",
    )
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--max-cases", type=int, default=0, help="0 means use the full split")
    parser.add_argument("--cases", type=str, default="", help="Comma-separated case ids; default is fold split")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne", "pca"])
    parser.add_argument("--gallery", type=str, default="all", choices=["all", "split"], help="Reduce against all real cases or only the query split")
    parser.add_argument("--seed", type=int, default=42)
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


def load_case_ids(args, study, metadata):
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
    return np.asarray(tensor.detach().cpu(), dtype=np.float32).reshape(-1)


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


def reduce_dims(features, method, seed=42):
    n_samples = features.shape[0]
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                random_state=seed,
                min_dist=0.1,
                n_neighbors=min(15, max(2, n_samples // 3)),
                metric="cosine",
                n_jobs=1,
            )
            return reducer.fit_transform(features), "umap"
        except Exception as exc:
            print(f"[warn] UMAP failed: {exc}; falling back to PCA")
            method = "pca"
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(4, n_samples // 4))
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=seed,
                metric="cosine",
                init="pca",
                learning_rate="auto",
                max_iter=1000,
            )
            return reducer.fit_transform(features), "tsne"
        except Exception as exc:
            print(f"[warn] t-SNE failed: {exc}; falling back to PCA")
            method = "pca"
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, "pca"


def overlay_stats(real_xy, gen_xy, paired_real_xy=None):
    real_std = np.std(real_xy, axis=0)
    gen_std = np.std(gen_xy, axis=0)
    stats = {
        "n_real": int(len(real_xy)),
        "n_gen": int(len(gen_xy)),
        "real_std_x": float(real_std[0]) if len(real_xy) else None,
        "real_std_y": float(real_std[1]) if len(real_xy) else None,
        "gen_std_x": float(gen_std[0]) if len(gen_xy) else None,
        "gen_std_y": float(gen_std[1]) if len(gen_xy) else None,
        "spread_ratio": float(np.mean(gen_std) / max(float(np.mean(real_std)), 1e-8)) if len(real_xy) and len(gen_xy) else None,
        "mean_pair_distance": None,
        "median_pair_distance": None,
    }
    if paired_real_xy is not None and len(paired_real_xy) == len(gen_xy) and len(gen_xy):
        pair_dist = np.linalg.norm(paired_real_xy - gen_xy, axis=1)
        stats["mean_pair_distance"] = float(np.mean(pair_dist))
        stats["median_pair_distance"] = float(np.median(pair_dist))
    return stats


def plot_overlay(real_xy, gen_xy, title, out_path, paired_real_xy=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(real_xy[:, 0], real_xy[:, 1], s=22, alpha=0.55, linewidths=0, c=REAL_COLOR, label="real")
    ax.scatter(gen_xy[:, 0], gen_xy[:, 1], s=42, alpha=0.92, linewidths=1.2, c=GEN_COLOR, label="generated", marker="x")
    if paired_real_xy is not None and len(paired_real_xy) == len(gen_xy):
        for src, dst in zip(paired_real_xy, gen_xy):
            ax.plot([src[0], dst[0]], [src[1], dst[1]], color="#9AA0A6", alpha=0.18, linewidth=0.6, zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_real_cloud(feature_dir, keep_ids=None):
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
    return ids, matrix


def collect_complete_cases(case_ids, metadata, paths):
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
                    "gene_real": flatten_feature(load_tensor(gene_path)),
                    "clinic_real": flatten_feature(load_tensor(clinic_path)),
                }
            )
        except Exception:
            skipped.append(case_id)
    return records, skipped


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
    print(f"[info] method={args.method} split={args.split} gallery={args.gallery}")

    case_ids = load_case_ids(args, study, metadata)
    poe_model = load_checkpoint(poe_ckpt, device)
    records, skipped = collect_complete_cases(case_ids, metadata, paths)
    keep_ids = None if args.gallery == "all" else set(case_ids)
    real_clouds = {
        "G": load_real_cloud(paths["gene_dir"], keep_ids),
        "C": load_real_cloud(paths["clinic_dir"], keep_ids),
    }
    print(f"[info] n_requested={len(case_ids)} n_loaded={len(records)} skipped={len(skipped)}")
    print(f"[info] n_real_G={len(real_clouds['G'][0])} n_real_C={len(real_clouds['C'][0])}")
    if len(records) < 4:
        raise RuntimeError(f"Need at least 4 cases for {args.method}, got {len(records)}")

    encoded_records = []
    for rec in records:
        x_path, x_omic, x_clinic = load_case_inputs(rec, device)
        encoded = encode_case(poe_model, x_path, x_omic, x_clinic)
        encoded_records.append((rec, encoded))
        del x_path, x_omic, x_clinic
        print(f"[encode] {rec['case_id']}", flush=True)
    subset_avails = {subset: make_avail(subset, device) for subset in subsets}
    summary_rows = []
    used_methods = set()
    for subset in subsets:
        avail = subset_avails[subset]
        real_bank = {"G": [], "C": []}
        gen_bank = {"G": [], "C": []}
        ids = []
        for rec, encoded in encoded_records:
            gene_gen, clinic_gen = reconstruct_from_encoding(poe_model, encoded, avail)
            ids.append(rec["case_id"])
            real_bank["G"].append(rec["gene_real"])
            real_bank["C"].append(rec["clinic_real"])
            gen_bank["G"].append(flatten_feature(gene_gen))
            gen_bank["C"].append(flatten_feature(clinic_gen))
            del gene_gen, clinic_gen
        subset_dir = output_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        for modality in SUBSET_EMBED[subset]:
            real_ids, real_mat = real_clouds[modality]
            gen_mat = np.stack(gen_bank[modality], axis=0)
            if real_mat.size == 0:
                raise RuntimeError(f"No real {modality} features found")
            stacked = np.concatenate([real_mat, gen_mat], axis=0)
            xy, used_method = reduce_dims(stacked, args.method, seed=args.seed)
            used_methods.add(used_method)
            n_real = len(real_ids)
            n_gen = len(ids)
            real_xy = xy[:n_real]
            gen_xy = xy[n_real:]
            real_index = {case_id: idx for idx, case_id in enumerate(real_ids)}
            paired_idx = [real_index[case_id] for case_id in ids if case_id in real_index]
            paired_gen = np.stack([gen_xy[i] for i, case_id in enumerate(ids) if case_id in real_index], axis=0) if paired_idx else gen_xy
            paired_real = real_xy[paired_idx] if paired_idx else None
            stats = overlay_stats(real_xy, gen_xy, paired_real_xy=paired_real)
            coords = pd.DataFrame(
                {
                    "kind": ["real"] * n_real + ["generated"] * n_gen,
                    "case_id": list(real_ids) + list(ids),
                    "x": np.concatenate([real_xy[:, 0], gen_xy[:, 0]]),
                    "y": np.concatenate([real_xy[:, 1], gen_xy[:, 1]]),
                }
            )
            coord_path = subset_dir / f"{modality}_{used_method}_coords.csv"
            fig_path = subset_dir / f"{modality}_{used_method}_overlay.png"
            coords.to_csv(coord_path, index=False)
            title = (
                f"{display} | {poe_model_name} | {subset} | {modality} vs {modality}' | "
                f"{used_method.upper()} | real={n_real} gen={n_gen}"
            )
            plot_overlay(real_xy, gen_xy, title, fig_path, paired_real_xy=paired_real)
            summary_rows.append(
                {
                    "dataset": display,
                    "study": study,
                    "fold": args.fold,
                    "poe_model": poe_model_name,
                    "eval_subset": subset,
                    "split": args.split,
                    "gallery": args.gallery,
                    "requested_method": args.method,
                    "used_method": used_method,
                    "modality": modality,
                    "figure": str(fig_path),
                    **stats,
                }
            )
            spread = stats["spread_ratio"]
            pair = stats["mean_pair_distance"]
            print(
                f"[plot] {subset} {modality}/{modality}' {used_method} real={n_real} gen={n_gen} "
                f"spread_ratio={None if spread is None else f'{spread:.3f}'} pair_dist={None if pair is None else f'{pair:.3f}'} -> {fig_path}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "embedding_overlay_summary.csv"
    meta_path = output_root / "embedding_overlay_meta.json"
    summary_df.to_csv(summary_path, index=False)
    meta = {
        "dataset": display,
        "study": study,
        "fold": args.fold,
        "poe_model": poe_model_name,
        "poe_ckpt": str(poe_ckpt),
        "eval_subsets": subsets,
        "split": args.split,
        "gallery": args.gallery,
        "requested_method": args.method,
        "used_methods": sorted(used_methods),
        "n_requested": len(case_ids),
        "n_loaded": len(records),
        "n_real_G": len(real_clouds["G"][0]),
        "n_real_C": len(real_clouds["C"][0]),
        "skipped": skipped,
        "summary": str(summary_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
