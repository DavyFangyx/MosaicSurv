"""Plot FigD generated-feature SurvPGC_F heatmaps from GT vs PoE-reconstructed gene/clinic features.

PoE only fills missing gene/clinic tokens. WSI patches stay real and are required
for SurvPGC_F heatmaps, even when the eval subset is C or G. Heatmaps come from a
frozen SurvPGC_F checkpoint.

Each comparison figure is:
    left  = original modality attention
    right = generated modality attention
    below each full map: 5 highest-attention patches
    heatmap colormap is jet (red=high attention, blue=low attention)

A standalone difference heatmap is also saved for each pair:
    G' - G or C' - C, using a diverging colormap centered at 0
    on a blank canvas (score difference only, no WSI overlay).

Comparisons by eval subset:
    P  -> G vs G' and C vs C'
    PC -> G vs G'
    PG -> C vs C'
    C  -> G vs G'
    G  -> C vs C'

Results go under results_display/FigD_Generated_Heatmaps/<dataset>/fold<k>/<poe_model>/attention/...

--poe-model 支持新注册键（mosaic_surv 家族）以及 Cfilm 时代旧拼写
（经 results_display/scripts/model_names.py 的 LEGACY_ALIASES 归一化）：
    mosaic_surv, mosaic_surv_single, mosaic_surv_single_enum,
    mosaic_surv_multi, mosaic_surv_twostage, mosaic_surv_frozen,
    mosaic_surv_kl, mosaic_surv_nojeffreys, mosaic_surv_detached
主模型 checkpoint 来自 Table1_Cindex_Main/<test-dir>；变体来自
Table4_Abaltion_Test/<study>__*__<variant-run-id>。

Example:
conda activate SurvPGC
python results_display/scripts/FigD_Generated_Heatmaps.py \
    --dataset LIHC \
    --fold 0 \
    --poe-model mosaic_surv \
    --eval-subset P,PC,PG,C,G\
    --max-cases 3

    --skip-render 时只会写 npy，不渲染差分图。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-figd")

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _dir in (_SCRIPT_DIR, str(PROJECT_ROOT)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from dataset_deployment.registry import DATASET_CONFIGS, infer_standard_paths
from dataset_deployment.workspace_features import resolve_case_feature_path
from models.missing_modality_baselines.common import apply_eval_subset
from model_names import MAIN_MODEL, canonical_model_key  # noqa: E402
from wsi_core.wsi_utils import sample_rois


GROUP_DIR = "Table1_Cindex_Main"
TEST_DIR_TEMPLATE = "L0Test"
# 变体模型（mosaic_surv_single / _multi / _twostage / ...）的 checkpoint 在
# Table4 消融批次里，run_id 可选 t025_alpha_learn / t028_orig / t028_alpha_learn。
VARIANT_GROUP_DIR = "Table4_Abaltion_Test"
DEFAULT_VARIANT_RUN_ID = "t028_orig"
OUTPUT_SERIES = "FigD_Generated_Heatmaps"
OUTPUT_EXPERIMENT = "attention"
ALLOWED_EVAL_SUBSETS = ("P", "PC", "PG", "C", "G")
SUBSET_COMPARISONS = {
    "P": (("G", "Agp"), ("C", "Acp")),
    "PC": (("G", "Agp"),),
    "PG": (("C", "Acp"),),
    "C": (("G", "Agp"),),
    "G": (("C", "Acp"),),
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
CANVAS_BG = (255, 255, 255)
TITLE_COLOR = (20, 20, 20)
ACCENT_COLOR = (40, 40, 40)
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def parse_args():
    parser = argparse.ArgumentParser(description="FigD GT vs PoE-generated SurvPGC_F heatmaps")
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
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--cases", type=str, default="", help="Comma-separated case ids; default is fold test split")
    parser.add_argument("--clinic-experiment", type=str, default="L0")
    parser.add_argument("--wsi-experiment", type=str, default="uni_v1")
    parser.add_argument("--gene-experiment", type=str, default="scFoundation_embedding_cell_norm")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--group-dir", type=str, default=GROUP_DIR)
    parser.add_argument("--test-dir", type=str, default=TEST_DIR_TEMPLATE)
    parser.add_argument(
        "--variant-run-id",
        type=str,
        default=DEFAULT_VARIANT_RUN_ID,
        help="Table4 ablation run_id for variant models (t025_alpha_learn / t028_orig / t028_alpha_learn)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--slide-root", type=str, default="")
    parser.add_argument("--patches-dir", type=str, default="")
    parser.add_argument("--pgc-ckpt", type=str, default="")
    parser.add_argument("--poe-ckpt", type=str, default="")
    parser.add_argument("--vis-level", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--diff-cmap", type=str, default="coolwarm")
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-ext", type=str, default="jpg")
    parser.add_argument("--save-raw-heatmaps", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
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


def find_pgc_ckpt(args, study):
    if args.pgc_ckpt:
        path = Path(args.pgc_ckpt)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    study_tag = study.replace("tcga_", "").upper()
    root = args.results_root / args.group_dir / args.test_dir
    candidates = [
        root / f"L0_{study_tag}_full_model_val" / f"{study}__L0__cell_norm__{args.wsi_experiment}" / "survpgc_f" / f"s_{args.fold}_checkpoint.pt",
        root / f"L0_{study_tag}_full_model_val" / f"{study}__{args.clinic_experiment}__cell_norm__{args.wsi_experiment}" / "survpgc_f" / f"s_{args.fold}_checkpoint.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.glob(f"*_{study_tag}_full_model_val/**/survpgc_f/s_{args.fold}_checkpoint.pt"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"SurvPGC_F checkpoint not found. Tried {candidates}")


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
        # 变体模型 checkpoint 在 Table4 消融批次里
        poe_root = (
            args.results_root
            / VARIANT_GROUP_DIR
            / f"{study}__cell_norm__{args.wsi_experiment}__{args.variant_run_id}"
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


def build_svs_index(slide_root):
    index = {}
    if not slide_root.exists():
        return index
    for path in slide_root.rglob("*.svs"):
        index[path.name] = path
        index[path.stem] = path
    return index


def load_case_ids(args, study, metadata):
    if args.cases:
        return [token.strip().upper()[:12] for token in args.cases.split(",") if token.strip()]
    split_csv = PROJECT_ROOT / "splits" / "5foldcv" / study / f"splits_{args.fold}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)
    split = pd.read_csv(split_csv)
    ids = [str(x).strip().upper()[:12] for x in split["test"].dropna().tolist()]
    available = set(metadata["case_id"].astype(str).str.upper().str[:12])
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


def attention_to_patch_scores(attn):
    if torch.is_tensor(attn):
        attn = attn.detach().cpu()
    scores = np.asarray(attn, dtype=np.float32)
    scores = np.squeeze(scores)
    if scores.ndim == 0:
        raise ValueError("Attention collapsed to a scalar")
    if scores.ndim == 1:
        return scores
    return scores.mean(axis=tuple(range(scores.ndim - 1)))


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def signed_scores_to_heatmap(delta):
    delta = np.asarray(delta, dtype=np.float32)
    max_abs = float(np.max(np.abs(delta)))
    if max_abs < 1e-8:
        return np.full_like(delta, 50.0)
    return 50.0 + 50.0 * (delta / max_abs)


def risk_from_logits(logits):
    hazards = torch.sigmoid(logits)
    return float((-torch.sum(torch.cumprod(1 - hazards, dim=1), dim=1)).item())


def make_avail(subset, device):
    dummy = {
        "wsi": torch.ones(1, dtype=torch.bool, device=device),
        "gene": torch.ones(1, dtype=torch.bool, device=device),
        "clinic": torch.ones(1, dtype=torch.bool, device=device),
    }
    return apply_eval_subset(dummy, subset)


def reconstruct_missing(poe_model, x_path, x_omic, x_clinic, avail):
    poe_model.eval()
    if hasattr(poe_model, "set_training_stage"):
        poe_model.set_training_stage("stage2")
    with torch.no_grad():
        x_omic_tok = poe_model._reshape_gene(x_omic.float())
        x_clinic_tok = poe_model._reshape_clinic(x_clinic.float())
        available_mask = poe_model._normalize_avail(avail, x_path.device)
        wsi_tokens = poe_model.wsi_resampler(x_path.float(), padding_mask=None)
        mu_list, logvar_list = poe_model._encode_tokens(wsi_tokens, x_omic_tok, x_clinic_tok)
        mu_joint, logvar_joint, poe_weights = poe_model.poe(
            mus=mu_list,
            logvars=logvar_list,
            available_mask=available_mask,
        )
        recon_gene, recon_clinic = poe_model._decode(mu_joint)[1:]
        recon_gene = recon_gene.reshape_as(x_omic_tok)
        recon_clinic = recon_clinic.reshape_as(x_clinic_tok)
        gene_keep = available_mask[:, 1].view(-1, *([1] * (x_omic_tok.ndim - 1)))
        clinic_keep = available_mask[:, 2].view(-1, *([1] * (x_clinic_tok.ndim - 1)))
        gene_out = torch.where(gene_keep, x_omic_tok, recon_gene)
        clinic_out = torch.where(clinic_keep, x_clinic_tok, recon_clinic)
        cached = {
            "mu_joint": mu_joint,
            "logvar_joint": logvar_joint,
            "poe_weights": poe_weights,
            "available_mask": available_mask,
        }
        return gene_out, clinic_out, cached


def run_pgc_attention(pgc_model, x_path, x_omic, x_clinic, bag_loss="nll_surv"):
    pgc_model.eval()
    with torch.no_grad():
        outputs = pgc_model(
            x_path=x_path,
            x_omic=x_omic,
            x_clinic=x_clinic,
            return_attn=True,
            bag_loss=bag_loss,
        )
    logits, _app, agp, _apg, _acc, acp, _apc = outputs
    return logits, attention_to_patch_scores(agp), attention_to_patch_scores(acp)


def save_image(image, path, ext):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    if ext.lower() in {"jpg", "jpeg"}:
        rgb.save(path, quality=95)
    else:
        rgb.save(path)


def load_font(size):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    try:
        import matplotlib
        candidates.insert(0, Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    except Exception:
        pass
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def resize_to_width(image, width):
    if image.width == width:
        return image
    height = max(1, int(round(image.height * (width / image.width))))
    return image.resize((width, height), RESAMPLE)




def extract_topk_patches(wsi_object, scores, coords, top_k, patch_size):
    sampled = sample_rois(np.asarray(scores, dtype=np.float32), np.asarray(coords), k=top_k, mode="topk", seed=1)
    patches = []
    for coord, score in zip(sampled["sampled_coords"], sampled["sampled_scores"]):
        xy = (int(coord[0]), int(coord[1]))
        patch = wsi_object.wsi.read_region(xy, 0, (patch_size, patch_size)).convert("RGB")
        patches.append({"image": patch, "coord": xy, "score": float(score)})
    return patches


def stack_patch_row(patches, width, gap=8, border_color=ACCENT_COLOR):
    if not patches:
        return None
    n = max(1, len(patches))
    patch_w = max(32, int((width - gap * (n - 1)) / n))
    tiles = []
    for item in patches:
        tile = item["image"].copy()
        tile = tile.resize((patch_w, patch_w), RESAMPLE)
        tile = ImageOps.expand(tile, border=3, fill=border_color)
        tiles.append(tile)
    row_w = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
    row = Image.new("RGB", (max(width, row_w), tiles[0].height), CANVAS_BG)
    x = 0
    for idx, tile in enumerate(tiles):
        row.paste(tile, (x, 0))
        x += tile.width + (gap if idx < len(tiles) - 1 else 0)
    if row.width != width:
        canvas = Image.new("RGB", (width, row.height), CANVAS_BG)
        canvas.paste(row, ((width - row.width) // 2, 0))
        return canvas
    return row


def labeled_column(heatmap, patches, title, col_w, font):
    heat = resize_to_width(heatmap.convert("RGB"), col_w)
    bar_h = 54
    bar = Image.new("RGB", (heat.width, bar_h), CANVAS_BG)
    draw = ImageDraw.Draw(bar)
    tw, th = text_size(draw, title, font)
    draw.text(((heat.width - tw) / 2, (bar_h - th) / 2), title, fill=TITLE_COLOR, font=font)
    patch_row = stack_patch_row(patches, heat.width, border_color=ACCENT_COLOR)
    extra = 0 if patch_row is None else (12 + patch_row.height)
    height = bar.height + heat.height + extra
    col = Image.new("RGB", (heat.width, height), CANVAS_BG)
    y = 0
    col.paste(bar, (0, y))
    y += bar.height
    col.paste(heat, (0, y))
    if patch_row is not None:
        y += heat.height + 12
        col.paste(patch_row, (0, y))
    return col


def compose_comparison_figure(gt_heatmap, gen_heatmap, gt_patches, gen_patches, gt_title, gen_title, header, col_w=900):
    title_font = load_font(28)
    header_font = load_font(32)
    left = labeled_column(gt_heatmap, gt_patches, gt_title, col_w, title_font)
    right = labeled_column(gen_heatmap, gen_patches, gen_title, col_w, title_font)
    gap = 28
    header_h = 64
    width = left.width + gap + right.width
    height = header_h + max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), CANVAS_BG)
    draw = ImageDraw.Draw(canvas)
    tw, th = text_size(draw, header, header_font)
    draw.text(((width - tw) / 2, (header_h - th) / 2), header, fill=(20, 20, 20), font=header_font)
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (left.width + gap, header_h))
    return canvas


def compose_single_figure(heatmap, title, header, col_w=900):
    title_font = load_font(28)
    header_font = load_font(32)
    column = labeled_column(heatmap, [], title, col_w, title_font)
    header_h = 64
    canvas = Image.new("RGB", (column.width, header_h + column.height), CANVAS_BG)
    draw = ImageDraw.Draw(canvas)
    tw, th = text_size(draw, header, header_font)
    draw.text(((column.width - tw) / 2, (header_h - th) / 2), header, fill=TITLE_COLOR, font=header_font)
    canvas.paste(column, (0, header_h))
    return canvas


def draw_heatmap(
    scores,
    coords,
    wsi_object,
    vis_level,
    alpha,
    cmap,
    max_size,
    patch_size,
    convert_to_percentiles=True,
    blank_canvas=False,
):
    heatmap = wsi_object.visHeatmap(
        scores=np.asarray(scores, dtype=np.float32),
        coords=np.asarray(coords),
        vis_level=vis_level,
        patch_size=(patch_size, patch_size),
        convert_to_percentiles=convert_to_percentiles,
        blur=False,
        segment=False,
        alpha=1.0 if blank_canvas else alpha,
        cmap=cmap,
        max_size=max_size,
        overlap=0.0,
        binarize=False,
        blank_canvas=blank_canvas,
    )
    return heatmap


def comparison_stem(modality):
    return f"{modality}_vs_{modality}p"


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
    slide_root = Path(args.slide_root) if args.slide_root else Path(config.raw.wsi_root)
    patches_dir = Path(args.patches_dir) if args.patches_dir else paths["workspace_root"] / "P" / f"{args.wsi_experiment}_h5"
    output_root = args.output_dir or (PROJECT_ROOT / "results_display" / OUTPUT_SERIES / display / f"fold{args.fold}" / poe_model_name / OUTPUT_EXPERIMENT)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    heatmap_cmap = args.cmap or "jet"
    diff_cmap = args.diff_cmap or "coolwarm"

    pgc_ckpt = find_pgc_ckpt(args, study)
    poe_ckpt = find_poe_ckpt(args, study, poe_model_name)
    print(f"[info] device={device}")
    print(f"[info] SurvPGC_F ckpt={pgc_ckpt}")
    print(f"[info] PoE ckpt={poe_ckpt}")
    print(f"[info] slide_root={slide_root}")
    print(f"[info] patches_dir={patches_dir}")
    print(f"[info] output_root={output_root}")
    print(f"[info] eval_subsets={subsets}")
    print("[info] SurvPGC_F heatmaps always use real WSI patches; C/G subsets still require SVS/coords.")

    pgc_model = load_checkpoint(pgc_ckpt, device)
    poe_model = load_checkpoint(poe_ckpt, device)
    svs_index = build_svs_index(slide_root) if not args.skip_render else {}
    case_ids = load_case_ids(args, study, metadata)
    rows = []
    subset_avails = {subset: make_avail(subset, device) for subset in subsets}

    if not args.skip_render:
        from wsi_core.WholeSlideImage import WholeSlideImage
    else:
        WholeSlideImage = None

    for case_id in case_ids:
        try:
            slide_id = first_slide_id(metadata, case_id)
            slide_stem = slide_id.replace(".svs", "")
            wsi_path = Path(paths["data_root_dir"]) / f"{slide_stem}.pt"
            gene_path = resolve_case_feature_path(paths["gene_dir"], case_id)
            clinic_path = resolve_case_feature_path(paths["clinic_dir"], case_id)
            coord_path = patches_dir / f"{slide_stem}.h5"
            svs_path = svs_index.get(slide_id) or svs_index.get(slide_stem)
            if gene_path is None or clinic_path is None:
                raise FileNotFoundError(f"missing gene/clinic features for {case_id}")
            if not wsi_path.exists():
                print(f"[skip] {case_id}: missing WSI features {wsi_path}")
                continue
            if not coord_path.exists():
                print(f"[skip] {case_id}: missing coords {coord_path}")
                continue
            if svs_path is None and not args.skip_render:
                print(f"[skip] {case_id}: missing SVS for {slide_id}")
                continue

            x_path = load_tensor(wsi_path).unsqueeze(0).to(device)
            x_omic = load_tensor(gene_path).unsqueeze(0).to(device)
            x_clinic = load_tensor(clinic_path).unsqueeze(0).to(device)
            with h5py.File(coord_path, "r") as handle:
                coords = handle["coords"][:]
            if coords.shape[0] != x_path.shape[1]:
                print(f"[skip] {case_id}: coord/feature mismatch {coords.shape[0]} vs {x_path.shape[1]}")
                continue

            gt_logits, gt_agp, gt_acp = run_pgc_attention(pgc_model, x_path, x_omic, x_clinic)
            gt_risk = risk_from_logits(gt_logits)
            wsi_object = None if args.skip_render else WholeSlideImage(str(svs_path))
            gt_scores = {"Agp": zscore(gt_agp), "Acp": zscore(gt_acp)}

            for subset in subsets:
                avail = subset_avails[subset]
                gene_gen, clinic_gen, _cached = reconstruct_missing(poe_model, x_path, x_omic, x_clinic, avail)
                gen_logits, gen_agp, gen_acp = run_pgc_attention(pgc_model, x_path, gene_gen, clinic_gen)
                gen_risk = risk_from_logits(gen_logits)
                gen_scores = {"Agp": zscore(gen_agp), "Acp": zscore(gen_acp)}
                case_dir = output_root / subset / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                np.save(case_dir / "gt_Agp.npy", gt_agp)
                np.save(case_dir / "gt_Acp.npy", gt_acp)
                np.save(case_dir / "gen_Agp.npy", gen_agp)
                np.save(case_dir / "gen_Acp.npy", gen_acp)
                np.save(case_dir / "diff_Agp.npy", gen_scores["Agp"] - gt_scores["Agp"])
                np.save(case_dir / "diff_Acp.npy", gen_scores["Acp"] - gt_scores["Acp"])

                comparison_files = []
                diff_files = []
                if not args.skip_render:
                    for modality, attn_key in SUBSET_COMPARISONS[subset]:
                        gt_heat = draw_heatmap(
                            gt_scores[attn_key],
                            coords,
                            wsi_object,
                            vis_level=args.vis_level,
                            alpha=args.alpha,
                            cmap=heatmap_cmap,
                            max_size=args.max_size,
                            patch_size=args.patch_size,
                        )
                        gen_heat = draw_heatmap(
                            gen_scores[attn_key],
                            coords,
                            wsi_object,
                            vis_level=args.vis_level,
                            alpha=args.alpha,
                            cmap=heatmap_cmap,
                            max_size=args.max_size,
                            patch_size=args.patch_size,
                        )
                        gt_patches = extract_topk_patches(wsi_object, gt_scores[attn_key], coords, args.top_k, args.patch_size)
                        gen_patches = extract_topk_patches(wsi_object, gen_scores[attn_key], coords, args.top_k, args.patch_size)
                        header = f"{display} | {case_id} | {subset} | {modality} vs {modality}'"
                        figure = compose_comparison_figure(
                            gt_heat,
                            gen_heat,
                            gt_patches,
                            gen_patches,
                            gt_title=f"{modality}  (original)",
                            gen_title=f"{modality}'  (generated)",
                            header=header,
                        )
                        figure_name = f"{comparison_stem(modality)}.{args.save_ext}"
                        save_image(figure, case_dir / figure_name, args.save_ext)
                        comparison_files.append(figure_name)
                        delta = gen_scores[attn_key] - gt_scores[attn_key]
                        diff_heat = draw_heatmap(
                            signed_scores_to_heatmap(delta),
                            coords,
                            wsi_object,
                            vis_level=args.vis_level,
                            alpha=args.alpha,
                            cmap=diff_cmap,
                            max_size=args.max_size,
                            patch_size=args.patch_size,
                            convert_to_percentiles=False,
                            blank_canvas=True,
                        )
                        diff_header = f"{display} | {case_id} | {subset} | {modality}' - {modality}"
                        diff_figure = compose_single_figure(
                            diff_heat,
                            title=f"{modality}' - {modality}  (red: generated higher, blue: original higher)",
                            header=diff_header,
                        )
                        diff_name = f"{modality}_diff.{args.save_ext}"
                        save_image(diff_figure, case_dir / diff_name, args.save_ext)
                        diff_files.append(diff_name)
                        if args.save_raw_heatmaps:
                            save_image(gt_heat, case_dir / f"gt_{attn_key}.{args.save_ext}", args.save_ext)
                            save_image(gen_heat, case_dir / f"gen_{attn_key}.{args.save_ext}", args.save_ext)
                            save_image(diff_heat, case_dir / f"raw_{modality}_diff.{args.save_ext}", args.save_ext)
                        for idx, item in enumerate(gt_patches, 1):
                            save_image(item["image"], case_dir / "patches" / f"gt_{attn_key}_{idx}.jpg", "jpg")
                        for idx, item in enumerate(gen_patches, 1):
                            save_image(item["image"], case_dir / "patches" / f"gen_{attn_key}_{idx}.jpg", "jpg")

                avail_dict = {name: bool(value.item()) for name, value in avail.items()}
                meta = {
                    "case_id": case_id,
                    "slide_id": slide_id,
                    "eval_subset": subset,
                    "poe_model": poe_model_name,
                    "fold": args.fold,
                    "n_patches": int(coords.shape[0]),
                    "gt_risk": gt_risk,
                    "gen_risk": gen_risk,
                    "avail": avail_dict,
                    "comparisons": [f"{modality} vs {modality}'" for modality, _attn in SUBSET_COMPARISONS[subset]],
                    "comparison_files": comparison_files,
                    "diff_files": diff_files,
                    "svs_path": str(svs_path) if svs_path is not None else "",
                    "pgc_ckpt": str(pgc_ckpt),
                    "poe_ckpt": str(poe_ckpt),
                }
                (case_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                rows.append(meta)
                print(
                    f"[ok] {case_id} subset={subset} patches={coords.shape[0]} "
                    f"gt_risk={gt_risk:.4f} gen_risk={gen_risk:.4f} files={comparison_files} diffs={diff_files}"
                )
        except Exception as exc:
            print(f"[fail] {case_id}: {exc}")
            rows.append({"case_id": case_id, "error": str(exc)})

    summary_path = output_root / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
