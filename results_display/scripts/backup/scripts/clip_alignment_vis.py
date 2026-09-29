"""
CLIP 对齐可视化脚本（适配当前 SurvFusion separate 实验目录与三模态 .pt embedding）

用途：
1. 从当前实验结果目录读取 Stage1 checkpoint。
2. 按训练时相同的 patient -> slide_id / case_id 逻辑读取 WSI、Gene、Clinic 三模态特征。
3. 对比随机初始化模型与 Stage1 训练后模型的投影空间分布。
4. 输出二维散点图 + 三模态平均余弦相似度热图，直观看 CLIP 对齐效果。

遍历某个 run 下所有模型：
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init
conda activate SurvPGC_results_display

python results_display/clip_alignment_vis.py \
    --exp_group ablation_Fusion_method \
    --run_name survfusion_separate_mhsa__heads2 \
    --scan_scope run \
    --method umap

遍历某个实验组下所有 run 与模型：
python results_display/clip_alignment_vis.py \
    --exp_group ablation_CLIP_weights \
    --scan_scope exp_group \
    --method umap


示例：
python results_display/clip_alignment_vis.py \
    --exp_group clinic_test \
    --run_name A_profile \
    --modality survfusion_separate \
    --fold 0 \
    --sample_set train \
    --method umap

"pca", "pacmap", "opentsne", "pymde", "umap", "tsne"
"""

import argparse
import copy
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJ_ROOT)

from dataset_deployment.registry import infer_standard_paths

DEFAULT_RESULTS_ROOT = os.path.join(PROJ_ROOT, "results")
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "aclip_effect")

_MODEL_MODULES = {
    "survfusion_separate": "models.deprecated_models.model_SurvFusion_separate",
    "survfusion_joint": "models.deprecated_models.model_SurvFusion_joint",
    "survfusion_noalign": "models.deprecated_models.model_SurvFusion_noalign",
}
_MODEL_CLASSES = {
    "survfusion_separate": "SurvFusion",
    "survfusion_joint": "SurvFusionJoint",
    "survfusion_noalign": "SurvFusionNoAlign",
}

_MODALITY_LABELS = ("WSI (I)", "Gene (T)", "Clinic (S)")
_MODALITY_COLORS = {
    "WSI (I)": "#D1495B",
    "Gene (T)": "#2A9D8F",
    "Clinic (S)": "#E9C46A",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Stage1 CLIP alignment for SurvFusion models")
    parser.add_argument("--method", choices=["pca", "pacmap", "opentsne", "pymde", "umap", "tsne"], default="pca")
    parser.add_argument("--study", type=str, default="tcga_coad")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample_set", choices=["all", "train", "val", "test", "trainval"], default="train")
    parser.add_argument("--num_patches", type=int, default=4096, help="Max WSI patches per patient, matched to training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wsi_dir", type=str, default=None)
    parser.add_argument("--gene_dir", type=str, default=None)
    parser.add_argument("--clinic_dir", type=str, default=None,
                        help="Direct clinic embedding dir")
    parser.add_argument("--label_file", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--exp_group", type=str, default="clinic_test")
    parser.add_argument("--run_name", type=str, default="A_profile")
    parser.add_argument("--modality", type=str, default="survfusion_separate")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Direct model result dir. If provided, overrides results_root/exp_group/run_name/modality")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--scan_scope",
        choices=["single", "run", "exp_group"],
        default="single",
        help="single: 仅跑一个模型; run: 扫描 results/<exp_group>/<run_name>/ 下所有模型; exp_group: 扫描 results/<exp_group>/ 下所有 run 的所有模型",
    )
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory. Default: results_display/aclip_effect/<exp_group>/<run_name>/<modality>")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    inferred = infer_standard_paths(args.study, PROJ_ROOT)
    if args.wsi_dir is None:
        args.wsi_dir = str(inferred["data_root_dir"])
    if args.gene_dir is None:
        args.gene_dir = str(inferred["gene_dir"])
    if args.clinic_dir is None:
        args.clinic_dir = str(inferred["clinic_dir"])
    if args.label_file is None:
        args.label_file = str(inferred["label_file"])
    if args.split_dir is None:
        args.split_dir = str(inferred["split_dir"])
    return args


def validate_dirs(args):
    required = {
        "wsi_dir": args.wsi_dir,
        "gene_dir": args.gene_dir,
        "clinic_dir": args.clinic_dir,
        "split_dir": args.split_dir,
    }
    missing = [f"{name}={path}" for name, path in required.items() if not os.path.isdir(path)]
    if missing:
        raise FileNotFoundError("以下输入目录不存在:\n" + "\n".join(missing))
    if not os.path.isfile(args.label_file):
        raise FileNotFoundError(f"label_file 不存在: {args.label_file}")


def read_label_df(label_file):
    label_df = pd.read_csv(label_file, low_memory=False)
    if "case_id" not in label_df.columns or "slide_id" not in label_df.columns:
        raise ValueError("label_file 必须包含 case_id 和 slide_id 列")
    return label_df


def build_patient_dict(label_df):
    patient_dict = {}
    temp_label_data = label_df.set_index("case_id")
    for patient in label_df.drop_duplicates(["case_id"])["case_id"]:
        slide_ids = temp_label_data.loc[patient, "slide_id"]
        if isinstance(slide_ids, str):
            slide_ids = np.array(slide_ids).reshape(-1)
        else:
            slide_ids = slide_ids.values
        patient_dict[patient] = slide_ids.tolist()
    return patient_dict


def load_case_ids(args, label_df):
    unique_case_ids = label_df["case_id"].dropna().astype(str).drop_duplicates().tolist()
    valid_set = set(unique_case_ids)
    if args.sample_set == "all":
        return unique_case_ids

    result_split_path = None
    if args.result_dir:
        candidate = os.path.join(args.result_dir, f"splits_{args.fold}.csv")
        if os.path.exists(candidate):
            result_split_path = candidate
    if result_split_path is None:
        result_split_path = os.path.join(args.results_root, args.exp_group, args.run_name, args.modality, f"splits_{args.fold}.csv")

    split_path = result_split_path if os.path.exists(result_split_path) else os.path.join(args.split_dir, f"splits_{args.fold}.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split 文件未找到: {split_path}")

    split_df = pd.read_csv(split_path)
    columns = ["train", "val"] if args.sample_set == "trainval" else [args.sample_set]
    case_ids = []
    for col in columns:
        if col not in split_df.columns:
            raise ValueError(f"split 文件中不存在列: {col}")
        raw_ids = split_df[col].dropna().astype(str).tolist()
        for raw_id in raw_ids:
            if raw_id in valid_set:
                case_ids.append(raw_id)
            else:
                case_ids.append(raw_id[:12])

    case_ids = list(dict.fromkeys(case_ids))
    return [case_id for case_id in case_ids if case_id in valid_set]


def resolve_result_dir(args):
    if args.result_dir:
        return args.result_dir
    return os.path.join(args.results_root, args.exp_group, args.run_name, args.modality)


def list_model_jobs(args):
    if args.scan_scope == "single":
        return [(args.run_name, args.modality, resolve_result_dir(args))]

    jobs = []
    if args.scan_scope == "run":
        run_dir = os.path.join(args.results_root, args.exp_group, args.run_name)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"run 目录不存在: {run_dir}")

        for model_name in sorted(os.listdir(run_dir)):
            model_dir = os.path.join(run_dir, model_name)
            if os.path.isdir(model_dir):
                jobs.append((args.run_name, model_name, model_dir))

        if not jobs:
            raise FileNotFoundError(f"未在 run 目录下找到模型目录: {run_dir}")
        return jobs

    exp_dir = os.path.join(args.results_root, args.exp_group)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"exp_group 目录不存在: {exp_dir}")

    for run_name in sorted(os.listdir(exp_dir)):
        run_dir = os.path.join(exp_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        for model_name in sorted(os.listdir(run_dir)):
            model_dir = os.path.join(run_dir, model_name)
            if os.path.isdir(model_dir):
                jobs.append((run_name, model_name, model_dir))

    if not jobs:
        raise FileNotFoundError(f"未在 exp_group 目录下找到任何模型目录: {exp_dir}")
    return jobs


def infer_modality_from_model_name(model_name, fallback):
    if model_name in _MODEL_MODULES:
        return model_name

    for candidate in _MODEL_MODULES:
        if model_name.startswith(candidate):
            return candidate

    lowered = model_name.lower()
    if "survfusion" in lowered:
        if "joint" in lowered:
            return "survfusion_joint"
        if "noalign" in lowered:
            return "survfusion_noalign"
        return "survfusion_separate"

    return fallback


def resolve_output_dir(args):
    if args.outdir:
        return args.outdir
    return os.path.join(DEFAULT_OUTPUT_ROOT, args.exp_group, args.run_name, args.modality)


def resolve_checkpoint_path(args):
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint_path}")
        return args.checkpoint_path

    result_dir = resolve_result_dir(args)
    candidates = [
        os.path.join(result_dir, f"s_{args.fold}_stage1_checkpoint.pt"),
        os.path.join(result_dir, f"s_{args.fold}_stage1_best.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("未找到 Stage1 checkpoint:\n" + "\n".join(candidates))


def _safe_torch_load(path):
    return torch.load(path, map_location="cpu")


def _load_wsi_for_case(wsi_dir, slide_ids, num_patches, rng):
    patch_features = []
    for slide_id in slide_ids:
        slide_stub = slide_id[:-4] if slide_id.endswith(".svs") else slide_id
        wsi_path = os.path.join(wsi_dir, f"{slide_stub}.pt")
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI 特征不存在: {wsi_path}")
        patch_features.append(_safe_torch_load(wsi_path).float())

    patch_features = torch.cat(patch_features, dim=0)
    if patch_features.shape[0] > num_patches:
        idx = np.sort(rng.choice(patch_features.shape[0], size=num_patches, replace=False))
        idx = torch.as_tensor(idx, dtype=torch.long)
        patch_features = patch_features.index_select(0, idx)
    return patch_features


def _load_gene_for_case(gene_dir, case_id):
    gene_path = os.path.join(gene_dir, f"{case_id[:12]}.pt")
    if not os.path.exists(gene_path):
        raise FileNotFoundError(f"Gene 特征不存在: {gene_path}")
    return _safe_torch_load(gene_path).float()


def _load_clinic_for_case(clinic_dir, case_id):
    clinic_path = os.path.join(clinic_dir, f"{case_id[:12]}.pt")
    if not os.path.exists(clinic_path):
        raise FileNotFoundError(f"Clinic 特征不存在: {clinic_path}")
    return _safe_torch_load(clinic_path).float()


def collect_features(case_ids, patient_dict, args):
    rng = np.random.default_rng(args.seed)
    wsi_list, gene_list, clinic_list, valid_ids = [], [], [], []
    skipped = []

    for case_id in case_ids:
        slide_ids = patient_dict.get(case_id)
        if not slide_ids:
            skipped.append((case_id, "missing_slide_ids"))
            continue
        try:
            wsi_features = _load_wsi_for_case(args.wsi_dir, slide_ids, args.num_patches, rng)
            gene_features = _load_gene_for_case(args.gene_dir, case_id)
            clinic_features = _load_clinic_for_case(args.clinic_dir, case_id)
        except FileNotFoundError as exc:
            skipped.append((case_id, str(exc)))
            continue

        wsi_list.append(wsi_features)
        gene_list.append(gene_features)
        clinic_list.append(clinic_features)
        valid_ids.append(case_id)

    if skipped:
        print(f"[警告] 共跳过 {len(skipped)} 个病例（缺少特征或 slide 映射）")
    if not valid_ids:
        raise ValueError("未找到任何可用于可视化的完整三模态病例")

    return wsi_list, gene_list, clinic_list, valid_ids


def infer_model_dims(wsi_list, gene_list, clinic_list):
    return {
        "wsi_embedding_dim": int(wsi_list[0].shape[-1]),
        "gene_embedding_dim": int(gene_list[0].shape[-1]),
        "clinic_embedding_dim": int(clinic_list[0].shape[-1]),
    }


def build_model(args, dims):
    module_name = _MODEL_MODULES.get(args.modality)
    if module_name is None:
        raise ValueError(f"当前脚本仅支持: {sorted(_MODEL_MODULES)}，实际得到 {args.modality}")

    module = importlib.import_module(module_name)
    model_kwargs = dict(
        wsi_embedding_dim=dims["wsi_embedding_dim"],
        gene_embedding_dim=dims["gene_embedding_dim"],
        clinic_embedding_dim=dims["clinic_embedding_dim"],
        projection_dim=args.projection_dim,
        num_classes=4,
        dropout=args.dropout,
        num_heads=args.num_heads,
    )
    if args.modality in {"survfusion_separate", "survfusion_joint"}:
        model_kwargs["temperature"] = args.temperature
    if args.modality == "survfusion_separate":
        model_kwargs["training_stage"] = "stage1"
    return getattr(module, _MODEL_CLASSES[args.modality])(**model_kwargs)


def load_trained_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        trained_model = checkpoint
    else:
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        trained_model = model
        trained_model.load_state_dict(state_dict, strict=True)

    if hasattr(trained_model, "training_stage"):
        trained_model.training_stage = "stage1"
    return trained_model


@torch.no_grad()
def extract_proj_features(model, wsi_list, gene_list, clinic_list, device):
    model = model.to(device)
    model.eval()

    image_proj_list = []
    gene_proj_list = []
    clinic_proj_list = []

    for wsi_feat, gene_feat, clinic_feat in zip(wsi_list, gene_list, clinic_list):
        x_path = wsi_feat.to(device)
        x_gene = gene_feat.to(device)
        x_clinic = clinic_feat.to(device)

        pooled_path = model.wsi_pooling(x_path).unsqueeze(0)
        pooled_gene = model.gene_pooling(x_gene).unsqueeze(0)
        pooled_clinic = model.clinic_pooling(x_clinic).unsqueeze(0)

        image_proj = F.normalize(model.norm_I(model.proj_I(pooled_path)), dim=-1)
        gene_proj = F.normalize(model.norm_T(model.proj_T(pooled_gene)), dim=-1)
        clinic_proj = F.normalize(model.norm_S(model.proj_S(pooled_clinic)), dim=-1)

        image_proj_list.append(np.asarray(image_proj.squeeze(0).cpu().tolist(), dtype=np.float32))
        gene_proj_list.append(np.asarray(gene_proj.squeeze(0).cpu().tolist(), dtype=np.float32))
        clinic_proj_list.append(np.asarray(clinic_proj.squeeze(0).cpu().tolist(), dtype=np.float32))

    return np.stack(image_proj_list), np.stack(gene_proj_list), np.stack(clinic_proj_list)


def reduce_dims(features, method):
    n_samples = features.shape[0]

    if method == "pacmap":
        try:
            import pacmap
            reducer = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=min(15, max(2, n_samples // 3)),
                MN_ratio=0.5,
                FP_ratio=2.0,
                random_state=42,
            )
            return reducer.fit_transform(features, init="pca")
        except ImportError:
            print("[警告] pacmap 未安装，回退到 PCA")
            method = "pca"

    if method == "opentsne":
        try:
            from openTSNE import TSNE as OpenTSNE
            reducer = OpenTSNE(
                n_components=2,
                perplexity=min(30, max(5, n_samples // 5)),
                metric="cosine",
                initialization="pca",
                random_state=42,
                n_jobs=4,
            )
            return np.asarray(reducer.fit(features))
        except ImportError:
            print("[警告] openTSNE 未安装，回退到 PCA")
            method = "pca"

    if method == "pymde":
        try:
            import pymde
            feat_t = torch.tensor(features, dtype=torch.float32)
            embedding = pymde.preserve_neighbors(
                feat_t,
                embedding_dim=2,
                n_neighbors=min(15, max(2, n_samples // 3)),
                device="cpu",
            ).embed(verbose=False)
            return embedding.numpy()
        except ImportError:
            print("[警告] pymde 未安装，回退到 PCA")
            method = "pca"

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                min_dist=0.1,
                n_neighbors=min(15, max(2, n_samples // 3)),
                metric="cosine",
                n_jobs=1,
            )
            return reducer.fit_transform(features)
        except Exception as exc:
            print(f"[警告] UMAP 失败: {exc}，回退到 PCA")
            method = "pca"

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(4, n_samples // 4))
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                metric="cosine",
                init="pca",
                learning_rate="auto",
                max_iter=1000,
            )
            return reducer.fit_transform(features)
        except ImportError:
            print("[警告] sklearn 不可用，回退到 PCA")
            method = "pca"

    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def modality_similarity_matrix(image_proj, gene_proj, clinic_proj):
    modalities = [image_proj, gene_proj, clinic_proj]
    matrix = np.zeros((3, 3), dtype=np.float32)
    for row, lhs in enumerate(modalities):
        for col, rhs in enumerate(modalities):
            matrix[row, col] = float(np.mean(np.sum(lhs * rhs, axis=1)))
    return matrix


def print_similarity_stats(tag, matrix):
    print(f"\n[{tag}] 三模态平均余弦相似度")
    print(f"  WSI-Gene   : {matrix[0, 1]:.4f}")
    print(f"  WSI-Clinic : {matrix[0, 2]:.4f}")
    print(f"  Gene-Clinic: {matrix[1, 2]:.4f}")


def plot_alignment_scatter(ax, coords_i, coords_t, coords_s, title):
    for coords, label in zip([coords_i, coords_t, coords_s], _MODALITY_LABELS):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=16,
            alpha=0.70,
            linewidths=0,
            c=_MODALITY_COLORS[label],
            label=label,
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)


def plot_similarity_heatmap(ax, matrix, title):
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(range(3), _MODALITY_LABELS, rotation=15)
    ax.set_yticks(range(3), _MODALITY_LABELS)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=9, color="black")
    return im


def make_figure(args, before_coords, after_coords, before_matrix, after_matrix, num_cases):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(
        f"CLIP Alignment Visualization | {args.study.upper()} | fold={args.fold} | {args.sample_set} | N={num_cases}\n"
        f"Result Dir: {resolve_result_dir(args)}",
        fontsize=12,
    )

    plot_alignment_scatter(axes[0, 0], *before_coords, "Before Stage1 (random init)")
    plot_alignment_scatter(axes[0, 1], *after_coords, f"After Stage1 ({args.method.upper()})")
    plot_similarity_heatmap(axes[1, 0], before_matrix, "Before Stage1: Avg cosine")
    heatmap = plot_similarity_heatmap(axes[1, 1], after_matrix, "After Stage1: Avg cosine")

    fig.colorbar(heatmap, ax=axes[1, :], shrink=0.8, fraction=0.05, pad=0.04)
    return fig


def run_once(args):
    args.outdir = resolve_output_dir(args)
    os.makedirs(args.outdir, exist_ok=True)
    validate_dirs(args)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Device: {device}")

    label_df = read_label_df(args.label_file)
    patient_dict = build_patient_dict(label_df)
    case_ids = load_case_ids(args, label_df)
    print(f"Study={args.study} | sample_set={args.sample_set} | requested cases={len(case_ids)}")

    print("加载三模态特征...")
    wsi_list, gene_list, clinic_list, valid_ids = collect_features(case_ids, patient_dict, args)
    dims = infer_model_dims(wsi_list, gene_list, clinic_list)
    print(
        f"有效病例数={len(valid_ids)} | "
        f"WSI dim={dims['wsi_embedding_dim']} | Gene shape={tuple(gene_list[0].shape)} | Clinic shape={tuple(clinic_list[0].shape)}"
    )

    print("构建随机初始化模型...")
    torch.manual_seed(args.seed)
    random_model = build_model(args, dims)

    checkpoint_path = resolve_checkpoint_path(args)
    print(f"加载 Stage1 checkpoint: {checkpoint_path}")
    trained_model = load_trained_model(build_model(args, dims), checkpoint_path)

    print("提取投影特征...")
    image_rand, gene_rand, clinic_rand = extract_proj_features(random_model, wsi_list, gene_list, clinic_list, device)
    image_train, gene_train, clinic_train = extract_proj_features(trained_model, wsi_list, gene_list, clinic_list, device)

    before_matrix = modality_similarity_matrix(image_rand, gene_rand, clinic_rand)
    after_matrix = modality_similarity_matrix(image_train, gene_train, clinic_train)
    print_similarity_stats("Before", before_matrix)
    print_similarity_stats("After", after_matrix)

    print(f"运行 {args.method.upper()} 降维...")
    before_emb = reduce_dims(np.concatenate([image_rand, gene_rand, clinic_rand], axis=0), args.method)
    after_emb = reduce_dims(np.concatenate([image_train, gene_train, clinic_train], axis=0), args.method)
    num_cases = len(valid_ids)

    before_coords = (
        before_emb[:num_cases],
        before_emb[num_cases:2 * num_cases],
        before_emb[2 * num_cases:],
    )
    after_coords = (
        after_emb[:num_cases],
        after_emb[num_cases:2 * num_cases],
        after_emb[2 * num_cases:],
    )

    print("绘制图像...")
    fig = make_figure(args, before_coords, after_coords, before_matrix, after_matrix, num_cases)
    out_name = f"clip_align_{args.method}_{args.study}_{args.sample_set}_fold{args.fold}_{args.run_name}_{args.modality}.png"
    out_path = os.path.join(args.outdir, out_name)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}")


def main():
    args = parse_args()

    if args.scan_scope == "single":
        run_once(args)
        return

    jobs = list_model_jobs(args)
    print(f"批量模式: scope={args.scan_scope} | 待处理模型数={len(jobs)}")

    ok = 0
    skipped_no_ckpt = 0
    failed = 0

    for idx, (run_name, model_name, model_dir) in enumerate(jobs, start=1):
        print(f"\n[{idx}/{len(jobs)}] run={run_name} | model={model_name}")

        one = copy.deepcopy(args)
        one.run_name = run_name
        one.modality = infer_modality_from_model_name(model_name, fallback=args.modality)
        one.result_dir = model_dir
        one.checkpoint_path = None

        if one.modality != model_name:
            print(f"[提示] 使用 modality={one.modality} 匹配模型目录 {model_name}")

        if args.outdir:
            one.outdir = os.path.join(args.outdir, run_name, model_name)
        else:
            one.outdir = os.path.join(DEFAULT_OUTPUT_ROOT, args.exp_group, run_name, model_name)

        try:
            run_once(one)
            ok += 1
        except FileNotFoundError as exc:
            msg = str(exc)
            if "未找到 Stage1 checkpoint" in msg or "checkpoint 不存在" in msg:
                skipped_no_ckpt += 1
                print(f"[跳过] 缺少 Stage1 checkpoint: {model_dir}")
                continue
            failed += 1
            print(f"[失败] {msg}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[失败] {type(exc).__name__}: {exc}")

    print("\n批量处理完成:")
    print(f"  成功: {ok}")
    print(f"  跳过(缺少Stage1 checkpoint): {skipped_no_ckpt}")
    print(f"  失败: {failed}")


if __name__ == "__main__":
    main()
