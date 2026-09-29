from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hgcn_paths

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dataset_deployment.registry import get_dataset_config, list_enabled_studies
from native_hgcn.edges import full_connect_edge_index, image_grid_edge_index
from native_hgcn.pack import event_from_censorship, pack_case
from native_hgcn.paths import HGCN_WSI_EXPERIMENT
from native_hgcn.rna import HGCN_RNA_EXPERIMENT
from simple_graph import GraphData


CLINIC_KEEP_DIR_NAMES = {f"L{i}" for i in range(6)}
HGCN_CLINIC_PKL_NAMES = (
    "ttt_cli_feas.pkl",
    "t_cli_feas.pkl",
    "x_cli.pkl",
    "edge_index_cli.pkl",
)
A_PIPELINE_OUTPUT_ROOT = Path(
    "/data/fangyuxuan/projects/medical_dl/trident_project/CONCH-main/projects/outputs"
)
T_IMG_FEA_NAME = "t_img_fea.pkl"
T_RNA_FEA_NAME = "t_rna_fea.pkl"
DEFAULT_WSI_EXPERIMENT = HGCN_WSI_EXPERIMENT
DEFAULT_GENE_EXPERIMENT = HGCN_RNA_EXPERIMENT
DEFAULT_CLINIC_SCHEME = "L0"


def resolve_study_name(study: str) -> str:
    study = str(study).strip()
    if not study:
        raise ValueError("study is required")
    if not study.startswith("tcga_"):
        study = f"tcga_{study}"
    return study


def _as_2d_feature(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D embedding, got shape {tuple(tensor.shape)}")
    return tensor


def _as_patch_dict(payload) -> dict:
    if not isinstance(payload, dict) or not payload:
        return {}
    out = {}
    for key, value in payload.items():
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        out[str(key)] = array
    return out


def load_t_img_fea(wsi_root: Path) -> dict:
    path = Path(wsi_root) / T_IMG_FEA_NAME
    if not path.is_file():
        return {}
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    return {str(case_id): _as_patch_dict(patches) for case_id, patches in payload.items()}


def load_t_rna_fea(gene_root: Path) -> dict:
    path = Path(gene_root) / T_RNA_FEA_NAME
    if not path.is_file():
        return {}
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    out = {}
    for case_id, rows in payload.items():
        array = np.asarray(rows, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        out[str(case_id)] = array
    return out


def graph_from_patch_dict(patches: dict) -> dict:
    if not patches:
        raise ValueError("empty patch dict")
    patch_ids = list(patches.keys())
    x_img = torch.tensor(
        np.stack([np.asarray(patches[key]).reshape(-1) for key in patch_ids], axis=0),
        dtype=torch.float32,
    )
    edge_index = image_grid_edge_index(patch_ids)
    return {
        "x": x_img,
        "edge_index": edge_index.long(),
        "source": T_IMG_FEA_NAME,
    }


def clinic_pkl_source_dir(study: str, scheme: str) -> Path:
    config = get_dataset_config(resolve_study_name(study))
    return A_PIPELINE_OUTPUT_ROOT / config.display_name / "A_manual" / "HGCN_clinic" / scheme


def load_clinic_pkl_bundle(clinic_root: Path) -> dict:
    missing = [name for name in HGCN_CLINIC_PKL_NAMES if not (clinic_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing HGCN clinic pkl under {clinic_root}: {missing}. "
            "Run A_pipeline/run.py hgcn_clinic, then copy the pkl files into hgcn data."
        )
    x_map = joblib.load(clinic_root / "x_cli.pkl")
    edge = torch.from_numpy(np.asarray(joblib.load(clinic_root / "edge_index_cli.pkl"), dtype=np.int64)).long()
    return {
        "x_cli": x_map,
        "edge_index_cli": edge,
        "source": str(clinic_root / "x_cli.pkl"),
    }


def clinic_graph_from_bundle(case_id: str, bundle: dict) -> dict | None:
    x_map = bundle["x_cli"]
    if case_id not in x_map:
        return None
    return {
        "x": _as_2d_feature(torch.from_numpy(np.asarray(x_map[case_id], dtype=np.float32))),
        "edge_index": bundle["edge_index_cli"].long(),
        "source": bundle["source"],
    }


def generate_clinic_pkl_dir(
    source_dir: Path,
    dest_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    label: str = "",
) -> dict[str, int]:
    stats = {"wrote": 0, "skipped": 0, "failed": 0}
    prefix = label or str(source_dir)
    missing = [name for name in HGCN_CLINIC_PKL_NAMES if not (source_dir / name).is_file()]
    if missing:
        print(f"[HGCN] {prefix} missing={missing}", flush=True)
        stats["failed"] += 1
        return stats
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for leftover in dest_dir.glob("*.pt"):
            leftover.unlink()
    for name in HGCN_CLINIC_PKL_NAMES:
        source = source_dir / name
        dest = dest_dir / name
        if dest.exists() and not force and dest.stat().st_mtime >= source.stat().st_mtime:
            stats["skipped"] += 1
            continue
        if dry_run:
            stats["wrote"] += 1
            continue
        dest.write_bytes(source.read_bytes())
        stats["wrote"] += 1
    print(
        f"[HGCN] {prefix} pkl wrote={stats['wrote']} skipped={stats['skipped']} failed={stats['failed']}",
        flush=True,
    )
    return stats


def _resolve_graph_metadata_path(study: str, metadata_csv: str | None = None, repo_root: Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else ROOT_DIR
    config = get_dataset_config(study)
    if metadata_csv:
        path = Path(metadata_csv)
    else:
        path = root / config.split_dir / "split_cohort_metadata.csv"
        if not path.exists():
            path = root / config.metadata_csv
    if not path.exists():
        raise FileNotFoundError(
            f"Missing HGCN cohort metadata: {path}. "
            f"Run dataset_deployment/scripts/generate_5fold_splits.py --study {study} first, "
            "or pass --metadata-csv explicitly."
        )
    return path


def assemble_case_graph(
    case_id: str,
    case_df: pd.DataFrame,
    *,
    wsi_root: Path,
    gene_root: Path,
    clinic_root: Path,
    clinic_bundle: dict | None = None,
    t_img_fea: dict | None = None,
    t_rna_fea: dict | None = None,
) -> GraphData:
    if t_img_fea is None:
        t_img_fea = load_t_img_fea(wsi_root)
    if t_rna_fea is None:
        t_rna_fea = load_t_rna_fea(gene_root)
    if clinic_bundle is None:
        clinic_bundle = load_clinic_pkl_bundle(clinic_root)

    x_img = None
    edge_index_image = None
    if case_id in t_img_fea and t_img_fea[case_id]:
        img_graph = graph_from_patch_dict(t_img_fea[case_id])
        x_img, edge_index_image = img_graph["x"], img_graph["edge_index"]

    x_rna = None
    edge_index_rna = None
    if case_id in t_rna_fea:
        x_rna = _as_2d_feature(torch.from_numpy(np.asarray(t_rna_fea[case_id], dtype=np.float32)))
        edge_index_rna = full_connect_edge_index(int(x_rna.shape[0]))

    clinic_graph = clinic_graph_from_bundle(case_id, clinic_bundle)
    x_cli = None
    edge_index_cli = None
    if clinic_graph is not None:
        x_cli = clinic_graph["x"]
        edge_index_cli = clinic_graph["edge_index"]
        if int(edge_index_cli.numel()) == 0:
            edge_index_cli = full_connect_edge_index(int(x_cli.shape[0]))

    if x_img is None and x_rna is None and x_cli is None:
        raise FileNotFoundError(
            f"Case {case_id} has no HGCN modality pkl under {wsi_root}, {gene_root}, {clinic_root}"
        )

    event = event_from_censorship(case_df["censorship"].iloc[0])
    survival_months = float(case_df["survival_months"].iloc[0])
    return pack_case(
        case_id,
        sur_type=event,
        survival_months=survival_months,
        x_img=x_img,
        edge_index_image=edge_index_image,
        x_rna=x_rna,
        edge_index_rna=edge_index_rna,
        x_cli=x_cli,
        edge_index_cli=edge_index_cli,
    )


def load_hgcn_graphs_from_dirs(
    study: str,
    *,
    data_root_dir: str | Path,
    gene_dir: str | Path,
    clinic_dir: str | Path,
    metadata_csv: str | None = None,
    repo_root: Path | None = None,
) -> tuple[list[str], dict[str, list[float]], dict[str, GraphData]]:
    study = resolve_study_name(study)
    root = Path(repo_root) if repo_root is not None else ROOT_DIR
    wsi_root = Path(data_root_dir)
    gene_root = Path(gene_dir)
    clinic_root = Path(clinic_dir)
    required = (
        (gene_root / T_RNA_FEA_NAME, "gene pkl", "generate_hgcn_rna_graph.py"),
        (clinic_root / "x_cli.pkl", "clinic pkl", "generate_hgcn_clinic.py"),
    )
    missing = [f"{description}: {path} (run SurvPGC_Workspace/{script})" for path, description, script in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing HGCN modality pkl files:\n  " + "\n  ".join(missing)
        )
    img_pkl = wsi_root / T_IMG_FEA_NAME
    if not img_pkl.exists():
        print(
            f"[HGCN] missing {T_IMG_FEA_NAME} under {wsi_root}; treating WSI as absent",
            flush=True,
        )

    metadata_path = _resolve_graph_metadata_path(study, metadata_csv, repo_root=root)
    metadata = pd.read_csv(metadata_path)
    required_columns = {"case_id", "slide_id", "censorship", "survival_months"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"{metadata_path} missing required columns: {sorted(missing_columns)}")
    metadata["case_id"] = metadata["case_id"].astype(str)
    label_df = metadata.drop_duplicates("case_id").copy().set_index("case_id")
    total_cases = metadata["case_id"].nunique()
    print(
        f"[HGCN] assemble study={study} metadata={metadata_path} total_cases={total_cases} "
        f"wsi={wsi_root} gene={gene_root} clinic={clinic_root}",
        flush=True,
    )

    clinic_bundle = load_clinic_pkl_bundle(clinic_root)
    t_img_fea = load_t_img_fea(wsi_root)
    t_rna_fea = load_t_rna_fea(gene_root)
    print(f"[HGCN] loaded {T_IMG_FEA_NAME}: cases={len(t_img_fea)} from {wsi_root}", flush=True)
    print(f"[HGCN] loaded {T_RNA_FEA_NAME}: cases={len(t_rna_fea)} from {gene_root}", flush=True)
    patients: list[str] = []
    sur_and_time: dict[str, list[float]] = {}
    all_data: dict[str, GraphData] = {}
    skipped = 0
    for index, (case_id, group) in enumerate(metadata.groupby("case_id", sort=False), start=1):
        case_id = str(case_id)
        try:
            data = assemble_case_graph(
                case_id,
                group,
                wsi_root=wsi_root,
                gene_root=gene_root,
                clinic_root=clinic_root,
                clinic_bundle=clinic_bundle,
                t_img_fea=t_img_fea,
                t_rna_fea=t_rna_fea,
            )
        except FileNotFoundError as exc:
            skipped += 1
            if skipped <= 5:
                print(f"[HGCN] skip {case_id}: {exc}", flush=True)
            continue
        all_data[case_id] = data
        patients.append(case_id)
        sur_and_time[case_id] = [
            event_from_censorship(label_df.loc[case_id, "censorship"]),
            float(label_df.loc[case_id, "survival_months"]),
        ]
        if index == 1 or index % 25 == 0 or index == total_cases:
            print(f"[HGCN] assembled {len(patients)}/{total_cases}: {case_id}", flush=True)

    if not patients:
        raise FileNotFoundError(
            f"No complete HGCN graphs for {study} under {wsi_root}, {gene_root}, {clinic_root}"
        )
    print(f"[HGCN] assemble finished: cases={len(patients)} skipped={skipped}", flush=True)
    return patients, sur_and_time, all_data


def hgcn_pack_dir(
    study: str,
    *,
    wsi_experiment: str,
    gene_experiment: str,
    clinic_scheme: str,
    repo_root: Path | None = None,
) -> Path:
    root = Path(repo_root) if repo_root is not None else ROOT_DIR
    study = resolve_study_name(study)
    return (
        hgcn_paths.hgcn_data_root(root)
        / study
        / "pack"
        / f"{wsi_experiment}__{gene_experiment}__{clinic_scheme}"
    )


def dump_study_pack(
    study: str,
    patients: list[str],
    sur_and_time: dict[str, list[float]],
    all_data: dict,
    *,
    wsi_experiment: str,
    gene_experiment: str,
    clinic_scheme: str,
    repo_root: Path | None = None,
    extra_meta: dict | None = None,
) -> Path:
    out_dir = hgcn_pack_dir(
        study,
        wsi_experiment=wsi_experiment,
        gene_experiment=gene_experiment,
        clinic_scheme=clinic_scheme,
        repo_root=repo_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(patients, out_dir / "patients.pkl")
    joblib.dump(sur_and_time, out_dir / "sur_and_time.pkl")
    joblib.dump(all_data, out_dir / "all_data.pkl")
    counts = {"img": 0, "rna": 0, "cli": 0, "any": 0}
    for data in all_data.values():
        for key in ("img", "rna", "cli"):
            if key in data.data_type:
                counts[key] += 1
        if data.data_type:
            counts["any"] += 1
    summary = {
        "study": resolve_study_name(study),
        "n_patients": len(patients),
        "n_graphs": len(all_data),
        "wsi_experiment": wsi_experiment,
        "gene_experiment": gene_experiment,
        "clinic_scheme": clinic_scheme,
        "modality_counts": counts,
        "event_encoding": "sur_and_time[case] = [event, survival_months]; event=1-censorship; 1=event, 0=censored",
        **(extra_meta or {}),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return out_dir


def assemble_and_dump_study_pack(
    study: str,
    *,
    wsi_experiment: str,
    gene_experiment: str,
    clinic_scheme: str,
    repo_root: Path | None = None,
    metadata_csv: str | None = None,
) -> Path:
    root = Path(repo_root) if repo_root is not None else ROOT_DIR
    study = resolve_study_name(study)
    hgcn_root = hgcn_paths.hgcn_data_root(root) / study
    patients, sur_and_time, all_data = load_hgcn_graphs_from_dirs(
        study,
        data_root_dir=hgcn_root / "P" / wsi_experiment,
        gene_dir=hgcn_root / "G" / gene_experiment,
        clinic_dir=hgcn_root / "C" / clinic_scheme,
        metadata_csv=metadata_csv,
        repo_root=root,
    )
    out_dir = dump_study_pack(
        study,
        patients,
        sur_and_time,
        all_data,
        wsi_experiment=wsi_experiment,
        gene_experiment=gene_experiment,
        clinic_scheme=clinic_scheme,
        repo_root=root,
        extra_meta={
            "wsi_dir": str(hgcn_root / "P" / wsi_experiment),
            "gene_dir": str(hgcn_root / "G" / gene_experiment),
            "clinic_dir": str(hgcn_root / "C" / clinic_scheme),
        },
    )
    print(f"[HGCN] packed {study} -> {out_dir} n={len(patients)}", flush=True)
    return out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble HGCN GraphData from native modality pkl files under hgcn data."
    )
    parser.add_argument("--study", action="append", default=[], help="Study id. Repeatable, or all.")
    parser.add_argument("--wsi-experiment", type=str, default=DEFAULT_WSI_EXPERIMENT)
    parser.add_argument("--gene-experiment", type=str, default=DEFAULT_GENE_EXPERIMENT)
    parser.add_argument("--clinic-scheme", type=str, default=DEFAULT_CLINIC_SCHEME)
    parser.add_argument("--repo-root", type=str, default=str(ROOT_DIR))
    parser.add_argument("--metadata-csv", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if not args.study or (len(args.study) == 1 and args.study[0] in {"all", "*"}):
        studies = list(list_enabled_studies())
    else:
        studies = [
            resolve_study_name(item)
            for study in args.study
            for item in study.split(",")
            if item.strip()
        ]
    clinic_scheme = args.clinic_scheme.strip().upper()
    if not clinic_scheme.startswith("L"):
        clinic_scheme = f"L{clinic_scheme}"
    if clinic_scheme not in CLINIC_KEEP_DIR_NAMES:
        raise SystemExit(f"Unsupported clinic scheme: {clinic_scheme}")
    print("[HGCN] studies=" + ", ".join(studies), flush=True)
    print(
        f"[HGCN] wsi={args.wsi_experiment} gene={args.gene_experiment} clinic={clinic_scheme}",
        flush=True,
    )
    for study in studies:
        assemble_and_dump_study_pack(
            study,
            wsi_experiment=args.wsi_experiment,
            gene_experiment=args.gene_experiment,
            clinic_scheme=clinic_scheme,
            repo_root=repo_root,
            metadata_csv=args.metadata_csv,
        )


if __name__ == "__main__":
    main()
