from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from simple_graph import GraphData

from .edges import HGCN_PAD_DIM, empty_edge_index, empty_node_features
from .paths import native_study_dir, resolve_study_name


def event_from_censorship(censorship) -> int:
    """HGCN stores event indicators, not TCGA censorship.

    metadata.csv uses 1=censored / 0=event. Cox / C-index need 1=event / 0=censored.
    """
    return int(1 - int(censorship))


def _to_float_2d(tensor: torch.Tensor | None, dim: int = HGCN_PAD_DIM) -> torch.Tensor:
    if tensor is None:
        return empty_node_features(dim)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(dtype=torch.float32)


def _to_edge(tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        return empty_edge_index()
    return tensor.to(dtype=torch.long)


def concat_slide_graphs(slide_graphs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep 8-neighborhood inside each slide; never connect patches across slides."""
    features = []
    edges = []
    offset = 0
    for x_img, edge_index in slide_graphs:
        x_img = _to_float_2d(x_img)
        n = int(x_img.shape[0])
        if n == 0:
            continue
        features.append(x_img)
        if edge_index is not None and int(edge_index.numel()):
            edges.append(edge_index.to(dtype=torch.long) + offset)
        offset += n
    if not features:
        return empty_node_features(), empty_edge_index()
    x_img = torch.cat(features, dim=0)
    nonempty = [edge for edge in edges if int(edge.numel())]
    edge_index = torch.cat(nonempty, dim=1) if nonempty else empty_edge_index()
    return x_img, edge_index


def pack_case(
    case_id: str,
    *,
    sur_type: int,
    survival_months: float,
    x_img: torch.Tensor | None = None,
    edge_index_image: torch.Tensor | None = None,
    x_rna: torch.Tensor | None = None,
    edge_index_rna: torch.Tensor | None = None,
    x_cli: torch.Tensor | None = None,
    edge_index_cli: torch.Tensor | None = None,
) -> GraphData:
    data_type: list[str] = []
    if x_img is not None and int(x_img.shape[0]) > 0:
        data_type.append("img")
    else:
        x_img = empty_node_features()
        edge_index_image = empty_edge_index()
    if x_rna is not None and int(x_rna.shape[0]) > 0:
        data_type.append("rna")
    else:
        x_rna = empty_node_features()
        edge_index_rna = empty_edge_index()
    if x_cli is not None and int(x_cli.shape[0]) > 0:
        data_type.append("cli")
    else:
        x_cli = empty_node_features()
        edge_index_cli = empty_edge_index()

    return GraphData(
        x_img=_to_float_2d(x_img),
        x_rna=_to_float_2d(x_rna),
        x_cli=_to_float_2d(x_cli),
        sur_type=torch.tensor([int(sur_type)], dtype=torch.long),
        data_id=str(case_id),
        data_type=data_type,
        edge_index_model=empty_edge_index(),
        edge_index_image=_to_edge(edge_index_image),
        edge_index_rna=_to_edge(edge_index_rna),
        edge_index_cli=_to_edge(edge_index_cli),
        surv_time=torch.tensor([float(survival_months)], dtype=torch.float32),
    )


def dump_pack(
    study: str,
    patients: list[str],
    sur_and_time: dict[str, list[float]],
    all_data: dict[str, GraphData],
    *,
    repo_root: str | Path | None = None,
    extra_meta: dict | None = None,
) -> Path:
    out_dir = native_study_dir(study, repo_root) / "pack"
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
        "modality_counts": counts,
        "pad_dim": HGCN_PAD_DIM,
        "event_encoding": "sur_and_time[case] = [event, survival_months]; event=1-censorship; 1=event, 0=censored",
        **(extra_meta or {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def load_metadata(study: str, repo_root: str | Path | None = None) -> pd.DataFrame:
    from dataset_deployment.registry import get_dataset_config

    study = resolve_study_name(study)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    config = get_dataset_config(study)
    path = root / config.metadata_csv
    df = pd.read_csv(path)
    required = {"case_id", "slide_id", "censorship", "survival_months"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["case_id"] = df["case_id"].astype(str)
    df["slide_id"] = df["slide_id"].astype(str)
    if "rna_file_name" in df.columns:
        df["rna_file_name"] = df["rna_file_name"].fillna("").astype(str)
    return df
