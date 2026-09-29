from __future__ import annotations

import sys
from pathlib import Path

from dataset_deployment.registry import get_dataset_config, list_enabled_studies


REPO_ROOT = Path(__file__).resolve().parents[3]
HGCN_THIRD_PARTY = REPO_ROOT / "models" / "missing_modality_baselines" / "third_party" / "HGCN"
CUT_AND_PRETRAIN_DIR = HGCN_THIRD_PARTY / "cut_and_pretrain"
NATIVE_OUTPUT_DIRNAME = "hgcn_native"
HGCN_WSI_EXPERIMENT = "kimianet"
DEFAULT_KIMIANET_WEIGHTS = Path(
    "/data/fangyuxuan/projects/medical_dl/trident_project/KimiaNet/KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth"
)
A_PIPELINE_ROOT = Path(
    "/data/fangyuxuan/projects/medical_dl/trident_project/CONCH-main/projects/A_pipeline"
)
A_PIPELINE_DATASETS_JSON = A_PIPELINE_ROOT / "datasets.json"

DISPLAY_NAME_TO_A_PIPELINE = {
    "TCGA-BRCA": "TCGA-BRCA",
    "TCGA-COAD": "TCGA-COAD",
    "TCGA-KICH": "TCGA-KICH",
    "TCGA-KIRC": "TCGA-KIRC",
    "TCGA-KIRP": "TCGA-KIRP",
    "TCGA_LIHC": "TCGA_LIHC",
    "TCGA-LIHC": "TCGA_LIHC",
    "TCGA-PRAD": "TCGA-PRAD",
    "TCGA-READ": "TCGA-READ",
    "TCGA-STAD": "TCGA-STAD",
}


def resolve_study_name(study: str) -> str:
    study = str(study).strip()
    if not study:
        raise ValueError("study is required")
    if not study.startswith("tcga_"):
        study = f"tcga_{study.lower()}"
    return study.lower()


def native_root(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return root / "SurvPGC_Workspace" / NATIVE_OUTPUT_DIRNAME


def native_study_dir(study: str, repo_root: str | Path | None = None) -> Path:
    return native_root(repo_root) / resolve_study_name(study)


def hgcn_wsi_dir(
    study: str,
    repo_root: str | Path | None = None,
    experiment: str = HGCN_WSI_EXPERIMENT,
) -> Path:
    import hgcn_paths

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return hgcn_paths.hgcn_data_root(root) / resolve_study_name(study) / "P" / experiment


def ensure_a_pipeline_on_path() -> Path:
    root = A_PIPELINE_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"A_pipeline is required for HGCN clinic encoding: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def ensure_cut_and_pretrain_on_path() -> Path:
    root = CUT_AND_PRETRAIN_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"HGCN cut_and_pretrain is missing: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def a_pipeline_dataset_name(study: str) -> str:
    config = get_dataset_config(resolve_study_name(study))
    return DISPLAY_NAME_TO_A_PIPELINE.get(config.display_name, config.display_name)


def enabled_studies() -> list[str]:
    return list(list_enabled_studies())
