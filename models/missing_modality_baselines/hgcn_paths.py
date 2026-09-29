from __future__ import annotations

from pathlib import Path


WORKSPACE_DIRNAME = "SurvPGC_Workspace"
HGCN_DATA_DIRNAME = "hgcn data"
PACK_FILENAMES = ("patients.pkl", "sur_and_time.pkl", "all_data.pkl")


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else repo_root_from_here()
    return (root / WORKSPACE_DIRNAME).resolve()


def hgcn_data_root(repo_root: str | Path | None = None) -> Path:
    return workspace_root(repo_root) / HGCN_DATA_DIRNAME


def _split_workspace_relative(path: Path, repo_root: Path) -> tuple[str, ...]:
    resolved = path.expanduser().resolve()
    workspace = (repo_root / WORKSPACE_DIRNAME).resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(
            f"HGCN path must live under {workspace}, got {resolved}"
        ) from exc
    parts = relative.parts
    if parts and parts[0] == HGCN_DATA_DIRNAME:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"HGCN path has no study/modality suffix: {resolved}")
    return parts


def remap_workspace_path_to_hgcn_data(
    path: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    """Map a public workspace path onto the mirrored HGCN data tree.

    SurvPGC_Workspace/<study>/C/L0
      -> SurvPGC_Workspace/hgcn data/<study>/C/L0

    Paths that already live under "hgcn data" are returned unchanged.
    """
    root = Path(repo_root).resolve() if repo_root is not None else repo_root_from_here()
    parts = _split_workspace_relative(Path(path), root)
    return (root / WORKSPACE_DIRNAME / HGCN_DATA_DIRNAME).joinpath(*parts)


def hgcn_pack_complete(path: str | Path) -> bool:
    pack_dir = Path(path)
    return pack_dir.is_dir() and all((pack_dir / name).is_file() for name in PACK_FILENAMES)


def hgcn_pack_missing(path: str | Path) -> list[str]:
    pack_dir = Path(path)
    missing = []
    if not pack_dir.exists():
        return [str(pack_dir)]
    for name in PACK_FILENAMES:
        candidate = pack_dir / name
        if not candidate.is_file():
            missing.append(str(candidate))
    return missing


def remap_hgcn_feature_dirs(
    *,
    data_root_dir: str | Path,
    clinic_dir: str | Path,
    gene_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Path]:
    return {
        "data_root_dir": remap_workspace_path_to_hgcn_data(data_root_dir, repo_root),
        "clinic_dir": remap_workspace_path_to_hgcn_data(clinic_dir, repo_root),
        "gene_dir": remap_workspace_path_to_hgcn_data(gene_dir, repo_root),
    }
