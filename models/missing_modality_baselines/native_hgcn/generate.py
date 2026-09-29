from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

_HERE = Path(__file__).resolve().parent
_BASELINE_DIR = _HERE.parent
if str(_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_DIR))

from dataset_deployment.registry import get_dataset_config, list_enabled_studies
from dataset_deployment.workspace_features import load_patient_allowlist

from .kimianet import load_kimianet
from .pack import load_metadata
from .paths import (
    DEFAULT_KIMIANET_WEIGHTS,
    HGCN_WSI_EXPERIMENT,
    REPO_ROOT,
    hgcn_wsi_dir,
    resolve_study_name,
)
from .wsi import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_MAGNIFICATION,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_TISSUE_THRESH,
    extract_slide_features,
    resolve_svs_path,
    slide_cache_dir,
)

T_IMG_FEA_NAME = "t_img_fea.pkl"


def _select_studies(study_args: list[str]) -> list[str]:
    if not study_args:
        raise ValueError("Pass --study tcga_kirc (repeatable) or --study all.")
    if len(study_args) == 1 and study_args[0] in {"all", "*"}:
        return list(list_enabled_studies())
    return [resolve_study_name(item) for study in study_args for item in study.split(",") if item.strip()]


def _as_float32_vector(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array


def merge_patient_patch_dict(
    slide_dicts: list[tuple[str, dict[str, np.ndarray]]] | list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate patches from every slide of one patient.

    Keys are `{slide_stem}/{patch_id}` so 8-neighborhood stays inside a slide.
    """
    merged: dict[str, np.ndarray] = {}
    for item in slide_dicts:
        if isinstance(item, tuple):
            slide_id, features = item
            stem = Path(str(slide_id)).name
            if stem.lower().endswith(".svs"):
                stem = stem[:-4]
            else:
                stem = Path(stem).stem
            prefix = f"{stem}/"
        else:
            features = item
            prefix = ""
        for raw_key, value in features.items():
            merged[f"{prefix}{raw_key}"] = _as_float32_vector(value)
    return merged


def generate_study(
    study: str,
    *,
    repo_root: Path | None = None,
    max_cases: int | None = None,
    save_patch: bool = False,
    force: bool = False,
    dry_run: bool = False,
    device: str | None = None,
    kimianet_weights: str | Path | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    magnification: int = DEFAULT_MAGNIFICATION,
    tissue_thresh: float = DEFAULT_TISSUE_THRESH,
    filter_method: str = DEFAULT_FILTER_METHOD,
    experiment: str = HGCN_WSI_EXPERIMENT,
) -> dict[str, int]:
    study = resolve_study_name(study)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    metadata = load_metadata(study, repo_root=root)
    allowlist = load_patient_allowlist(root / get_dataset_config(study).patient_table_csv)
    dest_dir = hgcn_wsi_dir(study, root, experiment=experiment)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / T_IMG_FEA_NAME
    if not dry_run:
        for leftover in dest_dir.glob("*.pt"):
            leftover.unlink()

    case_slides: dict[str, list[str]] = {}
    for _, row in metadata.iterrows():
        case_id = str(row["case_id"])
        if allowlist and case_id.upper() not in allowlist:
            continue
        if max_cases is not None and case_id not in case_slides and len(case_slides) >= int(max_cases):
            continue
        slide_id = str(row["slide_id"])
        case_slides.setdefault(case_id, [])
        if slide_id not in case_slides[case_id]:
            case_slides[case_id].append(slide_id)

    print(
        f"[native-hgcn] study={study} patients={len(case_slides)} dest={dest}",
        flush=True,
    )

    existing: dict[str, dict[str, np.ndarray]] = {}
    if dest.is_file() and not force:
        loaded = joblib.load(dest)
        if isinstance(loaded, dict):
            existing = loaded

    model = None
    torch_device = None
    if not dry_run:
        need_model = force or any(case_id not in existing for case_id in case_slides)
        if need_model:
            torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_kimianet(kimianet_weights or DEFAULT_KIMIANET_WEIGHTS, device=torch_device)
            print(
                f"[native-hgcn] KimiaNet weights={kimianet_weights or DEFAULT_KIMIANET_WEIGHTS} device={torch_device}",
                flush=True,
            )

    t_img_fea: dict[str, dict[str, np.ndarray]] = dict(existing) if not force else {}
    stats = {"wrote": 0, "skipped": 0, "failed": 0, "missing": 0}
    for case_id, slide_ids in tqdm(case_slides.items(), desc=f"{study} patients"):
        if not force and case_id in existing and existing[case_id]:
            t_img_fea[case_id] = existing[case_id]
            stats["skipped"] += 1
            continue
        if dry_run:
            stats["wrote"] += 1
            continue
        slide_dicts = []
        missing_slides = []
        try:
            for slide_id in slide_ids:
                svs_path = resolve_svs_path(study, slide_id)
                if svs_path is None:
                    missing_slides.append(slide_id)
                    continue
                cache_dir = slide_cache_dir(study, slide_id, repo_root=root)
                features = extract_slide_features(
                    svs_path,
                    cache_dir,
                    model_final=model,
                    device=torch_device,
                    patch_size=patch_size,
                    magnification=magnification,
                    scale_factor=DEFAULT_SCALE_FACTOR,
                    tissue_thresh=tissue_thresh,
                    method=filter_method,
                    save_patch=save_patch,
                    force=force,
                )
                if features:
                    slide_dicts.append((slide_id, features))
            if missing_slides and not slide_dicts:
                stats["missing"] += 1
                print(f"[native-hgcn] missing SVS for {case_id}: {missing_slides}", flush=True)
                continue
            merged = merge_patient_patch_dict(slide_dicts)
            if not merged:
                stats["missing"] += 1
                print(f"[native-hgcn] no patches for {case_id}", flush=True)
                continue
            t_img_fea[case_id] = merged
            stats["wrote"] += 1
            if missing_slides:
                print(f"[native-hgcn] partial SVS for {case_id}: missing={missing_slides}", flush=True)
        except Exception as exc:
            stats["failed"] += 1
            print(f"[native-hgcn] WSI failed {case_id}: {exc}", flush=True)

    if not dry_run and t_img_fea:
        if max_cases is None:
            keep = set(case_slides)
            t_img_fea = {case_id: patches for case_id, patches in t_img_fea.items() if case_id in keep}
        joblib.dump(t_img_fea, dest)
    elif not dry_run and dest.is_file() and not t_img_fea:
        dest.unlink()
    print(
        f"[native-hgcn] {study} wrote={stats['wrote']} skipped={stats['skipped']} "
        f"failed={stats['failed']} missing={stats['missing']} dest={dest}",
        flush=True,
    )
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build HGCN-native WSI patch features from original SVS files and write "
            "SurvPGC_Workspace/hgcn data/<study>/P/kimianet/t_img_fea.pkl."
        )
    )
    parser.add_argument("--study", action="append", default=[], help="Study id. Repeatable, or all.")
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--save-patch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--kimianet-weights", type=str, default=str(DEFAULT_KIMIANET_WEIGHTS))
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--magnification", type=int, default=DEFAULT_MAGNIFICATION)
    parser.add_argument("--tissue-thresh", type=float, default=DEFAULT_TISSUE_THRESH)
    parser.add_argument(
        "--filter-method",
        type=str,
        default=DEFAULT_FILTER_METHOD,
        choices=["rgb", "otsu", "adaptive"],
    )
    parser.add_argument("--experiment", type=str, default=HGCN_WSI_EXPERIMENT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    studies = _select_studies(args.study)
    for study in studies:
        generate_study(
            study,
            repo_root=Path(args.repo_root),
            max_cases=args.max_cases,
            save_patch=args.save_patch,
            force=args.force,
            dry_run=args.dry_run,
            device=args.device,
            kimianet_weights=args.kimianet_weights,
            patch_size=args.patch_size,
            magnification=args.magnification,
            tissue_thresh=args.tissue_thresh,
            filter_method=args.filter_method,
            experiment=args.experiment,
        )


if __name__ == "__main__":
    main()
