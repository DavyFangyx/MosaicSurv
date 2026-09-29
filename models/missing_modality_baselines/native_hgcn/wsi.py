from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import json
import numpy as np
import openslide
import torch
from torch.autograd import Variable
from torchvision import transforms

from dataset_deployment.registry import get_dataset_config

from .edges import image_grid_edge_index
from .paths import ensure_cut_and_pretrain_on_path, native_study_dir, resolve_study_name


DEFAULT_PATCH_SIZE = 512
DEFAULT_MAGNIFICATION = 10
DEFAULT_SCALE_FACTOR = 32
DEFAULT_TISSUE_THRESH = 0.35
DEFAULT_FILTER_METHOD = "rgb"


def _slide_stem(slide_id: str) -> str:
    name = Path(str(slide_id)).name
    if name.lower().endswith(".svs"):
        return name[:-4]
    return Path(name).stem


_SVS_INDEX: dict[str, dict[str, Path]] = {}


def _svs_index(study: str) -> dict[str, Path]:
    study = resolve_study_name(study)
    cached = _SVS_INDEX.get(study)
    if cached is not None:
        return cached
    config = get_dataset_config(study)
    wsi_root = Path(config.raw.wsi_root)
    index: dict[str, Path] = {}
    if wsi_root.is_dir():
        for path in wsi_root.rglob("*.svs"):
            index[path.name] = path
            index[path.stem] = path
    _SVS_INDEX[study] = index
    return index


def resolve_svs_path(study: str, slide_id: str) -> Path | None:
    name = Path(str(slide_id)).name
    stem = _slide_stem(slide_id)
    index = _svs_index(study)
    return index.get(name) or index.get(stem)


def slide_cache_dir(study: str, slide_id: str, repo_root: str | Path | None = None) -> Path:
    return native_study_dir(study, repo_root) / "wsi_cache" / _slide_stem(slide_id)


def load_cached_slide_features(cache_dir: Path) -> dict[str, np.ndarray] | None:
    pkl_path = cache_dir / "features.pkl"
    if not pkl_path.is_file():
        return None
    payload = joblib.load(pkl_path)
    if isinstance(payload, dict) and payload:
        return payload
    return None


def tiling(
    slide_filepath,
    magnification,
    patch_size,
    scale_factor=32,
    tissue_thresh=0.35,
    method="rgb",
    overview_level=5,
    coord_dir=None,
    overview_dir=None,
    mask_dir=None,
    patch_dir=None,
    filename=None,
    model_final=None,
    thumbnail_dir=None,
    device=None,
):
    """HGCN cut_and_pretrain.tiling, with feature extraction always enabled.

    Original HGCN nests KimiaNet encoding under `if patch_dir is not None` because
    `--save_patch` defaults to True. We keep the same cut/filter/encode steps, but
    extract features even when PNG patches are not written.
    """
    from utils.filters import RGB_filter, adaptive, otsu
    from utils.general import get_three_points, keep_patch, out_of_bound

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature = {}
    transform = transforms.Compose([transforms.ToTensor()])
    slide = openslide.open_slide(str(slide_filepath))
    if "aperio.AppMag" in slide.properties.keys():
        if slide.properties["aperio.AppMag"] == "40.000000":
            level0_magnification = int(float(slide.properties["aperio.AppMag"]))
        else:
            level0_magnification = int(slide.properties["aperio.AppMag"])
    elif "openslide.mpp-x" in slide.properties.keys():
        level0_magnification = 40 if int(np.floor(float(slide.properties["openslide.mpp-x"]) * 10)) == 2 else 20
    else:
        level0_magnification = 40

    if level0_magnification < magnification:
        print(f"{level0_magnification}<{magnification}? magnification should <= level0_magnification.")
        return {}
    patch_size_level0 = int(patch_size * (level0_magnification / magnification))

    downsamples = slide.level_downsamples
    overview_level = len(downsamples) - 1

    if overview_dir is not None:
        thumbnail = slide.get_thumbnail(slide.level_dimensions[overview_level]).convert("RGB")
        thumbnail = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)
        if thumbnail_dir is not None:
            cv2.imwrite(str(thumbnail_dir / f"{filename}.png"), thumbnail)
    else:
        thumbnail = None

    if patch_dir is not None:
        patch_dir = Path(patch_dir) / filename
        patch_dir.mkdir(parents=True, exist_ok=True)

    mask_filepath = str(mask_dir / f"{filename}.png") if mask_dir is not None else None
    if method == "adaptive":
        mask, color_bg = adaptive(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    elif method == "otsu":
        mask, color_bg = otsu(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    elif method == "rgb":
        mask, color_bg = RGB_filter(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    else:
        raise ValueError(f"filter method is wrong, {method}. ")
    mask_w, mask_h = mask.size
    mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_GRAY2BGR)
    mask_patch_size = int(((patch_size_level0 // scale_factor) * 2 + 1) // 2)
    num_step_x = int(mask_w // mask_patch_size)
    num_step_y = int(mask_h // mask_patch_size)

    coord_list = []
    print(f"Processing {filename}...")
    i = 0
    model_final.eval()
    with torch.no_grad():
        for row in range(num_step_y):
            for col in range(num_step_x):
                points_mask = get_three_points(col, row, mask_patch_size)
                row_start, row_end = points_mask[0][1], points_mask[1][1]
                col_start, col_end = points_mask[0][0], points_mask[1][0]
                patch_mask = mask[row_start:row_end, col_start:col_end]
                if keep_patch(patch_mask, tissue_thresh, color_bg):
                    points_level0 = get_three_points(col, row, patch_size_level0)
                    if out_of_bound(slide.dimensions[0], slide.dimensions[1], points_level0[1][0], points_level0[1][1]):
                        continue
                    coord_list.append({"row": row, "col": col, "x": points_level0[0][0], "y": points_level0[0][1]})
                    if overview_dir is not None:
                        points_thumbnail = get_three_points(
                            col, row, patch_size_level0 / slide.level_downsamples[overview_level]
                        )
                        cv2.rectangle(thumbnail, points_thumbnail[0], points_thumbnail[1], color=(255, 255, 255), thickness=3)
                    patch_level0 = slide.read_region(
                        location=points_level0[0],
                        level=0,
                        size=(patch_size_level0, patch_size_level0),
                    ).convert("RGB")
                    patch = patch_level0.resize(size=(patch_size, patch_size))
                    if patch_dir is not None:
                        patch.save(str(patch_dir / f"{row}_{col}.png"))
                    image = transform(patch).unsqueeze(0)
                    inputs = Variable(image).to(device)
                    x, _ = model_final(inputs)
                    feature[str(row) + "_" + str(col)] = x.cpu().detach().numpy().squeeze()
                    i += 1
                    print("\r" + str(i), end="", flush=True)

    coord_dict = {
        "slide_filepath": str(slide_filepath),
        "magnification": magnification,
        "magnification_level0": level0_magnification,
        "num_row": num_step_y,
        "num_col": num_step_x,
        "patch_size": patch_size,
        "patch_size_level0": patch_size_level0,
        "num_patches": len(coord_list),
        "coords": coord_list,
    }
    if coord_dir is not None:
        with open(Path(coord_dir) / f"{filename}.json", "w", encoding="utf-8") as fp:
            json.dump(coord_dict, fp)
    if thumbnail is not None and overview_dir is not None:
        cv2.imwrite(str(Path(overview_dir) / f"{filename}.png"), thumbnail)
    print(
        f"{filename} | mag0: {level0_magnification} | (rows, cols): {num_step_y}, {num_step_x} | "
        f"patch_size: {patch_size} | num_patches: {len(coord_list)}"
    )
    return feature


def extract_slide_features(
    slide_path: Path,
    cache_dir: Path,
    *,
    model_final,
    device,
    transform=None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    magnification: int = DEFAULT_MAGNIFICATION,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    tissue_thresh: float = DEFAULT_TISSUE_THRESH,
    method: str = DEFAULT_FILTER_METHOD,
    save_patch: bool = False,
    force: bool = False,
) -> dict[str, np.ndarray]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        cached = load_cached_slide_features(cache_dir)
        if cached is not None:
            return cached

    ensure_cut_and_pretrain_on_path()
    coord_dir = cache_dir / "coord"
    mask_dir = cache_dir / "mask"
    coord_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = cache_dir / "patch" if save_patch else None
    if patch_dir is not None:
        patch_dir.mkdir(parents=True, exist_ok=True)

    features = tiling(
        slide_filepath=slide_path,
        magnification=magnification,
        patch_size=patch_size,
        scale_factor=scale_factor,
        tissue_thresh=tissue_thresh,
        method=method,
        overview_level=5,
        coord_dir=coord_dir,
        overview_dir=None,
        mask_dir=mask_dir,
        patch_dir=patch_dir,
        filename=Path(slide_path).stem,
        model_final=model_final,
        thumbnail_dir=None,
        device=device,
    )
    if not features:
        raise RuntimeError(f"no tissue patches kept for {slide_path}")
    joblib.dump(features, cache_dir / "features.pkl")
    return features


def tensors_from_patch_dict(features: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    if not features:
        return torch.zeros((0, 1024), dtype=torch.float32), torch.zeros((2, 0), dtype=torch.long)
    patch_ids = list(features.keys())
    x_img = torch.tensor(np.stack([np.asarray(features[k]).reshape(-1) for k in patch_ids], axis=0), dtype=torch.float32)
    edge_index = image_grid_edge_index(patch_ids)
    return x_img, edge_index
