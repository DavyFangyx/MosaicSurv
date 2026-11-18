"""IDC derived-DICOM slide reader with an openslide-compatible surface.

CPTAC SMs are stored per pyramid level as separate tiled-full DICOM files in
one case dir (40x: ~0.25um, 10x: ~1um, 2.5x: ~4um, 256px tiles). DcmSlide
exposes the subset of the openslide API used by HGCN's tiling()/RGB_filter():

  level 0 = 40x layer (read_region slices the in-memory 10x layer and resizes)
  level 1 = 10x layer (in-memory full decode, exact 10x pixels)
  level 2 = 2.5x layer (in-memory, used by RGB_filter's 32x downsample mask)

HGCN semantics preserved: 512px patches on the 10x grid, same RGB tissue
filter, same KimiaNet encoding as the TCGA .svs pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import iter_pixels
from PIL import Image

TILE = 256
DOWNSAMPLE_10X = 4.0
DOWNSAMPLE_25X = 16.0


def _ybr_ict_to_rgb(arr):
    """Inverse DICOM YBR_ICT (lossless RGB transform) -> uint8 RGB."""
    y, cb, cr = arr[..., 0].astype(np.int32), arr[..., 1].astype(np.int32), arr[..., 2].astype(np.int32)
    g = y - (cb + cr) // 4
    r = cr + g
    b = cb + g
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


def _spacing_mm(ds) -> float | None:
    try:
        return float(ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0])
    except Exception:
        return None


def _decode_full(path: Path) -> np.ndarray | None:
    """Decode a whole tiled-full DICOM into one RGB array (small layers only)."""
    ds = pydicom.dcmread(str(path))
    cols, rows = int(ds.TotalPixelMatrixColumns), int(ds.TotalPixelMatrixRows)
    tiles_per_row = int(np.ceil(cols / TILE))
    tile_rows = int(np.ceil(rows / TILE))
    canvas = np.zeros((tile_rows * TILE, tiles_per_row * TILE, 3), dtype=np.uint8)
    for index, frame in enumerate(iter_pixels(ds)):
        if frame.ndim == 2:
            frame = np.dstack([frame] * 3)
        if ds.PhotometricInterpretation == "YBR_ICT":
            frame = _ybr_ict_to_rgb(frame)
        row, col = divmod(index, tiles_per_row)
        canvas[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE] = frame
    return canvas[:rows, :cols]


class DcmSlide:
    def __init__(self, dcm_path: Path):
        self.path = Path(dcm_path)
        self._ds0 = pydicom.dcmread(str(self.path), stop_before_pixels=True)
        self.dimensions = (
            int(self._ds0.TotalPixelMatrixColumns),
            int(self._ds0.TotalPixelMatrixRows),
        )
        spacing0 = _spacing_mm(self._ds0) or 0.00025
        self.properties = {"openslide.mpp-x": f"{spacing0 * 1000:.6f}"}
        self._slide_id = getattr(self._ds0, "SlideIdentifier", None)
        case_dir = self.path.parent.parent.parent
        self._mid_arr = None
        self._low_arr = None
        for candidate in case_dir.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(candidate), stop_before_pixels=True)
            except Exception:
                continue
            if self._slide_id is not None and getattr(ds, "SlideIdentifier", None) != self._slide_id:
                continue
            spacing = _spacing_mm(ds)
            if spacing is None:
                continue
            if 0.0008 < spacing < 0.0013 and candidate != self.path:  # ~1um = 10x
                self._mid_path = candidate
            elif 0.003 < spacing < 0.006:  # ~4um = 2.5x
                self._low_path = candidate

    @property
    def level_count(self) -> int:
        return 1 + int(hasattr(self, "_mid_path")) + int(hasattr(self, "_low_path"))

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        dims = [self.dimensions]
        if hasattr(self, "_mid_path"):
            dims.append((self.dimensions[0] // 4, self.dimensions[1] // 4))
        if hasattr(self, "_low_path"):
            dims.append((int(self.dimensions[0] / DOWNSAMPLE_25X), int(self.dimensions[1] / DOWNSAMPLE_25X)))
        return dims

    @property
    def level_downsamples(self) -> list[float]:
        ds = [1.0]
        if hasattr(self, "_mid_path"):
            ds.append(DOWNSAMPLE_10X)
        if hasattr(self, "_low_path"):
            ds.append(DOWNSAMPLE_25X)
        return ds

    def get_best_level_for_downsample(self, downsample: float) -> int:
        if hasattr(self, "_low_path") and downsample > (DOWNSAMPLE_10X + DOWNSAMPLE_25X) / 2:
            return 2
        if hasattr(self, "_mid_path") and downsample > 1.5:
            return 1
        return 0

    def _mid_image(self) -> np.ndarray | None:
        if not hasattr(self, "_mid_path"):
            return None
        if self._mid_arr is None:
            self._mid_arr = _decode_full(self._mid_path)
        return self._mid_arr

    def _low_image(self) -> np.ndarray | None:
        if not hasattr(self, "_low_path"):
            return None
        if self._low_arr is None:
            self._low_arr = _decode_full(self._low_path)
        return self._low_arr

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        if level == 1 and hasattr(self, "_mid_path"):
            arr = self._mid_image()
            x0, y0 = int(location[0]), int(location[1])
            patch = arr[y0:y0 + size[1], x0:x0 + size[0]]
            return Image.fromarray(patch)
        if level == 2 and hasattr(self, "_low_path"):
            arr = self._low_image()
            x0, y0 = int(location[0]), int(location[1])
            patch = arr[y0:y0 + size[1], x0:x0 + size[0]]
            return Image.fromarray(patch)
        # level 0 (40x): slice the 10x layer at 1/4 scale, then resize up to
        # the requested 40x size (content equals openslide's read+downsample).
        arr = self._mid_image()
        if arr is None:
            return Image.new("RGB", size, (255, 255, 255))
        f = int(round(DOWNSAMPLE_10X))
        x0, y0 = int(location[0]) // f, int(location[1]) // f
        w, h = int(np.ceil(size[0] / f)), int(np.ceil(size[1] / f))
        patch = arr[y0:y0 + h, x0:x0 + w]
        return Image.fromarray(patch).resize(size)

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        level = 2 if hasattr(self, "_low_path") else (1 if hasattr(self, "_mid_path") else 0)
        img = self.read_region((0, 0), level, self.level_dimensions[level])
        img.thumbnail(size)
        return img


def open_slide(path: Path):
    """openslide for .svs, DcmSlide for .dcm."""
    if path.suffix.lower() == ".dcm":
        return DcmSlide(path)
    import openslide

    return openslide.open_slide(str(path))
