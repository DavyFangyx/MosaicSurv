from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


HGCN_PAD_DIM = 1024


def parse_patch_coord(patch_id: str) -> tuple[int, int] | None:
    """Parse HGCN patch ids.

    Original tiling writes `{row}_{col}`; gendata.ipynb splits `{row}-{col}`.
    Treat `_` and `-` as grid separators, then keep the last two integers.
    """
    text = Path(str(patch_id)).name.strip()
    if text.lower().endswith('.png'):
        text = text[:-4]
    parts = [part for part in re.split(r"[_-]", text) if part != ""]
    ints = []
    for part in parts:
        if re.fullmatch(r"-?\d+", part):
            ints.append(int(part))
    if len(ints) < 2:
        return None
    return ints[-2], ints[-1]


def empty_node_features(dim: int = HGCN_PAD_DIM) -> torch.Tensor:
    return torch.zeros((0, int(dim)), dtype=torch.float32)


def empty_edge_index() -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long)


def full_connect_edge_index(n_nodes: int) -> torch.Tensor:
    """Directed full graph without self-loops, matching gendata.ipynb."""
    n_nodes = int(n_nodes)
    if n_nodes <= 1:
        return empty_edge_index()
    start: list[int] = []
    end: list[int] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                start.append(j)
                end.append(i)
    return torch.tensor([start, end], dtype=torch.long)


def _slide_namespace(patch_id: str) -> str:
    text = str(patch_id)
    if "/" not in text:
        return ""
    return text.rsplit("/", 1)[0]


def image_grid_edge_index(patch_ids: Sequence[str]) -> torch.Tensor:
    """8-neighborhood on the WSI patch grid. Missing neighbors are skipped."""
    index_of: dict[str, int] = {}
    coords: list[tuple[str, int, int]] = []
    for patch_id in patch_ids:
        coord = parse_patch_coord(patch_id)
        if coord is None:
            continue
        namespace = _slide_namespace(patch_id)
        key = f"{namespace}|{coord[0]}-{coord[1]}"
        if key in index_of:
            continue
        index_of[key] = len(coords)
        coords.append((namespace, coord[0], coord[1]))

    start: list[int] = []
    end: list[int] = []
    offsets = (
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
    )
    for namespace, row, col in coords:
        src = index_of[f"{namespace}|{row}-{col}"]
        for drow, dcol in offsets:
            dst = index_of.get(f"{namespace}|{row + drow}-{col + dcol}")
            if dst is None:
                continue
            start.append(src)
            end.append(dst)
    if not start:
        return empty_edge_index()
    return torch.tensor([start, end], dtype=torch.long)


def pad_vector(values: Sequence[float], dim: int = HGCN_PAD_DIM) -> np.ndarray:
    out = np.zeros((int(dim),), dtype=np.float32)
    n = min(len(values), int(dim))
    if n:
        out[:n] = np.asarray(values[:n], dtype=np.float32)
    return out


def diagonal_pad(values: Sequence[float | None], dim: int = HGCN_PAD_DIM) -> np.ndarray:
    n_cli = len(values)
    out = np.zeros((n_cli, int(dim)), dtype=np.float32)
    for i, value in enumerate(values):
        if value is None:
            continue
        if i >= int(dim):
            break
        out[i, i] = np.float32(value)
    return out
