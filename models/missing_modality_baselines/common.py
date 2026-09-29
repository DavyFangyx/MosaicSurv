from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import masked_mean
from utils.loss_func import CoxSurvLoss


MODALITY_NAMES = ("wsi", "gene", "clinic")
SUBSET_TO_MODALITIES = {
    "P": ("wsi",),
    "C": ("clinic",),
    "G": ("gene",),
    "PC": ("wsi", "clinic"),
    "PG": ("wsi", "gene"),
    "CG": ("gene", "clinic"),
    "PCG": ("wsi", "gene", "clinic"),
}
MODALITY_TO_LETTER = {"wsi": "P", "gene": "G", "clinic": "C"}
ALL_EVAL_SUBSETS = tuple(SUBSET_TO_MODALITIES.keys())
HGCN_SUBSET_TO_USE_TYPE = {
    "P": ["img"],
    "C": ["cli"],
    "G": ["rna"],
    "PC": ["img", "cli"],
    "PG": ["img", "rna"],
    "CG": ["rna", "cli"],
    "PCG": ["img", "rna", "cli"],
}


def as_bool_tensor(value, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.bool).view(-1)
    return torch.as_tensor(value, device=device, dtype=torch.bool).view(-1)


def normalize_avail(avail, *, device: torch.device) -> torch.Tensor:
    if avail is None:
        raise ValueError("Expected `avail` to be provided by the dataloader.")
    masks = []
    for name in MODALITY_NAMES:
        if name not in avail:
            raise KeyError(f"Missing availability key `{name}`.")
        masks.append(as_bool_tensor(avail[name], device=device))
    return torch.stack(masks, dim=1)


def safe_flatten(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x.reshape(x.shape[0], -1)


def maybe_tokenize_tokens(x: torch.Tensor, token_dim: int | None = None) -> torch.Tensor:
    if x is None:
        raise ValueError("Missing modality tensor.")
    x = x.float()
    if x.dim() == 2:
        if token_dim is not None and x.shape[1] % token_dim == 0:
            return x.reshape(x.shape[0], -1, token_dim)
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")


def tokenize_wsi(x_path: torch.Tensor, *, wsi_mask=None, resampler: WSIMILResampler | None = None) -> torch.Tensor:
    x_path = x_path.float()
    if x_path.dim() == 2:
        x_path = x_path.unsqueeze(0)
    if resampler is None:
        return masked_mean(x_path, wsi_mask).unsqueeze(1)
    return resampler(x_path, padding_mask=wsi_mask)


def cox_loss(risk: torch.Tensor, event_time: torch.Tensor, censor: torch.Tensor) -> torch.Tensor:
    return CoxSurvLoss(reduction="mean")(risk, event_time, censor)


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)


def prior_poe(mu_list: list[torch.Tensor], logvar_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    prior_mu = torch.zeros_like(mu_list[0])
    prior_logvar = torch.zeros_like(logvar_list[0])
    mus = torch.stack([*mu_list, prior_mu], dim=0)
    logvars = torch.stack([*logvar_list, prior_logvar], dim=0)
    return stable_poe(mus, logvars)


def stable_poe(mus: torch.Tensor, logvars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(mus) == 1:
        return mus[0], logvars[0]
    ln_inv_vars = torch.stack([-l for l in logvars])
    ln_var = -torch.logsumexp(ln_inv_vars, dim=0)
    joint_mu = (torch.exp(ln_inv_vars) * mus).sum(dim=0) * torch.exp(ln_var)
    return joint_mu, ln_var


def all_nonempty_subsets(items: Iterable[str]) -> list[tuple[str, ...]]:
    names = list(items)
    subsets: list[tuple[str, ...]] = []
    for r in range(1, len(names) + 1):
        subsets.extend(combinations(names, r))
    return subsets



def _canonical_subset_name(keep):
    if isinstance(keep, str):
        raw = keep.strip()
        upper = raw.upper().replace(" ", "")
        if upper in SUBSET_TO_MODALITIES:
            return upper
        names = tuple(token.strip().lower() for token in raw.replace(";", ",").split(",") if token.strip())
    else:
        names = tuple(str(name).strip().lower() for name in keep)
    letters = []
    seen = set()
    for name in names:
        letter = MODALITY_TO_LETTER.get(name)
        if letter is None:
            raise ValueError(f"Unsupported eval modality `{name}`. Expected wsi/gene/clinic or P/C/G subsets.")
        if letter not in seen:
            letters.append(letter)
            seen.add(letter)
    tag = "".join(sorted(letters, key=lambda item: "PCG".index(item)))
    if tag not in SUBSET_TO_MODALITIES:
        raise ValueError(f"Unsupported eval subset `{keep}`.")
    return tag


def parse_eval_modalities(raw):
    text = "off" if raw is None else str(raw).strip()
    if not text or text.lower() in {"off", "none", "false", "0"}:
        return ()
    if text.lower() == "all":
        return ALL_EVAL_SUBSETS
    tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
    if tokens and all(token.upper() in SUBSET_TO_MODALITIES for token in tokens):
        subsets = []
        seen = set()
        for token in tokens:
            name = token.upper()
            if name not in seen:
                subsets.append(name)
                seen.add(name)
        return tuple(subsets)
    return (_canonical_subset_name(text),)


def subset_keep_modalities(subset):
    return SUBSET_TO_MODALITIES[_canonical_subset_name(subset)]


def complete_avail_like(avail):
    if avail is None:
        raise ValueError("Expected `avail` to build a complete evaluation mask.")
    complete = {}
    for name in MODALITY_NAMES:
        if name not in avail:
            raise KeyError(f"Missing availability key `{name}`.")
        value = avail[name]
        if torch.is_tensor(value):
            complete[name] = torch.ones_like(value, dtype=torch.bool)
        else:
            complete[name] = True
    return complete


def apply_eval_subset(avail, subset):
    keep = set(subset_keep_modalities(subset))
    base = complete_avail_like(avail)
    restricted = {}
    for name in MODALITY_NAMES:
        value = base[name]
        if torch.is_tensor(value):
            restricted[name] = value.to(dtype=torch.bool) & (name in keep)
        else:
            restricted[name] = bool(value) and name in keep
    return restricted


def subset_result_dir(results_dir, subset):
    return Path(results_dir) / "eval_subsets" / _canonical_subset_name(subset)


def subset_summary_path(results_dir):
    return Path(results_dir) / "eval_subsets" / "summary_long.csv"


def hgcn_use_type_for_subset(subset):
    return list(HGCN_SUBSET_TO_USE_TYPE[_canonical_subset_name(subset)])


def eval_subset_metric_row(fold, subset, test_cindex, test_cindex_ipcw=0.0, test_IBS=0.0, test_iauc=0.0, test_iauc_list=0.0, test_loss=0.0, test_BS=0.0):
    return {
        "fold": int(fold),
        "subset": _canonical_subset_name(subset),
        "test_cindex": test_cindex,
        "test_cindex_ipcw": test_cindex_ipcw,
        "test_IBS": test_IBS,
        "test_iauc": test_iauc,
        "test_iauc_list": test_iauc_list,
        "test_loss": test_loss,
        "test_BS": test_BS,
    }



def write_eval_subset_outputs(results_dir, rows):
    import pandas as pd

    if not rows:
        return None
    results_dir = Path(results_dir)
    df = pd.DataFrame(rows)
    required = ["fold", "subset", "test_cindex", "test_cindex_ipcw", "test_IBS", "test_iauc", "test_iauc_list", "test_loss", "test_BS"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"eval subset rows missing columns: {missing}")
    df = df[required].copy()
    df["subset"] = [ _canonical_subset_name(value) for value in df["subset"] ]
    df = df.sort_values(["subset", "fold"]).reset_index(drop=True)
    summary_path = subset_summary_path(results_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    metric_cols = ["test_cindex", "test_cindex_ipcw", "test_IBS", "test_iauc", "test_iauc_list", "test_loss", "test_BS"]
    for subset, group in df.groupby("subset", sort=False):
        subset_df = group.sort_values("fold")[metric_cols].reset_index(drop=True)
        out_dir = subset_result_dir(results_dir, subset)
        out_dir.mkdir(parents=True, exist_ok=True)
        subset_df.to_csv(out_dir / "test_result.csv")
    return summary_path


def moment_match_gaussian(mus: list[torch.Tensor], logvars: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked_mu = torch.stack(mus, dim=0)
    stacked_logvar = torch.stack(logvars, dim=0)
    weights = torch.full((stacked_mu.shape[0], 1, 1), 1.0 / stacked_mu.shape[0], device=stacked_mu.device, dtype=stacked_mu.dtype)
    mean = (weights * stacked_mu).sum(dim=0)
    second_moment = (weights * (torch.exp(stacked_logvar) + stacked_mu.pow(2))).sum(dim=0)
    var = (second_moment - mean.pow(2)).clamp(min=1e-6)
    return mean, var.log().clamp(min=-4.0, max=2.0)


class CoxHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
