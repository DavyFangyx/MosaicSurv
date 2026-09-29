from __future__ import annotations

import torch
import torch.nn as nn

from models.missing_modality_baselines.common import ModalityMLP, normalize_avail, safe_flatten
from models.model_utils import WSIMILResampler, masked_mean


class ConcatMissingModalityBaseline(nn.Module):
    def __init__(
        self,
        *,
        wsi_embedding_dim: int = 1024,
        gene_embedding_dim: int = 3072,
        clinic_embedding_dim: int = 512,
        clinic_num_tokens: int | None = None,
        d_z: int = 128,
        mmhid: int = 256,
        dropout: float = 0.25,
        concat_wsi: str = "resampler",
        concat_impute: str = "zero",
        label_dim: int = 1,
    ):
        super().__init__()
        if concat_wsi not in {"meanpool", "resampler"}:
            raise ValueError(f"Unsupported concat_wsi `{concat_wsi}`.")
        if concat_impute not in {"zero", "mean"}:
            raise ValueError(f"Unsupported concat_impute `{concat_impute}`.")
        if label_dim != 1:
            raise ValueError("Concat baseline uses Cox risk with label_dim=1.")

        self.concat_wsi = concat_wsi
        self.concat_impute = concat_impute
        self.gene_embedding_dim = gene_embedding_dim
        self.clinic_embedding_dim = clinic_embedding_dim
        self.clinic_num_tokens = clinic_num_tokens
        self.d_z = d_z

        self.wsi_resampler = None
        if concat_wsi == "resampler":
            self.wsi_resampler = WSIMILResampler(
                input_dim=wsi_embedding_dim,
                token_dim=768,
                num_tokens=16,
                nhead=8,
                mlp_dim=1024,
                num_layers=2,
                dropout=dropout,
            )
            wsi_dim = 768
        else:
            wsi_dim = wsi_embedding_dim

        self.mlp_wsi = ModalityMLP(wsi_dim, d_z, dropout)
        self.mlp_gene = ModalityMLP(gene_embedding_dim, d_z, dropout)
        self.mlp_clinic = None
        self._clinic_flat_dim = clinic_embedding_dim * clinic_num_tokens if clinic_num_tokens is not None else None
        if self._clinic_flat_dim is not None:
            self.mlp_clinic = ModalityMLP(self._clinic_flat_dim, d_z, dropout)

        self.register_buffer("fill_wsi", torch.zeros(wsi_dim))
        self.register_buffer("fill_gene", torch.zeros(gene_embedding_dim))
        self.register_buffer("fill_clinic", torch.zeros(self._clinic_flat_dim or 1))

        self.fuse_fc = nn.Sequential(
            nn.Linear(3 * d_z, mmhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mmhid, mmhid),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(mmhid, label_dim)

    def _ensure_clinic_branch(self, flat_dim: int, device: torch.device) -> None:
        if self.mlp_clinic is not None and self._clinic_flat_dim == flat_dim:
            return
        self._clinic_flat_dim = flat_dim
        self.mlp_clinic = ModalityMLP(flat_dim, self.d_z, dropout=self.mlp_wsi.net[3].p)
        self.mlp_clinic = self.mlp_clinic.to(device)
        self.fill_clinic = torch.zeros(flat_dim, device=device)

    @torch.no_grad()
    def set_impute_stats(self, stats: dict[str, torch.Tensor]) -> None:
        if self.concat_impute != "mean":
            raise RuntimeError("Mean statistics are only valid for `concat_impute=mean`.")
        self.fill_wsi.copy_(stats["wsi"].detach().to(self.fill_wsi.device))
        self.fill_gene.copy_(stats["gene"].detach().to(self.fill_gene.device))
        clinic = stats["clinic"].detach().to(self.fill_clinic.device).reshape(-1)
        self._ensure_clinic_branch(int(clinic.numel()), self.fill_clinic.device)
        self.fill_clinic.copy_(clinic)

    @staticmethod
    def _fill(x: torch.Tensor, avail: torch.Tensor, fill_vec: torch.Tensor) -> torch.Tensor:
        keep = avail.view(-1, 1).to(dtype=x.dtype)
        return x * keep + fill_vec.unsqueeze(0) * (1.0 - keep)

    def _encode_wsi(self, x_path, wsi_mask=None):
        x_path = x_path.float()
        if x_path.dim() == 2:
            x_path = x_path.unsqueeze(0)
        if self.wsi_resampler is None:
            return masked_mean(x_path, wsi_mask)
        wsi_tokens = self.wsi_resampler(x_path, padding_mask=wsi_mask)
        return wsi_tokens.mean(dim=1)

    def _encode_clinic(self, x_clinic):
        x_clinic = x_clinic.float()
        if x_clinic.dim() == 3:
            flat = x_clinic.reshape(x_clinic.shape[0], -1)
            self._ensure_clinic_branch(flat.shape[1], x_clinic.device)
            return flat
        flat = safe_flatten(x_clinic)
        self._ensure_clinic_branch(flat.shape[1], x_clinic.device)
        return flat

    def forward(
        self,
        *,
        x_path=None,
        x_omic=None,
        x_clinic=None,
        data_WSI=None,
        data_omics=None,
        data_clinic=None,
        wsi_mask=None,
        avail=None,
        return_feats: bool = False,
        **_,
    ):
        x_path = x_path if x_path is not None else data_WSI
        x_omic = x_omic if x_omic is not None else data_omics
        x_clinic = x_clinic if x_clinic is not None else data_clinic
        if x_path is None or x_omic is None or x_clinic is None:
            raise KeyError("Expected WSI / Gene / Clinic inputs.")

        avail_mask = normalize_avail(avail, device=x_path.device)
        x_wsi = self._encode_wsi(x_path, wsi_mask=wsi_mask)
        x_gene = safe_flatten(x_omic.float())
        x_clinic = self._encode_clinic(x_clinic)

        x_wsi = self._fill(x_wsi, avail_mask[:, 0], self.fill_wsi)
        x_gene = self._fill(x_gene, avail_mask[:, 1], self.fill_gene)
        x_clinic = self._fill(x_clinic, avail_mask[:, 2], self.fill_clinic)

        h_w = self.mlp_wsi(x_wsi)
        h_g = self.mlp_gene(x_gene)
        h_c = self.mlp_clinic(x_clinic)
        fused = torch.cat([h_w, h_g, h_c], dim=1)
        risk = self.classifier(self.fuse_fc(fused))

        if return_feats:
            return risk, {"wsi": h_w, "gene": h_g, "clinic": h_c, "fused": fused}
        return risk


__all__ = ["ConcatMissingModalityBaseline"]
