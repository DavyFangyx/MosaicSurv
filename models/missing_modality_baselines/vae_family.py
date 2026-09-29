from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from pythae.models.base.base_utils import ModelOutput

from models.missing_modality_baselines.common import CoxHead, normalize_avail
from models.model_utils import WSIMILResampler, WSITargetPoolingHead, masked_mean, reparameterize
from utils.loss_func import CoxSurvLoss


BASE_DIR = Path(__file__).resolve().parent
MULTIVAE_SRC = BASE_DIR / "third_party" / "MultiVae" / "src"
if str(MULTIVAE_SRC) not in sys.path:
    sys.path.insert(0, str(MULTIVAE_SRC))

from multivae.data.datasets.base import IncompleteDataset
from multivae.models.mopoe.mopoe_config import MoPoEConfig
from multivae.models.mopoe.mopoe_model import MoPoE
from multivae.models.mvae.mvae_config import MVAEConfig
from multivae.models.mvae.mvae_model import MVAE as _MVAE
from multivae.models.nn.default_architectures import BaseDictDecoders, BaseDictEncoders


def _flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x.reshape(x.shape[0], -1)


def _get_joint_latent_params(model, inputs):
    """Return joint posterior parameters across MultiVae model variants."""
    if hasattr(model, "inference"):
        return model.inference(inputs)["joint"]

    # This vendored MVAE exposes the joint posterior through this method
    # instead of the inference() API used by MoPoE.
    if hasattr(model, "compute_mu_log_var_subset"):
        return model.compute_mu_log_var_subset(inputs, list(model.encoders.keys()))

    raise AttributeError(
        f"{type(model).__name__} exposes neither inference() nor "
        "compute_mu_log_var_subset()."
    )


class _SurvivalMultiVAEBase(nn.Module):
    model_cls = None
    config_cls = None

    def __init__(
        self,
        *,
        wsi_embedding_dim: int = 1024,
        gene_embedding_dim: int = 768,
        clinic_embedding_dim: int = 512,
        gene_num_tokens: int = 4,
        clinic_num_tokens: int = 6,
        latent_dim: int = 128,
        mmhid: int = 256,
        decoder_hidden_dim: int = 512,
        dropout: float = 0.1,
        wsi_resampler_tokens: int = 16,
        wsi_resampler_layers: int = 2,
        use_subsampling: bool = True,
        k_subsets: int = 2,
        beta: float = 1.0,
        warmup: int = 10,
        lambda_surv: float = 1.0,
    ):
        super().__init__()
        self.gene_num_tokens = gene_num_tokens
        self.clinic_num_tokens = clinic_num_tokens
        self.latent_dim = latent_dim
        self.lambda_surv = lambda_surv
        self.beta = beta
        self.warmup = warmup

        self.wsi_resampler = WSIMILResampler(
            input_dim=wsi_embedding_dim,
            token_dim=768,
            num_tokens=wsi_resampler_tokens,
            nhead=8,
            mlp_dim=1024,
            num_layers=wsi_resampler_layers,
            dropout=dropout,
        )
        self.wsi_target_head = WSITargetPoolingHead(token_dim=768, output_dim=768)
        self.cox_head = CoxHead(latent_dim, hidden_dim=mmhid, dropout=dropout)
        self.cox_loss = CoxSurvLoss(reduction="mean")

        input_dims = {
            "wsi": (768,),
            "gene": (gene_num_tokens * gene_embedding_dim,),
            "clinic": (clinic_num_tokens * clinic_embedding_dim,),
        }
        encoders = BaseDictEncoders(input_dims=input_dims, latent_dim=latent_dim)
        decoders = BaseDictDecoders(input_dims=input_dims, latent_dim=latent_dim)

        if self.config_cls is MVAEConfig:
            model_config = MVAEConfig(
                n_modalities=3,
                latent_dim=latent_dim,
                input_dims=input_dims,
                use_subsampling=use_subsampling,
                k=k_subsets,
                warmup=warmup,
                beta=beta,
            )
        else:
            model_config = MoPoEConfig(
                n_modalities=3,
                latent_dim=latent_dim,
                input_dims=input_dims,
                subsets=None,
                beta=beta,
                beta_style=1.0,
                modalities_specific_dim=None,
            )

        self.model = self.model_cls(model_config, encoders=encoders, decoders=decoders)

    def _prepare_inputs(self, x_path, x_omic, x_clinic, wsi_mask=None):
        x_path = x_path.float()
        if x_path.dim() == 2:
            x_path = x_path.unsqueeze(0)
        wsi_tokens = self.wsi_resampler(x_path, padding_mask=wsi_mask)
        wsi_feat = self.wsi_target_head(wsi_tokens, padding_mask=None)

        x_omic = _flatten_if_needed(x_omic.float())
        x_clinic = _flatten_if_needed(x_clinic.float())
        return wsi_feat, x_omic, x_clinic

    def _build_dataset(self, x_path, x_omic, x_clinic, avail, wsi_mask=None):
        wsi_feat, gene_feat, clinic_feat = self._prepare_inputs(x_path, x_omic, x_clinic, wsi_mask=wsi_mask)
        avail_mask = normalize_avail(avail, device=wsi_feat.device)

        data = {
            "wsi": torch.where(avail_mask[:, 0].view(-1, 1), wsi_feat, torch.zeros_like(wsi_feat)),
            "gene": torch.where(avail_mask[:, 1].view(-1, 1), gene_feat, torch.zeros_like(gene_feat)),
            "clinic": torch.where(avail_mask[:, 2].view(-1, 1), clinic_feat, torch.zeros_like(clinic_feat)),
        }
        masks = {
            "wsi": avail_mask[:, 0],
            "gene": avail_mask[:, 1],
            "clinic": avail_mask[:, 2],
        }
        return IncompleteDataset(data=data, masks=masks), avail_mask

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
        event_time=None,
        censor=None,
        epoch: int = 1,
        batch_ratio: float = 0.0,
        **kwargs,
    ):
        x_path = x_path if x_path is not None else data_WSI
        x_omic = x_omic if x_omic is not None else data_omics
        x_clinic = x_clinic if x_clinic is not None else data_clinic
        if x_path is None or x_omic is None or x_clinic is None:
            raise KeyError("Expected WSI / Gene / Clinic inputs.")
        if avail is None:
            raise ValueError("Expected `avail` from the dataloader.")

        inputs, _ = self._build_dataset(x_path, x_omic, x_clinic, avail, wsi_mask=wsi_mask)
        vae_out = self.model(inputs, epoch=epoch, batch_ratio=batch_ratio)
        joint_mu, joint_logvar = _get_joint_latent_params(self.model, inputs)
        z = reparameterize(joint_mu, joint_logvar, sample=self.training)
        risk = self.cox_head(z)

        surv_loss = None
        if event_time is not None and censor is not None:
            surv_loss = self.cox_loss(risk, event_time, censor)

        loss = vae_out.loss
        if surv_loss is not None:
            loss = loss + self.lambda_surv * surv_loss

        out = ModelOutput(
            loss=loss,
            vae_loss=vae_out.loss,
            surv_loss=surv_loss,
            risk=risk,
            z=z,
            vae_output=vae_out,
        )
        return out


class MVAEBaseline(_SurvivalMultiVAEBase):
    model_cls = _MVAE
    config_cls = MVAEConfig


class MoPoEBaseline(_SurvivalMultiVAEBase):
    model_cls = MoPoE
    config_cls = MoPoEConfig


__all__ = ["MVAEBaseline", "MoPoEBaseline"]
