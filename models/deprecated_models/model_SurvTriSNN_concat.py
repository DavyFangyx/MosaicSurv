from pathlib import Path

import torch
import torch.nn as nn

from models.model_single_clinic import SNNClinic
from models.model_single_gene import SNNGeneFM
from models.model_single_wsi import MLPWSI
# from models.model_TransMIL import TransMIL


def _resolve_wsi_embedding_dim(checkpoint_path: Path, wsi_embedding_dim: int = None) -> int:
    if wsi_embedding_dim is not None:
        return wsi_embedding_dim
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    if "wsi_projection_net.0.weight" in state_dict:
        return int(state_dict["wsi_projection_net.0.weight"].shape[1])
    return int(state_dict["_fc1.0.weight"].shape[1])


def _resolve_clinic_num_tokens(checkpoint_path: Path, clinic_num_tokens: int = None) -> int:
    if clinic_num_tokens is not None:
        return clinic_num_tokens
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    classifier_in_dim = int(state_dict["classifier.weight"].shape[1])
    hidden_dim = 256
    return classifier_in_dim // hidden_dim


def _load_checkpoint_state_dict(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value
    return cleaned_state_dict


def _load_encoder_weights(module: nn.Module, checkpoint_path: Path, skip_prefixes, module_name: str):
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    module_state = module.state_dict()
    filtered_state = {}

    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        if key in module_state and module_state[key].shape == value.shape:
            filtered_state[key] = value

    if not filtered_state:
        raise RuntimeError(f"No encoder weights matched for {module_name} from {checkpoint_path}")

    missing_keys, unexpected_keys = module.load_state_dict(filtered_state, strict=False)
    ignored_missing = [key for key in missing_keys if any(key.startswith(prefix) for prefix in skip_prefixes)]
    real_missing = [key for key in missing_keys if key not in ignored_missing]
    if real_missing or unexpected_keys:
        raise RuntimeError(
            f"Failed to load pretrained {module_name} from {checkpoint_path}\n"
            f"missing_keys={real_missing}\nunexpected_keys={unexpected_keys}"
        )


def _freeze_module(module: nn.Module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


class SurvTriSNNConcat(nn.Module):
    def __init__(
        self,
        study: str,
        current_fold: int,
        wsi_embedding_dim: int = None,
        gene_input_dim: int = 3072,
        clinic_input_dim: int = 512,
        clinic_num_tokens: int = None,
        num_classes: int = 4,
        single_model_size: str = "medium",
        dropout: float = 0.25,
        wsi_projection_dim: int = 256,
        selected_modalities: str = "wsi,gene,clinic",
    ):
        super().__init__()
        if not study:
            raise ValueError("`study` is required.")
        if current_fold is None or current_fold < 0:
            raise ValueError("`current_fold` must be a non-negative integer.")

        self.study = study
        self.current_fold = int(current_fold)
        self.selected_modalities = tuple(selected_modalities.split(","))

        repo_root = Path(__file__).resolve().parents[2]
        pretrained_root = repo_root / "results" / "Single_Multi_Test"
        wsi_checkpoint = pretrained_root / "WSItest_F" / f"{study}__uni_v2" / "mlp_wsi" / f"s_{self.current_fold}_checkpoint.pt"
        gene_checkpoint = pretrained_root / "Gengtest_F" / f"{study}__scFoundation_embedding_cell_raw" / "snn_gene_f" / f"s_{self.current_fold}_checkpoint.pt"
        clinic_checkpoint = pretrained_root / "Clinictest_Li" / f"{study}__L4" / "snn_clinic_flatten" / f"s_{self.current_fold}_checkpoint.pt"

        wsi_embedding_dim = _resolve_wsi_embedding_dim(wsi_checkpoint, wsi_embedding_dim)
        clinic_num_tokens = _resolve_clinic_num_tokens(clinic_checkpoint, clinic_num_tokens)

        clinic_size = single_model_size if single_model_size in {"small", "big"} else "small"
        gene_model_size = single_model_size if single_model_size in {"medium", "big"} else "medium"

        # self.wsi_encoder = TransMIL(input_dim=wsi_embedding_dim, n_classes=num_classes)
        self.wsi_encoder = None
        self.gene_encoder = None
        self.clinic_encoder = None

        self.wsi_feat_dim = wsi_projection_dim
        self.gene_feat_dim = 256
        self.clinic_feat_dim = 256 * clinic_num_tokens

        if "wsi" in self.selected_modalities:
            self.wsi_encoder = MLPWSI(
                wsi_embedding_dim=wsi_embedding_dim,
                dropout=dropout,
                num_classes=num_classes,
                wsi_projection_dim=wsi_projection_dim,
            )
            _load_encoder_weights(self.wsi_encoder, wsi_checkpoint, ("to_logits.", "feed_forward.", "layer_norm."), "wsi_encoder")
            _freeze_module(self.wsi_encoder)

        if "gene" in self.selected_modalities:
            self.gene_encoder = SNNGeneFM(
                input_dim=gene_input_dim,
                model_size=gene_model_size,
                n_classes=num_classes,
            )
            _load_encoder_weights(self.gene_encoder, gene_checkpoint, ("classifier.",), "gene_encoder")
            _freeze_module(self.gene_encoder)

        if "clinic" in self.selected_modalities:
            self.clinic_encoder = SNNClinic(
                input_dim=clinic_input_dim,
                model_size_omic=clinic_size,
                n_classes=num_classes,
                clinic_num_tokens=clinic_num_tokens,
                pooling="flatten",
            )
            _load_encoder_weights(self.clinic_encoder, clinic_checkpoint, ("classifier.",), "clinic_encoder")
            _freeze_module(self.clinic_encoder)

        feat_dims = {
            "wsi": self.wsi_feat_dim,
            "gene": self.gene_feat_dim,
            "clinic": self.clinic_feat_dim,
        }
        self.total_feat_dim = sum(feat_dims[name] for name in self.selected_modalities)

        hidden_dim = max(256, self.total_feat_dim // 4)
        self.classifier = nn.Sequential(
            nn.Linear(self.total_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.wsi_encoder is not None:
            _freeze_module(self.wsi_encoder)
        if self.gene_encoder is not None:
            _freeze_module(self.gene_encoder)
        if self.clinic_encoder is not None:
            _freeze_module(self.clinic_encoder)
        return self

    def _encode_modalities(self, x_path: torch.Tensor, x_omic: torch.Tensor, x_clinic: torch.Tensor):
        with torch.no_grad():
            features = {}
            if self.wsi_encoder is not None:
                wsi_embed = self.wsi_encoder.wsi_projection_net(x_path)
                features["wsi"] = torch.mean(wsi_embed, dim=1)
            if self.gene_encoder is not None:
                gene_feat, _ = self.gene_encoder(x_omic=x_omic, return_feats=True)
                features["gene"] = gene_feat
            if self.clinic_encoder is not None:
                clinic_feat, _ = self.clinic_encoder(x_clinic=x_clinic, return_feats=True)
                features["clinic"] = clinic_feat
        return features

    def forward(self, return_feats: bool = False, **kwargs):
        x_path = kwargs.get("x_path")
        if x_path is None:
            x_path = kwargs.get("data_WSI")
        x_omic = kwargs.get("x_omic")
        if x_omic is None:
            x_omic = kwargs.get("data_omics")
        x_clinic = kwargs.get("x_clinic")
        if x_clinic is None:
            x_clinic = kwargs.get("data_clinic")

        if "wsi" in self.selected_modalities and x_path is None:
            raise KeyError("Expected `x_path` or `data_WSI` in kwargs.")
        if "gene" in self.selected_modalities and x_omic is None:
            raise KeyError("Expected `x_omic` or `data_omics` in kwargs.")
        if "clinic" in self.selected_modalities and x_clinic is None:
            raise KeyError("Expected `x_clinic` or `data_clinic` in kwargs.")

        feature_dict = self._encode_modalities(x_path, x_omic, x_clinic)
        fused_feat = torch.cat([feature_dict[name] for name in self.selected_modalities], dim=1)
        logits = self.classifier(fused_feat)

        if return_feats:
            return fused_feat, logits
        return logits
