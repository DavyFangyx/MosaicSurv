from ast import Lambda
import copy
import numpy as np
import pdb
import os
import re
import time
import pandas as pd
from pathlib import Path
from utils.file_utils import _save_pkl
from custom_optims.radam import RAdam
from datasets.dataset_survival import SurvivalDatasetFactory
from dataset_deployment.registry import infer_standard_paths
from models.model_ABMIL import ABMIL
from models.model_single_clinic import CoxClinic, MLPClinic, SNNClinic
from models.model_single_gene import MLPGene, MLPGeneFM, SNNGene, SNNGeneFM
from models.model_single_wsi import MLPWSI
from models.model_MCATPathways import MCAT_Surv
from models.model_PORPOISE import PorpoiseMMF
from models.model_SurvPath import SurvPath
from models.model_SurvPGC_foundation import SurvPGC_F
from models.model_SurvPC import SurvPC
from models.model_SurvPC_foundation import SurvPC_F
from models.model_TransMIL import TransMIL
from models.model_SurvGC_foundation import SurvGC_F
from models.model_MLPPC_concat import MLPPC_concat
from models.deprecated_models.model_SurvFusion_separate import SurvFusion
from models.deprecated_models.model_SurvFusion_noalign import SurvFusionNoAlign
from models.deprecated_models.model_SurvFusion_joint import SurvFusionJoint
from models.deprecated_models.model_SurvTriSNN_concat import SurvTriSNNConcat
from models.deprecated_models.model_SurvTriSNN_mhsa import SurvTriSNNMHSA
from models.deprecated_models.model_SurvTriMLP_concat import SurvTriMLPConcat
from models.deprecated_models.model_SurvTriMLP_mhsa import SurvTriMLPMHSA
from models.model_SurvTriPoEVAE import SurvTriPoEVAE
from models.ablation_models.model_B_crossstage1 import SurvTriPoEVAE_BCrossStage1
from models.ablation_models.model_B_nopretrain import SurvTriPoEVAE_BNoPretrain
from models.ablation_models.model_B_kl import SurvTriPoEVAE_BKL
from models.ablation_models.model_B_single import SurvTriPoEVAE_BSingle
from models.ablation_models.model_B_multi import SurvTriPoEVAE_BMulti
from models.ablation_models.model_mosaic_surv_twostage import MosaicSurvTwostage
from models.ablation_models.model_mosaic_surv_frozen import MosaicSurvFrozen
from models.ablation_models.model_mosaic_surv_single import MosaicSurvSingle
from models.ablation_models.model_mosaic_surv_single_enum import MosaicSurvSingleEnum
from models.ablation_models.model_mosaic_surv_multi import MosaicSurvMulti
from models.ablation_models.model_mosaic_surv import MosaicSurv
from models.ablation_models.model_mosaic_surv_noenum import MosaicSurvNoEnum
from models.ablation_models.model_mosaic_surv_kl import MosaicSurvKL
from models.ablation_models.model_mosaic_surv_detached import MosaicSurvDetached
from models.missing_modality_baselines.concat import ConcatMissingModalityBaseline
from models.missing_modality_baselines.vae_family import MVAEBaseline, MoPoEBaseline
from models.missing_modality_baselines.common import (
    apply_eval_subset,
    parse_eval_modalities,
    safe_flatten,
    subset_result_dir,
)
from models.model_utils import PATTERN_NAMES, PATTERNS, pattern_to_mask
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score
from sksurv.util import Surv
from utils.survival_metrics import (
    AUC_LANDMARK_MONTHS,
    IBS_GRID_MONTHS,
    MIN_EVENTS_FOR_AUC,
    breslow_survival,
    compute_ibs,
    compute_landmark_aucs,
    interpolate_survival,
)

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss, NLLDiffSurvLoss, CoxSurvLoss
from utils.wandb_utils import finish_wandb_run, init_wandb_run
from utils.optuna_utils import log_optuna_analysis

import torch.optim as optim
try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


CSV_GENE_MODALITIES = {"mlp_gene", "snn_gene"}
GENE_F_MODALITIES = {"mlp_gene_f", "snn_gene_f"}
CLINIC_MODALITIES = {
    "mlp_clinic_mean", "mlp_clinic_flatten",
    "snn_clinic_mean", "snn_clinic_flatten",
    "clinic_cox",
}
TRIMODAL_MODALITIES = {
    "survtri_snn_concat",
    "survtri_snn_mhsa",
    "survtri_mlp_concat",
    "survtri_mlp_mhsa",
}
POE_NEW_A_MODALITIES = {"mosaic_surv_frozen"}
POE_NEW_B_MODALITIES = {
    "survtri_poe_vae_b_single",
    "survtri_poe_vae_b_multi",
    "mosaic_surv_twostage",
}
POE_NEW_C_MODALITIES = {
    "mosaic_surv_single",
    "mosaic_surv_single_enum",
    "mosaic_surv_multi",
    "mosaic_surv",
    "mosaic_surv_noenum",
    "mosaic_surv_kl",
    "mosaic_surv_detached",
    "mosaic_surv_nojeffreys",
    "mosaic_surv_nodropout",
}
POE_MULTI_PATTERN_MODALITIES = {
    "survtri_poe_vae_b_multi",
    "mosaic_surv_twostage",
    "mosaic_surv_frozen",
    "mosaic_surv_multi",
    "mosaic_surv",
    "mosaic_surv_single_enum",
    "mosaic_surv_kl",
    "mosaic_surv_detached",
    "mosaic_surv_nojeffreys",
    "mosaic_surv_nodropout",
}
POE_PGC_SELECTION_MODALITIES = {
    "mosaic_surv",
    "mosaic_surv_kl",
    "mosaic_surv_detached",
    "mosaic_surv_nojeffreys",
    "mosaic_surv_nodropout",
}
POE_MODALITIES = {
    "survtri_poe_vae",
    "survtri_poe_vae_b_kl",
    "survtri_poe_vae_b_crossstage1",
} | POE_NEW_A_MODALITIES | POE_NEW_B_MODALITIES | POE_NEW_C_MODALITIES
POE_CROSS_STAGE1_MODALITIES = {"survtri_poe_vae_b_crossstage1"}
POE_STAGE2_ONLY_MODALITIES = {"survtri_poe_vae_b_nopretrain"}
POE_ALL_MODALITIES = POE_MODALITIES | POE_STAGE2_ONLY_MODALITIES
MISSING_MODALITY_BASELINES = {"modality_concat", "mvae_poe", "mopoe"}
VAE_MISSING_MODALITY_BASELINES = {"mvae_poe", "mopoe"}


def _wandb_enabled(args):
    return wandb is not None and getattr(args, "wandb_run", None) is not None


def _wandb_log(args, payload, step=None):
    if not _wandb_enabled(args):
        return
    clean_payload = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()
        clean_payload[key] = value
    wandb.log(clean_payload, step=step)


def _optuna_report_and_prune(args, metric, step):
    trial = getattr(args, "optuna_trial", None)
    if trial is None:
        return
    trial.report(float(metric), step=step)
    if trial.should_prune():
        pruned_exception = getattr(args, "optuna_pruned_exception", None)
        if pruned_exception is None:
            raise RuntimeError("Optuna pruning requested but no pruning exception class was provided.")
        raise pruned_exception(f"Pruned at epoch {step} with val_cindex={metric:.4f}")


def _safe_cindex(event_times, censorships, risks):
    try:
        return concordance_index_censored(
            (1 - censorships).astype(bool),
            event_times,
            risks,
            tied_tol=1e-08,
        )[0]
    except ValueError:
        return float("nan")


@torch.no_grad()
def _evaluate_pattern_cindices(args, model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    collected = {s: {"risk": [], "event": [], "censor": []} for s in PATTERNS}
    for data in loader:
        data_WSI, mask, _, event_time, censor, data_omics, _, data_clinic, avail = _unpack_survtri_poe_vae_batch(device, data)
        avail = _build_survtri_poe_vae_avail(args, model, avail)
        model(
            x_path=data_WSI,
            x_omic=data_omics,
            x_clinic=data_clinic,
            wsi_mask=mask,
            avail=avail,
        )
        cached = model.get_cached_outputs()
        mu_list = cached["mu_list"]
        logvar_list = cached["logvar_list"]
        event_np = _tensor_to_numpy_safe(event_time.reshape(-1), dtype=np.float64)
        censor_np = _tensor_to_numpy_safe(censor.reshape(-1), dtype=np.float64)
        for s in PATTERNS:
            pm = pattern_to_mask(s, device=device).unsqueeze(0).expand(event_time.shape[0], -1)
            mu_s, _, _ = model.poe(mus=mu_list, logvars=logvar_list, available_mask=pm)
            risk_s = model.pattern_head(mu_s, pattern_id=s)
            collected[s]["risk"].append(_tensor_to_numpy_safe(risk_s.reshape(-1), dtype=np.float64))
            collected[s]["event"].append(event_np)
            collected[s]["censor"].append(censor_np)
    metrics = {}
    values = []
    for s in PATTERNS:
        name = PATTERN_NAMES[s]
        risks = np.concatenate(collected[s]["risk"], axis=0) if collected[s]["risk"] else np.array([])
        events = np.concatenate(collected[s]["event"], axis=0) if collected[s]["event"] else np.array([])
        censors = np.concatenate(collected[s]["censor"], axis=0) if collected[s]["censor"] else np.array([])
        cindex = _safe_cindex(events, censors, risks) if risks.size else float("nan")
        metrics[f"val/cindex_{name}"] = cindex
        if np.isfinite(cindex):
            values.append(cindex)
    metrics["val/cindex_weighted"] = float(np.mean(values)) if values else float("nan")
    metrics["val/cindex_worst"] = float(np.min(values)) if values else float("nan")
    if was_training:
        model.train()
    return metrics


def _tensor_to_numpy_safe(tensor, dtype=None):
    array = np.asarray(tensor.detach().cpu().tolist(), dtype=dtype)
    return array


def _init_poe_epoch_monitor():
    return {
        "count": 0,
        "mean_norm_sum": 0.0,
        "mean_std_sum": 0.0,
        "alpha_sum": np.zeros(3, dtype=np.float64),
        "rec_count": {"wsi": 0, "gene": 0, "clinic": 0},
        "rec_sum": {"wsi": 0.0, "gene": 0.0, "clinic": 0.0},
    }


def _update_poe_epoch_monitor(monitor, cached_outputs):
    mu_joint = cached_outputs["mu_joint"].detach()
    z_joint = cached_outputs["z_joint"].detach()
    poe_weights = cached_outputs["poe_weights"].detach()
    batch_size = mu_joint.shape[0]

    monitor["count"] += batch_size
    monitor["mean_norm_sum"] += torch.norm(mu_joint, dim=1).mean().item() * batch_size
    monitor["mean_std_sum"] += z_joint.var(dim=0, unbiased=False).mean().item() * batch_size
    monitor["alpha_sum"] += _tensor_to_numpy_safe(poe_weights.mean(dim=0), dtype=np.float64) * batch_size

    recon_losses = cached_outputs.get("recon_losses", {})
    for name in ("wsi", "gene", "clinic"):
        count = int(recon_losses.get(f"counts_{name}", batch_size))
        loss = recon_losses.get(name)
        if loss is None or not torch.is_tensor(loss) or not torch.isfinite(loss.detach()).all():
            continue
        monitor["rec_count"][name] += count
        monitor["rec_sum"][name] += float(loss.detach().item()) * count


def _finalize_poe_epoch_monitor(monitor):
    if monitor["count"] == 0:
        return {}
    denom = float(monitor["count"])
    metrics = {
        "z/mean_norm": monitor["mean_norm_sum"] / denom,
        "z/mean_std": monitor["mean_std_sum"] / denom,
        "poe/alpha_wsi": monitor["alpha_sum"][0] / denom,
        "poe/alpha_gene": monitor["alpha_sum"][1] / denom,
        "poe/alpha_clinic": monitor["alpha_sum"][2] / denom,
    }

    for name in ("wsi", "gene", "clinic"):
        rec_count = monitor["rec_count"][name]
        metrics[f"count/rec_{name}"] = rec_count
        if rec_count > 0:
            metrics[f"loss/rec_{name}"] = monitor["rec_sum"][name] / float(rec_count)
    return metrics


def _clone_args_for_subrun(args, results_dir, stage1_scan_tag=None):
    cloned = copy.copy(args)
    cloned.results_dir = results_dir
    cloned.wandb_run = None
    cloned.current_stage1_scan_tag = stage1_scan_tag
    return cloned


def _save_model_checkpoint(model, path, args):
    """Save missing-modality baselines without serializing closure-bearing objects."""
    if args.modality in MISSING_MODALITY_BASELINES:
        torch.save(
            {
                "checkpoint_format": "state_dict_v1",
                "modality": args.modality,
                "model_state_dict": model.state_dict(),
            },
            path,
        )
    else:
        torch.save(model, path)


def _load_model_checkpoint(model, path, args):
    """Load a checkpoint into the already-constructed model when possible."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(device)
    return checkpoint


def _cross_stage1_enabled(args):
    return args.modality in POE_CROSS_STAGE1_MODALITIES and args.poe_variant == "B"


def _normalize_stage1_study(study):
    study = str(study).strip()
    if not study:
        return None
    if not study.startswith("tcga_"):
        study = f"tcga_{study}"
    return study


def _get_poe_stage1_studies(args):
    if not _cross_stage1_enabled(args):
        return [args.study]

    raw_value = getattr(args, "poe_stage1_studies", "")
    studies = []
    seen = set()
    for token in str(raw_value).split(","):
        normalized = _normalize_stage1_study(token)
        if normalized and normalized not in seen:
            studies.append(normalized)
            seen.add(normalized)

    if not studies:
        raise ValueError("survtri_poe_vae_b_crossstage1 requires --poe_stage1_studies")
    return studies


def _normalize_stage1_studies_for_key(args):
    return sorted(_get_poe_stage1_studies(args))


def _sanitize_path_fragment(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def _get_poe_stage1_cache_tag(args):
    studies_tag = "-".join(_normalize_stage1_studies_for_key(args))
    parts = [
        args.modality,
        f"variant{getattr(args, 'poe_variant', 'B')}",
        f"studies{studies_tag}",
        f"sel{getattr(args, 'selected_modalities', 'wsi,gene,clinic')}",
        f"path{args.type_of_path}",
        f"seed{args.seed}",
        f"lr{args.lr_stage1}",
        f"epochs{args.max_epochs_stage1}",
        f"bs{args.batch_size_stage1}",
        f"drop{args.poe_modality_dropout}",
        f"miss{getattr(args, 'missing_mode', 'model_gen')}",
        f"mpat{getattr(args, 'missing_pattern', '') or 'none'}",
        f"mseed{getattr(args, 'missing_seed', args.seed)}",
        f"mmhid{args.poe_mmhid}",
        f"dec{args.poe_decoder_hidden_dim}",
        f"beta{args.poe_beta_target}",
        f"betafix{int(bool(getattr(args, 'betafix', False)))}",
        f"layers{args.poe_transformer_layers}",
    ]
    if getattr(args, 'alphafix', False):
        parts.extend([
            "alphafix1",
            f"alphapgc{getattr(args, 'alphapgc', '') or 'none'}",
        ])
    return "__".join(_sanitize_path_fragment(part) for part in parts)


def _get_poe_stage1_cache_root(args):
    # Cross-study Stage1 checkpoints are shared by all runs in one experiment
    # group, so keep them at results/<exp_group>/ instead of a project-level
    # cache.
    results_dir = Path(args.results_dir).resolve()
    exp_group = getattr(args, "exp_group", "default")
    return str(results_dir.parents[2] / exp_group)


def _get_poe_stage1_ckpt_path(args, cur):
    if _cross_stage1_enabled(args):
        return os.path.join(_get_poe_stage1_cache_root(args), f"fold_{cur}", f"s_{cur}_stage1_checkpoint.pt")
    return os.path.join(args.results_dir, f"s_{cur}_stage1_checkpoint.pt")


def _acquire_stage1_lock(lock_path):
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    else:
        os.close(fd)
        return True


def _release_stage1_lock(lock_path):
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _infer_workspace_experiments(args):
    return {
        "wsi_experiment": Path(args.data_root_dir).name,
        "clinic_experiment": Path(args.clinic_dir).name,
        "gene_experiment": Path(args.gene_dir).name,
    }


def _clone_args_for_study(args, study):
    cloned = copy.copy(args)
    experiments = _infer_workspace_experiments(args)
    paths = infer_standard_paths(
        study,
        ".",
        which_splits=args.which_splits,
        type_of_path=args.type_of_path,
        **experiments,
    )
    cloned.study = study
    cloned.combined_study = study
    cloned.label_file = str(paths["label_file"])
    cloned.omics_dir = str(paths["omics_dir"])
    cloned.split_dir = str(paths["split_dir"])
    cloned.data_root_dir = str(paths["data_root_dir"])
    cloned.clinic_dir = str(paths["clinic_dir"])
    cloned.gene_dir = str(paths["gene_dir"])
    cloned.clinical_file = str(paths["clinical_file"])
    return cloned


def _build_survival_dataset_factory(args):
    return SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        data_dir=args.data_root_dir,
        clinical_file=args.clinical_file,
        seed=args.seed,
        print_info=True,
        n_bins=args.n_classes,
        label_col=args.label_col,
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat=True if args.modality == "mcat" else False,
        is_survpath=True if args.modality == "survpath" else False,
        is_survpath_f=True if args.modality == "survpath_f" else False,
        is_survpgc=True if args.modality == "survpgc" else False,
        is_survpgc_f=True if args.modality in (
            "survpgc_f",
            "survfusion_separate",
            "survfusion_noalign",
            "survfusion_joint",
            "survtri_snn_concat",
            "survtri_snn_mhsa",
            "survtri_mlp_concat",
            "survtri_mlp_mhsa",
            "survtri_poe_vae",
            "modality_concat",
            "mvae_poe",
            "mopoe",
            "survtri_poe_vae_b_kl",
            "survtri_poe_vae_b_crossstage1",
            "survtri_poe_vae_b_nopretrain",
            "survtri_poe_vae_b_single",
            "survtri_poe_vae_b_multi",
            "mosaic_surv_twostage",
            "mosaic_surv_frozen",
            "mosaic_surv_single",
            "mosaic_surv_single_enum",
            "mosaic_surv_multi",
            "mosaic_surv",
            "mosaic_surv_kl",
            "mosaic_surv_noenum",
            "mosaic_surv_detached",
            "mosaic_surv_nojeffreys",
            "mosaic_surv_nodropout",
        ) else False,
        is_survpc=True if args.modality == "survpc" else False,
        is_survpc_f=True if args.modality == "survpc_f" else False,
        type_of_pathway=args.type_of_path,
    )


def _build_poe_stage1_loaders(args, cur):
    studies = _get_poe_stage1_studies(args)
    if len(studies) <= 1:
        return None

    train_splits = []
    val_splits = []
    for study in studies:
        study_args = _clone_args_for_study(args, study)
        study_factory = _build_survival_dataset_factory(study_args)
        study_args.dataset_factory = study_factory
        train_split, val_split, _ = study_factory.return_splits(
            study_args,
            csv_path=f"{study_args.split_dir}/splits_{cur}.csv",
            fold=cur,
        )
        train_splits.append(train_split)
        val_splits.append(val_split)

    if getattr(args, "weighted_sample", False):
        print(f"[TriPoEVAE] stage1 cross-study sampling disables weighted sampling: {', '.join(studies)}")

    train_dataset = ConcatDataset(train_splits)
    val_dataset = ConcatDataset(val_splits)
    train_loader = _get_split_loader(
        args,
        train_dataset,
        training=True,
        testing=False,
        weighted=False,
        batch_size=args.batch_size_stage1,
        disable_cox_batch_override=True,
    )
    val_loader = _get_split_loader(
        args,
        val_dataset,
        training=False,
        testing=False,
        batch_size=1,
        disable_cox_batch_override=True,
    )
    return train_loader, val_loader, studies


def _get_poe_step_metrics(model, total_loss, kl_value, phase, beta=None):
    cached = model.get_cached_outputs()
    recon_losses = cached["recon_losses"]
    obs_logvars = model.reconstruction_loss.logvars
    metrics = {
        "loss/total": float(total_loss),
        "loss/kl_or_jeffreys": float(kl_value),
        "logvar_obs_wsi": obs_logvars["wsi"].detach().item(),
        "logvar_obs_gene": obs_logvars["gene"].detach().item(),
        "logvar_obs_clinic": obs_logvars["clinic"].detach().item(),
        "trainer/phase": phase,
    }
    for name in ("wsi", "gene", "clinic"):
        rec_loss = recon_losses[name].detach()
        if torch.isfinite(rec_loss).all():
            metrics[f"loss/rec_{name}"] = rec_loss.item()
        rec_count = recon_losses.get(f"counts_{name}")
        if rec_count is not None:
            metrics[f"count/rec_{name}"] = int(rec_count)
    if beta is not None:
        metrics["poe/beta"] = float(beta)
    return metrics


@torch.no_grad()
def _set_concat_mean_impute_stats(args, model, train_loader):
    if args.modality != "modality_concat" or args.concat_impute != "mean":
        return

    device = next(model.parameters()).device
    wsi_feats = []
    gene_feats = []
    clinic_feats = []

    was_training = model.training
    model.eval()
    for data in train_loader:
        data_WSI, mask, _, _, _, data_omics, _, data_clinic, avail = _unpack_survtri_poe_vae_batch(device, data)
        avail_mask = torch.stack(
            [
                avail["wsi"].to(device=device, dtype=torch.bool),
                avail["gene"].to(device=device, dtype=torch.bool),
                avail["clinic"].to(device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        x_wsi = model._encode_wsi(data_WSI, wsi_mask=mask)
        x_gene = safe_flatten(data_omics.float())
        x_clinic = model._encode_clinic(data_clinic)

        if avail_mask[:, 0].any():
            wsi_feats.append(x_wsi[avail_mask[:, 0]].detach().cpu())
        if avail_mask[:, 1].any():
            gene_feats.append(x_gene[avail_mask[:, 1]].detach().cpu())
        if avail_mask[:, 2].any():
            clinic_feats.append(x_clinic[avail_mask[:, 2]].detach().cpu())

    stats = {
        "wsi": torch.cat(wsi_feats, dim=0).mean(dim=0) if wsi_feats else torch.zeros_like(model.fill_wsi.cpu()),
        "gene": torch.cat(gene_feats, dim=0).mean(dim=0) if gene_feats else torch.zeros_like(model.fill_gene.cpu()),
        "clinic": torch.cat(clinic_feats, dim=0).mean(dim=0) if clinic_feats else torch.zeros_like(model.fill_clinic.cpu()),
    }
    model.set_impute_stats(stats)
    if was_training:
        model.train()


def _unpack_survtri_poe_vae_batch(device, data):
    data_WSI = data[0].to(device)
    data_omics = data[1].to(device)
    data_clinic = data[2].to(device)
    y_disc, event_time, censor, clinical_data_list = data[3], data[4], data[5], data[6]
    if data[7][0, 0] == 1:
        mask = None
    else:
        mask = data[7].to(device=device, dtype=torch.bool)
    avail = {name: value.to(device=device, dtype=torch.bool) for name, value in data[8].items()}
    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic, avail



def _maybe_apply_eval_subset(args, avail):
    subset = getattr(args, "_eval_subset", None)
    if not subset or avail is None:
        return avail
    return apply_eval_subset(avail, subset)


def _metric_tuple_to_row(fold, subset, metrics):
    test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss = metrics
    return {
        "fold": int(fold),
        "subset": subset,
        "test_cindex": test_cindex,
        "test_cindex_ipcw": test_cindex_ipcw,
        "test_IBS": test_IBS,
        "test_iauc": test_iauc,
        "test_iauc_list": test_iauc_list,
        "test_loss": total_loss,
        "test_BS": test_BS,
    }


def _record_eval_subset_fold(args, cur, subset, metrics, results_dict):
    subset_dir = subset_result_dir(args.results_dir, subset)
    subset_dir.mkdir(parents=True, exist_ok=True)
    _save_pkl(str(subset_dir / f"split_{cur}_results.pkl"), results_dict)
    if not hasattr(args, "_eval_subset_fold_metrics") or args._eval_subset_fold_metrics is None:
        args._eval_subset_fold_metrics = []
    args._eval_subset_fold_metrics.append(_metric_tuple_to_row(cur, subset, metrics))


def _evaluate_requested_subsets(args, cur, model, test_loader, loss_fn, all_survival, default_results, default_metrics):
    subsets = parse_eval_modalities(getattr(args, "eval_modalities", "off"))
    if not subsets:
        return
    for subset in subsets:
        if subset == "PCG":
            results_dict = default_results
            metrics = default_metrics
        else:
            args._eval_subset = subset
            try:
                results_dict, test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss, _attn = _summary(
                    args, args.dataset_factory, model, args.modality, test_loader, loss_fn, all_survival
                )
                metrics = (test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss)
            finally:
                args._eval_subset = None
            print(f"Subset {subset} test c-index: {metrics[0]:.4f}")
        _record_eval_subset_fold(args, cur, subset, metrics, results_dict)


def _build_survtri_poe_vae_avail(args, model, raw_avail):
    selected_modalities = tuple(getattr(args, "selected_modalities", "wsi,gene,clinic").split(","))
    modality_names = ("wsi", "gene", "clinic")
    selected_mask = torch.tensor([name in selected_modalities for name in modality_names], dtype=torch.bool)
    selected_mask = selected_mask.to(device=next(iter(raw_avail.values())).device)

    final_mask = torch.stack([raw_avail[name].to(dtype=torch.bool) for name in modality_names], dim=1)
    final_mask = final_mask & selected_mask.unsqueeze(0)

    return {name: final_mask[:, idx] for idx, name in enumerate(modality_names)}



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
        - test_split :SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    _save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    return train_split, val_split, test_split


def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_diff_surv':
        loss_fn = NLLDiffSurvLoss(beta=args.beta_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    use_model_b_grouped_lr = (
        args.modality in POE_ALL_MODALITIES
        and args.modality not in POE_NEW_B_MODALITIES
        and getattr(args, "poe_variant", None) == "B"
        and hasattr(model, "get_stage2_param_groups")
    )
    if use_model_b_grouped_lr:
        encoder_lr_ratio = float(getattr(args, "poe_encoder_lr_ratio", 0.1))
        trainable_params = model.get_stage2_param_groups(args.lr, encoder_lr_ratio)
        encoder_lr = args.lr * encoder_lr_ratio
        print(
            f"\nInit optimizer ... Model B stage2 grouped LR: "
            f"head={args.lr:g}, encoder={encoder_lr:g} (ratio={encoder_lr_ratio:g})",
            end=' ',
        )
    else:
        print('\nInit optimizer ...', end=' ')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.opt == "adam":
        optimizer = optim.Adam(trainable_params, lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(trainable_params, lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(trainable_params, lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer


def _infer_clinic_shape(clinic_dir, check_n=20):
    """Infer clinic token count and feature dim from serialized tensors."""
    import glob

    paths = sorted(glob.glob(os.path.join(clinic_dir, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"No .pt under {clinic_dir}")

    def _shape_of(path):
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            tensor = tensor.get("features", next(iter(tensor.values())))
        tensor = tensor.float()
        if tensor.dim() == 1:
            return 1, int(tensor.shape[0])
        if tensor.dim() == 2:
            return int(tensor.shape[0]), int(tensor.shape[1])
        raise ValueError(f"Unsupported clinic tensor shape {tuple(tensor.shape)} in {path}")

    ref_shape = _shape_of(paths[0])
    for path in paths[1:check_n]:
        cur_shape = _shape_of(path)
        if cur_shape != ref_shape:
            raise ValueError(f"Clinic shape mismatch: {ref_shape} vs {cur_shape} ({path})")
    return ref_shape

def _init_model(args, current_fold=None):

    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    gene_fm_input_dim = 3072

    clinic_num_tokens = None
    clinic_feat_dim = None
    _PROMPT_CLINIC_MODALITIES = {
        'survpgc_f',
        'survgc_f',
        'survpc_f',
        'modality_concat',
        'mvae_poe',
        'mopoe',
    } | CLINIC_MODALITIES | TRIMODAL_MODALITIES | POE_ALL_MODALITIES
    if args.modality in _PROMPT_CLINIC_MODALITIES:
        clinic_num_tokens, clinic_feat_dim = _infer_clinic_shape(args.clinic_dir)
        print(f"[clinic] dir={args.clinic_dir} tokens={clinic_num_tokens} feat_dim={clinic_feat_dim}")

    if args.modality == "mlp_gene":
        model_dict = {
            "input_dim": omics_input_dim,
            "n_classes": args.n_classes,
            "projection_dim": 64,
            "dropout": args.encoder_dropout,
        }
        model = MLPGene(**model_dict)

    elif args.modality == "snn_gene":
        model_dict = {
            "omic_input_dim": omics_input_dim,
            "n_classes": args.n_classes,
            "model_size_omic": args.single_model_size if args.single_model_size in {"small", "big"} else "small",
        }
        model = SNNGene(**model_dict)

    elif args.modality == "mlp_gene_f":
        model_dict = {
            "input_dim": gene_fm_input_dim,
            "n_classes": args.n_classes,
            "model_size": args.single_model_size if args.single_model_size in {"medium", "big"} else "medium",
            "dropout": args.encoder_dropout,
            "use_input_ln": args.single_use_input_ln,
        }
        model = MLPGeneFM(**model_dict)

    elif args.modality == "snn_gene_f":
        model_dict = {
            "input_dim": gene_fm_input_dim,
            "n_classes": args.n_classes,
            "model_size": args.single_model_size if args.single_model_size in {"medium", "big"} else "medium",
            "use_input_ln": args.single_use_input_ln,
        }
        model = SNNGeneFM(**model_dict)

    elif args.modality in ["abmil_wsi"]:
        model_dict = {
            "device" : args.device, "df_comp": args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion, "path_input_dim": args.encoding_dim
        }
        model = ABMIL(**model_dict)

    # unimodal and multimodal baselines
    elif args.modality == "mlp_wsi":
        model_dict = {"wsi_embedding_dim":args.encoding_dim, "dropout":args.encoder_dropout, "device": args.device}
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi"]:
        model_dict = {"input_dim": args.encoding_dim, 'n_classes': args.n_classes}
        model = TransMIL(**model_dict)

    elif args.modality == "mcat":
        model_dict = {
            'fusion': args.fusion,
            'omic_sizes': args.omic_sizes,
            'n_classes': args.n_classes,
            'path_input_dim': args.encoding_dim,
        }
        model = MCAT_Surv(**model_dict)

    elif args.modality == "porpoise":
        model_dict = {
            'fusion': args.fusion,
            'omic_input_dim': 4999,
            'path_input_dim': args.encoding_dim,
            'n_classes': args.n_classes,
        }
        model = PorpoiseMMF(**model_dict)

    # survpath 
    elif args.modality == "survpath":
        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes, 'wsi_embedding_dim': args.encoding_dim}

        model = SurvPath(**model_dict)

    elif args.modality == 'survpgc_f':
        model_dict = {
            'num_classes': args.n_classes,
            'clinic_embedding_dim': clinic_feat_dim,
            'clinic_num_tokens': clinic_num_tokens,
            'wsi_embedding_dim': args.encoding_dim,
        }
        model = SurvPGC_F(**model_dict)

    elif args.modality == 'survpc':
        model_dict = {
            'num_classes': args.n_classes,
            'wsi_embedding_dim': args.encoding_dim,
        }
        model = SurvPC(**model_dict)

    elif args.modality == "survpc_f":
        model_dict = {
            'num_classes': args.n_classes,
            'clinic_embedding_dim': clinic_feat_dim,
            'clinic_num_tokens': clinic_num_tokens,
            'wsi_embedding_dim': args.encoding_dim,
        }
        model = SurvPC_F(**model_dict)

    elif args.modality == "mlppc_concat":
        model_dict = {
            'num_classes': args.n_classes,
            "clinic_dim": 24,
            'wsi_embedding_dim': args.encoding_dim,
        }
        model = MLPPC_concat(**model_dict)

    elif args.modality == "survgc_f":
        model_dict = {
            'num_classes': args.n_classes,
            'clinic_embedding_dim': clinic_feat_dim,
            'clinic_num_tokens': clinic_num_tokens,
        }
        model = SurvGC_F(**model_dict)

    elif args.modality == 'mlp_clinic_flatten':
        model_dict = {
            'input_dim': clinic_feat_dim,
            'n_classes': args.n_classes,
            'clinic_num_tokens': clinic_num_tokens,
            'dropout': 0.1,
            'pooling': 'flatten',
        }
        model = MLPClinic(**model_dict)

    elif args.modality == 'mlp_clinic_mean':
        model_dict = {
            'input_dim': clinic_feat_dim,
            'n_classes': args.n_classes,
            'clinic_num_tokens': clinic_num_tokens,
            'dropout': 0.1,
            'pooling': 'mean',
        }
        model = MLPClinic(**model_dict)

    elif args.modality == 'snn_clinic_mean':
        model_dict = {
            "input_dim": clinic_feat_dim,
            'n_classes': args.n_classes,
            'clinic_num_tokens': clinic_num_tokens,
            'model_size_omic': args.single_model_size if args.single_model_size in {"small", "big"} else "small",
            'pooling': 'mean',
        }
        model = SNNClinic(**model_dict)

    elif args.modality == 'snn_clinic_flatten':
        model_dict = {
            "input_dim": clinic_feat_dim,
            'n_classes': args.n_classes,
            'clinic_num_tokens': clinic_num_tokens,
            'model_size_omic': args.single_model_size if args.single_model_size in {"small", "big"} else "small",
            'pooling': 'flatten',
        }
        model = SNNClinic(**model_dict)

    elif args.modality == 'clinic_cox':
        model_dict = {'input_dim': clinic_feat_dim, 'clinic_num_tokens': clinic_num_tokens}
        model = CoxClinic(**model_dict)

    elif args.modality == 'survfusion_separate':
        model_dict = {
            'num_classes': args.n_classes,
            'wsi_embedding_dim': args.encoding_dim,
            'clinic_embedding_dim': clinic_feat_dim,
            'training_stage': 'stage1',
            'fusion_type': args.fusion_type,
            'num_heads': args.num_heads,
            'clip_weight_IT': args.clip_weight_IT,
            'clip_weight_IS': args.clip_weight_IS,
            'clip_weight_TS': args.clip_weight_TS,
        }
        model = SurvFusion(**model_dict)

    elif args.modality == 'survfusion_noalign':
        model_dict = {
            'num_classes': args.n_classes,
            'wsi_embedding_dim': args.encoding_dim,
            'clinic_embedding_dim': clinic_feat_dim,
            'num_heads': args.num_heads,
        }
        model = SurvFusionNoAlign(**model_dict)

    elif args.modality == 'survfusion_joint':
        model_dict = {
            'num_classes': args.n_classes,
            'wsi_embedding_dim': args.encoding_dim,
            'clinic_embedding_dim': clinic_feat_dim,
            'num_heads': args.num_heads,
        }
        model = SurvFusionJoint(**model_dict)

    elif args.modality == 'survtri_snn_concat':
        model_dict = {
            'study': args.study,
            'current_fold': current_fold,
            'num_classes': args.n_classes,
            'wsi_embedding_dim': args.encoding_dim,
            'clinic_num_tokens': clinic_num_tokens,
            'single_model_size': args.single_model_size,
            'dropout': args.encoder_dropout,
            'selected_modalities': args.selected_modalities,
        }
        model = SurvTriSNNConcat(**model_dict)

    elif args.modality == 'survtri_snn_mhsa':
        model_dict = {
            'study': args.study,
            'current_fold': current_fold,
            'num_classes': args.n_classes,
            'wsi_embedding_dim': args.encoding_dim,
            'clinic_num_tokens': clinic_num_tokens,
            'single_model_size': args.single_model_size,
            'dropout': args.encoder_dropout,
            'num_heads': args.num_heads,
            'selected_modalities': args.selected_modalities,
        }
        model = SurvTriSNNMHSA(**model_dict)

    elif args.modality == 'survtri_mlp_concat':
        model_dict = {
            'study': args.study,
            'current_fold': current_fold,
            'num_classes': args.n_classes,
            'wsi_embedding_dim': args.encoding_dim,
            'clinic_num_tokens': clinic_num_tokens,
            'single_model_size': args.single_model_size,
            'dropout': args.encoder_dropout,
            'selected_modalities': args.selected_modalities,
        }
        model = SurvTriMLPConcat(**model_dict)

    elif args.modality == 'survtri_mlp_mhsa':
        model_dict = {
            'study': args.study,
            'current_fold': current_fold,
            'num_classes': args.n_classes,
            'wsi_embedding_dim': args.encoding_dim,
            'clinic_num_tokens': clinic_num_tokens,
            'single_model_size': args.single_model_size,
            'dropout': args.encoder_dropout,
            'num_heads': args.num_heads,
            'selected_modalities': args.selected_modalities,
        }
        model = SurvTriMLPMHSA(**model_dict)

    elif args.modality == 'survtri_poe_vae':
        model_dict = {
            'clinic_num_tokens': clinic_num_tokens,
            'clinic_embedding_dim': clinic_feat_dim,
            'wsi_embedding_dim': args.encoding_dim,
            'latent_dim': 128,
            'mmhid': args.poe_mmhid,
            'label_dim': args.label_dim,
            'decoder_hidden_dim': args.poe_decoder_hidden_dim,
            'poe_variant': args.poe_variant,
            'poe_surv_lambda': args.poe_surv_lambda,
            'modality_dropout_prob': args.poe_modality_dropout,
            'transformer_dropout': args.encoder_dropout,
            'transformer_layers': args.poe_transformer_layers,
            'selected_modalities': args.selected_modalities,
            'alphafix': bool(getattr(args, 'alphafix', False)),
            'alphapgc': getattr(args, 'alphapgc', None),
        }
        model = SurvTriPoEVAE(**model_dict)

    elif args.modality == 'survtri_poe_vae_b_kl':
        args.poe_variant = "B"
        model_dict = {
            'clinic_num_tokens': clinic_num_tokens,
            'clinic_embedding_dim': clinic_feat_dim,
            'wsi_embedding_dim': args.encoding_dim,
            'latent_dim': 128,
            'mmhid': args.poe_mmhid,
            'label_dim': args.label_dim,
            'decoder_hidden_dim': args.poe_decoder_hidden_dim,
            'poe_variant': 'B',
            'poe_surv_lambda': args.poe_surv_lambda,
            'modality_dropout_prob': args.poe_modality_dropout,
            'transformer_dropout': args.encoder_dropout,
            'transformer_layers': args.poe_transformer_layers,
            'selected_modalities': args.selected_modalities,
            'alphafix': bool(getattr(args, 'alphafix', False)),
            'alphapgc': getattr(args, 'alphapgc', None),
        }
        model = SurvTriPoEVAE_BKL(**model_dict)

    elif args.modality == 'survtri_poe_vae_b_crossstage1':
        args.poe_variant = "B"
        model_dict = {
            'clinic_num_tokens': clinic_num_tokens,
            'clinic_embedding_dim': clinic_feat_dim,
            'wsi_embedding_dim': args.encoding_dim,
            'latent_dim': 128,
            'mmhid': args.poe_mmhid,
            'label_dim': args.label_dim,
            'decoder_hidden_dim': args.poe_decoder_hidden_dim,
            'poe_variant': 'B',
            'poe_surv_lambda': args.poe_surv_lambda,
            'modality_dropout_prob': args.poe_modality_dropout,
            'transformer_dropout': args.encoder_dropout,
            'transformer_layers': args.poe_transformer_layers,
            'selected_modalities': args.selected_modalities,
            'alphafix': bool(getattr(args, 'alphafix', False)),
            'alphapgc': getattr(args, 'alphapgc', None),
            'poe_stage1_studies': getattr(args, "poe_stage1_studies", ""),
        }
        model = SurvTriPoEVAE_BCrossStage1(**model_dict)

    elif args.modality == 'survtri_poe_vae_b_nopretrain':
        args.poe_variant = "B"
        model_dict = {
            'clinic_num_tokens': clinic_num_tokens,
            'clinic_embedding_dim': clinic_feat_dim,
            'wsi_embedding_dim': args.encoding_dim,
            'latent_dim': 128,
            'mmhid': args.poe_mmhid,
            'label_dim': args.label_dim,
            'decoder_hidden_dim': args.poe_decoder_hidden_dim,
            'poe_variant': 'B',
            'poe_surv_lambda': args.poe_surv_lambda,
            'modality_dropout_prob': args.poe_modality_dropout,
            'transformer_dropout': args.encoder_dropout,
            'transformer_layers': args.poe_transformer_layers,
            'selected_modalities': args.selected_modalities,
            'alphafix': bool(getattr(args, 'alphafix', False)),
            'alphapgc': getattr(args, 'alphapgc', None),
        }
        model = SurvTriPoEVAE_BNoPretrain(**model_dict)

    elif args.modality in {
        "survtri_poe_vae_b_single",
        "survtri_poe_vae_b_multi",
        "mosaic_surv_twostage",
        "mosaic_surv_frozen",
        "mosaic_surv_single",
        "mosaic_surv_single_enum",
        "mosaic_surv_multi",
        "mosaic_surv",
        "mosaic_surv_kl",
        "mosaic_surv_noenum",
        "mosaic_surv_detached",
        "mosaic_surv_nojeffreys",
    }:
        if args.modality in POE_NEW_A_MODALITIES:
            args.poe_variant = "A"
        elif args.modality in POE_NEW_B_MODALITIES:
            args.poe_variant = "B"
        else:
            args.poe_variant = "C"
        model_dict = {
            "clinic_num_tokens": clinic_num_tokens,
            "clinic_embedding_dim": clinic_feat_dim,
            "wsi_embedding_dim": args.encoding_dim,
            "latent_dim": 128,
            "mmhid": args.poe_mmhid,
            "label_dim": args.label_dim,
            "decoder_hidden_dim": args.poe_decoder_hidden_dim,
            "poe_variant": args.poe_variant,
            "poe_surv_lambda": args.poe_surv_lambda,
            "modality_dropout_prob": args.poe_modality_dropout,
            "transformer_dropout": args.encoder_dropout,
            "transformer_layers": args.poe_transformer_layers,
            "selected_modalities": args.selected_modalities,
            "alphafix": bool(getattr(args, "alphafix", False)),
            "alphapgc": getattr(args, "alphapgc", None),
        }
        model_cls = {
            "survtri_poe_vae_b_single": SurvTriPoEVAE_BSingle,
            "survtri_poe_vae_b_multi": SurvTriPoEVAE_BMulti,
            "mosaic_surv_twostage": MosaicSurvTwostage,
            "mosaic_surv_frozen": MosaicSurvFrozen,
            "mosaic_surv_single": MosaicSurvSingle,
            "mosaic_surv_single_enum": MosaicSurvSingleEnum,
            "mosaic_surv_multi": MosaicSurvMulti,
            "mosaic_surv": MosaicSurv,
            "mosaic_surv_kl": MosaicSurvKL,
            "mosaic_surv_noenum": MosaicSurvNoEnum,
            "mosaic_surv_detached": MosaicSurvDetached,
            "mosaic_surv_nojeffreys": MosaicSurv,
            "mosaic_surv_nodropout": MosaicSurv,
        }[args.modality]
        model = model_cls(**model_dict)

    elif args.modality == 'modality_concat':
        model_dict = {
            'wsi_embedding_dim': args.encoding_dim,
            'gene_embedding_dim': gene_fm_input_dim,
            'clinic_embedding_dim': clinic_feat_dim,
            'clinic_num_tokens': clinic_num_tokens,
            'd_z': 128,
            'mmhid': args.poe_mmhid,
            'dropout': args.encoder_dropout,
            'concat_wsi': args.concat_wsi,
            'concat_impute': args.concat_impute,
            'label_dim': args.label_dim,
        }
        model = ConcatMissingModalityBaseline(**model_dict)

    elif args.modality == 'mvae_poe':
        model_dict = {
            'wsi_embedding_dim': args.encoding_dim,
            'gene_embedding_dim': 768,
            'clinic_embedding_dim': clinic_feat_dim,
            'gene_num_tokens': 4,
            'clinic_num_tokens': clinic_num_tokens,
            'latent_dim': 128,
            'mmhid': args.poe_mmhid,
            'decoder_hidden_dim': args.poe_decoder_hidden_dim,
            'dropout': args.encoder_dropout,
            'use_subsampling': True,
            'k_subsets': 2,
            'beta': 1.0,
            'warmup': 10,
            'lambda_surv': 1.0,
        }
        model = MVAEBaseline(**model_dict)

    elif args.modality == 'mopoe':
        model_dict = {
            'wsi_embedding_dim': args.encoding_dim,
            'gene_embedding_dim': 768,
            'clinic_embedding_dim': clinic_feat_dim,
            'gene_num_tokens': 4,
            'clinic_num_tokens': clinic_num_tokens,
            'latent_dim': 128,
            'mmhid': args.poe_mmhid,
            'decoder_hidden_dim': args.poe_decoder_hidden_dim,
            'dropout': args.encoder_dropout,
            'beta': 1.0,
            'lambda_surv': 1.0,
        }
        model = MoPoEBaseline(**model_dict)

    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split, test_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    disable_cox_batch_override = args.modality in (POE_ALL_MODALITIES | MISSING_MODALITY_BASELINES)
    if train_split:
        train_loader = _get_split_loader(
            args,
            train_split,
            training=True,
            testing=False,
            weighted=args.weighted_sample,
            batch_size=args.batch_size,
            disable_cox_batch_override=disable_cox_batch_override,
        )
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(
            args,
            val_split,
            testing=False,
            batch_size=1,
            disable_cox_batch_override=disable_cox_batch_override,
        )
    else:
        val_loader = None

    if test_split:
        test_loader = _get_split_loader(
            args,
            test_split,
            testing=False,
            batch_size=1,
            disable_cox_batch_override=disable_cox_batch_override,
        )
    else:
        test_loader = None
    print('Done!')

    return train_loader, val_loader, test_loader

def _extract_survival_metadata(train_loader, val_loader, test_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
        - test_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy(),
        test_loader.dataset.metadata[test_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy(),
        test_loader.dataset.metadata[test_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _collect_train_survival_risks(args, model, train_loader, loss_fn):
    r"""
    Run the model over the training fold to collect (times, events, risks),
    used as the Breslow baseline cohort for cox_surv test metrics.
    """
    train_results, _, _, _, _, _, _, _, _ = _summary(
        args, args.dataset_factory, model, args.modality, train_loader, loss_fn, None
    )
    patient_ids = list(train_results)
    times = np.array([train_results[i]['time'] for i in patient_ids], dtype=np.float64)
    censorships = np.array([train_results[i]['censorship'] for i in patient_ids], dtype=np.float64)
    risks = np.array([train_results[i]['risk'] for i in patient_ids], dtype=np.float64)
    return times, (1.0 - censorships) > 0.5, risks

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - data_clinic : torch.Tensor
        - clinical_data_list : list
    
    """
    
    if modality in CSV_GENE_MODALITIES | GENE_F_MODALITIES:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        data_clinic = None
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in CLINIC_MODALITIES:
        data_WSI = data[0]
        mask = None
        data_clinic = data[1].to(device)
        data_omics = None
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["abmil_wsi", "mlp_wsi", "transmil_wsi"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        data_clinic = None
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["mcat"]:
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]
        data_clinic = None

        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality in ["porpoise"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        data_clinic = None
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]


    elif modality in ["survpath"]:
        data_WSI = data[0].to(device)
        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        data_clinic = None
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in [
        "survpgc_f",
        "survfusion_separate",
        "survfusion_noalign",
        "survfusion_joint",
        "survtri_snn_concat",
        "survtri_snn_mhsa",
        "survtri_mlp_concat",
        "survtri_mlp_mhsa",
        "survtri_poe_vae",
        "modality_concat",
        "mvae_poe",
        "mopoe",
    ]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        data_clinic = data[2].to(device)
        if data[7][0, 0] == 1:
            mask = None
        else:
            mask = data[7].to(device)

        y_disc, event_time, censor, clinical_data_list = data[3], data[4], data[5], data[6]

    elif modality in ["survpc", "survpc_f", "mlppc_concat"]:
        data_WSI = data[0].to(device)
        data_clinic = data[1].to(device)
        data_omics = None
        if data[6][0, 0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]


    elif modality in ["survgc_f"]:
        data_WSI = data[0].to(device)
        data_clinic = data[1].to(device)
        data_omics = data[2].to(device)
        mask = None

        y_disc, event_time, censor, clinical_data_list = data[3], data[4], data[5], data[6]

    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic

def _process_data_and_forward(args, model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    if modality in POE_ALL_MODALITIES | MISSING_MODALITY_BASELINES:
        data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic, avail = \
            _unpack_survtri_poe_vae_batch(device, data)
        if modality in POE_ALL_MODALITIES:
            avail = _build_survtri_poe_vae_avail(args, model, avail)
    else:
        data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic = _unpack_data(modality, device, data)
        avail = None

    if modality in ["mcat"]:
        out = model(
            x_path=data_WSI,
            x_omic1=data_omics[0],
            x_omic2=data_omics[1],
            x_omic3=data_omics[2],
            x_omic4=data_omics[3],
            x_omic5=data_omics[4],
            x_omic6=data_omics[5],
            return_attn=False
            )

    elif modality in ["porpoise"]:
        input_args = {"x_path": data_WSI.to(device)}
        input_args["x_omic"] = data_omics.to(device)
        input_args["return_attn"] = False
        out = model(**input_args)

    elif modality == 'survpath':
        input_args = {"x_path": data_WSI.to(device)}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        out = model(**input_args)

    elif modality == 'survpgc_f' and args.bag_loss == 'nll_surv':
        input_args = {"x_path": data_WSI.to(device)}
        input_args["x_omic"] = data_omics.to(device)
        input_args["x_clinic"] = data_clinic.to(device)
        input_args["return_attn"] = False
        input_args["bag_loss"] = args.bag_loss
        out = model(**input_args)

    elif modality == 'survpgc_f' and args.bag_loss == 'nll_diff_surv':
        input_args = {"x_path": data_WSI.to(device)}
        input_args["x_omic"] = data_omics.to(device)
        input_args["x_clinic"] = data_clinic.to(device)
        input_args["return_attn"] = False
        input_args["bag_loss"] = args.bag_loss
        tensor1, tensor2, out = model(**input_args)

    elif modality in  ["survpc", "survpc_f", "mlppc_concat"]:
        input_args = {"x_path": data_WSI.to(device)}
        input_args["x_clinic"] = data_clinic.to(device)
        input_args["return_attn"] = False
        out = model(**input_args)

    elif modality in ["survgc_f"]:
        input_args = {"x_path": data_WSI.to(device)}
        input_args["x_clinic"] = data_clinic.to(device)
        input_args["x_omic"] = data_omics.to(device)
        input_args["return_attn"] = False
        out = model(**input_args)

    elif modality == "survfusion_separate":
        out, _ = model(
            x_path=data_WSI.to(device),
            x_omic=data_omics.to(device),
            x_clinic=data_clinic.to(device),
            return_align_loss=False,
        )

    elif modality == "survfusion_noalign":
        out = model(
            x_path=data_WSI.to(device),
            x_omic=data_omics.to(device),
            x_clinic=data_clinic.to(device),
        )

    elif modality == "survfusion_joint":
        # 推理时只取 logits，丢弃 alignment_loss
        out, _ = model(
            x_path=data_WSI.to(device),
            x_omic=data_omics.to(device),
            x_clinic=data_clinic.to(device),
        )

    elif modality in TRIMODAL_MODALITIES:
        out = model(
            x_path=data_WSI.to(device),
            x_omic=data_omics.to(device),
            x_clinic=data_clinic.to(device),
        )

    elif modality in POE_ALL_MODALITIES:
        out = model(
            x_path=data_WSI.to(device),
            x_omic=data_omics.to(device),
            x_clinic=data_clinic.to(device),
            wsi_mask=mask,
            avail=avail,
        )

    elif modality == "modality_concat":
        out = model(
            x_path=data_WSI.to(device),
            x_omic=data_omics.to(device),
            x_clinic=data_clinic.to(device),
            wsi_mask=mask,
            avail=avail,
        )

    elif modality in VAE_MISSING_MODALITY_BASELINES:
        out_struct = model(
            x_path=data_WSI.to(device),
            x_omic=data_omics.to(device),
            x_clinic=data_clinic.to(device),
            wsi_mask=mask,
            avail=avail,
            event_time=event_time,
            censor=censor,
            epoch=1,
            batch_ratio=0.0,
        )
        out = out_struct.risk

    elif modality in CLINIC_MODALITIES:
        input_args = {"x_path": data_WSI.to(device)}
        input_args["x_clinic"] = data_clinic.to(device)
        input_args["return_attn"] = False
        out = model(**input_args)

    else:
        out = model(
            data_omics=data_omics,
            data_WSI=data_WSI,
            mask=mask
        )
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)

    if modality == 'survpgc_f' and args.bag_loss == 'nll_diff_surv':
        return tensor1, tensor2, out, y_disc, event_time, censor, clinical_data_list

    else:
        return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h, bag_loss='nll_surv'):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    if bag_loss == 'cox_surv':
        risk = _tensor_to_numpy_safe(h.reshape(-1), dtype=np.float64)
        return risk, None

    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = _tensor_to_numpy_safe(-torch.sum(survival, dim=1), dtype=np.float64)
    return risk, _tensor_to_numpy_safe(survival, dtype=np.float64)

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(_tensor_to_numpy_safe(censor, dtype=np.float64))
    all_event_times.append(_tensor_to_numpy_safe(event_time, dtype=np.float64))
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data


def _get_poe_beta(args, epoch):
    beta_target = float(args.poe_beta_target)
    if getattr(args, 'betafix', False):
        return beta_target
    warmup_epochs = max(0, int(args.warmup_epochs))
    if warmup_epochs <= 1:
        return beta_target
    progress = min(1.0, epoch / float(warmup_epochs - 1))
    return beta_target * progress

def _train_loop_survival(args, epoch, model, modality, loader, optimizer, scheduler, loss_fn):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    beta = _get_poe_beta(args, epoch) if modality in POE_MODALITIES else None
    poe_epoch_monitor = _init_poe_epoch_monitor() if modality in POE_ALL_MODALITIES else None
    surv_loss_total = 0.0
    surv_count = 0
    cox_skip_count = 0

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        if modality in VAE_MISSING_MODALITY_BASELINES:
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic, avail = \
                _unpack_survtri_poe_vae_batch(device, data)
            out_struct = model(
                x_path=data_WSI.to(device),
                x_omic=data_omics.to(device),
                x_clinic=data_clinic.to(device),
                wsi_mask=mask,
                avail=avail,
                event_time=event_time,
                censor=censor,
                epoch=epoch + 1,
                batch_ratio=float(batch_idx) / max(1, len(loader)),
            )
            h = out_struct.risk
            loss = out_struct.loss
            surv_loss_total += out_struct.surv_loss.item() * event_time.shape[0]
            surv_count += event_time.shape[0]
            loss_value = loss.item()
            risk, _ = _calculate_risk(h, 'cox_surv')
            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
                all_risk_scores, all_censorships, all_event_times, all_clinical_data,
                event_time, censor, risk, clinical_data_list
            )
            total_loss += loss_value * event_time.shape[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (batch_idx % 20) == 0:
                print(
                    "batch: {}, loss: {:.3f}, vae_loss: {:.3f}, surv_loss: {:.3f}".format(
                        batch_idx,
                        loss.item(),
                        out_struct.vae_loss.item(),
                        out_struct.surv_loss.item(),
                    )
                )
            continue

        # survfusion_joint: 联合训练，需同时拿到 alignment_loss
        if modality == 'survfusion_joint' and args.bag_loss == 'nll_surv':
            data_WSI, _, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic = \
                _unpack_data(modality, device, data)
            h, align_loss = model(x_path=data_WSI, x_omic=data_omics, x_clinic=data_clinic)
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            surv_loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = surv_loss.item()
            loss = surv_loss / y_disc.shape[0] + args.clip_lambda * align_loss
        elif args.bag_loss == 'nll_surv':
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(args, model, modality, device, data)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
        elif args.bag_loss == 'nll_diff_surv':
            tensor1, tensor2, h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(args, model, modality, device, data)
            loss = loss_fn(tensor1, tensor2, h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
        elif args.bag_loss == 'cox_surv':
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(args, model, modality, device, data)
            if modality in POE_ALL_MODALITIES and hasattr(model, "survival_loss_from_cached"):
                survival_loss = model.survival_loss_from_cached(event_time, censor, loss_fn)
                h = model.get_cached_outputs()["risk"]
            else:
                survival_loss = loss_fn(h=h, t=event_time, c=censor)
            surv_loss_total += survival_loss.item() * event_time.shape[0]
            surv_count += event_time.shape[0]
            if modality in POE_ALL_MODALITIES:
                loss = model.combine_loss(survival_loss, beta=beta)
            else:
                loss = survival_loss
            loss_value = loss.item()
        else:
            raise ValueError('Unsupported loss function:', args.bag_loss)
        
        risk, _ = _calculate_risk(h, args.bag_loss)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        if args.bag_loss == 'cox_surv':
            total_loss += loss_value * event_time.shape[0]
        else:
            total_loss += loss_value 

        loss.backward()
        optimizer.step()
        scheduler.step()

        if modality in ["survfusion_noalign", "survfusion_joint"]:
            processed = min(batch_idx * loader.batch_size, len(loader.dataset))
            print("batch: {}, loss: {:.3f}".format(processed, loss.item()))
        elif modality in POE_ALL_MODALITIES:
            cached = model.get_cached_outputs()
            pattern_surv = cached.get("pattern_surv") if isinstance(cached, dict) else None
            if isinstance(pattern_surv, dict) and int(pattern_surv.get("n_patterns", 1) or 0) == 0:
                cox_skip_count += 1
            _update_poe_epoch_monitor(poe_epoch_monitor, cached)
            _wandb_log(
                args,
                _get_poe_step_metrics(
                    model,
                    total_loss=loss.item(),
                    kl_value=cached["jeffreys"].item(),
                    phase=f"{args.poe_variant}_stage2",
                    beta=beta,
                ),
            )
            print(
                "batch: {}, loss: {:.3f}, recon: {:.3f}, jeffreys: {:.3f}".format(
                    batch_idx,
                    loss.item(),
                    cached["recon_total"].item(),
                    cached["jeffreys"].item(),
                )
            )
        elif (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except ValueError as exc:
        print(f"Warning: train c-index unavailable ({exc}); set to 0.")
        c_index = 0.

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    if modality in POE_ALL_MODALITIES:
        epoch_metrics = _finalize_poe_epoch_monitor(poe_epoch_monitor)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["train/c_index"] = c_index
        if surv_count > 0:
            epoch_metrics["loss/surv"] = surv_loss_total / surv_count
        epoch_metrics["trainer/phase"] = f"{args.poe_variant}_stage2"
        _wandb_log(args, epoch_metrics)

    return c_index, total_loss, cox_skip_count

def _calculate_metrics(args, loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores, train_survival_risks=None):
    r"""
    Calculate various survival metrics

    Metrics now follow the shared conventions in utils/survival_metrics.py:
      - IBS integrated over the monthly grid [1..60], only for folds whose
        test follow-up covers the grid
      - time-dependent (cumulative/dynamic) AUC at the landmarks 24 / 60
        months, only for folds whose test follow-up covers the landmark and
        with at least MIN_EVENTS_FOR_AUC events by the landmark
      - discrete models use their survival columns interpolated between bin
        edges; cox models use the Breslow baseline hazard estimated on the
        training fold (train_survival_risks)

    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array (censoring cohort, see _extract_survival_metadata)
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        - train_survival_risks : (times, events, risks) tuple of the training
          fold, required for cox_surv models; None leaves those metrics as 0.

    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float (nan if the fold cannot support the grid)
        - iauc : Float (mean over the available landmark AUCs)
        - iauc_list : List (per-landmark AUCs, [auc24, auc60], nan if unavailable)

    """

    bins_original = dataset_factory.bins
    edges = np.asarray(bins_original[1:], dtype=np.float64)  # [q1, q2, q3, tmax+eps]

    #---> delete the nans and corresponding elements from other arrays
    original_risk_scores = all_risk_scores
    nan_mask = np.argwhere(np.isnan(original_risk_scores))
    all_risk_scores = np.delete(all_risk_scores, nan_mask)
    all_censorships = np.delete(all_censorships, nan_mask)
    all_event_times = np.delete(all_event_times, nan_mask)
    if all_risk_by_bin_scores is not None:
        all_risk_by_bin_scores = np.delete(all_risk_by_bin_scores, nan_mask, axis=0)
    #<---

    #---> drop negative observed times (data quirk, a few BRCA rows) from the
    # IPCW-weighted metric path; the c-index above already used the full set
    keep = all_event_times >= 0.0
    if not keep.all():
        all_risk_scores = all_risk_scores[keep]
        all_censorships = all_censorships[keep]
        all_event_times = all_event_times[keep]
        if all_risk_by_bin_scores is not None:
            all_risk_by_bin_scores = all_risk_by_bin_scores[keep]
        if survival_train is not None:
            cohort_keep = survival_train["time"] >= 0.0
            survival_train = Surv.from_arrays(
                event=survival_train["event"][cohort_keep],
                time=survival_train["time"][cohort_keep],
            )
    #<---

    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except ValueError as exc:
        print(f'Warning: c-index unavailable ({exc}); set to 0.')
        c_index = 0.
    c_index_ipcw, BS, IBS, iauc, iauc_list = 0., 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except Exception as exc:
        print(f"Problem converting survival test datatype, so all metrics 0. ({exc})")
        return c_index, c_index_ipcw, BS, IBS, iauc, iauc_list

    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except Exception as exc:
        print(f'An error occured while computing c-index ipcw: {exc}')
        c_index_ipcw = 0.

    if survival_train is None:
        return c_index, c_index_ipcw, BS, IBS, iauc, iauc_list

    # survival estimates at the shared grid / landmarks
    estimate_grid = None
    if all_risk_by_bin_scores is not None:
        # discrete models: risk_by_bin columns are S(t) at the bin edges
        estimate_grid = interpolate_survival(all_risk_by_bin_scores, edges, IBS_GRID_MONTHS)
        estimate_landmarks = 1.0 - interpolate_survival(all_risk_by_bin_scores, edges, np.asarray(AUC_LANDMARK_MONTHS, dtype=np.float64))
    elif train_survival_risks is not None:
        train_times, train_events, train_risks = train_survival_risks
        estimate_grid = breslow_survival(train_times, train_events, train_risks, all_risk_scores, IBS_GRID_MONTHS)
        estimate_landmarks = 1.0 - breslow_survival(train_times, train_events, train_risks, all_risk_scores, np.asarray(AUC_LANDMARK_MONTHS, dtype=np.float64))

    if estimate_grid is None:
        return c_index, c_index_ipcw, BS, IBS, iauc, iauc_list

    test_min = float(all_event_times.min())
    test_max = float(all_event_times.max())

    # brier score / IBS over the monthly grid (only when the fold supports it)
    if test_min <= 1.0 and test_max >= 60.0:
        try:
            _, BS = brier_score(survival_train, survival_test, estimate=estimate_grid, times=IBS_GRID_MONTHS)
            IBS = compute_ibs(survival_train, survival_test, estimate_grid, IBS_GRID_MONTHS)
            if not np.isfinite(IBS):
                IBS = float('nan')
        except Exception as exc:
            print(f'An error occured while computing BS/IBS: {exc}')
            BS = 0.
            IBS = 0.
    else:
        BS, IBS = 0., 0.

    # landmark AUCs (risk direction: 1 - S)
    available_landmarks = []
    for landmark in AUC_LANDMARK_MONTHS:
        n_events = int(
            (((1.0 - all_censorships) > 0.5) & (all_event_times <= landmark)).sum()
        )
        if (
            test_min <= landmark <= test_max
            and n_events >= MIN_EVENTS_FOR_AUC.get(landmark, 5)
        ):
            available_landmarks.append(landmark)

    iauc_list = [float('nan')] * len(AUC_LANDMARK_MONTHS)
    iauc = 0.
    if available_landmarks:
        try:
            estimate = estimate_landmarks[:, [AUC_LANDMARK_MONTHS.index(lm) for lm in available_landmarks]]
            aucs = compute_landmark_aucs(survival_train, survival_test, estimate, available_landmarks)
            for landmark, value in zip(available_landmarks, aucs):
                iauc_list[AUC_LANDMARK_MONTHS.index(landmark)] = value
            finite = [value for value in iauc_list if np.isfinite(value)]
            iauc = float(np.mean(finite)) if finite else 0.
        except Exception as exc:
            print(f'An error occured while computing iauc: {exc}')
            iauc = 0.
            iauc_list = [0.] * len(AUC_LANDMARK_MONTHS)

    return c_index, c_index_ipcw, BS, IBS, iauc, iauc_list

def _summary(args, dataset_factory, model, modality, loader, loss_fn, survival_train=None, train_survival_risks=None):
    r"""
    Run a validation loop on the trained model

    Args:
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
        - train_survival_risks : (times, events, risks) tuple of the training
          fold for cox_surv Breslow baseline; None -> cox IBS/AUC stay 0.
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:
            if modality in VAE_MISSING_MODALITY_BASELINES:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic, avail = \
                    _unpack_survtri_poe_vae_batch(device, data)
                avail = _maybe_apply_eval_subset(args, avail)
                out_struct = model(
                    x_path=data_WSI.to(device),
                    x_omic=data_omics.to(device),
                    x_clinic=data_clinic.to(device),
                    wsi_mask=mask,
                    avail=avail,
                    event_time=event_time,
                    censor=censor,
                    epoch=1,
                    batch_ratio=0.0,
                )
                h = out_struct.risk
                loss = out_struct.loss
                loss_value = loss.item()
                total_loss += loss_value

                risk, survival = _calculate_risk(h, 'cox_surv')
                all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
                    all_risk_scores, all_censorships, all_event_times, all_clinical_data,
                    event_time, censor, risk, clinical_data_list
                )
                all_logits.append(_tensor_to_numpy_safe(h, dtype=np.float64))
                all_slide_ids.append(slide_ids.iloc[count])
                count += 1
                continue

            if modality in POE_ALL_MODALITIES | MISSING_MODALITY_BASELINES:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic, avail = \
                    _unpack_survtri_poe_vae_batch(device, data)
                avail = _maybe_apply_eval_subset(args, avail)
                if modality in POE_ALL_MODALITIES:
                    avail = _build_survtri_poe_vae_avail(args, model, avail)
            else:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic = _unpack_data(modality, device, data)
                avail = None

            if modality in ["mcat"]:
                h = model(
                    x_path=data_WSI,
                    x_omic1=data_omics[0],
                    x_omic2=data_omics[1],
                    x_omic3=data_omics[2],
                    x_omic4=data_omics[3],
                    x_omic5=data_omics[4],
                    x_omic6=data_omics[5],
                    return_attn=args.return_attn
                )

            elif modality in ["porpoise"]:
                input_args = {"x_path": data_WSI.to(device)}
                input_args["x_omic"] = data_omics.to(device)
                input_args["return_attn"] = False
                h = model(**input_args)

            elif modality == "survpath":
                input_args = {"x_path": data_WSI.to(device)}
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                h = model(**input_args)

            elif modality == "survpgc_f" and args.bag_loss == 'nll_surv':
                input_args = {"x_path": data_WSI.to(device)}
                input_args["x_omic"] = data_omics.to(device)
                input_args["x_clinic"] = data_clinic.to(device)
                input_args["return_attn"] = args.return_attn
                input_args["bag_loss"] = args.bag_loss
                h = model(**input_args)

            elif modality == "survpgc_f" and args.bag_loss == 'nll_diff_surv':
                input_args = {"x_path": data_WSI.to(device)}
                input_args["x_omic"] = data_omics.to(device)
                input_args["x_clinic"] = data_clinic.to(device)
                input_args["return_attn"] = args.return_attn
                input_args["bag_loss"] = args.bag_loss
                tensor1, tensor2, h = model(**input_args)

            elif modality in ["survpc", "survpc_f", "mlppc_concat"]:
                input_args = {"x_path": data_WSI.to(device)}
                input_args["x_clinic"] = data_clinic.to(device)
                input_args["return_attn"] = args.return_attn
                h = model(**input_args)

            elif modality in ["survgc_f"]:
                input_args = {"x_path": data_WSI.to(device)}
                input_args["x_clinic"] = data_clinic.to(device)
                input_args["x_omic"] = data_omics.to(device)
                input_args["return_attn"] = args.return_attn
                h = model(**input_args)

            elif modality == "survfusion_separate":
                h, _ = model(
                    x_path=data_WSI.to(device),
                    x_omic=data_omics.to(device),
                    x_clinic=data_clinic.to(device),
                    return_align_loss=False,
                )

            elif modality == "survfusion_noalign":
                h = model(
                    x_path=data_WSI.to(device),
                    x_omic=data_omics.to(device),
                    x_clinic=data_clinic.to(device),
                )

            elif modality == "survfusion_joint":
                h, _ = model(
                    x_path=data_WSI.to(device),
                    x_omic=data_omics.to(device),
                    x_clinic=data_clinic.to(device),
                )

            elif modality in TRIMODAL_MODALITIES:
                h = model(
                    x_path=data_WSI.to(device),
                    x_omic=data_omics.to(device),
                    x_clinic=data_clinic.to(device),
                )

            elif modality in POE_ALL_MODALITIES:
                h = model(
                    x_path=data_WSI.to(device),
                    x_omic=data_omics.to(device),
                    x_clinic=data_clinic.to(device),
                    wsi_mask=mask,
                    avail=avail,
                )

            elif modality == "modality_concat":
                h = model(
                    x_path=data_WSI.to(device),
                    x_omic=data_omics.to(device),
                    x_clinic=data_clinic.to(device),
                    wsi_mask=mask,
                    avail=avail,
                )

            elif modality in VAE_MISSING_MODALITY_BASELINES:
                out_struct = model(
                    x_path=data_WSI.to(device),
                    x_omic=data_omics.to(device),
                    x_clinic=data_clinic.to(device),
                    wsi_mask=mask,
                    avail=avail,
                    event_time=event_time,
                    censor=censor,
                    epoch=1,
                    batch_ratio=0.0,
                )
                h = out_struct.risk

            elif modality in ['mlp_clinic_mean', 'mlp_clinic_flatten', 'snn_clinic_mean', 'snn_clinic_flatten', 'clinic_cox']:
                input_args = {"x_path": data_WSI.to(device)}
                input_args['x_clinic'] = data_clinic.to(device)
                input_args["return_attn"] = args.return_attn
                h = model(**input_args)

            else:
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )

            if len(h.shape) == 1:
                h = h.unsqueeze(0)

            if args.bag_loss == 'nll_surv':
                loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            elif args.bag_loss == 'nll_diff_surv':
                loss = loss_fn(input1=tensor1, input2=tensor2, h=h, y=y_disc, t=event_time, c=censor)
            elif args.bag_loss == 'cox_surv':
                survival_loss = loss_fn(h=h, t=event_time, c=censor)
                if modality in POE_ALL_MODALITIES:
                    loss = model.combine_loss(survival_loss, beta=args.poe_beta_target)
                else:
                    loss = survival_loss
            loss_value = loss.item()
            if args.bag_loss != 'cox_surv':
                loss = loss / y_disc.shape[0]


            risk, risk_by_bin = _calculate_risk(h, args.bag_loss)
            if risk_by_bin is not None:
                all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(_tensor_to_numpy_safe(h, dtype=np.float64))
            if args.bag_loss == 'cox_surv':
                total_loss += loss_value * event_time.shape[0]
            else:
                total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0) if len(all_risk_by_bin_scores) > 0 else None
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc, iauc_list = _calculate_metrics(args, loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores, train_survival_risks)

    if args.return_attn == True:
        attn_matrix = [App, Agp, Apg, Acc, Acp, Apc]
        return patient_results, c_index, c_index2, BS, IBS, iauc, iauc_list, total_loss, attn_matrix
    else:
        attn_matrix = []
        return patient_results, c_index, c_index2, BS, IBS, iauc, iauc_list, total_loss, attn_matrix


def _train_loop_stage1(args, epoch, model, loader, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss = 0.

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()
        data_WSI, _, _, _, _, data_omics, _, data_clinic = _unpack_data("survfusion_separate", device, data)
        _, alignment_loss = model(x_path=data_WSI, x_omic=data_omics, x_clinic=data_clinic)
        alignment_loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += alignment_loss.item()
        print("Stage1 batch: {}, align_loss: {:.4f}".format(
            batch_idx * loader.batch_size, alignment_loss.item()))

    total_loss /= len(loader)
    print("Stage1 Epoch {}: train_align_loss={:.4f}".format(epoch, total_loss))
    return total_loss


def _val_loop_stage1(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for data in loader:
            data_WSI, _, _, _, _, data_omics, _, data_clinic = _unpack_data("survfusion_separate", device, data)
            _, alignment_loss = model(x_path=data_WSI, x_omic=data_omics, x_clinic=data_clinic)
            total_loss += alignment_loss.item()

    total_loss /= len(loader)
    return total_loss


def _train_loop_stage1_poe(args, epoch, model, loader, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.set_training_stage("stage1")
    total_loss = 0.
    beta = _get_poe_beta(args, epoch)
    poe_epoch_monitor = _init_poe_epoch_monitor()

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()
        data_WSI, mask, _, _, _, data_omics, _, data_clinic, avail = _unpack_survtri_poe_vae_batch(device, data)
        avail = _build_survtri_poe_vae_avail(args, model, avail)
        model(
            x_path=data_WSI,
            x_omic=data_omics,
            x_clinic=data_clinic,
            wsi_mask=mask,
            avail=avail,
        )
        vae_loss = model.get_vae_loss(beta=beta)
        vae_loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += vae_loss.item()
        cached = model.get_cached_outputs()
        _update_poe_epoch_monitor(poe_epoch_monitor, cached)
        _wandb_log(
            args,
            _get_poe_step_metrics(
                model,
                total_loss=vae_loss.item(),
                kl_value=cached["jeffreys"].item(),
                phase=f"{args.poe_variant}_stage1",
                beta=beta,
            ),
        )
        print(
            "TriPoEVAE Stage1 batch: {}, loss: {:.4f}, recon: {:.4f}, jeffreys: {:.4f}, beta: {:.3f}".format(
                batch_idx * loader.batch_size,
                vae_loss.item(),
                cached["recon_total"].item(),
                cached["jeffreys"].item(),
                beta,
            )
        )

    total_loss /= len(loader)
    print("TriPoEVAE Stage1 Epoch {}: train_vae_loss={:.4f}".format(epoch, total_loss))
    epoch_metrics = _finalize_poe_epoch_monitor(poe_epoch_monitor)
    epoch_metrics["epoch"] = epoch
    epoch_metrics["trainer/phase"] = f"{args.poe_variant}_stage1"
    _wandb_log(args, epoch_metrics)
    return total_loss


def _val_loop_stage1_poe(args, epoch, model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.set_training_stage("stage1")
    total_loss = 0.
    beta = _get_poe_beta(args, epoch)

    with torch.no_grad():
        for data in loader:
            data_WSI, mask, _, _, _, data_omics, _, data_clinic, avail = _unpack_survtri_poe_vae_batch(device, data)
            avail = _build_survtri_poe_vae_avail(args, model, avail)
            model(
                x_path=data_WSI,
                x_omic=data_omics,
                x_clinic=data_clinic,
                wsi_mask=mask,
                avail=avail,
            )
            total_loss += model.get_vae_loss(beta=beta).item()

    total_loss /= len(loader)
    return total_loss


def _step_stage1_poe(cur, args, model, train_loader, val_loader):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_stage1,
        weight_decay=args.reg,
    )
    total_steps = args.max_epochs_stage1 * len(train_loader)
    warmup_steps = len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    val_loss_min = float('inf')
    saved_checkpoints = []
    stage1_scan_enabled = bool(getattr(args, "poe_scan_stage1_ckpts", False)) and args.poe_variant == "B"
    ckpt_interval = max(1, int(getattr(args, "poe_stage1_ckpt_interval", 5)))
    stage1_scan_dir = None
    if stage1_scan_enabled:
        stage1_scan_dir = os.path.join(args.results_dir, "stage1_scan", f"fold_{cur}")
        os.makedirs(stage1_scan_dir, exist_ok=True)
    stage1_ckpt_path = _get_poe_stage1_ckpt_path(args, cur)
    os.makedirs(os.path.dirname(stage1_ckpt_path), exist_ok=True)

    for epoch in range(args.max_epochs_stage1):
        _train_loop_stage1_poe(args, epoch, model, train_loader, optimizer, scheduler)
        val_loss = _val_loop_stage1_poe(args, epoch, model, val_loader)
        print("TriPoEVAE Stage1 Val loss: {:.4f}".format(val_loss))
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            torch.save(model, stage1_ckpt_path)
            print("TriPoEVAE Stage1 Epoch {} is the best.".format(epoch))
        epoch_num = epoch + 1
        if stage1_scan_enabled and (epoch_num % ckpt_interval == 0 or epoch_num == args.max_epochs_stage1):
            scan_ckpt_path = os.path.join(stage1_scan_dir, f"ckpt_epoch_{epoch_num:03d}.pt")
            torch.save(model, scan_ckpt_path)
            saved_checkpoints.append((epoch_num, scan_ckpt_path))
            print(f"TriPoEVAE Stage1 checkpoint saved: epoch={epoch_num}, path={scan_ckpt_path}")

    return saved_checkpoints


def _run_poe_stage2_from_checkpoint(
    cur,
    args,
    loss_fn,
    model,
    train_loader,
    val_loader,
    test_loader,
    stage2_results_dir,
    stage1_scan_tag,
):
    stage2_args = _clone_args_for_subrun(args, stage2_results_dir, stage1_scan_tag=stage1_scan_tag)
    init_wandb_run(
        stage2_args,
        cur,
        stage_name="stage2",
        job_type="stage2_survival_finetune",
    )
    try:
        optimizer = _init_optim(stage2_args, model)
        lr_scheduler = _get_lr_scheduler(stage2_args, optimizer, train_loader)
        results_dict, metrics, attn_matrix = _step(
            cur,
            stage2_args,
            loss_fn,
            model,
            optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
            test_loader,
        )

        _save_pkl(os.path.join(stage2_results_dir, f"split_{cur}_results.pkl"), results_dict)
        val_result_path = os.path.join(stage2_results_dir, f"val_result_fold{cur}.csv")
        val_cindex = np.nan
        if os.path.exists(val_result_path):
            try:
                val_df = pd.read_csv(val_result_path)
                val_cindex = float(val_df["val_cindex"].iloc[0])
            except Exception:
                val_cindex = np.nan

        return {
            "results_dict": results_dict,
            "metrics": metrics,
            "attn_matrix": attn_matrix,
            "val_cindex": val_cindex,
            "results_dir": stage2_results_dir,
            "stage1_scan_tag": stage1_scan_tag,
        }
    finally:
        finish_wandb_run(stage2_args)


def _step_stage1(cur, args, model, train_loader, val_loader):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_stage1,
        weight_decay=args.reg,
    )
    total_steps = args.max_epochs_stage1 * len(train_loader)
    warmup_steps = len(train_loader)  # 1 epoch warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    val_loss_min = float('inf')
    stage1_ckpt_path = _get_poe_stage1_ckpt_path(args, cur)
    os.makedirs(os.path.dirname(stage1_ckpt_path), exist_ok=True)
    for epoch in range(args.max_epochs_stage1):
        _train_loop_stage1(args, epoch, model, train_loader, optimizer, scheduler)
        val_loss = _val_loop_stage1(model, val_loader)
        print("Stage1 Val align_loss: {:.4f}".format(val_loss))
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            torch.save(model, stage1_ckpt_path)
            print("Stage1 Epoch {} is the best.".format(epoch))


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler


def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader, test_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    all_survival = _extract_survival_metadata(train_loader, val_loader, test_loader)
    val_cindex_max = 0
    best_start_epoch = getattr(args, "save_best_from_epoch", 0)

    for epoch in range(args.max_epochs):
        train_cindex, train_loss, train_cox_skip_count = _train_loop_survival(args, epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
        results_dict_val, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_iauc_list, total_loss, attn_matrix = _summary(args, args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        print('Val loss:', total_loss, ', val_c_index:', val_cindex)
        epoch_log_step = getattr(args, "epoch_log_step_base", 0) + epoch
        epoch_metrics = {
            "epoch": epoch,
            "val/c_index": val_cindex,
            "val/loss": total_loss,
            "val/c_index_ipcw": val_cindex_ipcw,
        }
        selection_metric = val_cindex
        pattern_metrics = {}
        if args.modality in POE_MULTI_PATTERN_MODALITIES:
            pattern_metrics = _evaluate_pattern_cindices(args, model, val_loader)
            epoch_metrics.update(pattern_metrics)
            if args.modality in POE_PGC_SELECTION_MODALITIES:
                # Table1 / Optuna C-film report complete-modality PGC, not the 7-pattern mean.
                selection_metric = pattern_metrics.get("val/cindex_PGC", val_cindex)
            else:
                selection_metric = pattern_metrics.get("val/cindex_weighted", float("nan"))
            print(
                "Val cindex_weighted:",
                pattern_metrics.get("val/cindex_weighted"),
                ", val_cindex_PGC:",
                pattern_metrics.get("val/cindex_PGC"),
                ", selection:",
                selection_metric,
            )
        _wandb_log(args, epoch_metrics)
        trial = getattr(args, "optuna_trial", None)
        if trial is not None:
            log_optuna_analysis(
                "epoch",
                trial=trial.number,
                study=getattr(args, "study", None),
                fold=cur,
                epoch=epoch,
                train_cindex=train_cindex,
                train_loss=train_loss,
                val_cindex=val_cindex,
                val_loss=total_loss,
                val_cindex_weighted=pattern_metrics.get("val/cindex_weighted"),
                val_cindex_PGC=pattern_metrics.get("val/cindex_PGC"),
                selection_metric=selection_metric,
                batch_size=getattr(args, "batch_size", None),
                cox_skip_count=train_cox_skip_count,
            )
        if np.isfinite(selection_metric):
            _optuna_report_and_prune(args, selection_metric, epoch_log_step)
        # save the best trained model
        if epoch >= best_start_epoch and np.isfinite(selection_metric) and selection_metric >= val_cindex_max:
            val_cindex_max = selection_metric
            _save_model_checkpoint(
                model,
                os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)),
                args,
            )
            filename = os.path.join(args.results_dir, 'split_{}_results_val.pkl'.format(cur))
            _save_pkl(filename, results_dict_val)
            print('Epoch: {} is the best.'.format(epoch))
    final_df = pd.DataFrame([{
        'val_cindex': val_cindex_max,
        'val_cindex_ipcw': val_cindex_ipcw,
        'val_IBS': val_IBS,
        'val_iauc': val_iauc,
        'val_iauc_list': val_iauc_list,
        "val_loss": total_loss,
        'val_BS': val_BS
    }])
    final_df.to_csv(os.path.join(args.results_dir, 'val_result_fold{}.csv'.format(cur)))

    print('Start testing')
    model = _load_model_checkpoint(
        model,
        os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)),
        args,
    )
    train_survival_risks = None
    if args.bag_loss == 'cox_surv':
        print('Collecting train risks for the cox Breslow baseline')
        train_survival_risks = _collect_train_survival_risks(args, model, train_loader, loss_fn)
    eval_subsets = parse_eval_modalities(getattr(args, "eval_modalities", "off"))
    if eval_subsets:
        args._eval_subset = "PCG"
    try:
        results_dict, test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss, attn_matrix = _summary(args, args.dataset_factory, model, args.modality, test_loader, loss_fn, all_survival, train_survival_risks)
    finally:
        args._eval_subset = None
    print('Final test c-index: {:.4f}'.format(test_cindex))
    test_pattern_metrics = {}
    if args.modality in POE_MULTI_PATTERN_MODALITIES:
        test_pattern_metrics = _evaluate_pattern_cindices(args, model, test_loader)
        print(
            'Final test cindex_weighted:',
            test_pattern_metrics.get('val/cindex_weighted'),
            ', test_cindex_PGC:',
            test_pattern_metrics.get('val/cindex_PGC'),
        )
    trial = getattr(args, 'optuna_trial', None)
    if trial is not None:
        log_optuna_analysis(
            'fold_test',
            trial=trial.number,
            study=getattr(args, 'study', None),
            fold=cur,
            val_selection=val_cindex_max,
            test_cindex=test_cindex,
            test_cindex_weighted=test_pattern_metrics.get('val/cindex_weighted'),
            test_cindex_PGC=test_pattern_metrics.get('val/cindex_PGC'),
            batch_size=getattr(args, 'batch_size', None),
        )
    _evaluate_requested_subsets(
        args,
        cur,
        model,
        test_loader,
        loss_fn,
        all_survival,
        results_dict,
        (test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss),
    )

    return results_dict, (test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss), attn_matrix

def _train_val_test(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - test_cindex : Float
        - test_cindex2 : Float
        - test_BS : Float
        - test_IBS : Float
        - test_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    train_split, val_split, test_split = _get_splits(datasets, cur, args)

    #----> init loss function
    loss_fn = _init_loss_function(args)

    if args.modality in POE_ALL_MODALITIES | MISSING_MODALITY_BASELINES and args.bag_loss != 'cox_surv':
        raise ValueError(f"{args.modality} currently supports only `cox_surv`.")

    #----> init model
    model = _init_model(args, current_fold=cur)

    #---> init loaders (stage2 / normal loaders; train uses args.batch_size, val/test stay at batch_size=1)
    train_loader, val_loader, test_loader = _init_loaders(args, train_split, val_split, test_split)
    if args.modality == "modality_concat":
        _set_concat_mean_impute_stats(args, model, train_loader)

    try:
        if args.modality == 'survfusion_separate':
        # ── Stage 1: CLIP 对齐训练（使用更大的 batch_size）
            stage1_train_loader = _get_split_loader(
                args, train_split, training=True, testing=False,
                weighted=args.weighted_sample, batch_size=args.batch_size_stage1
            )
            _step_stage1(cur, args, model, stage1_train_loader, val_loader)

        # 加载 Stage1 最优 checkpoint，冻结参数，切换到 Stage2
            model = torch.load(
                os.path.join(args.results_dir, "s_{}_stage1_checkpoint.pt".format(cur)),
                weights_only=False
            )
            model.freeze_stage1_modules()
            model.set_training_stage("stage2")
            if torch.cuda.is_available():
                model = model.to(torch.device('cuda'))

        if args.modality in POE_MODALITIES and args.poe_variant in {'A', 'B'}:
            stage1_ckpt_path = _get_poe_stage1_ckpt_path(args, cur)
            stage1_ckpts = None
            if _cross_stage1_enabled(args):
                stage1_studies = _get_poe_stage1_studies(args)
                if args.study not in stage1_studies:
                    raise ValueError(
                        f"Current study `{args.study}` must be included in the fixed stage1 study pool: {', '.join(stage1_studies)}."
                    )
                if os.path.exists(stage1_ckpt_path):
                    print(f"[TriPoEVAE] reuse shared stage1 checkpoint: {stage1_ckpt_path}")
                else:
                    lock_path = stage1_ckpt_path + ".lock"
                    os.makedirs(os.path.dirname(stage1_ckpt_path), exist_ok=True)
                    if not _acquire_stage1_lock(lock_path):
                        print(f"[TriPoEVAE] waiting for shared stage1 checkpoint: {stage1_ckpt_path}")
                        while not os.path.exists(stage1_ckpt_path):
                            time.sleep(30)
                    else:
                        try:
                            init_wandb_run(
                                args,
                                cur,
                                stage_name="stage1",
                                job_type="stage1_vae_pretrain",
                            )
                            stage1_loaders = _build_poe_stage1_loaders(args, cur)
                            if stage1_loaders is None:
                                stage1_train_loader = _get_split_loader(
                                    args, train_split, training=True, testing=False,
                                    weighted=args.weighted_sample,
                                    batch_size=args.batch_size_stage1,
                                    disable_cox_batch_override=True,
                                )
                                stage1_val_loader = val_loader
                            else:
                                stage1_train_loader, stage1_val_loader, stage1_studies = stage1_loaders
                                print(f"[TriPoEVAE] stage1 studies: {', '.join(stage1_studies)}")
                            stage1_ckpts = _step_stage1_poe(cur, args, model, stage1_train_loader, stage1_val_loader)
                        finally:
                            finish_wandb_run(args)
                            _release_stage1_lock(lock_path)
            else:
                init_wandb_run(
                    args,
                    cur,
                    stage_name="stage1",
                    job_type="stage1_vae_pretrain",
                )
                try:
                    stage1_loaders = _build_poe_stage1_loaders(args, cur)
                    if stage1_loaders is None:
                        stage1_train_loader = _get_split_loader(
                            args, train_split, training=True, testing=False,
                            weighted=args.weighted_sample,
                            batch_size=args.batch_size_stage1,
                            disable_cox_batch_override=True,
                        )
                        stage1_val_loader = val_loader
                    else:
                        stage1_train_loader, stage1_val_loader, stage1_studies = stage1_loaders
                        print(f"[TriPoEVAE] stage1 studies: {', '.join(stage1_studies)}")
                    stage1_ckpts = _step_stage1_poe(cur, args, model, stage1_train_loader, stage1_val_loader)
                finally:
                    finish_wandb_run(args)

            model = torch.load(
                stage1_ckpt_path,
                weights_only=False
            )
            if args.poe_variant == 'A':
                model.freeze_backbone_for_probe()
                stage2_job_type = "stage2_linear_probe"
            else:
                model.set_training_stage("stage2")
                if args.modality in POE_NEW_B_MODALITIES:
                    model.freeze_backbone_keep_head()
                stage2_job_type = "stage2_survival_finetune"
            if torch.cuda.is_available():
                model = model.to(torch.device('cuda'))

            if args.poe_variant == 'B' and getattr(args, "poe_scan_stage1_ckpts", False):
                scan_root = os.path.join(args.results_dir, "stage2_scan", f"fold_{cur}")
                os.makedirs(scan_root, exist_ok=True)
                stage1_ckpts = stage1_ckpts or [(args.max_epochs_stage1, stage1_ckpt_path)]
                scan_rows = []
                scan_pick = []
                for stage1_epoch, ckpt_path in stage1_ckpts:
                    stage1_scan_tag = f"stage1{stage1_epoch:03d}pt"
                    stage2_dir = os.path.join(scan_root, stage1_scan_tag)
                    os.makedirs(stage2_dir, exist_ok=True)

                    stage2_model = torch.load(ckpt_path, weights_only=False)
                    stage2_model.set_training_stage("stage2")
                    if args.modality in POE_NEW_B_MODALITIES:
                        stage2_model.freeze_backbone_keep_head()
                    if torch.cuda.is_available():
                        stage2_model = stage2_model.to(torch.device('cuda'))

                    subrun_args = _clone_args_for_subrun(args, stage2_dir, stage1_scan_tag=stage1_scan_tag)
                    init_wandb_run(
                        subrun_args,
                        cur,
                        stage_name="stage2",
                        job_type=stage2_job_type,
                    )
                    try:
                        optimizer = _init_optim(subrun_args, stage2_model)
                        lr_scheduler = _get_lr_scheduler(subrun_args, optimizer, train_loader)
                        results_dict, metrics, attn_matrix = _step(
                            cur,
                            subrun_args,
                            loss_fn,
                            stage2_model,
                            optimizer,
                            lr_scheduler,
                            train_loader,
                            val_loader,
                            test_loader,
                        )
                        _save_pkl(os.path.join(stage2_dir, f"split_{cur}_results.pkl"), results_dict)
                        val_cindex = np.nan
                        val_csv = os.path.join(stage2_dir, f"val_result_fold{cur}.csv")
                        if os.path.exists(val_csv):
                            try:
                                val_df = pd.read_csv(val_csv)
                                val_cindex = float(val_df["val_cindex"].iloc[0])
                            except Exception:
                                val_cindex = np.nan
                        test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss = metrics
                        scan_rows.append({
                            "stage1_epoch": stage1_epoch,
                            "stage1_ckpt": ckpt_path,
                            "stage1_scan_tag": stage1_scan_tag,
                            "val_cindex": val_cindex,
                            "test_cindex": test_cindex,
                            "test_cindex_ipcw": test_cindex_ipcw,
                            "test_BS": test_BS,
                            "test_IBS": test_IBS,
                            "test_iauc": test_iauc,
                            "test_iauc_list": test_iauc_list,
                            "test_loss": total_loss,
                            "results_dir": stage2_dir,
                        })
                        scan_pick.append((results_dict, metrics, attn_matrix))
                    finally:
                        finish_wandb_run(subrun_args)

                scan_df = pd.DataFrame(scan_rows)
                scan_df.to_csv(os.path.join(scan_root, "scan_summary.csv"), index=False)
                best_idx = int(scan_df["val_cindex"].astype(float).fillna(-np.inf).to_numpy().argmax())
                return scan_pick[best_idx]

            init_wandb_run(
                args,
                cur,
                stage_name="stage2",
                job_type=stage2_job_type,
            )
        elif args.modality in POE_STAGE2_ONLY_MODALITIES:
            init_wandb_run(
                args,
                cur,
                stage_name="stage2",
                job_type="stage2_survival_finetune",
            )
        else:
            init_wandb_run(
                args,
                cur,
                stage_name="stage2" if args.modality in POE_MODALITIES and args.poe_variant == 'C' else None,
                job_type="stage2_joint_train" if args.modality in POE_MODALITIES and args.poe_variant == 'C' else None,
            )

        # survfusion_noalign / survfusion_joint：无 Stage1，直接走 Stage2 pipeline
        # （batch_size 由命令行 --batch_size 控制，建议设为 32）

        # ── Stage 2 / 普通模型：走现有 pipeline
        optimizer = _init_optim(args, model)
        lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

        results_dict, (test_cindex, test_cindex2, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss), attn_matrix = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)

        return results_dict, (test_cindex, test_cindex2, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss), attn_matrix
    finally:
        finish_wandb_run(args)
