import copy
import os
from pathlib import Path

from dataset_deployment.registry import infer_standard_paths, list_enabled_studies


OPTUNA_WSI_EXPERIMENT = "uni_v1"
OPTUNA_CLINIC_EXPERIMENT = "L0"
OPTUNA_GENE_EXPERIMENT = "scFoundation_embedding_cell_norm"

# Per-dataset 5-fold mean val C-index floors used by hard pruning.
# Edit a number here to change that dataset's cutoff. There is no CLI flag.
# The trial objective (mean of dataset means) is never thresholded.
OPTUNA_STUDY_MIN_CINDEX = {
    "tcga_kirp": 0.8115,
    "tcga_coad": 0.6611,
    "tcga_kirc": 0.6895,
    "tcga_brca": 0.6588,
    "tcga_lihc": 0.7040,
}


def ensure_optuna_available():
    try:
        import optuna  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Optuna is not installed in the current environment. "
            "Install it first, e.g. `python -m pip install optuna`."
        ) from exc


def build_optuna_components(args):
    import optuna

    if args.optuna_sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    elif args.optuna_sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported Optuna sampler `{args.optuna_sampler}`.")

    if args.optuna_pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.optuna_n_startup_trials,
            n_warmup_steps=args.optuna_n_warmup_steps,
            interval_steps=1,
        )
    elif args.optuna_pruner == "none":
        pruner = optuna.pruners.NopPruner()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported Optuna pruner `{args.optuna_pruner}`.")

    return sampler, pruner


def parse_optuna_studies(args):
    raw = getattr(args, "optuna_studies", "") or ""
    studies = []
    seen = set()
    for token in str(raw).split(","):
        study = token.strip()
        if not study:
            continue
        if not study.startswith("tcga_"):
            study = f"tcga_{study}"
        if study not in seen:
            studies.append(study)
            seen.add(study)
    if not studies:
        studies = [args.study]

    enabled = set(list_enabled_studies())
    unknown = [study for study in studies if study not in enabled]
    if unknown:
        raise ValueError(
            f"Unknown Optuna studies {unknown}. Expected one of: {sorted(enabled)}"
        )
    return studies


def get_optuna_study_min_cindex(study):
    try:
        return float(OPTUNA_STUDY_MIN_CINDEX[study])
    except KeyError as exc:
        raise KeyError(
            f"No Optuna min c-index for `{study}`. "
            "Add it to OPTUNA_STUDY_MIN_CINDEX in utils/optuna_utils.py."
        ) from exc


def bind_optuna_study_paths(args, study, repo_root="."):
    paths = infer_standard_paths(
        study,
        repo_root,
        which_splits=getattr(args, "which_splits", "5foldcv"),
        type_of_path=getattr(args, "type_of_path", "combine"),
        wsi_experiment=OPTUNA_WSI_EXPERIMENT,
        clinic_experiment=OPTUNA_CLINIC_EXPERIMENT,
        gene_experiment=OPTUNA_GENE_EXPERIMENT,
    )
    args.study = study
    args.label_file = str(paths["label_file"])
    args.clinical_file = str(paths["clinical_file"])
    args.omics_dir = str(paths["omics_dir"])
    args.split_dir = str(paths["split_dir"])
    args.data_root_dir = str(paths["data_root_dir"])
    args.clinic_dir = str(paths["clinic_dir"])
    args.gene_dir = str(paths["gene_dir"])
    return args


def default_optuna_study_name(args):
    betafix_tag = "_betafix" if getattr(args, "betafix", False) else ""
    # Alpha is searched per trial, so keep one study and do not embed a CLI alphapgc value.
    alphafix_tag = "_alphafix_search"
    studies = parse_optuna_studies(args)
    if len(studies) == 1:
        study_tag = studies[0]
    else:
        study_tag = "-".join(study.replace("tcga_", "") for study in studies)
    if getattr(args, "optuna_fold_mode", "mean_cv") == "single":
        fold_tag = f"_fold{args.optuna_fold}"
    else:
        fold_tag = "_meancv"
    return args.optuna_study_name or (
        f"{study_tag}_{args.modality}_{args.poe_variant}{betafix_tag}{alphafix_tag}{fold_tag}"
    )


def resolve_optuna_storage(args):
    if args.optuna_storage:
        return args.optuna_storage

    base_results = Path(args.results_dir)
    storage_dir = base_results / "optuna"
    storage_dir.mkdir(parents=True, exist_ok=True)
    study_name = default_optuna_study_name(args)
    return f"sqlite:///{(storage_dir / f'{study_name}.db').resolve()}"


def sample_fixed_poe_alpha(trial):
    # Keep --alphapgc as raw P/G/C weights. Optuna searches the intended mass
    # bands, then the same L1-normalize path writes --alphapgc for alphafix.
    pathology = trial.suggest_float("alphapgc_pathology", 0.45, 0.75)
    gene = trial.suggest_float("alphapgc_gene", 0.05, 0.20)
    clinic = trial.suggest_float("alphapgc_clinic", 0.20, 0.40)
    total = pathology + gene + clinic
    weights = [pathology / total, gene / total, clinic / total]
    return {
        "alphafix": True,
        "alphapgc": ",".join(f"{weight:.6g}" for weight in weights),
    }


def sample_survtri_poe_vae_model_c(trial, args):
    params = sample_fixed_poe_alpha(trial)
    params.update({
        "lr": trial.suggest_float("lr", 5e-5, 7e-4, log=True),
        "reg": trial.suggest_float("reg", 1e-5, 1e-3, log=True),
        "poe_surv_lambda": trial.suggest_float("poe_surv_lambda", 0.05, 2.0, log=True),
        "poe_beta_target": trial.suggest_float("poe_beta_target", 0.02, 1.0, log=True),
        "poe_modality_dropout": trial.suggest_float("poe_modality_dropout", 0.0, 0.4),
    })
    return params


def sample_survtri_poe_vae_model_a(trial, args):
    params = sample_fixed_poe_alpha(trial)
    params.update({
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "lr_stage1": trial.suggest_float("lr_stage1", 1e-5, 5e-4, log=True),
        "reg": trial.suggest_float("reg", 1e-6, 1e-2, log=True),
        "poe_beta_target": trial.suggest_float("poe_beta_target", 1e-2, 2.0, log=True),
        "poe_modality_dropout": trial.suggest_float("poe_modality_dropout", 0.0, 0.4),
        "batch_size_stage1": trial.suggest_categorical("batch_size_stage1", [16, 32, 64, 128]),
    })
    return params


def sample_survtri_poe_vae_model_b(trial, args):
    params = sample_fixed_poe_alpha(trial)
    params.update({
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "lr_stage1": trial.suggest_float("lr_stage1", 5e-5, 5e-4, log=True),
        "reg": trial.suggest_float("reg", 1e-5, 1e-3, log=True),
        "poe_beta_target": trial.suggest_float("poe_beta_target", 0.01, 0.5, log=True),
        "poe_modality_dropout": trial.suggest_float("poe_modality_dropout", 0.0, 0.4),
        "poe_mmhid": trial.suggest_categorical("poe_mmhid", [128, 256]),
        "poe_decoder_hidden_dim": trial.suggest_categorical("poe_decoder_hidden_dim", [256, 512]),
        "poe_encoder_lr_ratio": trial.suggest_categorical("poe_encoder_lr_ratio", [0.01, 0.1, 1.0, 10.0]),
        "batch_size_stage1": trial.suggest_categorical("batch_size_stage1", [16, 32, 64, 128]),
    })
    return params


MOSAIC_SURV_ALPHAPGC_CHOICES = (
    "0.33,0.33,0.33",
    "0.30,0.30,0.40",
    "0.30,0.40,0.30",
    "0.40,0.30,0.30",
    "0.20,0.30,0.50",
    "0.20,0.50,0.30",
    "0.30,0.20,0.50",
    "0.30,0.50,0.20",
    "0.50,0.20,0.30",
    "0.50,0.30,0.20",
    "0.30,0.50,0.20",
    "0.20,0.50,0.30",
    "0.30,0.20,0.50",
    "0.20,0.30,0.50",
)


def sample_discrete_poe_alpha(trial):
    return {
        "alphafix": False,
        "alphapgc": None,
    }


def sample_mosaic_surv_trial(trial, args):
    params = sample_discrete_poe_alpha(trial)
    params.update({
        # 学习率：保留经典量级，覆盖 5e-5 到 1e-3
        "lr": trial.suggest_categorical("lr", [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]),
        # L2正则：从 1e-5 到 5e-3，按经典倍数跳
        "reg": trial.suggest_categorical("reg", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]),
        # Cox损失权重：覆盖轻到重，保留你的基线 1.0
        "poe_surv_lambda": trial.suggest_categorical("poe_surv_lambda", [0.05, 0.1, 0.5, 1.0, 2.0]),
        # VAE正则：放宽到 3.0，按常用梯度走
        "poe_beta_target": trial.suggest_categorical("poe_beta_target", [0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]),
        # 模态丢弃：从 0.05 起步，按 0.05 步长走，非常漂亮
        "poe_modality_dropout": trial.suggest_categorical("poe_modality_dropout", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]),
        # 隐层宽度：保留你的备选
        "poe_mmhid": trial.suggest_categorical("poe_mmhid", [128, 256, 384]),
        # MosaicSurv 只有 stage2，真正生效的是 batch_size，不是 batch_size_stage1
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    })
    return params


def sample_survtri_poe_vae_trial(trial, args):
    if args.modality == "mosaic_surv":
        return sample_mosaic_surv_trial(trial, args)
    if args.poe_variant == "A":
        return sample_survtri_poe_vae_model_a(trial, args)
    if args.poe_variant == "B":
        return sample_survtri_poe_vae_model_b(trial, args)
    if args.poe_variant == "C":
        return sample_survtri_poe_vae_model_c(trial, args)
    raise ValueError(
        f"Unsupported Optuna combo modality=`{args.modality}` poe_variant=`{args.poe_variant}`."
    )


def build_trial_args(base_args, trial, sampled_params, pruned_exception_cls):
    trial_args = copy.deepcopy(base_args)
    for key, value in sampled_params.items():
        setattr(trial_args, key, value)

    trial_args.run_name = f"{base_args.run_name}_trial_{trial.number:04d}"
    trial_args.exp_group = f"{base_args.exp_group}_optuna"
    trial_args.optuna_base_run_name = base_args.run_name
    trial_args.optuna_experiment_tag = base_args.exp_group
    trial_args.optuna_trial_tag = f"trial_{trial.number:04d}"
    trial_args.use_optuna = True
    trial_args.optuna_trial = trial
    trial_args.optuna_pruned_exception = pruned_exception_cls
    return trial_args


def default_optuna_analysis_log_path(args, study_name=None):
    explicit = getattr(args, "optuna_analysis_log", None)
    if explicit:
        return Path(explicit)
    name = study_name or default_optuna_study_name(args)
    return Path(args.results_dir) / "optuna" / f"{name}.log"


def _format_analysis_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, dict)):
        return str(value).replace(" ", "")
    return str(value)


class OptunaAnalysisLogger:
    """Compact Optuna log for trial/hyperparam/score analysis."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, event, **fields):
        parts = [f"[analysis] event={event}"]
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={_format_analysis_value(value)}")
        self._fh.write(" ".join(parts) + chr(10))
        self._fh.flush()

    def close(self):
        if getattr(self, "_fh", None) is None:
            return
        self._fh.close()
        self._fh = None


_ANALYSIS_LOGGER = None


def get_optuna_analysis_logger():
    return _ANALYSIS_LOGGER


def log_optuna_analysis(event, **fields):
    logger = get_optuna_analysis_logger()
    if logger is None:
        return
    logger.log(event, **fields)


def attach_optuna_analysis_logger(args, study_name=None):
    global _ANALYSIS_LOGGER
    path = default_optuna_analysis_log_path(args, study_name=study_name)
    args.optuna_analysis_log = str(path)
    if _ANALYSIS_LOGGER is not None:
        current = Path(getattr(_ANALYSIS_LOGGER, "path", ""))
        if current.resolve() == path.resolve():
            return _ANALYSIS_LOGGER
        _ANALYSIS_LOGGER.close()
    _ANALYSIS_LOGGER = OptunaAnalysisLogger(path)
    return _ANALYSIS_LOGGER


def save_study_artifacts(study, output_dir):
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)

    with open(os.path.join(output_dir, "best_trial.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_value: {study.best_value}\n")
        f.write(f"best_trial_number: {study.best_trial.number}\n")
        f.write("best_params:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
