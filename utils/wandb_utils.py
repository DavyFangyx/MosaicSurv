import os

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def wandb_enabled(args):
    return wandb is not None and getattr(args, "wandb_mode", "disabled") != "disabled"


def finish_wandb_run(args):
    run = getattr(args, "wandb_run", None)
    if run is not None and wandb is not None:
        wandb.finish()
        args.wandb_run = None


def init_wandb_run(args, fold, stage_name=None, job_type=None):
    if not wandb_enabled(args):
        return None

    config = dict(vars(args))
    for key in ["dataset_factory", "wandb_run", "optuna_trial", "optuna_pruned_exception"]:
        config.pop(key, None)
    stage_tag = stage_name or "single"

    wandb_root = os.path.join(args.results_dir, "wandb")
    wandb_cache = os.path.join(wandb_root, "cache")
    os.makedirs(wandb_cache, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_root
    os.environ["WANDB_CACHE_DIR"] = wandb_cache

    settings = wandb.Settings(
        mode=args.wandb_mode,
        root_dir=wandb_root,
        start_method="thread",
        symlink=False,
    )

    is_optuna = bool(getattr(args, "use_optuna", False))
    if is_optuna:
        experiment_tag = getattr(args, "optuna_experiment_tag", args.exp_group)
        model_or_run = getattr(args, "optuna_base_run_name", args.run_name)
        trial = getattr(args, "optuna_trial", None)
        trial_number = trial.number if trial is not None else -1
        trial_tag = getattr(args, "optuna_trial_tag", f"trial_{trial_number:04d}")
        run_name = f"{model_or_run}_{trial_tag}_{args.study}_{stage_tag}_fold{fold}"
        group = f"{experiment_tag}_optuna"
        tags = [
            experiment_tag,
            "optuna",
            model_or_run,
            trial_tag,
            args.study,
            stage_tag,
            f"fold_{fold}",
            args.modality,
            f"poe_{getattr(args, 'poe_variant', 'na')}",
        ]
        config.update({
            "experiment_tag": experiment_tag,
            "run_kind": "optuna",
            "model": model_or_run,
            "trial": trial_number,
            "trial_tag": trial_tag,
            "study": args.study,
            "stage": stage_tag,
            "fold": fold,
            "modality": args.modality,
            "poe_variant": getattr(args, "poe_variant", "na"),
        })
    else:
        experiment_tag = args.exp_group
        run_name = f"{args.run_name}_{stage_tag}"
        stage1_scan_tag = getattr(args, "current_stage1_scan_tag", None)
        if stage1_scan_tag:
            run_name = f"{run_name}_{stage1_scan_tag}"
        run_name = f"{run_name}_fold{fold}"
        group = f"{args.exp_group}_train"
        tags = [
            args.exp_group,
            "train",
            args.run_name,
            stage_tag,
            f"fold_{fold}",
            args.modality,
            f"poe_{getattr(args, 'poe_variant', 'na')}",
        ]
        if stage1_scan_tag:
            tags.append(stage1_scan_tag)
        config.update({
            "experiment_tag": experiment_tag,
            "run_kind": "train",
            "model": args.run_name,
            "stage": stage_tag,
            "fold": fold,
            "modality": args.modality,
            "poe_variant": getattr(args, "poe_variant", "na"),
        })
        if stage1_scan_tag:
            config["stage1_scan_tag"] = stage1_scan_tag

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        job_type=job_type,
        name=run_name,
        dir=wandb_root,
        config=config,
        reinit=True,
        tags=tags,
        settings=settings,
    )
    args.wandb_run = run
    return run
