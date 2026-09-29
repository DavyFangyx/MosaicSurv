import os
from copy import deepcopy
from timeit import default_timer as timer
from warnings import simplefilter

import pandas as pd

from datasets.dataset_survival import SurvivalDatasetFactory
from main import _write_filter_log
from utils.core_utils import _train_val_test
from utils.general_utils import _get_start_end, _prepare_for_experiment
from utils.optuna_utils import (
    attach_optuna_analysis_logger,
    bind_optuna_study_paths,
    build_optuna_components,
    build_trial_args,
    default_optuna_study_name,
    ensure_optuna_available,
    get_optuna_study_min_cindex,
    log_optuna_analysis,
    parse_optuna_studies,
    resolve_optuna_storage,
    sample_survtri_poe_vae_trial,
    save_study_artifacts,
)
from utils.process_args import _process_args
from utils.missing_mask_protocol import prepare_missing_protocol
from utils.wandb_utils import finish_wandb_run

simplefilter(action="ignore", category=FutureWarning)


def _build_dataset_factory(args):
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
            "survtri_poe_vae_b_kl",
            "survtri_poe_vae_b_crossstage1",
            "survtri_poe_vae_b_nopretrain",
            "survtri_poe_vae_b_single",
            "survtri_poe_vae_b_multi",
            "survtri_poe_vae_b_film",
            "survtri_poe_vae_c_single",
            "survtri_poe_vae_c_multi",
            "survtri_poe_vae_c_film",
            "mvae_poe",
            "mopoe",
            "flex_moe",
            "hgcn",
            "concat_zero",
            "concat_mean",
        ) else False,
        is_survpc=True if args.modality == "survpc" else False,
        is_survpc_f=True if args.modality == "survpc_f" else False,
        type_of_pathway=args.type_of_path,
    )


def _reset_results_dir(args, base_results_dir, run_name):
    args.results_dir = base_results_dir
    args.run_name = run_name
    return args


def _eval_study_folds(trial_args, trial, folds, study_index, n_folds):
    fold_val_scores = []
    for fold_index, fold in enumerate(folds):
        trial_args.epoch_log_step_base = (
            study_index * n_folds * int(trial_args.max_epochs)
            + fold_index * int(trial_args.max_epochs)
        )
        try:
            datasets = trial_args.dataset_factory.return_splits(
                trial_args,
                csv_path=f"{trial_args.split_dir}/splits_{fold}.csv",
                fold=fold,
            )
            _train_val_test(datasets, fold, trial_args)
            val_result_path = os.path.join(trial_args.results_dir, f"val_result_fold{fold}.csv")
            val_df = pd.read_csv(val_result_path)
            best_val_cindex = float(val_df["val_cindex"].iloc[0])
            fold_val_scores.append(best_val_cindex)
            log_optuna_analysis(
                "fold_best",
                trial=trial.number,
                study=trial_args.study,
                fold=fold,
                val_cindex=best_val_cindex,
                results_dir=trial_args.results_dir,
            )
            if trial_args.wandb_run is not None:
                trial_args.wandb_run.summary["optuna/fold_val_cindex"] = best_val_cindex
                trial_args.wandb_run.summary["optuna/trial_number"] = trial.number
                trial_args.wandb_run.summary["optuna/fold"] = fold
                trial_args.wandb_run.summary["optuna/study"] = trial_args.study
        finally:
            finish_wandb_run(trial_args)
    return fold_val_scores


def _objective_factory(base_args):
    import optuna

    studies = parse_optuna_studies(base_args)
    base_results_dir = base_args.results_dir
    base_run_name = base_args.run_name

    def objective(trial):
        if base_args.modality not in {"survtri_poe_vae", "survtri_poe_vae_c_film"}:
            raise ValueError(
                "The minimal Optuna tuner currently supports only survtri_poe_vae "
                "and survtri_poe_vae_c_film."
            )

        sampled_params = sample_survtri_poe_vae_trial(trial, base_args)
        trial_args = build_trial_args(base_args, trial, sampled_params, optuna.TrialPruned)
        if "batch_size" in sampled_params:
            trial_args.batch_size = int(sampled_params["batch_size"])
        if "batch_size_stage1" in sampled_params:
            trial_args.batch_size_stage1 = int(sampled_params["batch_size_stage1"])
        trial.set_user_attr("sampled_params", sampled_params)
        trial.set_user_attr("optuna_studies", studies)
        print(
            f"[optuna] trial {trial.number} "
            f"studies={studies} "
            f"batch_size={trial_args.batch_size} "
            f"batch_size_stage1={trial_args.batch_size_stage1} "
            f"sampled={sampled_params}"
        )
        log_optuna_analysis(
            "trial_start",
            trial=trial.number,
            studies=studies,
            batch_size=trial_args.batch_size,
            batch_size_stage1=trial_args.batch_size_stage1,
            sampled=sampled_params,
        )

        if trial_args.optuna_fold_mode == "single":
            folds = [int(trial_args.optuna_fold)]
        else:
            folds = list(_get_start_end(trial_args))
        n_folds = max(len(folds), 1)

        study_means = {}
        fold_scores_by_study = {}

        for study_index, study in enumerate(studies):
            study_args = deepcopy(trial_args)
            study_args = bind_optuna_study_paths(study_args, study)
            study_args = _reset_results_dir(
                study_args,
                base_results_dir,
                f"{base_run_name}_trial_{trial.number:04d}_{study}",
            )
            study_args = prepare_missing_protocol(study_args)
            study_args = _prepare_for_experiment(study_args)
            study_args.dataset_factory = _build_dataset_factory(study_args)
            _write_filter_log(study_args)

            print(
                f"[optuna] trial {trial.number} study {study} "
                f"min_cindex={get_optuna_study_min_cindex(study):.4f} "
                f"folds={folds}"
            )
            fold_val_scores = _eval_study_folds(
                study_args, trial, folds, study_index, n_folds
            )
            mean_val_cindex = float(sum(fold_val_scores) / len(fold_val_scores))
            study_means[study] = mean_val_cindex
            fold_scores_by_study[study] = fold_val_scores
            trial.set_user_attr(f"{study}_mean_val_cindex", mean_val_cindex)
            trial.set_user_attr(f"{study}_fold_val_scores", fold_val_scores)
            print(
                f"[optuna] trial {trial.number} study {study} "
                f"mean_val_cindex={mean_val_cindex:.4f} folds={fold_val_scores}"
            )
            log_optuna_analysis(
                "study_mean",
                trial=trial.number,
                study=study,
                mean_val_cindex=mean_val_cindex,
                folds=fold_val_scores,
            )

            min_cindex = get_optuna_study_min_cindex(study)
            if mean_val_cindex < min_cindex:
                trial.set_user_attr("pruned_study", study)
                trial.set_user_attr("pruned_mean_val_cindex", mean_val_cindex)
                trial.set_user_attr("study_mean_val_cindex", study_means)
                log_optuna_analysis(
                    "trial_pruned",
                    trial=trial.number,
                    study=study,
                    mean_val_cindex=mean_val_cindex,
                    min_cindex=min_cindex,
                )
                raise optuna.TrialPruned(
                    f"Pruned after {study}: mean val_cindex={mean_val_cindex:.4f} "
                    f"< {min_cindex:.4f}"
                )

        objective_value = float(sum(study_means.values()) / len(study_means))
        trial.set_user_attr("study_mean_val_cindex", study_means)
        trial.set_user_attr("fold_val_scores_by_study", fold_scores_by_study)
        log_optuna_analysis(
            "trial_complete",
            trial=trial.number,
            objective=objective_value,
            study_means=study_means,
        )
        return objective_value

    return objective


def main(args):
    ensure_optuna_available()
    import optuna

    if args.modality not in {"survtri_poe_vae", "survtri_poe_vae_c_film"}:
        raise ValueError(
            "main_tune_optuna.py currently supports only "
            "`--modality survtri_poe_vae` or `--modality survtri_poe_vae_c_film`."
        )
    if args.modality == "survtri_poe_vae_c_film":
        args.poe_variant = "C"
    if args.poe_variant not in {"A", "B", "C"}:
        raise ValueError("main_tune_optuna.py currently supports only `--poe_variant A|B|C`.")

    studies = parse_optuna_studies(args)
    print(f"[optuna] studies in order: {studies}")
    for study in studies:
        print(f"[optuna] {study} min_cindex={get_optuna_study_min_cindex(study):.4f}")

    study_name = default_optuna_study_name(args)
    analysis_logger = attach_optuna_analysis_logger(args, study_name=study_name)
    print(f"[optuna] analysis log: {analysis_logger.path}")
    log_optuna_analysis(
        "study_start",
        study_name=study_name,
        studies=studies,
        modality=args.modality,
        poe_variant=args.poe_variant,
        max_epochs=args.max_epochs,
        k=args.k,
    )
    storage = resolve_optuna_storage(args)
    sampler, pruner = build_optuna_components(args)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=args.optuna_direction,
        load_if_exists=True,
    )
    study.optimize(_objective_factory(args), n_trials=args.optuna_trials)

    output_dir = os.path.join(args.results_dir, "optuna", study_name)
    save_study_artifacts(study, output_dir)
    print(f"Optuna study finished. Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    log_optuna_analysis(
        "study_finished",
        study_name=study_name,
        best_value=study.best_value,
        best_trial=study.best_trial.number,
        best_params=study.best_trial.params,
        artifacts_dir=output_dir,
    )


if __name__ == "__main__":
    start = timer()
    args = _process_args()
    args = prepare_missing_protocol(args)
    args_for_study = deepcopy(args)
    main(args_for_study)
    end = timer()
    print("finished optuna tuning!")
    print("end script")
    print("Script Time: %f seconds" % (end - start))
