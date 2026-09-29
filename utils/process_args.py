import argparse
import sys
from pathlib import Path

from dataset_deployment.registry import (
    DEFAULT_CLINIC_EXPERIMENT,
    DEFAULT_GENE_EXPERIMENT,
    DEFAULT_STUDY,
    DEFAULT_WSI_EXPERIMENT,
    infer_standard_paths,
    list_enabled_studies,
)

DEFAULT_RESULTS_DIR = './results'
DEFAULT_WSI_DIR = None
DEFAULT_CLINIC_DIR = None
DEFAULT_GENE_DIR = None

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, default=DEFAULT_STUDY, choices=list_enabled_studies(), help='study name')
    parser.add_argument('--task', type=str, default='survival', choices=['survival'])
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR, help='base results directory (default: ./results)')
    parser.add_argument('--exp_group', type=str, default='default',
                        help='experiment group; results are stored under results/{exp_group}/{run_name}/{modality}/')
    parser.add_argument('--run_name', type=str, default='default',
                        help='run name within the experiment group (e.g. O_origin, lr_0001)')
    parser.add_argument("--type_of_path", type=str, default="combine", choices=["xena", "hallmarks", "combine"])
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default=DEFAULT_WSI_DIR, help='data directory')
    parser.add_argument('--label_file', type=str, default=None, help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default=None, help='Path to dir with omics csv for all modalities')
    parser.add_argument('--clinic_dir', type=str, default=DEFAULT_CLINIC_DIR, help='Path to dir with clinical embedding')
    parser.add_argument('--gene_dir', type=str, default=DEFAULT_GENE_DIR, help='Path to dir with gene foundation model embedding')
    parser.add_argument('--clinical_file', type=str, default=None, help='Path to study clinical CSV')
    parser.add_argument('--data_pack_dir', type=str, default=None, help='Optional legacy HGCN pack directory; unused when P/C/G graph dirs are provided')

    parser.add_argument('--num_patches', type=int, default=4096, help='number of patches')
    parser.add_argument('--label_col', type=str, default="survival_months", help='type of survival (OS, DSS, PFI, PFS)')
    parser.add_argument("--wsi_projection_dim", type=int, default=256)
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)
    parser.add_argument('--single_model_size', type=str, default='small',
                        choices=['small', 'medium', 'big'],
                        help='hidden size preset for single-modality models')
    parser.add_argument('--single_use_input_ln', action='store_true', default=False,
                        help='apply LayerNorm on single-modality gene foundation inputs')
    parser.add_argument('--concat_wsi', type=str, default='resampler', choices=['meanpool', 'resampler'],
                        help='WSI frontend for the concat baseline; default is resampler')

    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--split_csv', type=str, default=None, help='single split csv for baselines that use one fold file')
    parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')
        
    #----> training related
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="radam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="L2", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=['ce_surv', "nll_surv", "nll_rank_surv", "rank_surv", "cox_surv", "nll_diff_surv"], default='nll_surv',
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.5, help='weight given to uncensored patients')
    parser.add_argument('--beta_surv', type=float, default=0.3, help='weight given to diff_loss')
    parser.add_argument('--reg', type=float, default=0.001, help='weight decay / L2 (default: 0.001)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)
    # Stage-1 params for two-stage training models (e.g. survfusion_separate)
    parser.add_argument('--lr_stage1', type=float, default=0.0001, help='learning rate for stage-1 alignment training')
    parser.add_argument('--max_epochs_stage1', type=int, default=40, help='max epochs for stage-1 alignment training')
    parser.add_argument('--batch_size_stage1', type=int, default=1, help='batch size for stage-1 VAE training')

    # ── SurvFusion 消融实验参数 ─────────────────────────────────────────────
    # 实验2: 联合训练时 alignment_loss 的权重 λ ∈ {0.01, 0.1, 0.5}
    parser.add_argument('--clip_lambda', type=float, default=0.1,
                        help='weight for alignment_loss in survfusion_joint: total = surv + λ * align')
    # 实验3: Stage2 融合方式消融
    parser.add_argument('--fusion_type', type=str, default='mhsa',
                        choices=['mhsa', 'concat', 'mean_concat'],
                        help='Stage2 fusion in survfusion_separate: mhsa | concat | mean_concat')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of attention heads for mhsa fusion, ablation in {2, 4, 8}')
    # 实验4: CLIP 三对损失权重消融 (w_IT, w_IS, w_TS) ∈ {(1,1,1),(2,2,1),(3,3,1)}
    parser.add_argument('--clip_weight_IT', type=float, default=1.0, help='CLIP I-T pair weight')
    parser.add_argument('--clip_weight_IS', type=float, default=1.0, help='CLIP I-S pair weight')
    parser.add_argument('--clip_weight_TS', type=float, default=1.0, help='CLIP T-S pair weight')
    parser.add_argument('--label_dim', type=int, default=1, help='survival head output dim; Cox uses 1')
    parser.add_argument('--poe_variant', type=str, default='A', choices=['A', 'B', 'C'],
                        help='TriPoEVAE training variant: A=freeze+linear probe, B=pretrain+finetune, C=joint train')
    parser.add_argument('--poe_surv_lambda', type=float, default=1.0,
                        help='survival loss weight for TriPoEVAE variant C')
    parser.add_argument('--poe_encoder_lr_ratio', type=float, default=0.1,
                        help='TriPoEVAE-B stage2: encoder/PoE/decoder LR = --lr * this ratio')
    parser.add_argument('--poe_modality_dropout', type=float, default=0.2,
                        help='modality dropout probability in TriPoEVAE VAE training')
    parser.add_argument('--poe_decoder_hidden_dim', type=int, default=512,
                        help='decoder hidden dim for TriPoEVAE')
    parser.add_argument('--poe_mmhid', type=int, default=256,
                        help='fusion hidden dim for TriPoEVAE survival head')
    parser.add_argument('--poe_beta_target', type=float, default=1.0,
                        help='target / fixed beta for TriPoEVAE Jeffreys term')
    parser.add_argument('--betafix', action='store_true', default=False,
                        help='disable TriPoEVAE Jeffreys beta warmup and keep beta at --poe_beta_target')
    parser.add_argument('--alphafix', action='store_true', default=False,
                        help='disable learnable TriPoEVAE PoE alpha and use --alphapgc')
    parser.add_argument('--alphapgc', type=str, default=None,
                        help='fixed PoE alpha as pathology,gene,clinic, e.g. 0.5,0.3,0.2; required with --alphafix except in main_tune_optuna.py')
    parser.add_argument('--poe_transformer_layers', type=int, default=1,
                        help='number of transformer layers for gene/clinic encoders in TriPoEVAE')
    parser.add_argument('--concat_impute', type=str, default='zero', choices=['zero', 'mean'],
                        help='fill mode for concat ablations')
    parser.add_argument('--poe_scan_stage1_ckpts', action='store_true', default=False,
                        help='for TriPoEVAE-B only: save stage1 checkpoints on a fixed interval and launch one stage2 run per checkpoint')
    parser.add_argument('--poe_stage1_ckpt_interval', type=int, default=5,
                        help='epoch interval for saving TriPoEVAE-B stage1 checkpoints when checkpoint scan is enabled')
    parser.add_argument('--poe_stage1_studies', type=str, default='',
                        help='comma-separated study list for the cross-study stage1 pretraining model')
    parser.add_argument('--missing_mode', type=str, default='model_gen',
                        choices=['model_gen', 'unified_mask_csv'],
                        help='model_gen uses per-model dropout; unified_mask_csv uses a shared exp-group mask csv')
    parser.add_argument('--missing_pattern', type=str, default='',
                        help='unified_mask_csv rates as wsi,gene,clinic percents, e.g. 60,0,0 or 20,20,20')
    parser.add_argument('--missing_seed', type=int, default=None,
                        help='seed for unified/model_gen missing masks; defaults to --seed')
    parser.add_argument('--eval_modalities', type=str, default='off',
                        help='inference-only missing subsets over complete split modalities: off | all | P,C,G,PC,PG,CG,PCG | wsi,clinic')
    parser.add_argument('--wandb_mode', type=str, default='disabled',
                        choices=['disabled', 'offline', 'online'],
                        help='Weights & Biases logging mode')
    parser.add_argument('--wandb_project', type=str, default='SurvPGC',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/team name')
    parser.add_argument('--optuna_trials', type=int, default=20,
                        help='number of Optuna trials to run in main_tune_optuna.py')
    parser.add_argument('--optuna_fold', type=int, default=0,
                        help='single fold index used by the minimal Optuna tuner')
    parser.add_argument('--optuna_fold_mode', type=str, default='mean_cv',
                        choices=['single', 'mean_cv'],
                        help='single uses one fold; mean_cv averages validation c-index across selected folds')
    parser.add_argument('--optuna_storage', type=str, default=None,
                        help='Optuna storage URL, e.g. sqlite:///results/optuna/study.db')
    parser.add_argument('--optuna_study_name', type=str, default=None,
                        help='Optuna study name; defaults to a study/variant-based name')
    parser.add_argument('--optuna_sampler', type=str, default='tpe',
                        choices=['tpe', 'random'],
                        help='Optuna sampler type')
    parser.add_argument('--optuna_pruner', type=str, default='median',
                        choices=['median', 'none'],
                        help='Optuna pruner type')
    parser.add_argument('--optuna_direction', type=str, default='maximize',
                        choices=['maximize', 'minimize'],
                        help='Optuna optimization direction')
    parser.add_argument('--optuna_n_startup_trials', type=int, default=5,
                        help='number of startup trials before pruning activates')
    parser.add_argument('--optuna_n_warmup_steps', type=int, default=3,
                        help='number of warmup epochs before pruning activates')
    parser.add_argument('--optuna_studies', type=str, default='',
                        help='comma-separated studies for one Optuna trial, evaluated in listed order; empty uses --study')
    parser.add_argument('--optuna_analysis_log', type=str, default=None,
                        help='compact Optuna analysis log path; default is results/optuna/<study_name>.log')

    #---> model related
    parser.add_argument('--fusion', type=str, default=None, choices=['concat', 'bilinear'])
    parser.add_argument(
        '--selected_modalities',
        type=str,
        default='wsi,gene,clinic',
        choices=['wsi', 'gene', 'clinic', 'wsi,gene', 'wsi,clinic', 'gene,clinic', 'wsi,gene,clinic'],
        help='selected modalities for SurvTri models',
    )
    parser.add_argument('--modality', type=str, default="survpgc_f",
                        choices=[
                            # unimodal G (csv)
                            'mlp_gene', 'snn_gene',
                            # unimodal G foundation embedding
                            'mlp_gene_f', 'snn_gene_f',
                            # unimodal WSI
                            'abmil_wsi', 'mlp_wsi', 'transmil_wsi',
                            # unimodal C
                            'mlp_clinic_mean', 'mlp_clinic_flatten',
                            'snn_clinic_mean', 'snn_clinic_flatten',
                            'clinic_cox',
                            # multimodal WSI+G baselines
                            'porpoise', 'survpath', 'mcat',
                            # main model WSI+G+C
                            'survpgc_f',
                            # two-stage CLIP alignment + survival (WSI+G+C)
                            'survfusion_separate',
                            # ablation exp1: no CLIP, end-to-end survival only
                            'survfusion_noalign',
                            # ablation exp2: joint CLIP + survival loss
                            'survfusion_joint',
                            # trimodal concat / attention baselines
                            'survtri_snn_concat',
                            'survtri_snn_mhsa',
                            'survtri_mlp_concat',
                            'survtri_mlp_mhsa',
                            'survtri_poe_vae',
                            'survtri_poe_vae_b_kl',
                            'survtri_poe_vae_b_crossstage1',
                            'survtri_poe_vae_b_nopretrain',
                            'survtri_poe_vae_b_single',
                            'survtri_poe_vae_b_multi',
                            'survtri_poe_vae_b_film',
                            'survtri_poe_vae_a_film',
                            'survtri_poe_vae_c_single',
                            'survtri_poe_vae_c_single_enum',
                            'survtri_poe_vae_c_multi',
                            'survtri_poe_vae_c_film',
                            'survtri_poe_vae_c_film_kl',
                            'survtri_poe_vae_c_film_noenum',
                            'survtri_poe_vae_c_film_surv0',
                            'survtri_poe_vae_c_film_beta0',
                            'modality_concat',
                            'mvae_poe',
                            'mopoe',
                            'hgcn',
                            # ablation WSI+C
                            'survpc_f',
                            # ablation G+C
                            'survgc_f',
                        ])
    parser.add_argument('--return_attn', type=str, default=False, help="Used for heatmap drawing")
    parser.add_argument('--encoding_dim', type=int, default=1024, help='WSI encoding dim, default: 1024')
    parser.add_argument('--use_nystrom', action='store_true', default=False, help='Use Nystrom attentin in SurvPath.')

    args = parser.parse_args()

    inferred_paths = infer_standard_paths(
        args.study,
        '.',
        which_splits=args.which_splits,
        type_of_path=args.type_of_path,
        wsi_experiment=DEFAULT_WSI_EXPERIMENT,
        clinic_experiment=DEFAULT_CLINIC_EXPERIMENT,
        gene_experiment=DEFAULT_GENE_EXPERIMENT,
    )

    if args.label_file is None:
        args.label_file = str(inferred_paths['label_file'])
    if args.omics_dir is None:
        args.omics_dir = str(inferred_paths['omics_dir'])
    if args.split_dir is None:
        args.split_dir = str(inferred_paths['split_dir'])
    if args.data_root_dir is None:
        args.data_root_dir = str(inferred_paths['data_root_dir'])
    if args.clinic_dir is None:
        args.clinic_dir = str(inferred_paths['clinic_dir'])
    if args.gene_dir is None:
        args.gene_dir = str(inferred_paths['gene_dir'])
    if args.clinical_file is None:
        args.clinical_file = str(inferred_paths['clinical_file'])

    if args.modality in {
        "survtri_poe_vae_b_kl",
        "survtri_poe_vae_b_crossstage1",
        "survtri_poe_vae_b_nopretrain",
        "survtri_poe_vae_b_single",
        "survtri_poe_vae_b_multi",
        "survtri_poe_vae_b_film",
    }:
        args.poe_variant = "B"
    elif args.modality == "survtri_poe_vae_a_film":
        args.poe_variant = "A"
    elif args.modality in {
        "survtri_poe_vae_c_single",
        "survtri_poe_vae_c_single_enum",
        "survtri_poe_vae_c_multi",
        "survtri_poe_vae_c_film",
        "survtri_poe_vae_c_film_kl",
        "survtri_poe_vae_c_film_noenum",
        "survtri_poe_vae_c_film_surv0",
        "survtri_poe_vae_c_film_beta0",
    }:
        args.poe_variant = "C"

    if args.alphafix:
        from models.model_utils import parse_alphapgc
        if args.alphapgc is None:
            # Optuna samples alphapgc per trial after argparse. Regular training still needs weights now.
            if Path(sys.argv[0]).name != "main_tune_optuna.py":
                raise ValueError(
                    "alphafix requires alphapgc weights for pathology,gene,clinic, e.g. 0.5,0.3,0.2"
                )
        else:
            args.alphapgc = ",".join(f"{weight:g}" for weight in parse_alphapgc(args.alphapgc))
    else:
        args.alphapgc = None

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args


'''
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init
conda activate SurvPGC

python main.py \
  --study tcga_coad \
  --modality survtri_mlp_mhsa \
  --selected_modalities wsi,gene \
  --type_of_path combine \
  --data_root_dir SurvPGC_Workspace/tcga_coad/P/uni_v2 \
  --num_heads 8 \
  --k 1

  --fusion bilinear \
  --fusion_type mean_concat \

                            'survtri_snn_concat',
                            'survtri_snn_mhsa',
                            'survtri_mlp_concat',
                            'survtri_mlp_mhsa',
choices=['wsi,gene', 'wsi,clinic', 'gene,clinic', 'wsi,gene,clinic'],

'''


'''
# unimodal G (csv)
'mlp_gene', 'snn_gene',

# unimodal G foundation embedding
'mlp_gene_f', 'snn_gene_f',

# unimodal WSI
abmil_wsi', 'mlp_wsi', 'transmil_wsi',

# unimodal C
'mlp_clinic_mean', 'mlp_clinic_flatten',
'snn_clinic_mean', 'snn_clinic_flatten',
'clinic_cox',

# 多模态基线（WSI + G）
porpoise
survpath
        # SurvPath：基因组 + 病理图像，以pathway为单位做注意力
        # 本项目最直接的前身方法，论文Table1中的重要对比基线

survpath_f（无）
        # SurvPath的foundation model版本，结构同上
        # _f 后缀表示使用了foundation model（如UNI/CONCH）编码的特征

# 本项目主模型（WSI + G + C）
survpgc_f
survfusion_f

'survfusion_mlpcontact', 
'survfusion_mhsaconcat', 
'survfusion_noalign'
                            'survtri_snn_concat',
                            'survtri_snn_mhsa',
                            'survtri_mlp_concat',
                            'survtri_mlp_mhsa',
# 消融：WSI + C（去掉G）
survpc
survpc_f

# 消融：WSI + C 直接拼接（无注意力）
mlppc_concat

# 消融：G + C（去掉WSI）
survgc_f

# 消融：只用C
clinic_mlp
clinic_snn
clinic_cox
'''
