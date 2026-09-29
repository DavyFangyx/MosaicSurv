#!/bin/bash
# bg_tune.sh — 后台启动 Optuna 调参
# 不走 configs/queue，也不改 run.sh / scheduler.sh。
# Alpha 不从命令行固定：A/B/C 每个 trial 采样
#   WSI/pathology 0.45-0.75, gene 0.05-0.20, clinic 0.20-0.40
# 再 L1 归一化成 --alphapgc，并打开 --alphafix。Model B 示例可显式写 --alphafix，
# 与设计说明 12.2 一致；不要再传 --alphapgc，权重仍由 trial 采样覆盖。
#
# 多数据集：每个 trial 按 --optuna_studies 出现顺序依次跑 5-fold。
# 某个数据集 5-fold mean val C-index 低于代码里的 OPTUNA_STUDY_MIN_CINDEX[study]
# 就立刻剪掉；5 个数据集 mean 的再平均只作为 objective，不做阈值校验。
# 改阈值：编辑 utils/optuna_utils.py 里的 OPTUNA_STUDY_MIN_CINDEX。
# 用法:
#   CUDA_VISIBLE_DEVICES=N bash bg_tune.sh <日志文件名> [main_tune_optuna.py 参数...]
# MODELB示例:
# Model B 搜索空间见 utils/optuna_utils.py::sample_survtri_poe_vae_model_b，
# 含 alphapgc 以及 lr / lr_stage1 / reg / poe_beta_target / poe_modality_dropout /
# poe_mmhid / poe_decoder_hidden_dim / poe_encoder_lr_ratio ∈ {0.01, 0.1, 1, 10}。
# 可显式传 --alphafix（保险，与 12.2 一致）；不要传 --alphapgc，权重由 trial 采样。
# 数据集顺序与阈值（5-fold mean val C-index 下限）：
#   kirp 0.8115, coad 0.6611, kirc 0.6895, brca 0.6588, lihc 0.7040
# 阈值在 utils/optuna_utils.py OPTUNA_STUDY_MIN_CINDEX 里面调节
''' 
CUDA_VISIBLE_DEVICES=7 bash bg_tune.sh gpu6_optunaB.log \
    --optuna_studies kirp,coad,kirc,brca,lihc \
    --modality survtri_poe_vae \
    --poe_variant B \
    --betafix \
    --alphafix \
    --optuna_fold_mode mean_cv \
    --optuna_pruner median \
    --optuna_n_startup_trials 5 \
    --optuna_trials 100 \
    --bag_loss cox_surv \
    --label_dim 1 \
    --max_epochs_stage1 10 \
    --max_epochs 25 \
    --warmup_epochs 3 \
    --k 5 \
    --batch_size 128 \
    --wandb_mode online \
    --wandb_project POE_MODELB_TUNE
''' 

# MODELC示例:
''' 
CUDA_VISIBLE_DEVICES=7 bash bg_tune.sh gpu7_optunaC.log \
    --optuna_studies kirp,lihc,coad,kirc,brca \
    --modality survtri_poe_vae \
    --poe_variant C \
    --betafix \
    --optuna_fold_mode mean_cv \
    --optuna_pruner median \
    --optuna_n_startup_trials 5 \
    --optuna_trials 15 \
    --bag_loss cox_surv \
    --label_dim 1 \
    --max_epochs 20 \
    --warmup_epochs 3 \
    --k 5 \
    --wandb_mode online \
    --wandb_project POE_MODELC_TUNE
''' 

# MODELC_FILM示例:
# 搜索空间见 utils/optuna_utils.py::sample_survtri_poe_vae_model_c_film：
# alpha 默认使用模型中的可学习 modality logits；只有固定 alpha 时才传 --alphafix；
# 其余 lr / reg / poe_surv_lambda / poe_beta_target / poe_modality_dropout /
# poe_mmhid ∈ {128, 256, 384}、batch_size ∈ {16, 32, 64, 128} 也都是离散采样。
# poe_variant 会被强制成 C。C-film 的 val_cindex / 选模 / 剪枝都用完整 PGC
# 主头 C-index，不再用 7-pattern 平均。完整训练日志仍写第一个参数；
# 超参/分数分析日志默认写 results/optuna/<study_name>.log。
# 停止: kill 3407781
''' 
CUDA_VISIBLE_DEVICES=7 bash bg_tune.sh gpu7_optunaCfilm.log \
    --optuna_studies kirp,lihc,coad,kirc,brca \
    --modality survtri_poe_vae_c_film \
    --optuna_fold_mode mean_cv \
    --optuna_pruner median \
    --optuna_n_startup_trials 5 \
    --optuna_trials 100 \
    --bag_loss cox_surv \
    --label_dim 1 \
    --max_epochs 20 \
    --warmup_epochs 3 \
    --k 5 \
    --wandb_mode online \
    --wandb_project POE_MODELC_FILM_TUNE

# 加上后人为固定
    --betafix \
    --alphafix \
''' 
set -euo pipefail

# Bound native math thread pools; each GPU may have multiple tuning processes.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

if [ "$#" -lt 1 ]; then
    echo "用法: CUDA_VISIBLE_DEVICES=N bash bg_tune.sh <日志文件名> [main_tune_optuna.py 参数...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$1"
shift

# ========== 可配置参数（命令行同名参数会覆盖这里） ==========
# 训练轮数
MAX_EPOCHS_STAGE1=40
MAX_EPOCHS=20
WARMUP_EPOCHS=3

# 每个 trial 按这个顺序跑数据集。路径由 infer_standard_paths 绑定：
# P/uni_v1、C/L0、G/scFoundation_embedding_cell_norm。不要再写死 LIHC 路径。
OPTUNA_STUDIES=kirp,coad,kirc,brca,lihc

# 5 折交叉验证
K=5

# WandB：disabled 不上传；offline 只写本地；online 上传训练曲线
WANDB_MODE=online
WANDB_PROJECT=SurvPGC_MultiVAE
WANDB_ENTITY=davyfangyuxuan-nanjing-university-of-aeronautics-and-ast

# Optuna：median 对 5 折 mean_cv 最稳；可改成 none 关闭剪枝
OPTUNA_PRUNER=median
OPTUNA_N_STARTUP_TRIALS=5
# ============================================================

CONDA_ENV_NAME="${CONDA_ENV_NAME:-SurvPGC}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -n "$CONDA_ENV_NAME" ] && [ "$CONDA_ENV_NAME" != "无" ] && [ "$CONDA_ENV_NAME" != "none" ]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[bg_tune.sh] conda not found, but CONDA_ENV_NAME=$CONDA_ENV_NAME was requested" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
fi

if [ "$#" -eq 0 ]; then
    echo "[bg_tune.sh] missing main_tune_optuna.py arguments" >&2
    echo "用法: CUDA_VISIBLE_DEVICES=N bash bg_tune.sh <日志文件名> [main_tune_optuna.py 参数...]" >&2
    exit 2
fi

TUNE_ARGS=(
    --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
    --max_epochs "$MAX_EPOCHS"
    --warmup_epochs "$WARMUP_EPOCHS"
    --optuna_studies "$OPTUNA_STUDIES"
    --k "$K"
    --optuna_pruner "$OPTUNA_PRUNER"
    --optuna_n_startup_trials "$OPTUNA_N_STARTUP_TRIALS"
    --wandb_mode "$WANDB_MODE"
    --wandb_project "$WANDB_PROJECT"
)
if [ -n "$WANDB_ENTITY" ]; then
    TUNE_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi

ANALYSIS_LOG=""
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    if [ "$arg" = "--optuna_analysis_log" ]; then
        next=$((i + 1))
        ANALYSIS_LOG="${!next}"
        break
    fi
done

nohup "$PYTHON_BIN" -u main_tune_optuna.py "${TUNE_ARGS[@]}" "$@" > "$LOG_FILE" 2>&1 &
PID=$!

echo "PID: $PID"
echo "训练日志: $SCRIPT_DIR/$LOG_FILE"
if [ -n "$ANALYSIS_LOG" ]; then
    echo "分析日志: $ANALYSIS_LOG"
else
    echo "分析日志: $SCRIPT_DIR/results/optuna/<study_name>.log"
fi
echo "epochs: stage1=$MAX_EPOCHS_STAGE1 max=$MAX_EPOCHS warmup=$WARMUP_EPOCHS"
echo "k=$K  optuna_pruner=$OPTUNA_PRUNER  optuna_n_startup_trials=$OPTUNA_N_STARTUP_TRIALS"
echo "optuna_studies=$OPTUNA_STUDIES"
echo "wandb: mode=$WANDB_MODE project=$WANDB_PROJECT"
echo "停止: kill $PID"
