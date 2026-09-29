#!/bin/bash
# run.sh — 唯一执行入口（单跑 / 批量都走这里）
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: bash run.sh <config.conf>"
    exit 2
fi

CONFIG_INPUT="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "$CONFIG_INPUT" ]; then
    echo "[run.sh] config not found: $CONFIG_INPUT"
    exit 2
fi

CONFIG_ABS="$(cd "$(dirname "$CONFIG_INPUT")" && pwd)/$(basename "$CONFIG_INPUT")"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/configs/defaults.conf"
# shellcheck disable=SC1090
source "$CONFIG_ABS"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/configs/presets.sh"

: "${EXP_GROUP:?EXP_GROUP is required in config}"
: "${RUN_NAME:?RUN_NAME is required in config}"
: "${PRESET:?PRESET is required in config}"

activate_conda_if_needed() {
    if [ -z "${CONDA_ENV_NAME:-}" ] || [ "${CONDA_ENV_NAME}" = "无" ] || [ "${CONDA_ENV_NAME}" = "none" ]; then
        return 0
    fi

    if ! command -v conda >/dev/null 2>&1; then
        echo "[run.sh] conda not found, but CONDA_ENV_NAME=$CONDA_ENV_NAME was requested" >&2
        exit 2
    fi

    # shellcheck disable=SC1090
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
}

require_command() {
    local exe_name="$1"
    if ! command -v "$exe_name" >/dev/null 2>&1; then
        echo "[run.sh] executable not found in PATH: $exe_name" >&2
        exit 2
    fi
}

resolve_path_or_default() {
    local current_value="$1"
    local default_value="$2"
    if [ -n "$current_value" ]; then
        printf '%s\n' "$current_value"
    else
        printf '%s\n' "$default_value"
    fi
}

require_path() {
    local target_path="$1"
    local description="$2"
    if [ ! -e "$target_path" ]; then
        echo "[run.sh] missing ${description}: $target_path" >&2
        exit 2
    fi
}

infer_wsi_encoding_dim() {
    local data_root_dir="$1"
    local python_bin="$2"
    "$python_bin" -c '
import glob
import os
import sys
import torch

data_root_dir = sys.argv[1]
pt_files = sorted(glob.glob(os.path.join(data_root_dir, "*.pt")))
if not pt_files:
    raise FileNotFoundError(f"No .pt files found under {data_root_dir}")

sample = torch.load(pt_files[0], map_location="cpu")
if sample.ndim < 2:
    raise ValueError(f"Expected WSI embedding tensor with ndim >= 2, got shape {tuple(sample.shape)} from {pt_files[0]}")

print(int(sample.shape[-1]))
' "$data_root_dir"
}

write_effective_config() {
    local out_file="$1"
    local key
    local tracked_keys=(
        CONDA_ENV_NAME
        PYTHON_BIN
        EXP_GROUP
        RUN_NAME
        PRESET
        MODEL
        RESULTS_SUBDIR
        RESULTS_BASE
        STUDY
        TASK
        N_CLASSES
        TESTING
        WHICH_SPLITS
        WSI_EXPERIMENT
        GENE_EXPERIMENT
        CLINIC_EXPERIMENT
        TYPE_OF_PATH
        LABEL_FILE_PATH
        OMICS_DIR_PATH
        DATA_ROOT_DIR_PATH
        CLINIC_DIR_PATH
        GENE_DIR_PATH
        SPLIT_DIR_PATH
        CLINICAL_FILE
        NUM_PATCHES
        LABEL_COL
        WSI_PROJECTION_DIM
        ENCODING_LAYER_1_DIM
        ENCODING_LAYER_2_DIM
        ENCODER_DROPOUT
        SINGLE_MODEL_SIZE
        SINGLE_USE_INPUT_LN
        ENCODING_DIM
        K
        K_START
        K_END
        MAX_EPOCHS
        LR
        SEED
        OPT
        REG_TYPE
        WEIGHTED_SAMPLE
        BATCH_SIZE
        BAG_LOSS
        ALPHA_SURV
        BETA_SURV
        REG
        LR_SCHEDULER
        WARMUP_EPOCHS
        LR_STAGE1
        MAX_EPOCHS_STAGE1
        BATCH_SIZE_STAGE1
        FUSION
        SELECTED_MODALITIES
        FUSION_TYPE
        NUM_HEADS
        CLIP_LAMBDA
        CLIP_WEIGHT_IT
        CLIP_WEIGHT_IS
        CLIP_WEIGHT_TS
        LABEL_DIM
        POE_VARIANT
        POE_SURV_LAMBDA
        POE_ENCODER_LR_RATIO
        POE_MODALITY_DROPOUT
        MISSING_MODE
        MISSING_PATTERN
        MISSING_SEED
        EVAL_MODALITIES
        POE_DECODER_HIDDEN_DIM
        POE_MMHID
        POE_BETA_TARGET
        BETAFIX
        ALPHAFIX
        ALPHAPGC
        POE_TRANSFORMER_LAYERS
        POE_STAGE1_STUDIES
        WANDB_MODE
        WANDB_PROJECT
        WANDB_ENTITY
        RETURN_ATTN
        USE_NYSTROM
        CONCAT_WSI
        CONCAT_IMPUTE
        DATA_PACK_DIR_PATH
        LABEL_FILE
        OMICS_DIR
        DATA_ROOT_DIR
        CLINIC_DIR
        GENE_DIR
        SPLIT_DIR
        DATA_PACK_DIR
        CUDA_VISIBLE_DEVICES
    )

    {
        echo "# merged from configs/defaults.conf + $CONFIG_ABS + configs/presets.sh"
        for key in "${tracked_keys[@]}"; do
            if [ "${!key+x}" = x ]; then
                printf '%s=%q\n' "$key" "${!key}"
            fi
        done

        printf 'EXTRA_ARGS=('
        if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
            local arg
            for arg in "${EXTRA_ARGS[@]}"; do
                printf ' %q' "$arg"
            done
            printf ' '
        fi
        printf ')\n'
    } > "$out_file"
}

append_bool_flag() {
    local var_name="$1"
    local flag_name="$2"
    if [ "${!var_name}" = "true" ]; then
        cmd+=("$flag_name")
    fi
}

activate_conda_if_needed
require_command "$PYTHON_BIN"

STUDY_SUBTYPE="${STUDY#tcga_}"
LABEL_FILE="$(resolve_path_or_default "${LABEL_FILE_PATH}" "$SCRIPT_DIR/datasets_csv/metadata/${STUDY}.csv")"
OMICS_DIR="$(resolve_path_or_default "${OMICS_DIR_PATH}" "$SCRIPT_DIR/datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY_SUBTYPE}")"
DATA_ROOT_DIR="$(resolve_path_or_default "${DATA_ROOT_DIR_PATH}" "$SCRIPT_DIR/SurvPGC_Workspace/${STUDY}/P/${WSI_EXPERIMENT}")"
CLINIC_DIR="$(resolve_path_or_default "${CLINIC_DIR_PATH}" "$SCRIPT_DIR/SurvPGC_Workspace/${STUDY}/C/${CLINIC_EXPERIMENT}")"
GENE_DIR="$(resolve_path_or_default "${GENE_DIR_PATH}" "$SCRIPT_DIR/SurvPGC_Workspace/${STUDY}/G/${GENE_EXPERIMENT}")"
CLINICAL_FILE="$SCRIPT_DIR/datasets_csv/clinical_data/${STUDY}_clinical.csv"
SPLIT_DIR="$(resolve_path_or_default "${SPLIT_DIR_PATH}" "$SCRIPT_DIR/splits/${WHICH_SPLITS}/${STUDY}")"
DATA_PACK_DIR="${DATA_PACK_DIR_PATH:-}"

EXTRA_ARGS=()
apply_preset "$PRESET"

: "${MODEL:?preset did not set MODEL}"
: "${RESULTS_SUBDIR:?preset did not set RESULTS_SUBDIR}"

OUT_DIR="$SCRIPT_DIR/${RESULTS_BASE#./}/${EXP_GROUP}/${RUN_NAME}/${RESULTS_SUBDIR}"
mkdir -p "$OUT_DIR"

if [ -f "$OUT_DIR/.done" ]; then
    echo "[SKIP] already done: $OUT_DIR"
    exit 0
fi

require_path "$SPLIT_DIR" "split_dir"

require_path "$LABEL_FILE" "label_file"
require_path "$CLINICAL_FILE" "clinical_file"
require_path "$OMICS_DIR" "omics_dir"
if [ "$MODEL" = "hgcn" ]; then
    if [ "$WSI_EXPERIMENT" = "uni_v1" ]; then
        echo "[run.sh] HGCN uses native KimiaNet pkl; overriding WSI_EXPERIMENT=uni_v1 -> kimianet"
        WSI_EXPERIMENT="kimianet"
        DATA_ROOT_DIR="$SCRIPT_DIR/SurvPGC_Workspace/${STUDY}/P/${WSI_EXPERIMENT}"
    fi
    case "$GENE_EXPERIMENT" in
        scFoundation*)
            echo "[run.sh] HGCN uses native MSigDB pkl; overriding GENE_EXPERIMENT=$GENE_EXPERIMENT -> msigdb_gsea_families"
            GENE_EXPERIMENT="msigdb_gsea_families"
            GENE_DIR="$SCRIPT_DIR/SurvPGC_Workspace/${STUDY}/G/${GENE_EXPERIMENT}"
            ;;
    esac
    remap_to_hgcn_data() {
        local src_path="$1"
        "$PYTHON_BIN" -c 'from pathlib import Path; import sys; repo=Path(sys.argv[1]).resolve(); src=Path(sys.argv[2]).expanduser(); src = src if src.is_absolute() else (repo/src); src=src.resolve(); workspace=(repo/"SurvPGC_Workspace").resolve(); parts=src.relative_to(workspace).parts; parts = parts[1:] if parts and parts[0]=="hgcn data" else parts; print(workspace / "hgcn data" / Path(*parts))' "$SCRIPT_DIR" "$src_path"
    }
    HGCN_DATA_ROOT_DIR="$(remap_to_hgcn_data "$DATA_ROOT_DIR")"
    HGCN_CLINIC_DIR="$(remap_to_hgcn_data "$CLINIC_DIR")"
    HGCN_GENE_DIR="$(remap_to_hgcn_data "$GENE_DIR")"
    require_path "$HGCN_CLINIC_DIR" "HGCN remapped clinic_dir"
    require_path "$HGCN_GENE_DIR" "HGCN remapped gene_dir"
    if [ ! -d "$HGCN_DATA_ROOT_DIR" ] || [ ! -f "$HGCN_DATA_ROOT_DIR/t_img_fea.pkl" ]; then
        echo "[run.sh] missing HGCN WSI pkl under $HGCN_DATA_ROOT_DIR; treating WSI as absent" >&2
    fi
    if [ ! -f "$HGCN_CLINIC_DIR/x_cli.pkl" ]; then
        echo "[run.sh] missing HGCN clinic pkl: $HGCN_CLINIC_DIR/x_cli.pkl" >&2
        echo "[run.sh] run: python SurvPGC_Workspace/generate_hgcn_clinic.py --study $STUDY" >&2
        exit 2
    fi
    if [ ! -f "$HGCN_GENE_DIR/t_rna_fea.pkl" ]; then
        echo "[run.sh] missing HGCN gene pkl: $HGCN_GENE_DIR/t_rna_fea.pkl" >&2
        echo "[run.sh] run: python SurvPGC_Workspace/generate_hgcn_rna_graph.py --study $STUDY" >&2
        exit 2
    fi
    DATA_ROOT_DIR="$HGCN_DATA_ROOT_DIR"
    CLINIC_DIR="$HGCN_CLINIC_DIR"
    GENE_DIR="$HGCN_GENE_DIR"
    ENCODING_DIM="${ENCODING_DIM:-1024}"
else
    require_path "$DATA_ROOT_DIR" "data_root_dir"
    require_path "$CLINIC_DIR" "clinic_dir"
    require_path "$GENE_DIR" "gene_dir"

    INFERRED_WSI_DIM="$(infer_wsi_encoding_dim "$DATA_ROOT_DIR" "$PYTHON_BIN")"
    if [ -z "${ENCODING_DIM:-}" ] || [ "$ENCODING_DIM" = "auto" ]; then
        ENCODING_DIM="$INFERRED_WSI_DIM"
    elif [ "$ENCODING_DIM" != "$INFERRED_WSI_DIM" ]; then
        echo "[run.sh] override ENCODING_DIM from $ENCODING_DIM to inferred $INFERRED_WSI_DIM for $DATA_ROOT_DIR"
        ENCODING_DIM="$INFERRED_WSI_DIM"
    fi
fi

cp "$CONFIG_ABS" "$OUT_DIR/config.snapshot"
git -C "$SCRIPT_DIR" rev-parse HEAD > "$OUT_DIR/git.sha" 2>/dev/null || echo "n/a" > "$OUT_DIR/git.sha"
date +'%F %T %z' > "$OUT_DIR/started_at"
write_effective_config "$OUT_DIR/effective_config.txt"

cmd=(
    "$PYTHON_BIN" -u main.py
    --study "$STUDY"
    --task "$TASK"
    --n_classes "$N_CLASSES"
    --results_dir "$RESULTS_BASE"
    --exp_group "$EXP_GROUP"
    --run_name "$RUN_NAME"
    --type_of_path "$TYPE_OF_PATH"
    --data_root_dir "$DATA_ROOT_DIR"
    --label_file "$LABEL_FILE"
    --omics_dir "$OMICS_DIR"
    --clinical_file "$CLINICAL_FILE"
    --clinic_dir "$CLINIC_DIR"
    --gene_dir "$GENE_DIR"
    --num_patches "$NUM_PATCHES"
    --label_col "$LABEL_COL"
    --wsi_projection_dim "$WSI_PROJECTION_DIM"
    --encoding_layer_1_dim "$ENCODING_LAYER_1_DIM"
    --encoding_layer_2_dim "$ENCODING_LAYER_2_DIM"
    --encoder_dropout "$ENCODER_DROPOUT"
    --single_model_size "$SINGLE_MODEL_SIZE"
    --k "$K"
    --k_start "$K_START"
    --k_end "$K_END"
    --split_dir "$SPLIT_DIR"
    --which_splits "$WHICH_SPLITS"
    --max_epochs "$MAX_EPOCHS"
    --lr "$LR"
    --seed "$SEED"
    --opt "$OPT"
    --reg_type "$REG_TYPE"
    --batch_size "$BATCH_SIZE"
    --bag_loss "$BAG_LOSS"
    --alpha_surv "$ALPHA_SURV"
    --beta_surv "$BETA_SURV"
    --reg "$REG"
    --lr_scheduler "$LR_SCHEDULER"
    --warmup_epochs "$WARMUP_EPOCHS"
    --wandb_mode "$WANDB_MODE"
    --wandb_project "$WANDB_PROJECT"
    --selected_modalities "$SELECTED_MODALITIES"
    --missing_mode "$MISSING_MODE"
    --modality "$MODEL"
    --encoding_dim "$ENCODING_DIM"
    "${EXTRA_ARGS[@]}"
)

if [ -n "${WANDB_ENTITY}" ]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
fi

if [ -n "${MISSING_PATTERN}" ]; then
    cmd+=(--missing_pattern "$MISSING_PATTERN")
fi

if [ -n "${MISSING_SEED}" ]; then
    cmd+=(--missing_seed "$MISSING_SEED")
fi

if [ -n "${EVAL_MODALITIES}" ] && [ "${EVAL_MODALITIES}" != "off" ]; then
    cmd+=(--eval_modalities "$EVAL_MODALITIES")
fi

append_bool_flag TESTING --testing
append_bool_flag WEIGHTED_SAMPLE --weighted_sample
append_bool_flag USE_NYSTROM --use_nystrom
append_bool_flag SINGLE_USE_INPUT_LN --single_use_input_ln
append_bool_flag BETAFIX --betafix
append_bool_flag ALPHAFIX --alphafix

if [ -n "${ALPHAPGC}" ]; then
    cmd+=(--alphapgc "$ALPHAPGC")
fi

if [ "${RETURN_ATTN}" = "true" ]; then
    cmd+=(--return_attn True)
fi

if [ -n "${DATA_PACK_DIR}" ]; then
    cmd+=(--data_pack_dir "$DATA_PACK_DIR")
fi

if [ -n "${FUSION}" ] && [ "$MODEL" = "porpoise" -o "$MODEL" = "mcat" ]; then
    :
fi

CMD_LINE="$(printf '%q ' "${cmd[@]}")"
CMD_LINE="${CMD_LINE% }"

{
    echo "[CMD] $CMD_LINE"
    echo "[WSI] data_root_dir=$DATA_ROOT_DIR encoding_dim=$ENCODING_DIM"
    echo "============================================================"
    "${cmd[@]}"
} 2>&1 | tee "$OUT_DIR/run.log"
status=${PIPESTATUS[0]}

date +'%F %T %z' > "$OUT_DIR/ended_at"

if [ "$status" -eq 0 ]; then
    touch "$OUT_DIR/.done"
    echo "[DONE] ${EXP_GROUP}/${RUN_NAME}/${RESULTS_SUBDIR}"
    exit 0
fi

echo "[FAIL] ${EXP_GROUP}/${RUN_NAME}/${RESULTS_SUBDIR} exit=$status"
exit "$status"
