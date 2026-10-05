#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Table4（_Modality_Missing 子实验）MosaicSurv 消融生成：
# Table2 式的缺失模态评测（EVAL_MODALITIES=all + MISSING_MODE=model_gen，
# 每个模型只训一次三模态，推理期生成缺失模态并跑 7 个子集 eval）。
# 仅 A_readout 组（读出层变体，设计动机即缺失模态推理）+ 主模型同批次参照。
# 协议与 Table2 对齐：CLINIC=L0 / GENE=scFoundation_embedding_gene_raw / WSI=uni_v1，
# 公共超参统一来自 configs/z_exp_gen/mosaic_hparams.sh。
# 结果目录：results/Table4_Abaltion_Test/_Modality_Missing/{study}__L0__gene_raw__uni_v1/。

SCRIPT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 统一公共超参（与主模型一致）
# shellcheck disable=SC1091
source "$SCRIPT_DIR/mosaic_hparams.sh"

MOSAIC_EXTRA_LINES=$'LR='"$MOSAIC_LR"$'\nREG='"$MOSAIC_REG"$'\nPOE_SURV_LAMBDA='"$MOSAIC_POE_SURV_LAMBDA"$'\nPOE_BETA_TARGET='"$MOSAIC_POE_BETA_TARGET"$'\nPOE_MODALITY_DROPOUT='"$MOSAIC_POE_MODALITY_DROPOUT"$'\nPOE_MMHID='"$MOSAIC_POE_MMHID"$'\nBATCH_SIZE='"$MOSAIC_BATCH_SIZE"$'\nALPHAFIX='"$MOSAIC_ALPHAFIX"$'\nALPHAPGC='"$MOSAIC_ALPHAPGC"$'\nBETAFIX='"$MOSAIC_BETAFIX"

EXP_GROUP="${EXP_GROUP:-Table4_Abaltion_Test/_Modality_Missing}"

if [ -n "${STUDIES:-}" ]; then
    read -r -a studies <<< "$STUDIES"
else
    studies=(tcga_brca tcga_coad tcga_kirc tcga_kirp tcga_lihc)
fi

if [ -n "${PRESETS:-}" ]; then
    read -r -a presets <<< "$PRESETS"
else
    # A_readout 组（读出层变体）+ 主模型同批次参照
    presets=(
        mosaic_surv
        mosaic_surv_single
        mosaic_surv_single_enum
        mosaic_surv_noenum
        mosaic_surv_multi
    )
fi

CLINIC_EXPERIMENT="${CLINIC_EXPERIMENT:-L0}"
GENE_EXPERIMENT="${GENE_EXPERIMENT:-scFoundation_embedding_gene_raw}"
WSI_EXPERIMENT="${WSI_EXPERIMENT:-uni_v1}"
BATCH_SIZE="${BATCH_SIZE:-$MOSAIC_BATCH_SIZE}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
EVAL_MODALITIES="${EVAL_MODALITIES:-all}"
MISSING_MODE="${MISSING_MODE:-model_gen}"
BATCH_SIZE_STAGE1="${BATCH_SIZE_STAGE1:-128}"
MAX_EPOCHS_STAGE1="${MAX_EPOCHS_STAGE1:-10}"

OUT_DIR="${OUT_DIR:-$SCRIPT_ROOT/configs/queue}"
mkdir -p "$OUT_DIR"

create_conf() {
    local target="$1"
    local content="$2"
    if [ -e "$target" ]; then
        skipped=$((skipped + 1))
        return
    fi
    printf '%s\n' "$content" > "$target"
    created=$((created + 1))
}

seq=0
created=0
skipped=0

for study in "${studies[@]}"; do
    for preset in "${presets[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "table4_abaltion_test_modality_missing__%03d__%s__%s.conf" "$seq" "$study" "$preset")
        target="$OUT_DIR/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${study}__${CLINIC_EXPERIMENT}__gene_raw__${WSI_EXPERIMENT}
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$GENE_EXPERIMENT
WSI_EXPERIMENT=$WSI_EXPERIMENT
WANDB_MODE=disabled
BAG_LOSS=cox_surv
BATCH_SIZE=$BATCH_SIZE
MAX_EPOCHS=$MAX_EPOCHS
WARMUP_EPOCHS=$WARMUP_EPOCHS
BATCH_SIZE_STAGE1=$BATCH_SIZE_STAGE1
MAX_EPOCHS_STAGE1=$MAX_EPOCHS_STAGE1
EVAL_MODALITIES=$EVAL_MODALITIES
MISSING_MODE=$MISSING_MODE
$MOSAIC_EXTRA_LINES
EOF
)"
    done
done

echo "Generated $created new configs in $OUT_DIR"
echo "Total indexed configs this round: $seq"
echo "Skipped existing configs: $skipped"
echo "Studies: ${studies[*]}"
echo "Presets: ${presets[*]}"
echo "Experiment group: $EXP_GROUP"
echo "Run name pattern: {study}__${CLINIC_EXPERIMENT}__gene_raw__${WSI_EXPERIMENT}"
