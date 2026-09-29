#!/bin/bash
# 生成 SurvTriPoEVAE A/B/C 及 B_nopretrain 消融的批量配置
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

EXP_GROUP="${EXP_GROUP:-SurvTriPoEVAE_ABC}"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

PRESETS=(
    survtri_poe_vae_A
    survtri_poe_vae_B
    survtri_poe_vae_B_nopretrain
    survtri_poe_vae_C
)

CLINIC_EXPERIMENT="${CLINIC_EXPERIMENT:-L4}"
GENE_EXPERIMENT="${GENE_EXPERIMENT:-scFoundation_embedding_cell_norm}"
WSI_EXPERIMENT="${WSI_EXPERIMENT:-uni_v1}"
WANDB_MODE="${WANDB_MODE:-online}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_SIZE_STAGE1="${BATCH_SIZE_STAGE1:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
MAX_EPOCHS_STAGE1="${MAX_EPOCHS_STAGE1:-10}"

seq=0
created=0
skipped=0

for study in "${STUDIES[@]}"; do
    for preset in "${PRESETS[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "poe_abc__%03d__%s__%s.conf" "$seq" "$study" "$preset")
        target="$OUT_DIR/$fname"

        if [ -e "$target" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        cat > "$target" <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${study}__L4__cell_norm__uni_v1__${preset}
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$GENE_EXPERIMENT
WSI_EXPERIMENT=$WSI_EXPERIMENT
WANDB_MODE=$WANDB_MODE
WANDB_PROJECT=${WANDB_PROJECT:-SurvPGC_MultiVAE}
BATCH_SIZE=$BATCH_SIZE
BATCH_SIZE_STAGE1=$BATCH_SIZE_STAGE1
MAX_EPOCHS=$MAX_EPOCHS
MAX_EPOCHS_STAGE1=$MAX_EPOCHS_STAGE1
EOF
        created=$((created + 1))
    done
done

echo "Generated $created new configs in $OUT_DIR (total indexed: $seq, skipped existing: $skipped)"
