#!/bin/bash
# 历史 FigB 配置生成器，不属于当前实验计划；不再生成实验说明。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

CLINIC_EXPERIMENTS=(
    L0
    L1
    L2
    L3
    L4
    L5
    D0
    D1
    D2
    D3
    D4
    D5
)

MODELS=(
    mlp_clinic_mean
    mlp_clinic_flatten
    snn_clinic_mean
    snn_clinic_flatten
)

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

for study in "${STUDIES[@]}"; do
    for clinic in "${CLINIC_EXPERIMENTS[@]}"; do
        for preset in "${MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "figb_clinic_prompt__%03d__%s__%s__%s.conf" "$seq" "$study" "$clinic" "$preset")
            target="$OUT_DIR/$fname"
            create_conf "$target" "$(cat <<EOF
EXP_GROUP='FigB_Clinical Prompt Test'
RUN_NAME=${study}__${clinic}
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$clinic
WSI_EXPERIMENT=uni_v1
GENE_EXPERIMENT=scFoundation_embedding_gene_raw
WHICH_SPLITS=5foldcv
K=5
WANDB_MODE=disabled
EOF
)"
        done
    done
done


echo "Enabled studies: ${#STUDIES[@]}"
printf 'Studies:'
for study in "${STUDIES[@]}"; do
    printf ' %s' "$study"
done
printf '\n'

echo "Generated $created new configs in $OUT_DIR"
echo "Total indexed configs this round: $seq"
echo "Skipped existing configs: $skipped"
echo "Expected result folders:"
echo "  results/FigB_Clinical Prompt Test/<study>__<Lx_or_Dx>/<clinic_model>/"
