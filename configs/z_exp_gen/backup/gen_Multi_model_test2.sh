#!/bin/bash
# configs/z_exp_gen/gen_Multi_model_test2.sh
# 生成 Multi_model_test2 的 config：按注册表启用的数据集批量展开
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

EXP_GROUP="Multi_model_test2"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

MODELS=(
    survtri_snn_concat
    survtri_mlp_concat
    survtri_mlp_mhsa
)

MODALITIES=(
    wsi,gene
    wsi,clinic
    gene,clinic
)

CLINIC_EXPERIMENT="L4"
GENE_EXPERIMENT="scFoundation_embedding_gene_norm"
WSI_EXPERIMENT="uni_v2"
NUM_HEADS=8
BATCH_SIZE=1

seq=0
created=0
skipped=0

for study in "${STUDIES[@]}"; do
    for selected_modalities in "${MODALITIES[@]}"; do
        modality_tag="${selected_modalities//,/_}"

        for preset in "${MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "multi_model_test2__%03d__%s__%s__%s.conf" "$seq" "$study" "$modality_tag" "$preset")
            target="$OUT_DIR/$fname"

            if [ -e "$target" ]; then
                skipped=$((skipped + 1))
                continue
            fi

            cat > "$target" <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${study}__${modality_tag}__L4__gene_norm__uni_v2
PRESET=$preset
STUDY=$study
SELECTED_MODALITIES=$selected_modalities
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$GENE_EXPERIMENT
WSI_EXPERIMENT=$WSI_EXPERIMENT
NUM_HEADS=$NUM_HEADS
BATCH_SIZE=$BATCH_SIZE
EOF
            created=$((created + 1))
        done
    done
done

echo "Generated $created new configs in $OUT_DIR  (Multi_model_test2 total: $seq, skipped existing: $skipped)"
