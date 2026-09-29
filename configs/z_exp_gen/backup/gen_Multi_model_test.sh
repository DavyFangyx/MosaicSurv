#!/bin/bash
# configs/z_exp_gen/gen_Multi_model_test.sh
# 生成 Multi_model_test 的 config：按注册表启用的数据集批量展开
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

EXP_GROUP="Multi_model_test"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

HEADS=(
    4
    8
    16
)

MODELS=(
    survpgc_f
    survtri_mlp_concat
    survtri_mlp_mhsa
    survtri_snn_concat
    survtri_snn_mhsa
)

CLINIC_EXPERIMENT="L4"
GENE_EXPERIMENT="scFoundation_embedding_gene_norm"
WSI_EXPERIMENT="uni_v2"
BATCH_SIZE=1

seq=0
created=0
skipped=0

for study in "${STUDIES[@]}"; do
    for heads in "${HEADS[@]}"; do
        for preset in "${MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "multi_model_test__%03d__%s__h%02d__%s.conf" "$seq" "$study" "$heads" "$preset")
            target="$OUT_DIR/$fname"

            if [ -e "$target" ]; then
                skipped=$((skipped + 1))
                continue
            fi

            cat > "$target" <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${study}__L4__gene_norm__uni_v2__h$(printf "%02d" "$heads")
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$GENE_EXPERIMENT
WSI_EXPERIMENT=$WSI_EXPERIMENT
NUM_HEADS=$heads
BATCH_SIZE=$BATCH_SIZE
EOF
            created=$((created + 1))
        done
    done
done

echo "Generated $created new configs in $OUT_DIR  (Multi_model_test total: $seq, skipped existing: $skipped)"
