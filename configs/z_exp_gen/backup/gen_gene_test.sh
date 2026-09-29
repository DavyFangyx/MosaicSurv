#!/bin/bash
# exp_gen/gen_gene_test.sh
# 生成 gene_test 的所有 config：4 个 gene 特征 × 8 个模型 = 32 个 config
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

GENES=(
    scFoundation_embedding_cell_norm
    scFoundation_embedding_cell_raw
    scFoundation_embedding_gene_norm
    scFoundation_embedding_gene_raw
)

MODELS=(
    omics
    snn
    porpoise
    survpath
    mcat
    survpgc_f
    survgc_f
    survfusion_separate_mhsa
)

TYPE_OF_PATH=combine

seq=0
for gene in "${GENES[@]}"; do
    for preset in "${MODELS[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "gene__%03d__%s__%s.conf" "$seq" "$gene" "$preset")
        cat > "$OUT_DIR/$fname" <<EOF
EXP_GROUP=gene_test
RUN_NAME=${gene}_${TYPE_OF_PATH}
PRESET=$preset
GENE_EXPERIMENT=$gene
TYPE_OF_PATH=$TYPE_OF_PATH
BATCH_SIZE=1
EOF
    done
done

echo "Generated $seq configs in $OUT_DIR  (gene_test: ${#GENES[@]} × ${#MODELS[@]})"
