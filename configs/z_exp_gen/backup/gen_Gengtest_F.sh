#!/bin/bash
# exp_gen/gen_Gengtest_F.sh
# 生成 Gengtest_F 的 config：按注册表启用的数据集批量展开
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

GENES=(
    scFoundation_embedding_cell_norm
    scFoundation_embedding_cell_raw
    scFoundation_embedding_gene_norm
    scFoundation_embedding_gene_raw
)

MODELS=(
    mlp_gene_f
    snn_gene_f
)

seq=0
created=0
skipped=0
for study in "${STUDIES[@]}"; do
    for gene in "${GENES[@]}"; do
        for preset in "${MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "gengtest_f__%03d__%s__%s__%s.conf" "$seq" "$study" "$gene" "$preset")
            target="$OUT_DIR/$fname"

            if [ -e "$target" ]; then
                skipped=$((skipped + 1))
                continue
            fi

            cat > "$target" <<EOF
EXP_GROUP=Gengtest_F
RUN_NAME=${study}__${gene}
PRESET=$preset
STUDY=$study
GENE_EXPERIMENT=$gene
EOF
            created=$((created + 1))
        done
    done
done

echo "Generated $created new configs in $OUT_DIR  (Gengtest_F total: $seq, skipped existing: $skipped)"
