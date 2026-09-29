#!/bin/bash
# exp_gen/gen_Gene_singile.sh
# 生成 Gene_singile 的 config：2 个单模态 G 模型，使用 defaults.conf 默认值
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

MODELS=(
    omics
    snn
)

seq=0
for preset in "${MODELS[@]}"; do
    seq=$((seq + 1))
    fname=$(printf "genesingle__%03d__%s.conf" "$seq" "$preset")
    cat > "$OUT_DIR/$fname" <<EOF
EXP_GROUP=Gene_singile
RUN_NAME=default
PRESET=$preset
EOF
done

echo "Generated $seq configs in $OUT_DIR  (Gene_singile: 1 × ${#MODELS[@]})"
