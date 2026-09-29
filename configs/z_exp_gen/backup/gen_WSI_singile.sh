#!/bin/bash
# exp_gen/gen_WSI_singile.sh
# 生成 WSI_singile 的 config：3 个单模态 WSI 模型，使用 defaults.conf 默认值
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

MODELS=(
    mlp_wsi
    abmil_wsi
    transmil_wsi
)

seq=0
for preset in "${MODELS[@]}"; do
    seq=$((seq + 1))
    fname=$(printf "wsisingle__%03d__%s.conf" "$seq" "$preset")
    cat > "$OUT_DIR/$fname" <<EOF
EXP_GROUP=WSI_singile
RUN_NAME=default
PRESET=$preset
EOF
done

echo "Generated $seq configs in $OUT_DIR  (WSI_singile: 1 × ${#MODELS[@]})"
