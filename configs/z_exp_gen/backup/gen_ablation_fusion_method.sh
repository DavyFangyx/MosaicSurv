#!/bin/bash
# exp_gen/gen_ablation_fusion_method.sh
# 生成 ablation_Fusion_method 的 config：3 模型 × 4 个 NUM_HEADS = 12 个
# 方案 B：MHSA / CONCAT / MEAN 各跑 4 次 heads（CONCAT/MEAN 实际不读 heads，形式对称）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

HEADS=(2 4 8 16)
MODELS=(
    survfusion_separate_mhsa
    survfusion_separate_concat
    survfusion_separate_mean
)

seq=0
for h in "${HEADS[@]}"; do
    for preset in "${MODELS[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "ablfuse__%03d__%s__heads%02d.conf" "$seq" "$preset" "$h")
        cat > "$OUT_DIR/$fname" <<EOF
EXP_GROUP=ablation_Fusion_method
RUN_NAME=${preset}__heads${h}
PRESET=$preset
NUM_HEADS=$h
EOF
    done
done

echo "Generated $seq configs in $OUT_DIR  (ablation_Fusion_method: ${#HEADS[@]} × ${#MODELS[@]})"
