#!/bin/bash
# exp_gen/gen_ablation_model_structure.sh
# 生成 ablation_Model_structure 的 config：3 模型 × 4 个 CLIP_LAMBDA = 12 个
# 方案 B：NOALIGN / JOINT / SEPARATE_MHSA 各跑 4 次 λ（NOALIGN/SEPARATE_MHSA 实际不读 λ，形式对称）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

LAMBDAS=(0.1 0.01 0.02 0.05)
MODELS=(
    survfusion_noalign
    survfusion_joint
    survfusion_separate_mhsa
)

seq=0
for lam in "${LAMBDAS[@]}"; do
    for preset in "${MODELS[@]}"; do
        seq=$((seq + 1))
        lam_tag=$(echo "$lam" | tr '.' '_')
        fname=$(printf "ablmodel__%03d__%s__lam%s.conf" "$seq" "$preset" "$lam_tag")
        cat > "$OUT_DIR/$fname" <<EOF
EXP_GROUP=ablation_Model_structure
RUN_NAME=${preset}__lam${lam_tag}
PRESET=$preset
CLIP_LAMBDA=$lam
EOF
    done
done

echo "Generated $seq configs in $OUT_DIR  (ablation_Model_structure: ${#LAMBDAS[@]} × ${#MODELS[@]})"
