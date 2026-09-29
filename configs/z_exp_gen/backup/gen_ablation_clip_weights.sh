#!/bin/bash
# exp_gen/gen_ablation_clip_weights.sh
# 生成 ablation_CLIP_weights 的 config：仅 survfusion_separate_mhsa，{1,2,3}^3 = 27 组
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

seq=0
for it in 1 2 3; do
    for is in 1 2 3; do
        for ts in 1 2 3; do
            seq=$((seq + 1))
            fname=$(printf "ablclipw__%03d__IT%d_IS%d_TS%d.conf" "$seq" "$it" "$is" "$ts")
            cat > "$OUT_DIR/$fname" <<EOF
EXP_GROUP=ablation_CLIP_weights
RUN_NAME=IT${it}_IS${is}_TS${ts}
PRESET=survfusion_separate_mhsa
CLIP_WEIGHT_IT=$it
CLIP_WEIGHT_IS=$is
CLIP_WEIGHT_TS=$ts
EOF
        done
    done
done

echo "Generated $seq configs in $OUT_DIR  (ablation_CLIP_weights: 27 = {1,2,3}^3)"
