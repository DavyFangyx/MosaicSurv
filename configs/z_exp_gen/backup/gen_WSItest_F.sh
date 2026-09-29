#!/bin/bash
# exp_gen/gen_WSItest_F.sh
# 生成 WSItest_F 的 config：按注册表启用的数据集批量展开
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

WSI_EXPERIMENTS=(
    uni_v1
    uni_v2
)

MODELS=(
    abmil_wsi
    mlp_wsi
    transmil_wsi
)

seq=0
created=0
skipped=0
for study in "${STUDIES[@]}"; do
    for wsi in "${WSI_EXPERIMENTS[@]}"; do
        for preset in "${MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "wsitest_f__%03d__%s__%s__%s.conf" "$seq" "$study" "$wsi" "$preset")
            target="$OUT_DIR/$fname"

            if [ -e "$target" ]; then
                skipped=$((skipped + 1))
                continue
            fi

            cat > "$target" <<EOF
EXP_GROUP=WSItest_F
RUN_NAME=${study}__${wsi}
PRESET=$preset
STUDY=$study
WSI_EXPERIMENT=$wsi
EOF
            created=$((created + 1))
        done
    done
done

echo "Generated $created new configs in $OUT_DIR  (WSItest_F total: $seq, skipped existing: $skipped)"
