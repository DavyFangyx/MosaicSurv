#!/bin/bash
# exp_gen/gen_Clinictest_Li.sh
# 生成 Clinictest_Li 的 config：按注册表启用的数据集批量展开
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

CLINICS=(
    L0
    L1
    L2
    L3
    L4
    L5
    D0
    D1
    D2
    D3
    D4
    D5
)

MODELS=(
    mlp_clinic_mean
    mlp_clinic_flatten
    snn_clinic_mean
    snn_clinic_flatten
)

seq=0
created=0
skipped=0
for study in "${STUDIES[@]}"; do
    for clinic in "${CLINICS[@]}"; do
        for preset in "${MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "clinictest_li__%03d__%s__%s__%s.conf" "$seq" "$study" "$clinic" "$preset")
            target="$OUT_DIR/$fname"

            if [ -e "$target" ]; then
                skipped=$((skipped + 1))
                continue
            fi

            cat > "$target" <<EOF
EXP_GROUP=Clinictest_Li
RUN_NAME=${study}__${clinic}
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$clinic
EOF
            created=$((created + 1))
        done
    done
done

echo "Generated $created new configs in $OUT_DIR  (Clinictest_Li total: $seq, skipped existing: $skipped)"
