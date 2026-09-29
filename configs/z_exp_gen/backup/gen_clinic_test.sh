#!/bin/bash
# exp_gen/gen_clinic_test.sh
# 生成 clinic_test 的 config：7 个 clinic embedding × 8 个含 clinic 模态对照的模型 = 56 个 config
# CLINIC_EXPERIMENT 取 SurvPGC_Workspace/C/<basename> 的 basename
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

CLINICS=(
    C_noIntent
    C_noTreat
    C_intentOnly
    C_treatment
    O_noTreatOnly
    O_origin
    O_simple
)

MODELS=(
    clinic_mlp
    clinic_snn
    porpoise
    survpath
    mcat
    survpgc_f
    survpc_f
    survgc_f
)

seq=0
for clinic in "${CLINICS[@]}"; do
    for preset in "${MODELS[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "clinic__%03d__%s__%s.conf" "$seq" "$clinic" "$preset")
        cat > "$OUT_DIR/$fname" <<EOF
EXP_GROUP=clinic_test
RUN_NAME=$clinic
PRESET=$preset
CLINIC_EXPERIMENT=$clinic
BATCH_SIZE=1
EOF
    done
done

echo "Generated $seq configs in $OUT_DIR  (clinic_test: ${#CLINICS[@]} × ${#MODELS[@]})"
