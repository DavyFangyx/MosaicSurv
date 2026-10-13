#!/bin/bash
# configs/z_exp_gen/gen_Single_modal_full_test.sh
# 基于严格筛选后的 split 结果，生成全部可用数据集的单模态基线配置。
#
# 规则：
# - 仅使用 splits/5foldcv/汇总.csv 中 status=ok 且 eligible_cases>0 的数据集
# - 当前等价于 10 个数据集（排除 tcga_stad，含 LUAD/LUSC）
# - 单模态按 3 个模态展开：
#   - WSI
#   - Gene
#   - Clinic
# - 其中每个模态保留当前项目已有的全部单模态模型与输入变体
#
# 结果目录：
# - results/WSItest_F/<study>__<wsi_experiment>/<wsi_model>/
# - results/Genetest/<study>__<gene_experiment>/<gene_f_model>/
# - results/Genetest/<study>__csvraw/<gene_csv_model>/
# - results/Clinictest_Li/<study>__<clinic_experiment>/<clinic_model>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

SPLIT_SUMMARY_CSV="${SPLIT_SUMMARY_CSV:-$SCRIPT_DIR/splits/5foldcv/汇总.csv}"
INCLUDE_CLINIC_COX="${INCLUDE_CLINIC_COX:-true}"

if [ ! -f "$SPLIT_SUMMARY_CSV" ]; then
    echo "[gen_Single_modal_full_test] missing split summary: $SPLIT_SUMMARY_CSV" >&2
    exit 2
fi

mapfile -t STUDIES < <(python - <<'PY' "$SPLIT_SUMMARY_CSV"
import pandas as pd
import sys
df = pd.read_csv(sys.argv[1])
df = df[(df["status"] == "ok") & (df["eligible_cases"] > 0)]
for study in df["study"].astype(str).tolist():
    print(study)
PY
)

if [ "${#STUDIES[@]}" -eq 0 ]; then
    echo "[gen_Single_modal_full_test] no eligible studies found in $SPLIT_SUMMARY_CSV" >&2
    exit 2
fi

WSI_EXPERIMENTS=(
    uni_v1
)

WSI_MODELS=(
    abmil_wsi
    mlp_wsi
    transmil_wsi
)

GENE_F_EXPERIMENTS=(
    scFoundation_embedding_cell_norm
    scFoundation_embedding_cell_raw
    scFoundation_embedding_gene_norm
    scFoundation_embedding_gene_raw
)

GENE_F_MODELS=(
    mlp_gene_f
    snn_gene_f
)

GENE_CSVRAW_MODELS=(
    mlp_gene
    snn_gene
)

CLINIC_EXPERIMENTS=(
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

CLINIC_MODELS=(
    mlp_clinic_mean
    mlp_clinic_flatten
    snn_clinic_mean
    snn_clinic_flatten
)

if [ "$INCLUDE_CLINIC_COX" = "true" ]; then
    CLINIC_MODELS+=(clinic_cox)
fi

create_conf() {
    local target="$1"
    local content="$2"
    if [ -e "$target" ]; then
        skipped=$((skipped + 1))
        return
    fi
    printf '%s\n' "$content" > "$target"
    created=$((created + 1))
}

seq=0
created=0
skipped=0

# ---- WSItest_F ----
for study in "${STUDIES[@]}"; do
    for wsi in "${WSI_EXPERIMENTS[@]}"; do
        for preset in "${WSI_MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "wsitest_f__%03d__%s__%s__%s.conf" "$seq" "$study" "$wsi" "$preset")
            target="$OUT_DIR/$fname"
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=WSItest_F
RUN_NAME=${study}__${wsi}
PRESET=$preset
STUDY=$study
WSI_EXPERIMENT=$wsi
EOF
)"
        done
    done
done

# ---- Genetest: gene foundation ----
for study in "${STUDIES[@]}"; do
    for gene in "${GENE_F_EXPERIMENTS[@]}"; do
        for preset in "${GENE_F_MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "genetest__%03d__%s__%s__%s.conf" "$seq" "$study" "$gene" "$preset")
            target="$OUT_DIR/$fname"
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=Genetest
RUN_NAME=${study}__${gene}
PRESET=$preset
STUDY=$study
GENE_EXPERIMENT=$gene
EOF
)"
        done
    done
done

# ---- Genetest: gene csvraw ----
for study in "${STUDIES[@]}"; do
    for preset in "${GENE_CSVRAW_MODELS[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "genetest__%03d__%s__csvraw__%s.conf" "$seq" "$study" "$preset")
        target="$OUT_DIR/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=Genetest
RUN_NAME=${study}__csvraw
PRESET=$preset
STUDY=$study
EOF
)"
    done
done

# ---- Clinictest_Li ----
for study in "${STUDIES[@]}"; do
    for clinic in "${CLINIC_EXPERIMENTS[@]}"; do
        for preset in "${CLINIC_MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "clinictest_li__%03d__%s__%s__%s.conf" "$seq" "$study" "$clinic" "$preset")
            target="$OUT_DIR/$fname"
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=Clinictest_Li
RUN_NAME=${study}__${clinic}
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$clinic
EOF
)"
        done
    done
done

echo "Eligible studies: ${#STUDIES[@]}"
printf 'Studies:'
for study in "${STUDIES[@]}"; do
    printf ' %s' "$study"
done
printf '\n'

echo "Generated $created new configs in $OUT_DIR"
echo "Total indexed configs this round: $seq"
echo "Skipped existing configs: $skipped"
echo "Expected result folders:"
echo "  results/WSItest_F/<study>__uni_v1/<wsi_model>/"
echo "  results/Genetest/<study>__<gene_embedding>/<gene_f_model>/"
echo "  results/Genetest/<study>__csvraw/<gene_csv_model>/"
echo "  results/Clinictest_Li/<study>__<Lx_or_Dx>/<clinic_model>/"
