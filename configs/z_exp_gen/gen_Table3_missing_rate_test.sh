#!/bin/bash
# configs/z_exp_gen/gen_Table3_missing_rate_test.sh
# 生成 Table3 缺失配比实验的批量 config。
#
# 模型开关：在下面 PRESETS 数组里注释掉哪一行，那个模型就不生成。
# 命令行仍可用 PRESETS="..." 临时覆盖数组，不改文件。
#
# 训练缺失配比（MISSING_MODE=unified_mask_csv）：
#   60,0,0 / 0,60,0 / 0,0,60 / 30,30,0 / 30,0,30 / 0,30,30 / 20,20,20
# 各模型共用 results/Table3_MissingRate/masks/ 下同一组 split mask
# （路径只由 exp_group/study/pattern/seed/fold 决定，与模型无关）。
#
# 测试 7 子集由 EVAL_MODALITIES=all 共用同一次 5-fold 训练，
# 不再按测试子集拆成独立训练任务。
#
# 结果矩阵：7 个训练配比 x N 个模型 x 7 个测试子集 x 5 个数据集。
# 训练任务：7 个配比 x N 个模型 x 5 个数据集。
#
# POE 家族只保留主模型 C_film（其余 B/C 系已注释，降级到 Table4 消融）；
# C_film 的公共超参（lr/reg/batch/poe_*）来自 configs/z_exp_gen/cfilm_hparams.sh，
# 其中 POE_MODALITY_DROPOUT 固定写 0（unified_mask_csv 下缺失由共享 mask 决定）。
#
# 用法：
#   bash configs/z_exp_gen/gen_Table3_missing_rate_test.sh
#   PRESETS="survtri_poe_vae_B_single survtri_poe_vae_B_multi" STUDIES="tcga_kirp" bash ...
#   PRESETS="survtri_poe_vae_B hgcn" bash ...   # 临时覆盖，不改文件里的注释
#
# 结果目录：
#   results/Table3_MissingRate/<study>__<clinic>__<gene_tag>__<wsi>__pW_G_C/<model_subdir>/
#   <model_subdir> 由 PRESETS 决定，例如 survtri_poe_vae_b_single / hgcn
#   完整三模态写 test_result.csv
#   7 个测试子集写 eval_subsets/{P,C,G,PC,PG,CG,PCG}/test_result.csv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

# Cfilm 公共超参（C 系 preset 专用；Test1-4 共用单一来源）
# shellcheck disable=SC1090
source "$SCRIPT_DIR/configs/z_exp_gen/cfilm_hparams.sh"
# 追加到 C 系 preset 的 extra_lines 末尾，覆盖主模板里的同名键。
# unified_mask_csv 下缺失由共享 mask csv 决定，drop_prob 不生效，
# 所以 POE_MODALITY_DROPOUT 写 0（与 Cfilm_Hparam_Eval/gen_Table3 惯例一致）。
CFILM_EXTRA_LINES=$'LR='"$CFILM_LR"$'\nREG='"$CFILM_REG"$'\nPOE_SURV_LAMBDA='"$CFILM_POE_SURV_LAMBDA"$'\nPOE_BETA_TARGET='"$CFILM_POE_BETA_TARGET"$'\nPOE_MMHID='"$CFILM_POE_MMHID"$'\nBATCH_SIZE='"$CFILM_BATCH_SIZE"$'\nALPHAFIX='"$CFILM_ALPHAFIX"$'\nALPHAPGC='"$CFILM_ALPHAPGC"$'\nBETAFIX='"$CFILM_BETAFIX"$'\nPOE_MODALITY_DROPOUT=0'

EXP_GROUP="${EXP_GROUP:-Table3_MissingRate}"

if [ -n "${STUDIES:-}" ]; then
    read -r -a STUDIES <<< "$STUDIES"
else
    STUDIES=(tcga_brca tcga_coad tcga_kirc tcga_kirp tcga_lihc)
fi

if [ -n "${PRESETS:-}" ]; then
    read -r -a PRESETS <<< "$PRESETS"
else
    # 注释掉哪一行，那个模型就不生成。
    # POE 家族只保留主模型 C_film（其余 B/C 系已注释，降级到 Table4 消融）；
    # hgcn 为与 POE 家族无关的基线。
    PRESETS=(
#        survtri_poe_vae_B_single
#        survtri_poe_vae_B_multi
#        survtri_poe_vae_B_film
#        survtri_poe_vae_C_single
#        survtri_poe_vae_C_multi
        survtri_poe_vae_C_film
#        survtri_poe_vae_B
#        modality_concat_zero
        modality_concat_mean
#        mvae_poe
#        mopoe
        hgcn
    )
fi

MISSING_PATTERNS=(
    "60,0,0"
    "0,60,0"
    "0,0,60"
    "30,30,0"
    "30,0,30"
    "0,30,30"
    "20,20,20"
)

CLINIC_EXPERIMENT="${CLINIC_EXPERIMENT:-L0}"
GENE_EXPERIMENT="${GENE_EXPERIMENT:-scFoundation_embedding_gene_raw}"
WSI_EXPERIMENT="${WSI_EXPERIMENT:-uni_v1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
EVAL_MODALITIES="${EVAL_MODALITIES:-all}"
MISSING_MODE="${MISSING_MODE:-unified_mask_csv}"
MISSING_SEED="${MISSING_SEED:-1}"
K="${K:-5}"
BATCH_SIZE_STAGE1="${BATCH_SIZE_STAGE1:-$BATCH_SIZE}"
MAX_EPOCHS_STAGE1="${MAX_EPOCHS_STAGE1:-10}"

case "$GENE_EXPERIMENT" in
    scFoundation_embedding_cell_norm) GENE_TAG="cell_norm" ;;
    scFoundation_embedding_cell_raw) GENE_TAG="cell_raw" ;;
    scFoundation_embedding_gene_norm) GENE_TAG="gene_norm" ;;
    scFoundation_embedding_gene_raw) GENE_TAG="gene_raw" ;;
    *) GENE_TAG="$GENE_EXPERIMENT" ;;
esac

pattern_tag() {
    local pattern="$1"
    printf 'p%s' "${pattern//,/_}"
}

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

preset_result_folder() {
    local preset="$1"
    case "$preset" in
        survtri_poe_vae_B) echo "survtri_poe_vae__B" ;;
        survtri_poe_vae_B_single) echo "survtri_poe_vae_b_single" ;;
        survtri_poe_vae_B_multi) echo "survtri_poe_vae_b_multi" ;;
        survtri_poe_vae_B_film) echo "survtri_poe_vae_b_film" ;;
        survtri_poe_vae_C_single) echo "survtri_poe_vae_c_single" ;;
        survtri_poe_vae_C_multi) echo "survtri_poe_vae_c_multi" ;;
        survtri_poe_vae_C_film) echo "survtri_poe_vae_c_film" ;;
        modality_concat_zero) echo "modality_concat__resampler__zero" ;;
        modality_concat_mean) echo "modality_concat__resampler__mean" ;;
        *) echo "$preset" ;;
    esac
}

seq=0
created=0
skipped=0

if [ "${#PRESETS[@]}" -eq 0 ]; then
    echo "No presets selected. Uncomment at least one model in PRESETS, or pass PRESETS=..." >&2
    exit 1
fi

for study in "${STUDIES[@]}"; do
    for pattern in "${MISSING_PATTERNS[@]}"; do
        tag="$(pattern_tag "$pattern")"
        for preset in "${PRESETS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "table3_missing_rate__%04d__%s__%s__%s.conf" "$seq" "$study" "$tag" "$preset")
            target="$OUT_DIR/$fname"
            extra_lines=""
            gene_experiment="$GENE_EXPERIMENT"
            gene_tag="$GENE_TAG"
            wsi_experiment="$WSI_EXPERIMENT"
            case "$preset" in
                survtri_poe_vae_B|survtri_poe_vae_B_single|survtri_poe_vae_B_multi|survtri_poe_vae_B_film)
                    extra_lines=$'BAG_LOSS=cox_surv\nBATCH_SIZE_STAGE1='"$BATCH_SIZE_STAGE1"$'\nMAX_EPOCHS_STAGE1='"$MAX_EPOCHS_STAGE1"
                    ;;
                survtri_poe_vae_C_single|survtri_poe_vae_C_multi|survtri_poe_vae_C_film)
                    extra_lines=$'BAG_LOSS=cox_surv\nBATCH_SIZE_STAGE1='"$BATCH_SIZE_STAGE1"$'\nMAX_EPOCHS_STAGE1='"$MAX_EPOCHS_STAGE1"$'\n'"$CFILM_EXTRA_LINES"
                    ;;
                modality_concat_zero)
                    extra_lines=$'CONCAT_WSI=resampler\nCONCAT_IMPUTE=zero'
                    ;;
                modality_concat_mean)
                    extra_lines=$'CONCAT_WSI=resampler\nCONCAT_IMPUTE=mean'
                    ;;
                hgcn)
                    extra_lines=$'BATCH_SIZE=32'
                    ;;
            esac
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${study}__${CLINIC_EXPERIMENT}__${gene_tag}__${wsi_experiment}__${tag}
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
WANDB_MODE=$WANDB_MODE
BAG_LOSS=cox_surv
BATCH_SIZE=$BATCH_SIZE
MAX_EPOCHS=$MAX_EPOCHS
WARMUP_EPOCHS=$WARMUP_EPOCHS
K=$K
EVAL_MODALITIES=$EVAL_MODALITIES
MISSING_MODE=$MISSING_MODE
MISSING_PATTERN=$pattern
MISSING_SEED=$MISSING_SEED
POE_MODALITY_DROPOUT=0
$extra_lines
EOF
)"
        done
    done
done

echo "Generated $created new configs in $OUT_DIR"
echo "Total indexed configs this round: $seq"
echo "Skipped existing configs: $skipped"
echo "Experiment group: $EXP_GROUP"
echo "Presets: ${PRESETS[*]}"
echo "Clinic embedding: $CLINIC_EXPERIMENT"
echo "Gene embedding: $GENE_EXPERIMENT"
echo "WSI embedding: $WSI_EXPERIMENT"
echo "Eval modalities: $EVAL_MODALITIES"
echo "Missing mode: $MISSING_MODE"
echo "Missing seed: $MISSING_SEED"
echo "Folds: $K"
echo "Result matrix: ${#MISSING_PATTERNS[@]} train patterns x ${#PRESETS[@]} models x 7 test subsets x ${#STUDIES[@]} studies"
echo "Training jobs: ${#MISSING_PATTERNS[@]} patterns x ${#PRESETS[@]} models x ${#STUDIES[@]} studies = $seq"
echo "Expected result folders:"
for preset in "${PRESETS[@]}"; do
    folder="$(preset_result_folder "$preset")"
    echo "  results/$EXP_GROUP/<study>__${CLINIC_EXPERIMENT}__${GENE_TAG}__${WSI_EXPERIMENT}__pW_G_C/$folder/"
done
echo "Each trained model writes PCG to test_result.csv and 7 shared-inference subsets to eval_subsets/{P,C,G,PC,PG,CG,PCG}/"
