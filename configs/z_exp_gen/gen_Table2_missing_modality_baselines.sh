#!/bin/bash
# configs/z_exp_gen/gen_Table2_missing_modality_baselines.sh
# 生成 Table2 缺失模态实验的批量 config。
#
# 模型开关：在下面 PRESETS 数组里注释掉哪一行，那个模型就不生成。
# 命令行仍可用 PRESETS="..." 临时覆盖数组，不改文件。
#
# POE 家族只保留主模型 MosaicSurv（其余 B/C 系已注释，降级到 Table4 消融）；
# MosaicSurv 的公共超参（lr/reg/batch/poe_*）来自 configs/z_exp_gen/mosaic_hparams.sh。
#
# 结果矩阵是 7 个测试子集 x N 个方法 x 5 个数据集。
# 训练任务只有 N 个方法 x 5 个数据集：
# 每个模型只训一次三模态，推理期 7 个子集由 EVAL_MODALITIES=all 共用这次训练。
# 训练缺失配比开关默认关闭（MISSING_MODE=model_gen），留给 Table3。
#
# 用法：
#   bash configs/z_exp_gen/gen_Table2_missing_modality_baselines.sh
#   PRESETS="survtri_poe_vae_B_single survtri_poe_vae_B_multi" STUDIES="tcga_kirp" bash ...
#   PRESETS="survtri_poe_vae_B hgcn" bash ...   # 临时覆盖，不改文件里的注释
#
# 结果目录：
#   results/Table2_Baselines/<study>__<clinic>__<gene_tag>__<wsi>/<preset_folder>/
#   完整三模态写 test_result.csv
#   7 个测试子集写 eval_subsets/{P,C,G,PC,PG,CG,PCG}/test_result.csv
#
# HGCN 读的是原生三模态 pkl，不是 UNI / scFoundation 的 .pt：
#   python SurvPGC_Workspace/generate_native_hgcn_graph.py
#   python SurvPGC_Workspace/generate_hgcn_rna_graph.py
#   python SurvPGC_Workspace/generate_hgcn_clinic.py
# 训练入口仍传普通 P/C/G 路径，内部映射到
#   SurvPGC_Workspace/hgcn data/<study>/P/kimianet/t_img_fea.pkl
#   SurvPGC_Workspace/hgcn data/<study>/G/msigdb_gsea_families/t_rna_fea.pkl
#   SurvPGC_Workspace/hgcn data/<study>/C/L{k}/x_cli.pkl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

# MosaicSurv 公共超参（C 系 preset 专用；Test1-4 共用单一来源）
# shellcheck disable=SC1090
source "$SCRIPT_DIR/configs/z_exp_gen/mosaic_hparams.sh"
# 追加到 C 系 preset 的 extra_lines 末尾，覆盖主模板里的同名键
MOSAIC_EXTRA_LINES=$'LR='"$MOSAIC_LR"$'\nREG='"$MOSAIC_REG"$'\nPOE_SURV_LAMBDA='"$MOSAIC_POE_SURV_LAMBDA"$'\nPOE_BETA_TARGET='"$MOSAIC_POE_BETA_TARGET"$'\nPOE_MODALITY_DROPOUT='"$MOSAIC_POE_MODALITY_DROPOUT"$'\nPOE_MMHID='"$MOSAIC_POE_MMHID"$'\nBATCH_SIZE='"$MOSAIC_BATCH_SIZE"$'\nALPHAFIX='"$MOSAIC_ALPHAFIX"$'\nALPHAPGC='"$MOSAIC_ALPHAPGC"$'\nBETAFIX='"$MOSAIC_BETAFIX"

EXP_GROUP="${EXP_GROUP:-Table2_Baselines}"

if [ -n "${STUDIES:-}" ]; then
    read -r -a STUDIES <<< "$STUDIES"
else
    STUDIES=(tcga_brca tcga_coad tcga_kirc tcga_kirp tcga_lihc)
fi

if [ -n "${PRESETS:-}" ]; then
    read -r -a PRESETS <<< "$PRESETS"
else
    # 注释掉哪一行，那个模型就不生成。
    # POE 家族除主模型 MosaicSurv 外全部注释不用（B/C 系降级到 Table4 消融）；
    # 其余为与 POE 家族无关的缺失模态基线。
    PRESETS=(
#        survtri_poe_vae_B_single
#        survtri_poe_vae_B_multi
#        mosaic_surv_twostage
#        mosaic_surv_single
#        mosaic_surv_multi
        mosaic_surv
#        survtri_poe_vae_B
        modality_concat_zero
        modality_concat_mean
        mvae_poe
        mopoe
        hgcn
    )
fi

CLINIC_EXPERIMENT="${CLINIC_EXPERIMENT:-L0}"
GENE_EXPERIMENT="${GENE_EXPERIMENT:-scFoundation_embedding_gene_raw}"
WSI_EXPERIMENT="${WSI_EXPERIMENT:-uni_v1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
EVAL_MODALITIES="${EVAL_MODALITIES:-all}"
MISSING_MODE="${MISSING_MODE:-model_gen}"
BATCH_SIZE_STAGE1="${BATCH_SIZE_STAGE1:-$BATCH_SIZE}"
MAX_EPOCHS_STAGE1="${MAX_EPOCHS_STAGE1:-10}"

case "$GENE_EXPERIMENT" in
    scFoundation_embedding_cell_norm) GENE_TAG="cell_norm" ;;
    scFoundation_embedding_cell_raw) GENE_TAG="cell_raw" ;;
    scFoundation_embedding_gene_norm) GENE_TAG="gene_norm" ;;
    scFoundation_embedding_gene_raw) GENE_TAG="gene_raw" ;;
    *) GENE_TAG="$GENE_EXPERIMENT" ;;
esac

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
        mosaic_surv_twostage) echo "mosaic_surv_twostage" ;;
        mosaic_surv_single) echo "mosaic_surv_single" ;;
        mosaic_surv_multi) echo "mosaic_surv_multi" ;;
        mosaic_surv) echo "mosaic_surv" ;;
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
    for preset in "${PRESETS[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "table2_baselines__%03d__%s__%s.conf" "$seq" "$study" "$preset")
        target="$OUT_DIR/$fname"
        extra_lines=""
        gene_experiment="$GENE_EXPERIMENT"
        gene_tag="$GENE_TAG"
        wsi_experiment="$WSI_EXPERIMENT"
        case "$preset" in
            survtri_poe_vae_B|survtri_poe_vae_B_single|survtri_poe_vae_B_multi|mosaic_surv_twostage)
                extra_lines=$'BAG_LOSS=cox_surv\nBATCH_SIZE_STAGE1='"$BATCH_SIZE_STAGE1"$'\nMAX_EPOCHS_STAGE1='"$MAX_EPOCHS_STAGE1"
                ;;
            mosaic_surv|mosaic_surv_single|mosaic_surv_multi|mosaic_surv_noenum|mosaic_surv_kl|mosaic_surv_detached|mosaic_surv_nojeffreys)
                extra_lines=$'BAG_LOSS=cox_surv\nBATCH_SIZE_STAGE1='"$BATCH_SIZE_STAGE1"$'\nMAX_EPOCHS_STAGE1='"$MAX_EPOCHS_STAGE1"$'\n'"$MOSAIC_EXTRA_LINES"
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
RUN_NAME=${study}__${CLINIC_EXPERIMENT}__${gene_tag}__${wsi_experiment}
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
EVAL_MODALITIES=$EVAL_MODALITIES
MISSING_MODE=$MISSING_MODE
$extra_lines
EOF
)"
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
echo "Result matrix: 7 subsets x ${#PRESETS[@]} methods x ${#STUDIES[@]} studies"
echo "Training jobs: ${#PRESETS[@]} methods x ${#STUDIES[@]} studies = $seq"
echo "Expected result folders:"
for preset in "${PRESETS[@]}"; do
    folder="$(preset_result_folder "$preset")"
    echo "  results/$EXP_GROUP/<study>__${CLINIC_EXPERIMENT}__${GENE_TAG}__${WSI_EXPERIMENT}/$folder/"
done
echo "Each trained model writes PCG to test_result.csv and 7 shared-inference subsets to eval_subsets/{P,C,G,PC,PG,CG,PCG}/"
