#!/bin/bash
# configs/z_exp_gen/Table4_Abaltion_Test/_common.bash
# 共享的 Table4_Abaltion_Test 配置生成逻辑。
# 全部消融 preset 的公共超参（lr/reg/batch/poe_*）统一来自
# configs/z_exp_gen/mosaic_hparams.sh，与主模型 MosaicSurv 保持一致。

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat <<'EOF'
[_common.bash] 这是共享脚本，不会单独生成 config。

请运行下面任一入口脚本：
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_brca_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_coad_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_kich_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_kirc_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_kirp_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_lihc_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_prad_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_read_ablation_test.sh
EOF
    exit 2
fi

# MosaicSurv 公共超参（Test1-4 共用单一来源；全部消融 preset 使用）
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../mosaic_hparams.sh"

normalize_stage1_pool_tag() {
    local raw_value="$1"
    local -a studies=()
    local token study seen
    for token in ${raw_value//,/ }; do
        study="${token// /}"
        [ -z "$study" ] && continue
        if [[ "$study" != tcga_* ]]; then
            study="tcga_${study}"
        fi
        seen=false
        for token in "${studies[@]}"; do
            if [[ "$token" == "$study" ]]; then
                seen=true
                break
            fi
        done
        if [ "$seen" = false ]; then
            studies+=("$study")
        fi
    done
    printf '%s\n' "${studies[@]}"
}

generate_table4_ablation_test_configs() {
    : "${STUDY:?STUDY is required}"
    : "${EXP_GROUP:?EXP_GROUP is required}"

    local script_dir out_dir clinic_experiment gene_experiment wsi_experiment
    local batch_size max_epochs warmup_epochs run_name_base file_prefix
    local seq created skipped preset fname target gene_tag stage1_pool stage1_pool_tag

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    out_dir="${OUT_DIR:-$script_dir/configs/queue}"
    clinic_experiment="${CLINIC_EXPERIMENT:-L0}"
    gene_experiment="${GENE_EXPERIMENT:-scFoundation_embedding_cell_norm}"
    wsi_experiment="${WSI_EXPERIMENT:-uni_v1}"
    batch_size="${MOSAIC_BATCH_SIZE}"
    max_epochs="${MAX_EPOCHS:-20}"
    warmup_epochs="${WARMUP_EPOCHS:-3}"
    stage1_pool="${POE_STAGE1_STUDIES:-brca,coad,kirc,kirp,lihc}"
    stage1_pool_tag=" $(normalize_stage1_pool_tag "$stage1_pool" | tr '\n' ' ') "

    case "$gene_experiment" in
        scFoundation_embedding_cell_norm) gene_tag="cell_norm" ;;
        scFoundation_embedding_cell_raw) gene_tag="cell_raw" ;;
        scFoundation_embedding_gene_norm) gene_tag="gene_norm" ;;
        scFoundation_embedding_gene_raw) gene_tag="gene_raw" ;;
        *) gene_tag="$gene_experiment" ;;
    esac

    run_name_base="${RUN_NAME_BASE:-${STUDY}__${gene_tag}__${wsi_experiment}}"
    file_prefix="${FILE_PREFIX:-Table4_Abaltion_Test_${STUDY#tcga_}_ablation_test}"
    # FILE_PREFIX 不含 study 时补上，避免 5 个 study 的 conf 同名互踩
    # （只有第一个 study 会写入，其余全部被 create_conf 跳过）
    case "$file_prefix" in
        *"${STUDY#tcga_}"*) ;;
        *) file_prefix="${file_prefix}_${STUDY#tcga_}" ;;
    esac

    local -a ablation_presets
    if [[ -n "${PRESETS:-}" ]]; then
        read -r -a ablation_presets <<< "$PRESETS"
    else
        ablation_presets=(
            mosaic_surv_single
            mosaic_surv_single_enum
            mosaic_surv_noenum
            mosaic_surv
            mosaic_surv_nodropout
            mosaic_surv_multi
            mosaic_surv_twostage
            mosaic_surv_frozen
            mosaic_surv_kl
            mosaic_surv_nojeffreys
            mosaic_surv_detached
        )
    fi

    # MosaicSurv 公共超参（全部消融 preset 与主模型 MosaicSurv 保持一致；
    # nojeffreys/detached 等消融维度由 preset 内部的覆盖逻辑处理）
    local poe_beta_target="${MOSAIC_POE_BETA_TARGET}"
    local poe_surv_lambda="${MOSAIC_POE_SURV_LAMBDA}"
    # stage1 预训练 batch 保持历史默认 128，不随 MosaicSurv 的 stage2 batch 变化
    local batch_size_stage1="${BATCH_SIZE_STAGE1:-128}"
    local max_epochs_stage1="${MAX_EPOCHS_STAGE1:-10}"

    mkdir -p "$out_dir"

    create_conf() {
        local target_path="$1"
        local content="$2"
        if [ -e "$target_path" ]; then
            skipped=$((skipped + 1))
            return
        fi
        printf '%s\n' "$content" > "$target_path"
        created=$((created + 1))
    }

    seq=0
    created=0
    skipped=0

    for preset in "${ablation_presets[@]}"; do
        if [[ "$preset" == "survtri_poe_vae_B_crossstage1" ]]; then
            local study_tag="$(normalize_stage1_pool_tag "$STUDY")"
            if [[ "$stage1_pool_tag" != *" $study_tag "* ]]; then
                continue
            fi
        fi
        local preset_beta="$poe_beta_target"
        local preset_lambda="$poe_surv_lambda"
        case "$preset" in
            mosaic_surv_nojeffreys) preset_beta=0 ;;
        esac
        seq=$((seq + 1))
        fname=$(printf "%s__%03d__%s.conf" "$file_prefix" "$seq" "$preset")
        target="$out_dir/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$run_name_base
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$clinic_experiment
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
BAG_LOSS=cox_surv
BATCH_SIZE=$batch_size
BATCH_SIZE_STAGE1=$batch_size_stage1
MAX_EPOCHS=$max_epochs
MAX_EPOCHS_STAGE1=$max_epochs_stage1
WARMUP_EPOCHS=$warmup_epochs
POE_BETA_TARGET=$preset_beta
POE_SURV_LAMBDA=$preset_lambda
LR=$MOSAIC_LR
REG=$MOSAIC_REG
POE_MODALITY_DROPOUT=$MOSAIC_POE_MODALITY_DROPOUT
POE_MMHID=$MOSAIC_POE_MMHID
ALPHAFIX=$MOSAIC_ALPHAFIX
ALPHAPGC=$MOSAIC_ALPHAPGC
BETAFIX=$MOSAIC_BETAFIX
POE_STAGE1_STUDIES=${POE_STAGE1_STUDIES:-brca,coad,kirc,kirp,lihc}
EOF
)"
    done

    echo "Generated $created new configs in $out_dir"
    echo "Total indexed configs this round: $seq"
    echo "Skipped existing configs: $skipped"
    echo "Study: $STUDY"
    echo "Experiment group: $EXP_GROUP"
    echo "Clinic embedding: $clinic_experiment"
    echo "Gene embedding: $gene_experiment"
    echo "WSI embedding: $wsi_experiment"
    echo "Run name base: $run_name_base"
}
