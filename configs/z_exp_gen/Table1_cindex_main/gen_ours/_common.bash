#!/bin/bash
# configs/z_exp_gen/Table1_cindex_main/gen_ours/_common.bash
# 共享的 SurvTriPoEVAE 配置生成逻辑。
# MosaicSurv 已提升为主模型：POE 家族只生成 mosaic_surv，
# 其余 A/B/C 系变体全部注释（消融见 Table4_Abaltion_Test）。
# MosaicSurv 的公共超参统一来自 configs/z_exp_gen/mosaic_hparams.sh。

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat <<'EOF'
[_common.bash] 这是共享脚本，不会单独生成 config。

请运行下面任一入口脚本：
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_brca_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_coad_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_kich_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_kirc_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_kirp_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_lihc_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_prad_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_read_poe_model_val.sh
EOF
    exit 2
fi

# MosaicSurv 公共超参（C 系 preset 使用；Test1-4 共用单一来源）
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../mosaic_hparams.sh"

generate_poe_model_val_configs() {
    : "${STUDY:?STUDY is required}"
    : "${EXP_GROUP:?EXP_GROUP is required}"

    local script_dir out_dir clinic_experiment gene_experiment wsi_experiment
    local batch_size batch_size_stage1
    local model_b_lr model_b_lr_stage1 model_b_reg
    local model_b_beta model_b_dropout model_b_mmhid model_b_decoder_hidden_dim
    local model_b_encoder_lr_ratio
    local model_c_batch_size model_c_lr model_c_reg
    local model_c_beta model_c_lambda model_c_betafix
    local max_epochs max_epochs_stage1 warmup_epochs run_name_base file_prefix exp_group_prefix
    local seq created skipped preset fname target

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
    out_dir="${OUT_DIR:-$script_dir/configs/queue}"
    clinic_experiment="${CLINIC_EXPERIMENT:-L0}"
    gene_experiment="${GENE_EXPERIMENT:-scFoundation_embedding_cell_norm}"
    wsi_experiment="${WSI_EXPERIMENT:-uni_v1}"
    batch_size="${BATCH_SIZE:-128}"
    batch_size_stage1="${BATCH_SIZE_STAGE1:-128}"
    max_epochs="${MAX_EPOCHS:-20}"
    max_epochs_stage1="${MAX_EPOCHS_STAGE1:-10}"
    warmup_epochs="${WARMUP_EPOCHS:-3}"
    # model_b_lr="${MODEL_B_LR:-6e-4}"
    # model_b_lr_stage1="${MODEL_B_LR_STAGE1:-8e-5}"
    # model_b_reg="${MODEL_B_REG:-7e-5}"
    # model_b_beta="${MODEL_B_POE_BETA_TARGET:-0.0164}"
    # model_b_dropout="${MODEL_B_POE_MODALITY_DROPOUT:-0.25}"
    # model_b_mmhid="${MODEL_B_POE_MMHID:-128}"
    # model_b_decoder_hidden_dim="${MODEL_B_POE_DECODER_HIDDEN_DIM:-512}"
    # model_b_encoder_lr_ratio="${MODEL_B_POE_ENCODER_LR_RATIO:-0.1}"
    # model_c_batch_size="${MODEL_C_BATCH_SIZE:-16}"
    # model_c_lr="${MODEL_C_LR:-1.49e-4}"
    # model_c_reg="${MODEL_C_REG:-1.25e-5}"
    # model_c_beta="${MODEL_C_POE_BETA_TARGET:-0.268}"
    # model_c_lambda="${MODEL_C_POE_SURV_LAMBDA:-0.361}"
    # model_c_betafix="${MODEL_C_BETAFIX:-true}"
    # alphafix="${ALPHAFIX:-true}"
    # alphapgc="${ALPHAPGC:-0.5,0.3,0.2}"
    exp_group_prefix=""
    if [[ "$clinic_experiment" == "L0" ]]; then
        exp_group_prefix="L0_"
    fi

    run_name_base="${RUN_NAME_BASE:-${STUDY}__${clinic_experiment}__cell_norm__${wsi_experiment}}"
    if [[ "$EXP_GROUP" != "${exp_group_prefix}"* ]]; then
        EXP_GROUP="${exp_group_prefix}${EXP_GROUP}"
    fi
    file_prefix="${FILE_PREFIX:-${exp_group_prefix}${STUDY#tcga_}_poe_model_val}"

    # MosaicSurv 已提升为主模型：POE 家族除 mosaic_surv 外全部注释不用，
    # 其余 A/B/C 系变体作为消融对象见 Table4_Abaltion_Test。
    local -a poe_presets=(
#        survtri_poe_vae_A
#        survtri_poe_vae_B
#        survtri_poe_vae_C
#        survtri_poe_vae_B_single
#        survtri_poe_vae_B_multi
#        mosaic_surv_twostage
#        mosaic_surv_single
#        mosaic_surv_multi
        mosaic_surv
    )

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

    for preset in "${poe_presets[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "%s__%03d__PCG__%s.conf" "$file_prefix" "$seq" "$preset")
        target="$out_dir/$fname"
        # MosaicSurv 主模型与 C 系消融（single/single_enum/multi/noenum/kl/detached/
        # nojeffreys）使用 MOSAIC_* 公共超参；twostage(两阶段 B) 与 frozen(预训练冻结 A)
        # 走各自分支，遗留 survtri_poe_vae_C* preset 保持原行为。
        if [[ "$preset" == survtri_poe_vae_C* \
              || "$preset" == mosaic_surv \
              || "$preset" == mosaic_surv_single \
              || "$preset" == mosaic_surv_single_enum \
              || "$preset" == mosaic_surv_multi \
              || "$preset" == mosaic_surv_noenum \
              || "$preset" == mosaic_surv_kl \
              || "$preset" == mosaic_surv_detached \
              || "$preset" == mosaic_surv_nojeffreys ]]; then
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${run_name_base}__${preset}
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$clinic_experiment
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
BAG_LOSS=cox_surv
BATCH_SIZE=$MOSAIC_BATCH_SIZE
BATCH_SIZE_STAGE1=$batch_size_stage1
MAX_EPOCHS=$max_epochs
MAX_EPOCHS_STAGE1=$max_epochs_stage1
WARMUP_EPOCHS=$warmup_epochs
LR=$MOSAIC_LR
REG=$MOSAIC_REG
POE_SURV_LAMBDA=$MOSAIC_POE_SURV_LAMBDA
POE_BETA_TARGET=$MOSAIC_POE_BETA_TARGET
POE_MODALITY_DROPOUT=$MOSAIC_POE_MODALITY_DROPOUT
POE_MMHID=$MOSAIC_POE_MMHID
ALPHAFIX=$MOSAIC_ALPHAFIX
ALPHAPGC=$MOSAIC_ALPHAPGC
BETAFIX=$MOSAIC_BETAFIX
EOF
)"
        elif [[ "$preset" == survtri_poe_vae_B* || "$preset" == mosaic_surv_twostage ]]; then
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${run_name_base}__${preset}
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
EOF
)"
            # LR=$model_b_lr
            # LR_STAGE1=$model_b_lr_stage1
            # REG=$model_b_reg
            # POE_BETA_TARGET=$model_b_beta
            # BETAFIX=$model_c_betafix
            # POE_MODALITY_DROPOUT=$model_b_dropout
            # POE_MMHID=$model_b_mmhid
            # POE_DECODER_HIDDEN_DIM=$model_b_decoder_hidden_dim
            # POE_ENCODER_LR_RATIO=$model_b_encoder_lr_ratio
            # ALPHAFIX=$alphafix
            # ALPHAPGC=$alphapgc
        else
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${run_name_base}__${preset}
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
EOF
)"
            # ALPHAFIX=$alphafix
            # ALPHAPGC=$alphapgc
        fi
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
