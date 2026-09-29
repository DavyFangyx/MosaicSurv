# configs/presets.sh
# 定义 apply_preset <preset_name>：
#   - 设置 MODEL（传给 main.py 的 --modality）
#   - 设置 EXTRA_ARGS 数组（追加到命令末尾）
#   - 设置 RESULTS_SUBDIR（用于对齐 Python 侧真实结果目录）

POE_STAGE1_STUDIES="${POE_STAGE1_STUDIES:-brca,coad,kirc,kirp,lihc}"

format_poe_stage1_studies_tag() {
    local raw_value="$1"
    local -a studies=()
    local token study
    for token in ${raw_value//,/ }; do
        study="${token// /}"
        [ -z "$study" ] && continue
        if [[ "$study" != tcga_* ]]; then
            study="tcga_${study}"
        fi
        local seen=false
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

apply_preset() {
    local preset="$1"

    EXTRA_ARGS=()
    RESULTS_SUBDIR=""

    case "$preset" in
        # ========== 单模态 C ==========
        mlp_clinic_mean)
            MODEL="mlp_clinic_mean"
            RESULTS_SUBDIR="$MODEL"
            ;;
        mlp_clinic_flatten)
            MODEL="mlp_clinic_flatten"
            RESULTS_SUBDIR="$MODEL"
            ;;
        snn_clinic_mean)
            MODEL="snn_clinic_mean"
            RESULTS_SUBDIR="$MODEL"
            ;;
        snn_clinic_flatten)
            MODEL="snn_clinic_flatten"
            RESULTS_SUBDIR="$MODEL"
            ;;
        clinic_cox)
            MODEL="clinic_cox"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            ;;

        # ========== 单模态 G ==========
        mlp_gene)
            MODEL="mlp_gene"
            RESULTS_SUBDIR="$MODEL"
            ;;
        snn_gene)
            MODEL="snn_gene"
            RESULTS_SUBDIR="$MODEL"
            ;;
        mlp_gene_f)
            MODEL="mlp_gene_f"
            RESULTS_SUBDIR="$MODEL"
            ;;
        snn_gene_f)
            MODEL="snn_gene_f"
            RESULTS_SUBDIR="$MODEL"
            ;;

        # ========== 单模态 WSI ==========
        mlp_wsi)
            MODEL="mlp_wsi"
            RESULTS_SUBDIR="$MODEL"
            ;;
        abmil_wsi)
            MODEL="abmil_wsi"
            RESULTS_SUBDIR="$MODEL"
            ;;
        transmil_wsi)
            MODEL="transmil_wsi"
            RESULTS_SUBDIR="$MODEL"
            ;;

        # ========== 多模态基线 WSI+G ==========
        porpoise)
            MODEL="porpoise"
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(--fusion "$FUSION")
            ;;
        survpath)
            MODEL="survpath"
            RESULTS_SUBDIR="$MODEL"
            ;;
        mcat)
            MODEL="mcat"
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(--fusion "$FUSION")
            ;;

        # ========== 主模型 / 消融 ==========
        survpgc_f)
            MODEL="survpgc_f"
            RESULTS_SUBDIR="$MODEL"
            ;;
        survpc_f)
            MODEL="survpc_f"
            RESULTS_SUBDIR="$MODEL"
            ;;
        survgc_f)
            MODEL="survgc_f"
            RESULTS_SUBDIR="$MODEL"
            ;;
        survtri_snn_concat)
            MODEL="survtri_snn_concat"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${MODEL}__${SELECTED_MODALITIES//,/_}"
            fi
            ;;
        survtri_snn_mhsa)
            MODEL="survtri_snn_mhsa"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${MODEL}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --num_heads "$NUM_HEADS"
            )
            ;;
        survtri_mlp_concat)
            MODEL="survtri_mlp_concat"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${MODEL}__${SELECTED_MODALITIES//,/_}"
            fi
            ;;
        survtri_mlp_mhsa)
            MODEL="survtri_mlp_mhsa"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${MODEL}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --num_heads "$NUM_HEADS"
            )
            ;;
        survtri_poe_vae_A)
            MODEL="survtri_poe_vae"
            POE_VARIANT="A"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="${MODEL}__${POE_VARIANT}"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_B)
            MODEL="survtri_poe_vae"
            POE_VARIANT="B"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="${MODEL}__${POE_VARIANT}"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_encoder_lr_ratio "$POE_ENCODER_LR_RATIO"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_C)
            MODEL="survtri_poe_vae"
            POE_VARIANT="C"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="${MODEL}__${POE_VARIANT}"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_B_nopretrain)
            MODEL="survtri_poe_vae_b_nopretrain"
            POE_VARIANT="B"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            # This ablation skips stage1 pretraining by design, so stage1 args are omitted.
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_encoder_lr_ratio "$POE_ENCODER_LR_RATIO"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
            )
            ;;
        survtri_poe_vae_B_kl)
            MODEL="survtri_poe_vae_b_kl"
            POE_VARIANT="B"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_encoder_lr_ratio "$POE_ENCODER_LR_RATIO"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_B_crossstage1)
            MODEL="survtri_poe_vae_b_crossstage1"
            POE_VARIANT="B"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            STAGE1_TAG="$(format_poe_stage1_studies_tag "$POE_STAGE1_STUDIES" | tr '\n' '_')"
            STAGE1_TAG="${STAGE1_TAG%_}"
            if [ -n "$STAGE1_TAG" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__stage1_${STAGE1_TAG}"
            fi
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_encoder_lr_ratio "$POE_ENCODER_LR_RATIO"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --poe_stage1_studies "$POE_STAGE1_STUDIES"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;

        survtri_poe_vae_A_film)
            MODEL="survtri_poe_vae_a_film"
            POE_VARIANT="A"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_B_single)
            MODEL="survtri_poe_vae_b_single"
            POE_VARIANT="B"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_B_multi)
            MODEL="survtri_poe_vae_b_multi"
            POE_VARIANT="B"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_B_film)
            MODEL="survtri_poe_vae_b_film"
            POE_VARIANT="B"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_C_single|survtri_poe_vae_C_single_enum)
            if [ "$PRESET" = "survtri_poe_vae_C_single_enum" ]; then
                MODEL="survtri_poe_vae_c_single_enum"
            else
                MODEL="survtri_poe_vae_c_single"
            fi
            POE_VARIANT="C"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_C_multi)
            MODEL="survtri_poe_vae_c_multi"
            POE_VARIANT="C"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_C_film|survtri_poe_vae_C_film_noenum|survtri_poe_vae_C_film_kl|survtri_poe_vae_C_film_beta0|survtri_poe_vae_C_film_surv0)
            if [ "$PRESET" = "survtri_poe_vae_C_film_kl" ]; then
                MODEL="survtri_poe_vae_c_film_kl"
            elif [ "$PRESET" = "survtri_poe_vae_C_film_surv0" ]; then
                MODEL="survtri_poe_vae_c_film_surv0"
            elif [ "$PRESET" = "survtri_poe_vae_C_film_beta0" ]; then
                # 独立 modality 名：python 按 modality 建结果目录，
                # 若复用 c_film 会覆盖主模型结果
                MODEL="survtri_poe_vae_c_film_beta0"
            else
                MODEL="survtri_poe_vae_c_film"
            fi
            if [ "$PRESET" = "survtri_poe_vae_C_film_noenum" ]; then
                MODEL="survtri_poe_vae_c_film_noenum"
            elif [ "$PRESET" = "survtri_poe_vae_C_film_beta0" ]; then
                POE_BETA_TARGET=0
            fi
            RESULTS_SUBDIR="$MODEL"
            POE_VARIANT="C"
            BAG_LOSS="cox_surv"
            if [ -z "${RESULTS_SUBDIR:-}" ]; then RESULTS_SUBDIR="$MODEL"; fi
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;

        # ========== SurvFusion 变体 ==========
        survfusion_noalign)
            MODEL="survfusion_noalign"
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(
                --num_heads "$NUM_HEADS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survfusion_joint)
            MODEL="survfusion_joint"
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(
                --num_heads "$NUM_HEADS"
                --clip_lambda "$CLIP_LAMBDA"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survfusion_separate_mhsa)
            MODEL="survfusion_separate"
            RESULTS_SUBDIR="${MODEL}_mhsa_$(printf '%g' "$CLIP_WEIGHT_IT")_$(printf '%g' "$CLIP_WEIGHT_IS")_$(printf '%g' "$CLIP_WEIGHT_TS")"
            FUSION_TYPE="mhsa"
            EXTRA_ARGS=(
                --fusion_type "$FUSION_TYPE"
                --num_heads "$NUM_HEADS"
                --clip_weight_IT "$CLIP_WEIGHT_IT"
                --clip_weight_IS "$CLIP_WEIGHT_IS"
                --clip_weight_TS "$CLIP_WEIGHT_TS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survfusion_separate_concat)
            MODEL="survfusion_separate"
            FUSION_TYPE="concat"
            RESULTS_SUBDIR="${MODEL}_${FUSION_TYPE}"
            EXTRA_ARGS=(
                --fusion_type "$FUSION_TYPE"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survfusion_separate_mean)
            MODEL="survfusion_separate"
            FUSION_TYPE="mean_concat"
            RESULTS_SUBDIR="${MODEL}_${FUSION_TYPE}"
            EXTRA_ARGS=(
                --fusion_type "$FUSION_TYPE"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;

        # ========== 缺失模态基线（不含 flexmoe）==========
        modality_concat|modality_concat_zero|modality_concat_mean|modality_concat_resampler_zero|modality_concat_resampler_mean|modality_concat_meanpool_zero|modality_concat_meanpool_mean)
            case "$preset" in
                modality_concat_meanpool_zero)
                    CONCAT_WSI="meanpool"
                    CONCAT_IMPUTE="zero"
                    ;;
                modality_concat_meanpool_mean)
                    CONCAT_WSI="meanpool"
                    CONCAT_IMPUTE="mean"
                    ;;
                modality_concat_zero|modality_concat_resampler_zero)
                    CONCAT_WSI="resampler"
                    CONCAT_IMPUTE="zero"
                    ;;
                modality_concat_mean|modality_concat_resampler_mean)
                    CONCAT_WSI="resampler"
                    CONCAT_IMPUTE="mean"
                    ;;
            esac
            MODEL="modality_concat"
            BAG_LOSS="cox_surv"
            LABEL_DIM=1
            RESULTS_SUBDIR="${MODEL}__${CONCAT_WSI}__${CONCAT_IMPUTE}"
            EXTRA_ARGS=(
                --concat_wsi "$CONCAT_WSI"
                --concat_impute "$CONCAT_IMPUTE"
                --label_dim "$LABEL_DIM"
            )
            ;;
        mvae_poe)
            MODEL="mvae_poe"
            BAG_LOSS="cox_surv"
            LABEL_DIM=1
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
            )
            ;;
        mopoe)
            MODEL="mopoe"
            BAG_LOSS="cox_surv"
            LABEL_DIM=1
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
            )
            ;;
        hgcn)
            MODEL="hgcn"
            RESULTS_SUBDIR="$MODEL"
            BATCH_SIZE=32
            ;;

        *)
            echo "[presets.sh] Unknown preset: $preset" >&2
            exit 2
            ;;
    esac
}
