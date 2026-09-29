#!/bin/bash
# configs/z_exp_gen/Table1_cindex_main/gen_baselines/_common.bash
# 共享的全模型验证 config 生成逻辑。

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat <<'EOF'
[_common.bash] 这是共享脚本，不会单独生成 config。

请运行下面任一入口脚本：
  bash configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_tcga_brca_full_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_tcga_coad_full_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_tcga_kich_full_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_tcga_kirc_full_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_tcga_kirp_full_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_tcga_lihc_full_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_tcga_prad_full_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_tcga_read_full_model_val.sh
EOF
    exit 2
fi

generate_full_model_val_configs() {
    : "${STUDY:?STUDY is required}"
    : "${EXP_GROUP:?EXP_GROUP is required}"

    local script_dir out_dir batch_size clinic_experiment gene_experiment wsi_experiment
    local run_name_wsi run_name_clinic run_name_gene_raw run_name_gene_f run_name_multi gene_tag
    local file_prefix exp_group_prefix seq created skipped preset modality_tag fname target run_name

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
    out_dir="${OUT_DIR:-$script_dir/configs/queue}"
    batch_size="${BATCH_SIZE:-1}"
    clinic_experiment="${CLINIC_EXPERIMENT:-L0}"
    gene_experiment="${GENE_EXPERIMENT:-scFoundation_embedding_cell_norm}"
    wsi_experiment="${WSI_EXPERIMENT:-uni_v1}"
    exp_group_prefix=""
    if [[ "$clinic_experiment" == "L0" ]]; then
        exp_group_prefix="L0_"
    fi
    case "$gene_experiment" in
        scFoundation_embedding_cell_norm)
            gene_tag="cell_norm"
            ;;
        scFoundation_embedding_cell_raw)
            gene_tag="cell_raw"
            ;;
        scFoundation_embedding_gene_norm)
            gene_tag="gene_norm"
            ;;
        scFoundation_embedding_gene_raw)
            gene_tag="gene_raw"
            ;;
        *)
            gene_tag="$gene_experiment"
            ;;
    esac

    run_name_wsi="${RUN_NAME_WSI:-${STUDY}__${wsi_experiment}}"
    run_name_clinic="${RUN_NAME_CLINIC:-${STUDY}__${clinic_experiment}}"
    run_name_gene_raw="${RUN_NAME_GENE_RAW:-${STUDY}__csvraw}"
    run_name_gene_f="${RUN_NAME_GENE_F:-${STUDY}__${gene_experiment}}"
    run_name_multi="${RUN_NAME_MULTI:-${STUDY}__${clinic_experiment}__${gene_tag}__${wsi_experiment}}"
    if [[ "$EXP_GROUP" != "${exp_group_prefix}"* ]]; then
        EXP_GROUP="${exp_group_prefix}${EXP_GROUP}"
    fi
    file_prefix="${FILE_PREFIX:-${exp_group_prefix}${STUDY#tcga_}_full_model_val}"

    local -a wsi_models=(
        abmil_wsi
#        mlp_wsi
        transmil_wsi
    )

    local -a clinic_models=(
#        mlp_clinic_mean
        mlp_clinic_flatten
#        snn_clinic_mean
        snn_clinic_flatten
#        clinic_cox
    )

    local -a gene_models=(
#        mlp_gene
#        snn_gene
        mlp_gene_f
        snn_gene_f
    )

    local -a multi_models=(
        survpc_f
        porpoise
        survpath
        mcat
        survgc_f
        survpgc_f
        hgcn
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

    for preset in "${wsi_models[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "%s__%03d__P__%s.conf" "$file_prefix" "$seq" "$preset")
        target="$out_dir/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$run_name_wsi
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$clinic_experiment
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
BATCH_SIZE=$batch_size
EOF
)"
    done

    for preset in "${clinic_models[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "%s__%03d__C__%s.conf" "$file_prefix" "$seq" "$preset")
        target="$out_dir/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$run_name_clinic
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$clinic_experiment
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
BATCH_SIZE=$batch_size
EOF
)"
    done

    for preset in "${gene_models[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "%s__%03d__G__%s.conf" "$file_prefix" "$seq" "$preset")
        target="$out_dir/$fname"
        run_name="$run_name_gene_f"
        case "$preset" in
            mlp_gene|snn_gene)
                run_name="$run_name_gene_raw"
                ;;
        esac
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$run_name
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$clinic_experiment
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
BATCH_SIZE=$batch_size
EOF
)"
    done

    for preset in "${multi_models[@]}"; do
        seq=$((seq + 1))
        modality_tag="PCG"
        case "$preset" in
            survpc_f)
                modality_tag="PC"
                ;;
            porpoise|survpath|mcat)
                modality_tag="PG"
                ;;
            survgc_f)
                modality_tag="CG"
                ;;
            # hgcn 与 survpgc_f 同组：三模态 PCG（默认值即 PCG，此处显式列出）
            hgcn|survpgc_f)
                modality_tag="PCG"
                ;;
        esac

        fname=$(printf "%s__%03d__%s__%s.conf" "$file_prefix" "$seq" "$modality_tag" "$preset")
        target="$out_dir/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$run_name_multi
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$clinic_experiment
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
BATCH_SIZE=$batch_size
EOF
)"
    done

    echo "Generated $created new configs in $out_dir"
    echo "Total indexed configs this round: $seq"
    echo "Skipped existing configs: $skipped"
    echo "Study: $STUDY"
    echo "Clinic embedding: $clinic_experiment"
    echo "Gene embedding: $gene_experiment"
    echo "WSI embedding: $wsi_experiment"
    echo "Run name (WSI): $run_name_wsi"
    echo "Run name (Clinic): $run_name_clinic"
    echo "Run name (Gene raw): $run_name_gene_raw"
    echo "Run name (Gene FM): $run_name_gene_f"
    echo "Run name (Multi): $run_name_multi"
    echo "Experiment group: $EXP_GROUP"
}
