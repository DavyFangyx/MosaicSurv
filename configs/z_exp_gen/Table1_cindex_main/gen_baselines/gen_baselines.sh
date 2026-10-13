#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$SCRIPT_DIR/gen_tcga_brca_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_coad_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_kich_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_kirc_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_kirp_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_lihc_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_luad_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_lusc_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_prad_full_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_read_full_model_val.sh"
