#!/bin/bash
set -euo pipefail

# Table1 ours 入口。
# POE 家族只生成主模型 MosaicSurv（其余 A/B/C 系已降级到 Table4 消融）。
# MosaicSurv 公共超参见 configs/z_exp_gen/mosaic_hparams.sh。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$SCRIPT_DIR/gen_tcga_brca_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_coad_poe_model_val.sh"
#bash "$SCRIPT_DIR/gen_tcga_kich_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_kirc_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_kirp_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_lihc_poe_model_val.sh"
#bash "$SCRIPT_DIR/gen_tcga_prad_poe_model_val.sh"
#bash "$SCRIPT_DIR/gen_tcga_read_poe_model_val.sh"
