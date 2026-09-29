#!/bin/bash
# 生成 BRCA / COAD / KIRC / KIRP / LIHC 的 SurvTriPoEVAE 单模态 B/C 配置
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$SCRIPT_DIR/Table1_cindex_main/gen_ours/gen_tcga_brca_poe_single_modal_BC.sh"
bash "$SCRIPT_DIR/Table1_cindex_main/gen_ours/gen_tcga_coad_poe_single_modal_BC.sh"
bash "$SCRIPT_DIR/Table1_cindex_main/gen_ours/gen_tcga_kirc_poe_single_modal_BC.sh"
bash "$SCRIPT_DIR/Table1_cindex_main/gen_ours/gen_tcga_kirp_poe_single_modal_BC.sh"
bash "$SCRIPT_DIR/Table1_cindex_main/gen_ours/gen_tcga_lihc_poe_single_modal_BC.sh"

echo "[gen_SurvTriPoEVAE_single_modal_BC.sh] done."
。；
