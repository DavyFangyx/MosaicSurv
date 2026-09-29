#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_single_modal_common.bash"

STUDY="tcga_brca"
EXP_GROUP="${EXP_GROUP:-L0_BRCA_poe_single_model_val}"

generate_poe_single_modal_bc_configs
