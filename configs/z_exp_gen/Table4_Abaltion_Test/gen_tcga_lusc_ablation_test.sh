#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.bash"

STUDY="tcga_lusc"
EXP_GROUP="${EXP_GROUP:-Table4_Abaltion_Test}"

generate_table4_ablation_test_configs
