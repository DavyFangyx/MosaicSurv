#!/bin/bash
set -euo pipefail

# POE 家族只生成主模型 MosaicSurv（其余 A/B/C 系已降级到 Table4 消融）。
# MosaicSurv 公共超参见 configs/z_exp_gen/mosaic_hparams.sh。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.bash"

STUDY="tcga_luad"
EXP_GROUP="${EXP_GROUP:-L0_LUAD_poe_model_val}"

generate_poe_model_val_configs
