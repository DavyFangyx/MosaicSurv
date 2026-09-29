#!/bin/bash
set -euo pipefail

# Model B/C 超参由 _common.bash 写入；当前已注释，需要时再启用。
# Model B：LR=6e-4, LR_STAGE1=8e-5, Reg=7e-5, Beta=0.0164,
# Dropout=0.25, MMHID=128, DecoderHidden=512。
# Model C：Batch=16, LR=1.49e-4, Reg=1.25e-5,
# Beta=0.268 (BETAFIX=true), Lambda=0.361。A 仍用默认 batch 128。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.bash"

STUDY="tcga_kirp"
EXP_GROUP="${EXP_GROUP:-L0_KIRP_poe_model_val}"

generate_poe_model_val_configs
