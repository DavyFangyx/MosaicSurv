#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Table4 MosaicSurv 消融生成。
# MosaicSurv 公共超参数统一来自 configs/z_exp_gen/mosaic_hparams.sh（单组超参），
# 不再按 Cfilm_Hparam_Eval 遗留清单（旧名，未随本次改名迁移）的 T4 多行逐组生成。

bash "$SCRIPT_DIR/Table4_Abaltion_Test/gen_ablation_test.sh"
