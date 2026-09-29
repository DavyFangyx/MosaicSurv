#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Table4 Cfilm 消融生成。
# Cfilm 公共超参数统一来自 configs/z_exp_gen/cfilm_hparams.sh（单组超参），
# 不再按 Cfilm_Hparam_Eval 清单的 T4 多行逐组生成。

bash "$SCRIPT_DIR/Table4_Abaltion_Test/gen_ablation_test.sh"
