#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Table4（_Cindex_Main 子实验）MosaicSurv 消融生成：
# 完整模态 C-index 评测，A/B/C 三组消融全部进行。
# MosaicSurv 公共超参数统一来自 configs/z_exp_gen/mosaic_hparams.sh（单组超参），
# 不再按 Cfilm_Hparam_Eval 遗留清单（旧名，未随本次改名迁移）的 T4 多行逐组生成。
# 结果目录：results/Table4_Abaltion_Test/_Cindex_Main/{study}__cell_norm__uni_v1/。

# 兄弟实验（_Modality_Missing 子实验，仅 A 组、Table2 式缺失模态评测）见
# gen_Table4_Abaltion_Test_Modality_Missing.sh。

export EXP_GROUP="Table4_Abaltion_Test/_Cindex_Main"

bash "$SCRIPT_DIR/Table4_Abaltion_Test/gen_ablation_test.sh"
