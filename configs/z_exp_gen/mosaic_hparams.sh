#!/bin/bash
# configs/z_exp_gen/mosaic_hparams.sh
# MosaicSurv 主模型最终超参数 —— Test1-4 的唯一来源与入口，直接修改本文件即可。
# 取值来源：optuna t018_alpha_beta_learn（Cfilm_Hparam_Eval/cfilm_table1_hparams.csv）
#
# 各 test 生成脚本 source 本文件后，把 mosaic_conf_lines 的输出
# （或各 MOSAIC_* 变量）嵌入生成的 .conf；键名与 run.sh 读取的 .conf 键一致。
# 不做环境变量覆盖：生成的 .conf 永远只取这里定义的 MOSAIC_* 值，
# 避免与各 test 脚本自带的同名变量（如 BATCH_SIZE=128）打架。
#
# 最终生效层级（后者覆盖前者）：
#   configs/defaults.conf < 生成的 .conf < configs/presets.sh 个别硬编码
#   （如 hgcn 固定 BATCH_SIZE=32，不受本文件影响）

MOSAIC_LR=0.001
MOSAIC_REG=0.0001
MOSAIC_POE_SURV_LAMBDA=1.0
MOSAIC_POE_BETA_TARGET=0.1
MOSAIC_POE_MODALITY_DROPOUT=0.35
MOSAIC_POE_MMHID=128
MOSAIC_BATCH_SIZE=16
MOSAIC_ALPHAFIX=false
MOSAIC_ALPHAPGC=""
MOSAIC_BETAFIX=false

# 输出 MosaicSurv 公共超参的 .conf 行（每行一个 KEY=VALUE）。
# 生成器可整段嵌入，随后再追加个别覆盖行（如 Table4 nojeffreys 的 POE_BETA_TARGET=0、
# Table3 unified_mask_csv 的 POE_MODALITY_DROPOUT=0），后写者生效。
mosaic_conf_lines() {
    printf 'LR=%s\n' "$MOSAIC_LR"
    printf 'REG=%s\n' "$MOSAIC_REG"
    printf 'POE_SURV_LAMBDA=%s\n' "$MOSAIC_POE_SURV_LAMBDA"
    printf 'POE_BETA_TARGET=%s\n' "$MOSAIC_POE_BETA_TARGET"
    printf 'POE_MODALITY_DROPOUT=%s\n' "$MOSAIC_POE_MODALITY_DROPOUT"
    printf 'POE_MMHID=%s\n' "$MOSAIC_POE_MMHID"
    printf 'BATCH_SIZE=%s\n' "$MOSAIC_BATCH_SIZE"
    printf 'ALPHAFIX=%s\n' "$MOSAIC_ALPHAFIX"
    printf 'ALPHAPGC=%s\n' "$MOSAIC_ALPHAPGC"
    printf 'BETAFIX=%s\n' "$MOSAIC_BETAFIX"
}
