#!/bin/bash
# configs/z_exp_gen/cfilm_hparams.sh
# Cfilm 主模型最终超参数 —— Test1-4 的唯一来源与入口，直接修改本文件即可。
#
# 各 test 生成脚本 source 本文件后，把 cfilm_conf_lines 的输出
# （或各 CFILM_* 变量）嵌入生成的 .conf；键名与 run.sh 读取的 .conf 键一致。
# 不做环境变量覆盖：生成的 .conf 永远只取这里定义的 CFILM_* 值，
# 避免与各 test 脚本自带的同名变量（如 BATCH_SIZE=128）打架。
#
# 最终生效层级（后者覆盖前者）：
#   configs/defaults.conf < 生成的 .conf < configs/presets.sh 个别硬编码
#   （如 hgcn 固定 BATCH_SIZE=32，不受本文件影响）

CFILM_LR=0.001
CFILM_REG=1e-05
CFILM_POE_SURV_LAMBDA=2.0
CFILM_POE_BETA_TARGET=0.1
CFILM_POE_MODALITY_DROPOUT=0.35
CFILM_POE_MMHID=128
CFILM_BATCH_SIZE=16
CFILM_ALPHAFIX=false
CFILM_ALPHAPGC=""
CFILM_BETAFIX=false

# 输出 Cfilm 公共超参的 .conf 行（每行一个 KEY=VALUE）。
# 生成器可整段嵌入，随后再追加个别覆盖行（如 Table4 beta0 的 POE_BETA_TARGET=0、
# Table3 unified_mask_csv 的 POE_MODALITY_DROPOUT=0），后写者生效。
cfilm_conf_lines() {
    printf 'LR=%s\n' "$CFILM_LR"
    printf 'REG=%s\n' "$CFILM_REG"
    printf 'POE_SURV_LAMBDA=%s\n' "$CFILM_POE_SURV_LAMBDA"
    printf 'POE_BETA_TARGET=%s\n' "$CFILM_POE_BETA_TARGET"
    printf 'POE_MODALITY_DROPOUT=%s\n' "$CFILM_POE_MODALITY_DROPOUT"
    printf 'POE_MMHID=%s\n' "$CFILM_POE_MMHID"
    printf 'BATCH_SIZE=%s\n' "$CFILM_BATCH_SIZE"
    printf 'ALPHAFIX=%s\n' "$CFILM_ALPHAFIX"
    printf 'ALPHAPGC=%s\n' "$CFILM_ALPHAPGC"
    printf 'BETAFIX=%s\n' "$CFILM_BETAFIX"
}
