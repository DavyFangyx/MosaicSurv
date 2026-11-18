#!/bin/bash
# configs/z_exp_gen/gen_all_L1.sh
#
# 生成 “C 模态改用 L1 临床 prompt” 的整套实验配置（只生成 config，不训练）。
#
# L0/L1 的差别只在临床 prompt 字段数：L0=4 个 token，L1=10 个 token
# （L1 = L0 + primary_diagnosis / morphology / tissue_or_organ_of_origin /
#  laterality / year_of_diagnosis / age_at_diagnosis）。训练代码通过
#  utils/core_utils.py::_infer_clinic_shape 自动读取 token 数，模型结构会随
#  10×512 自动适配，不需要改代码或超参。
#
# 重要：L1 必须写入与 L0 独立的实验组/结果目录。Table4 的 RUN_NAME 不含
# clinic_experiment，stage1 cache 的 key 也不含，混用会覆盖 L0 结果或误用
# L0 的 stage1 checkpoint。
#
# 用法（项目根目录）：
#   bash configs/z_exp_gen/gen_all_L1.sh
#   STUDIES="tcga_coad tcga_kirc" bash configs/z_exp_gen/gen_all_L1.sh   # 试点
#   OUT_DIR=/path/to/queue bash configs/z_exp_gen/gen_all_L1.sh
#
# 生成目录默认 configs/queue_L1。
# 训练：bash run.sh <conf>；批跑时建议给 scheduler 指定独立的 running/done/failed：
#   RUNNING_DIR=configs/L1/running DONE_DIR=configs/L1/done \
#   FAIL_DIR=configs/L1/failed bash scheduler.sh configs/queue_L1
#
# 结果目录（与 L0 批次并列，不覆盖旧结果）：
#   Table1: results/Table1_Cindex_Main/L1Test/L1_<STUDY>_{full,poe}_model_val/
#   Table2: results/Table2_Baselines_L1/<study>__L1__gene_raw__uni_v1/
#   Table3: results/Table3_MissingRate_L1/<study>__L1__gene_raw__uni_v1__pW_G_C/
#   Table4: results/Table4_Abaltion_Test_L1/_Cindex_Main|_Modality_Missing/
#
# 展示输出统一写 results_display_l1/，不要写 results_display/。
#
# Table1 的 P-only / G-only / P+G 模型不消费 C，默认不生成重训配置，
# 由 configs/z_exp_gen/link_table1_L0_shared_models.sh 把 L0 结果硬链到 L1Test。
# 如需全部重跑：TABLE1_REUSE_L0=0 bash configs/z_exp_gen/gen_all_L1.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

export CLINIC_EXPERIMENT=L1
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue_L1}"
mkdir -p "$OUT_DIR"
export OUT_DIR

if [ -n "${STUDIES:-}" ]; then
    read -r -a studies <<< "$STUDIES"
else
    studies=(tcga_brca tcga_coad tcga_kirc tcga_kirp tcga_lihc tcga_luad tcga_lusc)
fi

echo "============================================================"
echo "[L1] clinic experiment : $CLINIC_EXPERIMENT"
echo "[L1] studies          : ${studies[*]}"
echo "[L1] config output    : $OUT_DIR"
echo "============================================================"

# ---- Table1：完整模态对比 ----
# 每个 study 单独一个 EXP_GROUP，目录名必须满足 results_display 的
# L{level}_{STUDY}_{full|poe}_model_val 约定。
# Table1 的 P/G/PG 模型不消费 C，结果与 L0 完全等价；默认在生成阶段跳过，
# 后续用 link_table1_L0_shared_models.sh 把 L0 结果硬链进 L1Test。
TABLE1_SKIP_TAGS=""
if [ "${TABLE1_REUSE_L0:-1}" = "1" ]; then
    TABLE1_SKIP_TAGS="P,G,PG"
fi
for study in "${studies[@]}"; do
    token="${study#tcga_}"
    upper="$(printf '%s' "$token" | tr '[:lower:]' '[:upper:]')"

    SKIP_MODALITY_TAGS="$TABLE1_SKIP_TAGS" \
    EXP_GROUP="Table1_Cindex_Main/L1Test/L1_${upper}_full_model_val" \
    FILE_PREFIX="L1_${token}_full_model_val" \
        bash "$SCRIPT_DIR/configs/z_exp_gen/Table1_cindex_main/gen_baselines/gen_${study}_full_model_val.sh"

    EXP_GROUP="Table1_Cindex_Main/L1Test/L1_${upper}_poe_model_val" \
    FILE_PREFIX="L1_${token}_poe_model_val" \
        bash "$SCRIPT_DIR/configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_${study}_poe_model_val.sh"
done

# ---- Table2：缺失模态基线（完整病例训练，七子集推理） ----
EXP_GROUP=Table2_Baselines_L1 STUDIES="${studies[*]}" \
    bash "$SCRIPT_DIR/configs/z_exp_gen/gen_Table2_missing_modality_baselines.sh"

# ---- Table3：训练缺失配比 ----
EXP_GROUP=Table3_MissingRate_L1 STUDIES="${studies[*]}" \
    bash "$SCRIPT_DIR/configs/z_exp_gen/gen_Table3_missing_rate_test.sh"

# ---- Table4：消融（完整模态 + Table2 式缺失评测） ----
EXP_GROUP="Table4_Abaltion_Test_L1/_Cindex_Main" STUDIES="${studies[*]}" \
    bash "$SCRIPT_DIR/configs/z_exp_gen/gen_Table4_Abaltion_Test_Cindex_Main.sh"

EXP_GROUP="Table4_Abaltion_Test_L1/_Modality_Missing" STUDIES="${studies[*]}" \
    bash "$SCRIPT_DIR/configs/z_exp_gen/gen_Table4_Abaltion_Test_Modality_Missing.sh"

echo ""
echo "[L1] config generation done: $OUT_DIR"
echo "[L1] configs: $(find "$OUT_DIR" -maxdepth 1 -name '*.conf' | wc -l)"
if [ "${TABLE1_REUSE_L0:-1}" = "1" ]; then
    echo "[L1] Table1 只保留消费 C 的模型：clinic-only / SurvPC / SurvGC / SurvPGC / HGCN / MosaicSurv。"
    echo "[L1] 训练后运行 link_table1_L0_shared_models.sh 复用 L0 的 P/G/PG 结果。"
else
    echo "[L1] Table1 全部模型都会重训（TABLE1_REUSE_L0=0）。"
fi
echo "[L1] 展示输出请写 results_display_l1/。"
