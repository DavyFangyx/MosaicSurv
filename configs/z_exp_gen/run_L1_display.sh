#!/bin/bash
# configs/z_exp_gen/run_L1_display.sh
#
# L1 批次展示入口：所有输出统一写 results_display_l1/，不写 results_display/。
# 只读 results/ 下的 L1 训练结果，不触发训练。
#
# 用法（项目根目录；建议在 SurvPGC_results_display 环境）：
#   bash configs/z_exp_gen/run_L1_display.sh table1
#   bash configs/z_exp_gen/run_L1_display.sh table2
#   bash configs/z_exp_gen/run_L1_display.sh table3
#   bash configs/z_exp_gen/run_L1_display.sh table4
#   bash configs/z_exp_gen/run_L1_display.sh figc
#   bash configs/z_exp_gen/run_L1_display.sh all
#
# 常用环境变量：
#   PYTHON_BIN=python            解释器
#   OUT_ROOT=results_display_l1  输出根目录
#   GPU=4                        Table1 cox 前向使用的 GPU
#   SKIP_COX=1                   Table1 跳过 cox Breslow 前向（IBS/AUC 显示为 -）
#   CONFIRM_ARCHIVE=1            显式确认：先列旧产物清单，再归档到
#                                AnysoleWorkspace/_trash/<date>/ 后重新生成。
#                                默认 0，不替用户确认；存在旧产物时展示脚本会
#                                打印完整清单并退出。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/results_display_l1}"
GPU="${GPU:-4}"
STAGE="${1:-all}"
case "$GPU" in
    1|2|3|4) ;;
    *)
        echo "[L1-DISPLAY] GPU 限定为 1/2/3/4，当前 GPU=$GPU" >&2
        exit 2
        ;;
esac

CONFIRM_ARCHIVE="${CONFIRM_ARCHIVE:-0}"
archive_flag=()
if [ "$CONFIRM_ARCHIVE" = "1" ]; then
    archive_flag=(--confirm-archive)
fi

mkdir -p "$OUT_ROOT"

run_table1() {
    local cox_flag=()
    if [ "${SKIP_COX:-0}" = "1" ]; then
        cox_flag=(--skip-cox)
    fi
    "$PYTHON_BIN" results_display/scripts/Table1_Generate_All.py \
        --test-dir L1Test \
        --output_root "$OUT_ROOT" \
        --gpu "$GPU" \
        "${cox_flag[@]}" \
        "${archive_flag[@]}"
}

run_table2() {
    "$PYTHON_BIN" results_display/scripts/Table2_Modality_Missing_Tset.py \
        --group-dir Table2_Baselines_L1 \
        --run-suffix L1__gene_raw__uni_v1 \
        --output_root "$OUT_ROOT"
}

run_table3() {
    "$PYTHON_BIN" results_display/scripts/Table3_Train_MissingRate.py \
        --group-dir Table3_MissingRate_L1 \
        --output_root "$OUT_ROOT" \
        "${archive_flag[@]}"
}

run_table4() {
    "$PYTHON_BIN" results_display/scripts/Table4_Abaltion_Test_Cindex_Main.py \
        --group-dir Table4_Abaltion_Test_L1/_Cindex_Main \
        --display-dir Table4_Abaltion_Test/_Cindex_Main \
        --output-root "$OUT_ROOT"
    "$PYTHON_BIN" results_display/scripts/Table4_Abaltion_Test_Modality_Missing.py \
        --group-dir Table4_Abaltion_Test_L1/_Modality_Missing \
        --output-root "$OUT_ROOT"
}

run_figc() {
    "$PYTHON_BIN" results_display/scripts/FigC_KM_curves.py \
        --test-dir L1Test \
        --dataset all \
        --output-root "$OUT_ROOT/FigC_KM_curves" \
        "${archive_flag[@]}"
}

case "$STAGE" in
    table1) run_table1 ;;
    table2) run_table2 ;;
    table3) run_table3 ;;
    table4) run_table4 ;;
    figc)   run_figc ;;
    all)
        run_table1
        run_table2
        run_table3
        run_table4
        run_figc
        ;;
    *)
        echo "usage: bash configs/z_exp_gen/run_L1_display.sh {table1|table2|table3|table4|figc|all}" >&2
        exit 2
        ;;
esac

echo "[L1-DISPLAY] done: $STAGE -> $OUT_ROOT"
