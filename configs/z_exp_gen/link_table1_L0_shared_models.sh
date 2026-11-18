#!/bin/bash
# configs/z_exp_gen/link_table1_L0_shared_models.sh
#
# 把 Table1 L0 批次中“不消费 C 模态”的模型目录物化到 L1Test，供
# results_display 直接扫描。默认用硬链接（cp -al），不占额外空间；跨文件
# 系统时自动退回复制。
#
# 复用的模型（输入只有 P / G / P+G，与 C 编码无关）：
#   P  : abmil_wsi, transmil_wsi
#   G  : mlp_gene_f, snn_gene_f
#   PG : porpoise, survpath, mcat
# 不复用：clinic-only、SurvPC、SurvGC、SurvPGC、HGCN、MosaicSurv —— 这些必须
# 用 L1 重训，否则模型结构与 L1 的 10 个 clinic token 不匹配。
#
# 用法（项目根目录）：
#   bash configs/z_exp_gen/link_table1_L0_shared_models.sh
#   MODE=copy bash configs/z_exp_gen/link_table1_L0_shared_models.sh   # 复制而非硬链
#   STUDIES="tcga_coad tcga_kirc" bash configs/z_exp_gen/link_table1_L0_shared_models.sh
#
# 注意：脚本只写 results/Table1_Cindex_Main/L1Test/，不会动 L0Test。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
GROUP_DIR="${GROUP_DIR:-Table1_Cindex_Main}"
L0_TEST_DIR="${L0_TEST_DIR:-L0Test}"
L1_TEST_DIR="${L1_TEST_DIR:-L1Test}"
MODE="${MODE:-link}"
GENE_RUN_DIR="scFoundation_embedding_cell_norm"

if [ -n "${STUDIES:-}" ]; then
    read -r -a studies <<< "$STUDIES"
else
    studies=(tcga_brca tcga_coad tcga_kirc tcga_kirp tcga_lihc tcga_luad tcga_lusc)
fi

P_MODELS=(abmil_wsi transmil_wsi)
G_MODELS=(mlp_gene_f snn_gene_f)
PG_MODELS=(porpoise survpath mcat)

materialize() {
    local src="$1"
    local dst="$2"
    if [ ! -d "$src" ]; then
        echo "[WARN] missing L0 model dir: $src" >&2
        return 0
    fi
    if [ -e "$dst" ]; then
        echo "[SKIP] exists: $dst"
        return 0
    fi
    mkdir -p "$(dirname "$dst")"
    if [ "$MODE" = "copy" ]; then
        cp -a "$src" "$dst"
    else
        # 先链到唯一临时名，成功后再 rename；失败时保留临时目录并报告，
        # 不执行删除（项目规则禁止 rm）。
        local tmp="${dst}.__l1tmp.$$.$RANDOM"
        if [ -e "$tmp" ]; then
            echo "[ERROR] temp path already exists: $tmp" >&2
            return 1
        fi
        if cp -al "$src" "$tmp" 2>/dev/null; then
            mv "$tmp" "$dst"
        else
            if [ -e "$tmp" ]; then
                echo "[WARN] cp -al left partial temp (not deleted): $tmp" >&2
            fi
            cp -a "$src" "$dst"
        fi
    fi
    echo "[OK] $dst <- $src"
}

for study in "${studies[@]}"; do
    token="${study#tcga_}"
    upper="$(printf '%s' "$token" | tr '[:lower:]' '[:upper:]')"
    src_full="$RESULTS_ROOT/$GROUP_DIR/$L0_TEST_DIR/L0_${upper}_full_model_val"
    dst_full="$RESULTS_ROOT/$GROUP_DIR/$L1_TEST_DIR/L1_${upper}_full_model_val"

    if [ ! -d "$src_full" ]; then
        echo "[WARN] missing L0 study dir: $src_full" >&2
        continue
    fi
    mkdir -p "$dst_full"

    # P-only
    for model in "${P_MODELS[@]}"; do
        materialize "$src_full/${study}__uni_v1/$model" \
                    "$dst_full/${study}__uni_v1/$model"
    done

    # G-only
    for model in "${G_MODELS[@]}"; do
        materialize "$src_full/${study}__${GENE_RUN_DIR}/$model" \
                    "$dst_full/${study}__${GENE_RUN_DIR}/$model"
    done

    # P+G：L0 run 名含 L0，L1 树里改成 L1 run 名，保持目录命名一致。
    for model in "${PG_MODELS[@]}"; do
        materialize "$src_full/${study}__L0__cell_norm__uni_v1/$model" \
                    "$dst_full/${study}__L1__cell_norm__uni_v1/$model"
    done
done

echo "[DONE] Table1 L0 shared models materialized into:"
echo "       $RESULTS_ROOT/$GROUP_DIR/$L1_TEST_DIR"
