#!/bin/bash
# bg.sh — 后台启动 scheduler
# 用法: CUDA_VISIBLE_DEVICES=N bash bg.sh
set -euo pipefail

# Bound native math thread pools; each GPU may have multiple training processes.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

if [ "$#" -ne 0 ]; then
    echo "用法: CUDA_VISIBLE_DEVICES=N bash bg.sh" >&2
    exit 1
fi

GPU_ID="${CUDA_VISIBLE_DEVICES:-}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "错误: 请先设置单个 GPU，例如 CUDA_VISIBLE_DEVICES=4 bash bg.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

shopt -s nullglob
next_worker=1
for existing_log in "$SCRIPT_DIR"/GPU"${GPU_ID}"_worker*.log; do
    log_name="$(basename "$existing_log")"
    if [[ "$log_name" =~ ^GPU${GPU_ID}_worker_?([0-9]+)\.log$ ]]; then
        worker_id="${BASH_REMATCH[1]}"
        if [ "$worker_id" -ge "$next_worker" ]; then
            next_worker=$((worker_id + 1))
        fi
    fi
done

LOG_FILE="GPU${GPU_ID}_worker${next_worker}.log"

nohup bash scheduler.sh > "$LOG_FILE" 2>&1 &
PID=$!

echo "PID: $PID"
echo "日志: $SCRIPT_DIR/$LOG_FILE"
echo "停止: kill $PID"
