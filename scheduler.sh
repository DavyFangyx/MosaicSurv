#!/bin/bash
# scheduler.sh — 串行执行 queue 中的全部 config，失败任务不阻塞后续任务
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

QUEUE_DIR="${1:-configs/queue}"
# 支持为不同表征/实验批次指定独立的中间目录，避免与旧批次的同名 config 互踩。
RUNNING_DIR="${RUNNING_DIR:-configs/running}"
DONE_DIR="${DONE_DIR:-configs/done}"
FAIL_DIR="${FAIL_DIR:-configs/failed}"
mkdir -p "$QUEUE_DIR" "$RUNNING_DIR" "$DONE_DIR" "$FAIL_DIR"

echo "============================================================"
echo "[scheduler] queue: $QUEUE_DIR"
echo "[scheduler] GPU:   ${CUDA_VISIBLE_DEVICES:-default}"
echo "[scheduler] start: $(date +'%F %T %z')"
echo "============================================================"

processed=0
failed=0
shopt -s nullglob

count_running() {
    local running_files=("$RUNNING_DIR"/*.conf)
    printf '%s\n' "${#running_files[@]}"
}

while :; do
    queue_files=("$QUEUE_DIR"/*.conf)
    cfg="${queue_files[0]:-}"
    if [ -z "$cfg" ]; then
        running_now="$(count_running)"
        if [ "$running_now" -gt 0 ]; then
            echo "[scheduler] queue empty, but $running_now task(s) still running; waiting..."
            sleep 10
            continue
        fi
        echo "[scheduler] queue empty, stop."
        break
    fi

    name="$(basename "$cfg")"
    claimed="$RUNNING_DIR/$name"

    if ! mv "$cfg" "$claimed" 2>/dev/null; then
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date +'%F %T %z')] START  $name"
    echo "------------------------------------------------------------"

    if bash run.sh "$claimed"; then
        mv "$claimed" "$DONE_DIR/$name"
        echo "[$(date +'%F %T %z')] OK     $name"
        processed=$((processed + 1))
    else
        mv "$claimed" "$FAIL_DIR/$name"
        echo "[$(date +'%F %T %z')] FAIL   $name"
        failed=$((failed + 1))
    fi
done

remaining_files=("$QUEUE_DIR"/*.conf)
remaining="${#remaining_files[@]}"

echo ""
echo "============================================================"
echo "[scheduler] finished: $(date +'%F %T %z')"
echo "[scheduler] processed=$processed failed=$failed queue_remaining=$remaining"
echo "============================================================"
