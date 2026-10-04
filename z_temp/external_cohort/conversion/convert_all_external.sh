#!/usr/bin/env bash
# convert_all_external.sh
# 全量把 /data/lizhe/.../eval_tiff 下的 31 张扁平 BigTIFF 转成 openslide 可读金字塔 tiled BigTIFF。
#
#   cd <本目录> && nohup bash convert_all_external.sh > logs/_batch.log 2>&1 &
#
# 特性:
#   * 逐张串行, 单张失败不中断(最后汇总失败名单); 输出已存在则跳过 => 可直接重跑续传
#   * 单张日志落在 logs/<stem>.log
#   * 文件名前导下划线/特殊字符安全(全程数组+引号, 不做字符串拼接)
#
# 时间: 实测单张 8.27GB 约 70s(冷缓存) => 31 张串行约 50-60 分钟。
# 想并行(时间约减半, 内存 96 核/1TB 完全够):
#   ls /data/lizhe/Medteam_projects/test/eval_tiff/*/*.tif | \
#     xargs -P 2 -I{} /data/fangyuxuan/miniconda3/envs/trident/bin/python \
#       convert_flat_to_pyramid.py {} --outdir <OUT> --skip-existing
set -uo pipefail

PY=/data/fangyuxuan/miniconda3/envs/trident/bin/python
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)                 # .../SurvPGC_github_init
SCRIPT="$HERE/convert_flat_to_pyramid.py"
SRC=/data/lizhe/Medteam_projects/test/eval_tiff    # 只读!
OUT="$ROOT/SurvPGC_Workspace/external_fukun"
LOGDIR="$HERE/logs"

mkdir -p "$OUT" "$LOGDIR"
mapfile -t FILES < <(find "$SRC" -mindepth 2 -maxdepth 2 -name '*.tif' | sort)
echo "[$(date '+%F %T')] found ${#FILES[@]} source files -> $OUT"

failed=()
for f in "${FILES[@]}"; do
    stem=$(basename "$f" .tif)
    if [[ -f "$OUT/${stem}_pyr.tif" ]]; then
        echo "[skip]         $stem (output exists)"
        continue
    fi
    echo "[$(date '+%F %T')] start $stem"
    if "$PY" "$SCRIPT" "$f" --outdir "$OUT" > "$LOGDIR/${stem}.log" 2>&1; then
        sz=$(stat -c %s "$OUT/${stem}_pyr.tif" 2>/dev/null || echo 0)
        echo "[$(date '+%F %T')] OK    $stem  ($((sz / 1000000)) MB)"
    else
        echo "[$(date '+%F %T')] FAIL  $stem  (see logs/${stem}.log)"
        failed+=("$stem")
    fi
done

echo "[$(date '+%F %T')] DONE. ok=$(( ${#FILES[@]} - ${#failed[@]} ))/${#FILES[@]} failures=${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then printf '  FAILED: %s\n' "${failed[@]}"; fi
exit 0
