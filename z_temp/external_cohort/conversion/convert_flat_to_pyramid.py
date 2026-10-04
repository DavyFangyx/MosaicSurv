#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_flat_to_pyramid.py
==========================
把"扁平单页 BigTIFF"(Motic 导出的 LZW 未分块 tif)转成 **openslide 4.x 可读的
金字塔 tiled BigTIFF**, 供 TRIDENT 管线使用。

只用 trident env 里**已有**的 tifffile + imagecodecs + numpy, 不装任何新依赖。

实测得到的三条硬约束(细节见 plan.md)
------------------------------------
1) 金字塔层必须带 `subfiletype=1` (FILETYPE_REDUCEDIMAGE)。
   openslide 4.0 的 generic-TIFF reader:
       - 普通 IFD 链(无 subfiletype) → 只认第 1 层 (level_count=1)   ✗ 本脚本曾踩坑
       - SubIFD 金字塔              → 也只认第 1 层                  ✗
       - IFD 链 + subfiletype=1     → 正确识别全部层                 ✓
2) 每层尺寸用 **floor 除法**(如 62755 → 31377), 不要用 ceil。
   因为 openslide 的 level_downsamples = level0_dim / level_dim, 而
   TRIDENT `WSIPatcher._prepare()` 里 `int(level_downsamples[level])` 是**截断**取整:
       floor: 62755/31377 = 2.00003 → int()=2  ✓  读图尺寸正确
       ceil : 62755/31378 = 1.99997 → int()=1  ✗  读图尺寸算成 2 倍(静默错误!)
3) mpp 通过 TIFF 的 XResolution/ResolutionUnit 写进去 → openslide 暴露成
   `tiff.XResolution` / `tiff.ResolutionUnit` → TRIDENT `_fetch_mpp()` 读得到。
   默认 mpp=0.25 (40x), 这样 seg@10x 的 downsample = 1.0/0.25 = 4.0
   正好命中 4x 层(否则会掉回 level0 读 2048px 大图, 慢到不可用)。

用法
----
  python convert_flat_to_pyramid.py /path/in.tif --outdir /path/out
  python convert_flat_to_pyramid.py --dry-run /path/in.tif     # 只看计划不写盘

层级结构由 `--levels` 决定(downsample 因子, level0 隐含为 1):
  --levels 2,4,8,16,32,64   → 2x 步进金字塔 (默认; 仿 Aperio/常见 WSI)
  --levels 4,16,32          → 无 2x 层, 结构对齐内部 TCGA svs (见 plan.md ②)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

# --- BigTIFF tag 常量 -------------------------------------------------------
FILETYPE_REDUCEDIMAGE = 1  # TIFFTAG_SUBFILETYPE 值, openslide 以此判定金字塔层
RESUNIT_CENTIMETER = 'CENTIMETER'


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------------------
# 图像处理
# ---------------------------------------------------------------------------
def block_mean(img: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """按 fh×fw 块做均值下采样, **尺寸用 floor 除法** (= 约束 (2)), 返回 uint8。

    用 cv2.INTER_AREA: 整数倍降采样时数学上等价于 block mean(实测 2x 时与
    numpy 块均值逐像素一致, 仅 rounding 差 ≤1), 但直接输出 uint8、多线程、
    峰值内存只有 numpy .mean() 的零头 —— 后者对 17.6GB 的 level0 会materialize
    出 ~35GB 的 float64 中间量, 实测卡住 1 分钟以上。
    非整数倍(奇数边长)时按面积加权, 比"裁掉多余行列再块均值"更合理。
    """
    h, w = img.shape[:2]
    nh, nw = h // fh, w // fw
    if nh < 1 or nw < 1:
        raise ValueError(f'level too small: {h}x{w} with factor {fh}x{fw}')
    if fh == fw == 1:
        return img
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# 写盘
# ---------------------------------------------------------------------------
def write_pyramid(path: Path, level0: np.ndarray, factors: list[int], args) -> list[int]:
    """边建层边写盘(写入顺序 = IFD 顺序), 峰值内存 ≈ level0 + 1 个下级。

    返回实际写出的 downsample 因子列表。
    """
    xres = 1e4 / args.mpp                 # px per cm  (1 cm = 1e4 µm)
    base_kw = dict(
        tile=(args.tile, args.tile),
        photometric='rgb',
        planarconfig='contig',
        resolution=(xres, xres),
        resolutionunit=RESUNIT_CENTIMETER,
    )
    if args.compression == 'jpeg':
        base_kw['compression'] = ('jpeg', args.quality)
    else:
        base_kw['compression'] = args.compression
        base_kw['predictor'] = True       # 水平差分, 对 WSI 提升约 20-30%
    desc = (f'converted_from={args.input.name}; mpp={args.mpp}; '
            f'levels=1,{",".join(str(f) for f in factors)}; '
            f'tile={args.tile}; compression={args.compression}')

    h0, w0 = level0.shape[:2]
    written = [1]
    tmp = path.with_suffix(path.suffix + '.part')
    with tifffile.TiffWriter(tmp, bigtiff=True) as tw:
        t0 = time.time()
        tw.write(level0, description=desc, **base_kw)
        log(f'    wrote level0 (ds=1x)   {w0}x{h0:<7d} {time.time() - t0:6.1f}s  '
            f'file={tmp.stat().st_size / 1e9:.3f} GB')
        cur, cur_factor = level0, 1
        for f in factors:
            step = f // cur_factor
            if step * cur_factor != f:
                raise ValueError(f'level factor {f} is not an integer multiple of {cur_factor}')
            t0 = time.time()
            nxt = block_mean(cur, step, step)
            t_build = time.time() - t0
            if min(nxt.shape[0], nxt.shape[1]) < args.min_level_dim:
                log(f'    stop: level {f}x would be {nxt.shape[1]}x{nxt.shape[0]} '
                    f'< min-level-dim {args.min_level_dim}')
                break
            t0 = time.time()
            tw.write(nxt, subfiletype=FILETYPE_REDUCEDIMAGE, **base_kw)  # ← 关键, 见 (1)
            t_write = time.time() - t0
            log(f'    wrote level{len(written)} (ds={f:>3d}x) {nxt.shape[1]}x{nxt.shape[0]:<7d} '
                f'actual_ds={w0 / nxt.shape[1]:.5f} int()={int(w0 / nxt.shape[1])}  '
                f'build={t_build:5.1f}s write={t_write:6.1f}s  '
                f'file={tmp.stat().st_size / 1e9:.3f} GB')
            written.append(f)
            cur, cur_factor = nxt, f
    os.replace(tmp, path)
    return written


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def verify(path: Path, verbose: bool = True) -> dict:
    """用 openslide 回读校验(若 openslide 不可用则返回 {})。"""
    try:
        import openslide
    except ImportError:
        return {}
    out: dict = {}
    with openslide.OpenSlide(str(path)) as o:
        out['level_count'] = o.level_count
        out['dimensions'] = o.dimensions
        out['level_dimensions'] = o.level_dimensions
        out['level_downsamples'] = [round(d, 5) for d in o.level_downsamples]
        out['int_downsamples'] = [int(d) for d in o.level_downsamples]
        out['tiff.XResolution'] = o.properties.get('tiff.XResolution')
        out['tiff.ResolutionUnit'] = o.properties.get('tiff.ResolutionUnit')
        out['openslide.mpp-x'] = o.properties.get('openslide.mpp-x')
        if o.level_count > 1:
            r = np.array(o.read_region((0, 0), 1, (64, 64)).convert('RGB'))
            out['read_level1_shape'] = r.shape
    if verbose:
        for k, v in out.items():
            print(f'    {k}: {v}')
    return out


# ---------------------------------------------------------------------------
def convert_one(inp: Path, outdir: Path, args) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else outdir / f'{inp.stem}_pyr.tif'

    if out.exists() and args.skip_existing:
        log(f'skip (exists): {out}')
        return 0

    log(f'== {inp.name}  ({inp.stat().st_size / 1e9:.2f} GB)')
    with tifffile.TiffFile(inp) as tf:
        page = tf.pages[0]
        if len(tf.pages) != 1:
            log(f'    WARNING: source has {len(tf.pages)} pages, only page 0 will be converted')
        shape = page.shape
        if page.samplesperpixel != 3 or page.bitspersample != 8:
            raise ValueError(f'unexpected samples/bits: {page.samplesperpixel}/'
                             f'{page.bitspersample} (only 8-bit RGB supported)')
        log(f'    source {shape[1]}x{shape[0]} {page.dtype} '
            f'compression={page.compression.name} tiled={page.is_tiled}')

    # 逐条 strip 并行解码(不落全图的中间副本)
    t0 = time.time()
    log(f'    decoding source strips (maxworkers={args.jobs}) ...')
    img = tifffile.imread(inp, maxworkers=args.jobs)
    log(f'    decoded in {time.time() - t0:.1f}s  array={img.shape} ({img.nbytes / 1e9:.1f} GB)')

    factors = [int(x) for x in args.levels.split(',') if x.strip()]

    log('    building + writing tiled BigTIFF:')
    t0 = time.time()
    written = write_pyramid(out, img, factors, args)
    del img
    log(f'    pyramid done in {time.time() - t0:.1f}s  size={out.stat().st_size / 1e9:.3f} GB')

    log('    verifying with openslide:')
    v = verify(out)
    if v and v.get('level_count', 0) != len(written):
        log(f'    !! WARNING: openslide sees {v.get("level_count")} levels, expected {len(written)}')
        return 2
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path, help='扁平单页 BigTIFF 源文件')
    p.add_argument('--outdir', type=Path,
                   default=Path('/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/'
                                'SurvPGC_Workspace/external_fukun'),
                   help='输出目录 (默认 .../SurvPGC_Workspace/external_fukun)')
    p.add_argument('--output', type=Path, default=None,
                   help='显式指定输出文件路径 (默认 <outdir>/<stem>_pyr.tif)')
    p.add_argument('--levels', default='2,4,8,16,32,64',
                   help='相对 level0 的 downsample 因子, 逗号分隔 (默认 2x 步进)')
    p.add_argument('--min-level-dim', type=int, default=128,
                   help='最小边长低于此值就停止建层 (默认 128)')
    p.add_argument('--tile', type=int, default=256, help='tile 边长 (默认 256)')
    p.add_argument('--compression', default='jpeg',
                   choices=['jpeg', 'lzw', 'deflate'], help='压缩方式 (默认 jpeg)')
    p.add_argument('--quality', type=int, default=85, help='JPEG 质量 (默认 85)')
    p.add_argument('--mpp', type=float, default=0.25,
                   help='写入 TIFF 的 µm/px, 决定 TRIDENT 认到的 mag (默认 0.25 → 40x)')
    p.add_argument('--jobs', type=int, default=16, help='strip 解码线程数 (默认 16)')
    p.add_argument('--skip-existing', action='store_true', help='输出已存在则跳过(断点续跑)')
    p.add_argument('--dry-run', action='store_true', help='只打印层级计划, 不写盘')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        print(f'ERROR: input not found: {args.input}', file=sys.stderr)
        return 1

    if args.dry_run:
        with tifffile.TiffFile(args.input) as tf:
            h, w = tf.pages[0].shape[:2]
        factors = [int(x) for x in args.levels.split(',') if x.strip()]
        print(f'{args.input.name}: {w}x{h}')
        print(f'  outdir  : {args.outdir}')
        print(f'  levels  : 1 (level0, {w}x{h})')
        cur_f, cur_w, cur_h = 1, w, h
        for f in factors:
            step = f // cur_f
            cur_w, cur_h = cur_w // step, cur_h // step
            print(f'            {f}x -> {cur_w}x{cur_h}  '
                  f'(actual ds={w / cur_w:.5f}, int()={int(w / cur_w)})')
            cur_f = f
            if min(cur_w, cur_h) < args.min_level_dim:
                print(f'            (stop: min dim < {args.min_level_dim})')
                break
        return 0

    return convert_one(args.input, args.outdir, args)


if __name__ == '__main__':
    sys.exit(main())
