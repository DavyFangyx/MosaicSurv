#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_pyramid.py
=================
转换后校验脚本。做四件事:

 1. openslide 能否打开; level_count / 各层尺寸 / level_downsamples / mpp 属性
 2. 层级是否精确 2x 步进(允许 ±1px)
 3. **用 TRIDENT 自己的类**跑一遍 level 选择, 确认 --mag 20 会命中哪一层
    (调用真实的 WSI.get_best_level_and_custom_downsample, 见
     TRIDENT/trident/wsi_objects/WSI.py:585 和 WSIPatcher.py:350)
 4. 两种金字塔设计的 patch 像素差异: (a) level0 大图 + resize (内部 TCGA 走的路径)
    vs (b) 直接读 2x 层 —— 用于判定"是否要保留精确 2x 层"

用法:  python verify_pyramid.py <pyramid.tif> [--patch-size 512] [--n 12]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import openslide


def fmt(o):
    print(f'  file            : {o._filename if hasattr(o, "_filename") else ""}')
    print(f'  openslide lib   : {openslide.__library_version__}')
    print(f'  dimensions      : {o.dimensions}')
    print(f'  level_count     : {o.level_count}')
    for i, (d, ds) in enumerate(zip(o.level_dimensions, o.level_downsamples)):
        print(f'    L{i}: {d[0]:>6d} x {d[1]:<6d}  downsample={ds:.6f}  int()={int(ds)}')
    for k in ('tiff.XResolution', 'tiff.ResolutionUnit', 'openslide.mpp-x',
              'openslide.objective-power'):
        if k in o.properties:
            print(f'  {k:22s}: {o.properties[k]}')


def check_2x(o) -> bool:
    ok = True
    for i in range(o.level_count - 1):
        w0, h0 = o.level_dimensions[i]
        w1, h1 = o.level_dimensions[i + 1]
        dw, dh = w0 / w1, h0 / h1
        good = abs(w0 / 2 - w1) <= 1 and abs(h0 / 2 - h1) <= 1
        print(f'  L{i}->L{i+1}: {w0}x{h0} -> {w1}x{h1}  ratio=({dw:.5f},{dh:.5f})  '
              f'exact-2x±1px={"OK" if good else "NO"}')
        ok &= bool(good)
    return ok


def trident_level_choice(path: str, mag: float = 20.0, mpp: float = 0.25,
                         patch_size: int = 512) -> None:
    """用 TRIDENT 真实代码复算 --mag 20 时的 level 与读图尺寸。"""
    print('\n[TRIDENT level selection]')
    try:
        sys.path.insert(0, '/data/fangyuxuan/projects/medical_dl/trident_project/TRIDENT')
        from trident.wsi_objects.WSIFactory import load_wsi
        wsi = load_wsi(slide_path=path, lazy_init=False)
        print(f'  TRIDENT sees: width={wsi.width} height={wsi.height} '
              f'mpp={wsi.mpp} mag={wsi.mag} levels={wsi.level_count}')
        print(f'  level_downsamples={[round(d, 5) for d in wsi.level_downsamples]}')
    except Exception as e:                                    # pragma: no cover
        print(f'  (TRIDENT import failed: {type(e).__name__}: {e})')
        print('  -> falling back to a verbatim copy of the logic')
        with openslide.OpenSlide(path) as o:
            ds_list, dims = o.level_downsamples, o.level_dimensions
            mpp_x = 1e4 / float(o.properties['tiff.XResolution'])
            mag_v = 40 if mpp_x < 0.3 else (20 if mpp_x < 0.6 else 10)

        class _S:  # minimal stand-in
            level_downsamples = ds_list

            def get_best_level_and_custom_downsample(self, downsample, tolerance=0.01):
                ld = self.level_downsamples
                for i, d in enumerate(ld):
                    if abs(d - downsample) <= tolerance:
                        return i, 1.0
                if downsample >= ld[0]:
                    cl = cd = None
                    for i, d in enumerate(ld):
                        if d <= downsample:
                            cl, cd = i, d
                        else:
                            break
                    if cl is not None:
                        return cl, downsample / cd
                for i, d in enumerate(ld):
                    if d >= downsample:
                        return i, d / downsample
                raise ValueError('no level')

        wsi = _S()

    # --mag 20 语义: WSIPatcher.__init__ 用 src_mag=wsi.mag, dst_mag=20
    src_mag = getattr(wsi, 'mag', 40)
    src_ps, dst_ps = 10 / src_mag, 10 / mag
    downsample = dst_ps / src_ps
    level, custom = wsi.get_best_level_and_custom_downsample(downsample, tolerance=0.1)
    level_ds = int(wsi.level_downsamples[level])
    patch_size_src = round(patch_size * downsample)
    patch_size_level = round(patch_size_src / level_ds)
    print(f'  src_mag={src_mag} dst_mag={mag} -> downsample={downsample}')
    print(f'  => level={level} (level_downsamples={round(wsi.level_downsamples[level], 5)}), '
          f'custom_downsample={custom} [discarded by TRIDENT]')
    print(f'  => read_region size = {patch_size_level}px @ L{level}, '
          f'then cv2.resize -> {patch_size}px')
    # 反向: 若该层的 int(downsample) 被截断成别的值, 读图尺寸会错
    for i, d in enumerate(wsi.level_downsamples):
        print(f'     L{i}: ds={d:.5f} int()={int(d)}  '
              f'{"<-- exact match" if abs(d - downsample) <= 0.1 else ""}')


def compare_designs(path: str, patch_size: int = 512, n: int = 12, seed: int = 0) -> None:
    """对比 (a) 读 level0 大图再 resize 与 (b) 直接读 2x 层 的像素差。"""
    with openslide.OpenSlide(path) as o:
        if o.level_count < 2:
            print('\n[design compare] skipped: no level-1')
            return
        w, h = o.dimensions
        rng = np.random.default_rng(seed)
        ds = o.level_downsamples[1]
        diffs, rel = [], []
        print(f'\n[design compare] patch {patch_size}px @20x  '
              f'(a)=read {patch_size*2}px@L0+INTER_LINEAR  (b)=read {patch_size}px@L{1}')
        for _ in range(n):
            x = int(rng.integers(0, w - patch_size * 4))
            y = int(rng.integers(0, h - patch_size * 4))
            a = np.array(o.read_region((x, y), 0, (patch_size * 2, patch_size * 2))
                         .convert('RGB'))
            a = cv2.resize(a, (patch_size, patch_size))     # TRIDENT 内部路径
            bx, by = int(round(x / ds)), int(round(y / ds))
            b = np.array(o.read_region((bx, by), 1, (patch_size, patch_size))
                         .convert('RGB'))
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            diffs.append(d.mean())
            rel.append((d > 8).mean() * 100)
        print(f'  mean|Δ| = {np.mean(diffs):.2f} gray levels (per-patch {np.round(diffs, 2).tolist()})')
        print(f'  pixels with |Δ|>8: {np.mean(rel):.2f}% of pixels')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--patch-size', type=int, default=512)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--skip-compare', action='store_true')
    a = ap.parse_args()

    print(f'=== {os.path.basename(a.path)} ({os.path.getsize(a.path)/1e9:.3f} GB) ===')
    with openslide.OpenSlide(a.path) as o:
        fmt(o)
        print('\n[2x step check]')
        check_2x(o)
    trident_level_choice(a.path, patch_size=a.patch_size)
    if not a.skip_compare:
        compare_designs(a.path, patch_size=a.patch_size, n=a.n)


if __name__ == '__main__':
    main()
