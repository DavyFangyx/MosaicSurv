"""1) seg 路径模拟 2) 几张致密片子的 JPEG 体积密度估计"""
import sys, numpy as np, tifffile, imagecodecs
sys.path.insert(0,'/data/fangyuxuan/projects/medical_dl/trident_project/TRIDENT')
from trident.wsi_objects.WSIFactory import load_wsi

P='/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/external_fukun/2211834_20240105093257_pyr.tif'
wsi=load_wsi(slide_path=P, lazy_init=False)
print('=== seg 路径 (hest, target_mag=10, input_size=512) 模拟 ===')
print(f'wsi.mpp={wsi.mpp} wsi.mag={wsi.mag}')
pat = wsi.create_patcher(patch_size=512, src_pixel_size=wsi.mpp, dst_pixel_size=10/10)
print(f'  downsample={pat.downsample:.4f} -> level={pat.level} (ds={wsi.level_downsamples[pat.level]:.5f}) '
      f'read {pat.patch_size_level}px @L{pat.level}  '
      f'覆盖 level0 {int(pat.patch_size_level*wsi.level_downsamples[pat.level])}px (期望 2048) '
      f'{"OK" if abs(pat.patch_size_level*wsi.level_downsamples[pat.level]-2048)<5 else "*** WRONG ***"}')
print('\n=== feat 路径 (mag=20, patch=512) 模拟 ===')
pat2 = wsi.create_patcher(patch_size=512, src_mag=wsi.mag, dst_mag=20)
print(f'  downsample={pat2.downsample:.4f} -> level={pat2.level} (ds={wsi.level_downsamples[pat2.level]:.5f}) '
      f'read {pat2.patch_size_level}px @L{pat2.level}  '
      f'覆盖 level0 {int(pat2.patch_size_level*wsi.level_downsamples[pat2.level])}px (期望 1024) '
      f'{"OK" if abs(pat2.patch_size_level*wsi.level_downsamples[pat2.level]-1024)<3 else "*** WRONG ***"}')
t,_ = pat2.get_tile_xy(40000, 40000)
print(f'  get_tile_xy(40000,40000) -> array {np.asarray(t).shape}')
