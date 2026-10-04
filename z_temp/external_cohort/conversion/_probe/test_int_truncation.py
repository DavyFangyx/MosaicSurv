"""证明: 金字塔层尺寸用 ceil 会让 TRIDENT 的 int(level_downsamples) 截断, 静默读错面积。
用真实 TRIDENT 类跑, 造两个只差 1px 的小金字塔: floor(506) vs ceil(507)。"""
import sys, numpy as np, tifffile, cv2
sys.path.insert(0,'/data/fangyuxuan/projects/medical_dl/trident_project/TRIDENT')
from trident.wsi_objects.WSIFactory import load_wsi
import openslide

W0, H0 = 1013, 1007          # 奇数边长
img0 = (np.random.default_rng(0).random((H0, W0, 3))*255).astype('uint8')
RES = (40000, 40000)

def build(path, w1, h1):
    lv1 = cv2.resize(img0, (w1, h1), interpolation=cv2.INTER_AREA)
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for i, lv in enumerate([img0, lv1]):
            tw.write(lv, tile=(256,256), photometric='rgb', resolution=RES,
                     resolutionunit='CENTIMETER', compression=('jpeg',85),
                     **({'subfiletype':1} if i else {}))

for tag, w1, h1 in [('floor', W0//2, H0//2), ('ceil', -(-W0//2), -(-H0//2))]:
    p=f'/tmp/trunc_{tag}.tif'
    build(p, w1, h1)
    with openslide.OpenSlide(p) as o:
        ds = o.level_downsamples
    wsi = load_wsi(slide_path=p, lazy_init=False)
    # 复刻 WSIPatcher.__init__ 的那几行
    src_ps, dst_ps = 10/wsi.mag, 10/20.0          # --mag 20
    downsample = dst_ps/src_ps
    level, _ = wsi.get_best_level_and_custom_downsample(downsample, tolerance=0.1)
    lds = int(wsi.level_downsamples[level])
    ps_src = round(512*downsample)
    ps_level = round(ps_src/lds)
    px_area = (ps_level*ds[level])**2
    print(f'{tag:>5s}: L1={w1}x{h1} openslide_ds={ds[1]:.5f} int()={int(ds[1])} | '
          f'TRIDENT level={level} patch_size_src={ps_src} read={ps_level}px@L{level} '
          f'-> 覆盖 level0 {int(ps_level*ds[level])}px (期望 {2*512}) '
          f'{"OK" if abs(ps_level*ds[level]-1024)<3 else "*** 错 4 倍 ***"}')
