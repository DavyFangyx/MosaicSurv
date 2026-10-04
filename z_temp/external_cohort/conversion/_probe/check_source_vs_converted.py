"""端到端像素校验: 源 tif 原始像素 vs 转换后 level0 (排除转置/错位/丢内容)"""
import numpy as np, tifffile, zarr, openslide
SRC='/data/lizhe/Medteam_projects/test/eval_tiff/lung/2211834_20240105093257.tif'
OUT='/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/external_fukun/2211834_20240105093257_pyr.tif'
with tifffile.TiffFile(SRC) as tf:
    z=zarr.open(tf.aszarr(), mode='r')
    o=openslide.OpenSlide(OUT)
    assert (z.shape[1], z.shape[0]) == o.dimensions, (z.shape, o.dimensions)
    print('shape match:', o.dimensions)
    for (x,y) in [(5000,5000),(30000,40000),(58000,88000),(0,0),(40000,2000)]:
        s=np.asarray(z[y:y+512, x:x+512])
        c=np.array(o.read_region((x,y),0,(512,512)).convert('RGB'))
        d=np.abs(s.astype(np.int16)-c.astype(np.int16))
        # 同时比"转置假设"下的差异, 排除 x/y 搞反
        print(f'  ({x:>5d},{y:>5d}) src.mean={s.mean():6.2f} conv.mean={c.mean():6.2f} '
              f'mean|Δ|={d.mean():5.2f} max|Δ|={d.max():3d} PSNR={10*np.log10(255**2/max(d.mean()**2,1e-9)):5.1f}dB')
    o.close()
