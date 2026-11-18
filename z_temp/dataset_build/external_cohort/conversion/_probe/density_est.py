"""估计全量输出体积: 从几张(最致密的)源图取真实组织区域, 按 Q85 编码测 bytes/px。"""
import numpy as np, tifffile, imagecodecs
SRC='/data/lizhe/Medteam_projects/test/eval_tiff'
files=[('lung','2211834_20240105093257'), ('liver','2110282_20240116085741'),
       ('kidney','2212456_20240105104036'), ('kidney','2213740_20240105104021'),
       ('liver','2102850_20240116095752')]
XRES=40000
for organ,stem in files:
    p=f'{SRC}/{organ}/{stem}.tif'
    with tifffile.TiffFile(p) as tf:
        pg=tf.pages[0]; H,W=pg.shape[:2]
        # 取中间区域(多半是组织) + 一个偏角区域
        regs=[]
        for cy,cx in [(0.5,0.5),(0.35,0.4),(0.65,0.6)]:
            y,x=int(H*cy),int(W*cx)
            regs.append(pg.asarray()[y:y+2048, x:x+2048] if False else None)
        # tifffile 无法只读子区域, 用 zarr store 按 strip 读
        store=tf.aszarr()
        import zarr
        z=zarr.open(store, mode='r')
        tot_raw=0; tot_jpg=0
        for cy,cx in [(0.5,0.5),(0.35,0.4),(0.65,0.6)]:
            y,x=int(H*cy),int(W*cx)
            sub=np.asarray(z[y:y+1024, x:x+1024])
            enc=imagecodecs.jpeg_encode(sub, level=85, bitspersample=8, colorspace='rgb')
            tot_raw+=sub.nbytes; tot_jpg+=len(enc)
        ratio=tot_raw/tot_jpg
        px=H*W
        est_gb=px*3/ratio/1e9
        print(f'{organ:>6s}/{stem[:18]:18s} {W}x{H} px={px/1e9:.2f}G  JPEG_ratio={ratio:5.1f}x  '
              f'-> level0 ~{est_gb:5.2f} GB, 金字塔总计 ~{est_gb*1.34:5.2f} GB')
