"""分解 (a)内部路径 vs (b)2x层 的差异来源: 重采样核 / 亚像素相位 / JPEG+内容"""
import numpy as np, cv2, openslide
P='/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/external_fukun/2211834_20240105093257_pyr.tif'
PS=512
o=openslide.OpenSlide(P); W,H=o.dimensions
rng=np.random.default_rng(7)
def tissued(x,y):
    t=np.array(o.read_region((x,y),4,(32,32)).convert('L'))
    return t.mean()<220
print(f'{"region":>14s} {"mean":>6s} {"std":>6s} | {"Lin-vs-Area":>11s} {"Area-vs-L1":>11s} {"Lin-vs-L1":>10s} {"bestshift":>9s} {"aligned":>8s}')
acc=[]
for k in range(8):
    while True:
        x=int(rng.integers(0,W-PS*4)); y=int(rng.integers(0,H-PS*4))
        if tissued(x,y): break
    A=np.array(o.read_region((x,y),0,(PS*2,PS*2)).convert('RGB'))
    A_lin=cv2.resize(A,(PS,PS),interpolation=cv2.INTER_LINEAR)     # TRIDENT 内部走法
    A_area=cv2.resize(A,(PS,PS),interpolation=cv2.INTER_AREA)      # 理想 2x 面积均值
    bx,by=int(round(x/2.000027)),int(round(y/2.000027))
    B=np.array(o.read_region((bx-4,by-4),1,(PS+8,PS+8)).convert('RGB'))
    best=(1e9,0,0)
    for dy in range(9):
        for dx in range(9):
            c=B[dy:dy+PS,dx:dx+PS].astype(np.int16)
            d=np.abs(A_lin.astype(np.int16)-c).mean()
            if d<best[0]: best=(d,dx-4,dy-4)
    d_lin=np.abs(A_lin.astype(np.int16)-A_area.astype(np.int16)).mean()
    d_area=np.abs(A_area.astype(np.int16)-B[4:4+PS,4:4+PS].astype(np.int16)).mean()
    d_l1=np.abs(A_lin.astype(np.int16)-B[4:4+PS,4:4+PS].astype(np.int16)).mean()
    print(f'{k:>14d} {A.mean():6.1f} {A.std():6.1f} | {d_lin:11.2f} {d_area:11.2f} {d_l1:10.2f} {str((best[1],best[2])):>9s} {best[0]:8.2f}')
    acc.append((d_lin,d_area,d_l1,best[0]))
a=np.array(acc).mean(axis=0)
print(f'\n{"MEAN":>14s} {"":>6s} {"":>6s} | {a[0]:11.2f} {a[1]:11.2f} {a[2]:10.2f} {"":>9s} {a[3]:8.2f}')
print('\nLin-vs-Area = 同一相位下 cv2 INTER_LINEAR(内部走法) 与 INTER_AREA(理想2x均值) 之差')
print('Area-vs-L1  = 同一相位下 理想2x均值 与 文件里 2x 层 之差(=JPEG/编码误差)')
print('Lin-vs-L1   = 实际(未对齐)差异; bestshift/aligned = 允许±4px平移搜索后的最小差')
o.close()
