"""正确版: openslide read_region 的 location 是 **level0 坐标**, 脚本自己会除 downsample。
 (a) 内部 TCGA 路径 = read_region((x,y), 0, (2*PS,2*PS)) + cv2.resize -> PS
 (b) 2x 层路径    = read_region((x,y), 1, (PS,PS))          [同一个 level0 坐标]
"""
import numpy as np, cv2, openslide
P='/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/external_fukun/2211834_20240105093257_pyr.tif'
o=openslide.OpenSlide(P); W,H=o.dimensions
rng=np.random.default_rng(7)
def tissued(x,y):
    return np.array(o.read_region((x,y),4,(32,32)).convert('L')).mean()<220
print(f'{"PS":>5s} {"region":>7s} {"std":>6s} | {"Lin_vs_Area":>11s} {"L1_vs_Area":>10s} {"L1_vs_Lin":>9s}')
for PS in (256, 512):
    acc=[]
    for k in range(8):
        while True:
            x=int(rng.integers(0,W-PS*4)); y=int(rng.integers(0,H-PS*4))
            if tissued(x,y): break
        A=np.array(o.read_region((x,y),0,(PS*2,PS*2)).convert('RGB'))
        A_lin=cv2.resize(A,(PS,PS),interpolation=cv2.INTER_LINEAR)
        A_area=cv2.resize(A,(PS,PS),interpolation=cv2.INTER_AREA)
        B=np.array(o.read_region((x,y),1,(PS,PS)).convert('RGB'))     # 同一 level0 坐标
        d_ka=np.abs(A_lin.astype(np.int16)-A_area.astype(np.int16)).mean()
        d_1a=np.abs(B.astype(np.int16)-A_area.astype(np.int16)).mean()
        d_1l=np.abs(B.astype(np.int16)-A_lin.astype(np.int16)).mean()
        print(f'{PS:>5d} {k:>7d} {A.std():6.1f} | {d_ka:11.2f} {d_1a:10.2f} {d_1l:9.2f}')
        acc.append((d_ka,d_1a,d_1l))
    a=np.array(acc).mean(axis=0)
    print(f'{PS:>5d} {"MEAN":>7s} {"":>6s} | {a[0]:11.2f} {a[1]:10.2f} {a[2]:9.2f}\n')
o.close()
