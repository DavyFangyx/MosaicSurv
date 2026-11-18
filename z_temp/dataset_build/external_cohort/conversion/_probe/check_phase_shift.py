"""真实 patch 网格(x=1024k, 与 TRIDENT 实际坐标一致)下, 比 L1 读图与 L0+resize 的相位关系"""
import numpy as np, cv2, openslide
OUT='/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/external_fukun/2211834_20240105093257_pyr.tif'
PS=512; o=openslide.OpenSlide(OUT); W,H=o.dimensions
ds=o.level_downsamples[1]
print(f'ds(L1)={ds:.6f}  x/ds for x=1024k -> k*512/1.0000135')
for k in [1, 5, 30, 60]:
    x, y = 1024*k, 1024*7
    if x+PS*2 > W: break
    A=np.array(o.read_region((x,y),0,(PS*2,PS*2)).convert('RGB'))
    A=cv2.resize(A,(PS,PS),interpolation=cv2.INTER_LINEAR)
    shifts={}
    for name, loc in [('as-is (x,y)', (x,y)), ('int(x/ds) floor', (int(x/ds), int(y/ds))),
                      ('x/2 - 1 px', (x//2-1, y//2)), ('x/2 + 1 px', (x//2+1, y//2))]:
        B=np.array(o.read_region(loc,1,(PS,PS)).convert('RGB'))
        shifts[name]=np.abs(A.astype(np.int16)-B.astype(np.int16)).mean()
    best=min(shifts, key=shifts.get)
    print(f'  x={x:>6d} k={k:>3d} | ' + '  '.join(f'{n}={v:5.2f}' for n,v in shifts.items()) + f'   BEST={best}')
o.close()
