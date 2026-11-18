"""Probe: can openslide read tifffile-written pyramidal TIFFs? Which flavours work?"""
import numpy as np, tifffile, imagecodecs, os, sys, time
import openslide

OUT = os.path.dirname(os.path.abspath(__file__))
print('openslide-python', openslide.__version__, 'lib', openslide.__library_version__)
print('tifffile', tifffile.__version__, 'imagecodecs', imagecodecs.__version__)

# realistic tissue image: grab a chunk from an internal TCGA svs
SVS = '/data/fangyuxuan/projects/medical_dl/trident_project/wsis_flat/TCGA-2Z-A9J1-01Z-00-DX1.07E7992F-1AC5-4E6F-8AC5-4A5C77B0DA0A.svs'
import glob
svs = sorted(glob.glob('/data/fangyuxuan/projects/medical_dl/trident_project/wsis_flat/*.svs'))[0]
s = openslide.OpenSlide(svs)
base = np.array(s.read_region((20000, 20000), 0, (4096, 4096)).convert('RGB'))
print('base', base.shape, base.dtype, 'mean', base.mean().round(1))
s.close()

def build_levels(img, n=6):
    lv = [img]
    for _ in range(n-1):
        h, w = lv[-1].shape[:2]
        # floor-style 2x shrink (matches what we want from converters)
        nh, nw = max(h//2, 1), max(w//2, 1)
        t = lv[-1][:nh*2, :nw*2].reshape(nh, 2, nw, 2, 3).mean(axis=(1, 3))
        lv.append(t.astype(np.uint8))
    return lv

levels = build_levels(base)
print('levels', [l.shape for l in levels])
RES = (40000, 40000)  # px per cm -> mpp 0.25

def write_chain(path, comp, subfiletype=None):
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for i, lv in enumerate(levels):
            kw = dict(tile=(256, 256), compression=comp, photometric='rgb',
                      resolution=RES, resolutionunit='CENTIMETER')
            if subfiletype is not None:
                kw['subfiletype'] = 1 if i else 0
            tw.write(lv, **kw)

def write_subifd(path, comp):
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        tw.write(levels[0], tile=(256, 256), compression=comp, photometric='rgb',
                 resolution=RES, resolutionunit='CENTIMETER',
                 subifds=len(levels)-1)
        for lv in levels[1:]:
            tw.write(lv, tile=(256, 256), compression=comp, photometric='rgb',
                     subfiletype=1)

cases = []
for comp in ['jpeg', 'lzw', 'deflate', 'zstd']:
    cases.append((f'chain_{comp}.tif', lambda p, c=comp: write_chain(p, c)))
cases.append(('chain_jpeg_subft.tif', lambda p: write_chain(p, 'jpeg', subfiletype=True)))
cases.append(('subifd_jpeg.tif', lambda p: write_subifd(p, 'jpeg')))

for name, fn in cases:
    p = os.path.join(OUT, name)
    try:
        t0 = time.time(); fn(p); tw_t = time.time()-t0
    except Exception as e:
        print(f'{name:26s} WRITE-FAIL {type(e).__name__}: {str(e)[:60]}'); continue
    sz = os.path.getsize(p)
    try:
        o = openslide.OpenSlide(p)
        info = f'OK levels={o.level_count} dims={o.dimensions} ld={[round(d,5) for d in o.level_downsamples]}'
        r = np.array(o.read_region((0, 0), min(1, o.level_count-1), (64, 64)).convert('RGB'))
        info += f' read={r.shape} mean={r.mean():.1f}'
        xr = o.properties.get('tiff.XResolution'); ru = o.properties.get('tiff.ResolutionUnit')
        info += f' XRes={xr} unit={ru}'
        o.close()
    except Exception as e:
        info = f'OPENSLIDE-FAIL {type(e).__name__}: {str(e)[:70]}'
    print(f'{name:26s} {sz/1e6:8.2f} MB  write={tw_t:5.1f}s  {info}')
