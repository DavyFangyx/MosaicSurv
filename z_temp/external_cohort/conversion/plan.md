# 外部数据集 WSI 金字塔化转换 —— 调研 / 试转 / 全量方案

日期: 2026-10-01
目录: `z_temp/external_cohort/conversion/`
状态: **单张试转已完成并验证通过; 全量 31 张等待批准, 尚未执行。**

---

## 0. 结论速览(TL;DR)

| 项 | 结论 |
|---|---|
| TRIDENT `--mag 20` 语义 | **不是** `read_region(downsample=2)` 插值, 也 **不是** openslide 的 `getBestLevelForDownsample`(最近层)。是 `get_best_level_and_custom_downsample(2.0, tol=0.1)`: 命中"精确 2x 层"就用它; 否则退到"≤2.0 的最大层"(即 level0), 再靠 `cv2.resize` 补足倍率。**4x 层永远不会被选中。** |
| 是否需要精确 2x 层 | **需要, 且推荐要**。有 2x 层 → `read 512px@L1`(原生 20x); 无 2x 层 → `read 1024px@L0 + resize`(内部 TCGA 的走法)。两者 patch 内容实测仅差 **~3 个灰度级**(Q85 JPEG 量级), 但后者读图量 **4 倍**。 |
| 层级结构 | `1,2,4,8,16,32,64` 步进, **每层尺寸必须 floor 取整**(否则 TRIDENT 静默读错 4 倍面积, 已用真实 TRIDENT 代码证明) |
| 工具 | 系统无 vips, **不装任何东西**: 用 trident env 已有的 `tifffile + imagecodecs + cv2` 搞定了(已端到端验证) |
| 单张试转 | 8.27GB → **1.25GB / 47s**(热缓存; 冷缓存约 70s), openslide 7 层全部正确 |
| 全量 ETA | 约 **50–70 分钟**(串行, 31 张 365.7GB); 输出预计 **60–100 GB**(/data 尚有 1.2T) |

---

## ① TRIDENT `--mag 20` 的准确语义(源码行号)

代码路径: `run_batch_of_slides.py` → `Processor.run_patching_job` / `run_feature_extraction_job`
→ `WSI.extract_tissue_coords` → `WSI.create_patcher` → `WSIPatcher`。

### 1.1 倍率是怎么算出来的

`trident/wsi_objects/WSIPatcher.py:124-143`

```python
self.src_pixel_size = 10 / src_mag          # src_mag = wsi.mag (由 mpp 推出; mpp=0.25 → 40)
self.dst_pixel_size = 10 / dst_mag          # dst_mag = --mag 20 → 0.5
self.downsample = self.dst_pixel_size / self.src_pixel_size     # = 2.0
self.patch_size_src = round(patch_size * self.downsample)       # 512*2 = 1024 (level0 空间的图幅)
self.level, self.patch_size_level, self.overlap_level = self._prepare()
```

注意 `--mag` 的默认值是 `run_batch_of_slides.py:85` 的 `20.0`; `--patch_size` 默认 `512`(`:87`)。
**(任务书里写的 `--patch_size 256` 与项目实际不符 —— 内部 P 模态用的是
`TRIDENT_workspace/20.0x_512px_0px_overlap/_config_coords.json` 里的 `patch_size: 512`。
下文一律按 512 说明; 换 256 只是数字等比缩小, 结论不变。)**

### 1.2 层选择:`get_best_level_and_custom_downsample`

`trident/wsi_objects/WSIPatcher.py:350-355`

```python
def _prepare(self):
    level, _ = self.wsi.get_best_level_and_custom_downsample(self.downsample, tolerance=0.1)
    level_downsample = int(self.wsi.level_downsamples[level])       # ← 注意是 int() 截断
    patch_size_level = round(self.patch_size_src / level_downsample)
    overlap_level = round(self.overlap_src / level_downsample)
    return level, patch_size_level, overlap_level
```

`trident/wsi_objects/WSI.py:585-644` 的判定顺序:

1. 先找 **精确匹配**: `abs(level_downsample - downsample) <= tolerance(0.1)` → 返回该层;
2. 否则若 `downsample >= level_downsamples[0]`, 取 **≤ downsample 的最大层**(向下取整, **不是最近层**);
3. 否则(要放大)取 ≥ downsample 的最小层。

返回值第二项 `custom_downsample` 在 `:351` 被 `_` **丢弃**了。真正补足残余倍率的,
是 `WSIPatcher.get_tile_xy`(`:367-383`):

```python
tile = self.wsi.read_region(location=(x, y), level=self.level,
                            size=(self.patch_size_level, self.patch_size_level), ...)
tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))   # 默认 INTER_LINEAR
```

**所以既不是 `read_region(..., downsample=2)` 插值取图, 也不是 `getBestLevelForDownsample(2)` 取最近层**
(后者在内部 TCGA 上会取到 4x 层 = 10x, 那才是灾难; TRIDENT 用的是"不超过目标倍率的最大层", 方向相反)。

### 1.3 落到两种数据上的实际行为

| 数据 | `level_downsamples` | mag 20 选中 | 实际读图 | 覆盖 level0 |
|---|---|---|---|---|
| 内部 TCGA svs | `[1, 4.0001, 16.001, 32.0]` | **L0** | `read 1024px@L0` + `cv2.resize→512` | 1024 px |
| 本方案 2x 金字塔 | `[1, 2.00003, 4.0001, 8, 16, 32, 64]` | **L1** | `read 512px@L1`(resize 变 no-op) | 1024 px ✓ |

两条路径取到的 **level0 覆盖范围完全一致(都是 1024px = 40x 下的 512@20x 图幅)**,
差别只在"谁来降采样": 内部是 `cv2.resize`, 外部是转换时建好的 2x 层。
实测二者 patch 差异 ~3 个灰度级(见 ②)。

### 1.4 seg / coords 步骤读什么

* **seg**: `trident/wsi_objects/WSI.py:320-326` 用 `create_patcher(patch_size=seg_model.input_size,
  src_pixel_size=self.mpp, dst_pixel_size=10/target_mag)` —— 注意用的是 **mpp 原值**(不是 mag)。
  hest 模型 `target_mag=10`(`trident/segmentation_models/load.py:131`)、`input_size=512`,
  内部 seg 配置也是 `seg_mag: 10`。于是 downsample = `1.0 / 0.25 = 4.0`, 需要**恰好 4x 的层**。
  实测: 选中 L2(ds=4.00012), `read 512px@L2`, 覆盖 2048 px ✓。
* **coords**: `WSIPatcher._compute_masked`(`:277-323`)纯粹是 **GeoJSON 多边形 vs level0 坐标的几何运算**,
  `coords_only=True` 时 `__getitem__` 直接 `return x, y` **不读像素**。所以 coords 步骤与金字塔层级无关,
  只跟 `patch_size_src`(level0 步长)有关。
* **`wsi.mpp` 的来源**: `OpenSlideWSI._fetch_mpp`(`OpenSlideWSI.py:102-131`)读
  `openslide.mpp-x` → `tiff.XResolution` + `tiff.ResolutionUnit`;
  `WSI._fetch_magnification`(`WSI.py:257-276`)按 mpp 分档: `<0.3 → 40`。
  **⇒ 转换时必须写入 mpp=0.25 的 XResolution, 否则 seg 的 4.0 匹配不上, 会掉回 level0
  去读 2048px 的大图(慢 ~16 倍), 这是最容易踩的坑。**

### 1.5 `run_batch_of_slides.py` 的输入目录约定

* `--wsi_dir`: WSI 所在目录, **不允许嵌套**(`run_batch_of_slides.py:54`)→ 转换输出用扁平目录即可。
* 后端按扩展名自动选: `.tif/.tiff/.svs/...` → `OpenSlideWSI`(`WSIFactory.py:10,82-84`)。
* `--job_dir` 下产出: seg 的 GeoJSON/contours、`<mag>x_<patch>px_<overlap>px_overlap>/patches/<name>_patches.h5`、
  `features_<encoder>/`。名字来自 `wsi.name` = **文件名 stem**。

---

## ② 金字塔层级结构决策

### 2.1 决策

* **层级**: level0(40x 原生) + `2,4,8,16,32,64` 共 7 层(2x 步进, 够缩略图/可视化用)。
* **尺寸取整: 每层一律 floor**(`floor(w/2)`, `floor(h/2)` …), 这是硬要求, 见 2.2。
* **每层必须带 `TIFFTAG_SUBFILETYPE = 1`**(FILETYPE_REDUCEDIMAGE), 见 2.3。
* **写入 mpp = 0.25**(XResolution=40000 px/cm + ResolutionUnit=centimeter)。

### 2.2 为什么必须 floor —— `int()` 截断地雷(本次最关键的发现)

`WSIPatcher.py:352` 是 `int(level_downsamples[level])`(截断), 而 openslide 的
`level_downsamples = level0 尺寸 / 该层尺寸`:

* floor: `62755 / 31377 = 2.00003` → `int() = 2` ✓ `patch_size_level = round(1024/2) = 512` ✓
* ceil : `62755 / 31378 = 1.99997` → `int() = 1` ✗ `patch_size_level = round(1024/1) = 1024`
  → **在 L1 上读 1024px(覆盖 2045 个 level0 像素, 是预期的 2 倍边长 = 4 倍面积),
  再 resize 回 512 —— 等效倍率变成 10x, 而且不报任何错。**

用真实 TRIDENT 类造了两个只差 1px 的小金字塔实测(`_probe/test_int_truncation.py`):

```
floor: L1=506x503 openslide_ds=2.00198 int()=2 | level=1 read=512px@L1 -> 覆盖 level0 1025px (期望 1024) OK
 ceil: L1=507x504 openslide_ds=1.99802 int()=1 | level=1 read=1024px@L1 -> 覆盖 level0 2045px (期望 1024) *** 错 4 倍 ***
```

顺带确认: 内部 Aperio TCGA svs 也正是 floor 取整(156086→39021、104610→26152、76778→19194 全是 floor),
所以 floor 是与内部数据一致的约定。

> 若改用 vips 路线, **必须先验证** `int(level_downsamples) == 名义值`。
> libvips 缩小尺寸惯用 ROUND_UP(ceil), 一旦是 ceil 就会命中上述地雷。
> 直接跑 `verify_pyramid.py <out.tif>` 看 `int_downsamples` 一栏即可。

### 2.3 为什么必须写 `subfiletype=1`(否则白转)

openslide 4.0 的 generic-TIFF reader 实测(`_probe/test_openslide_pyramid.py`):

| 文件结构 | openslide 看到的层数 |
|---|---|
| 普通 IFD 链(无 subfiletype) | **1 层** ✗ |
| SubIFD 金字塔 | **1 层** ✗ |
| IFD 链 + 每级 `subfiletype=1` | **全部层** ✓ |

两个 env 的 openslide 都是 4.0.x(SurvPGC env lib 4.0.1 / trident env lib 4.0.0), 行为一致。

### 2.4 为什么最终选"带精确 2x 层"(而不是完全复刻内部的无-2x 结构)

一开始的担心是"有 2x 层会让外部数据和内部数据走不同代码路径"。实测把两方面都量化了
(`_probe/design_decompose2.py`, 注意 `read_region` 的 location 是 **level0 坐标**, 脚本内部自己除 downsample):

* `cv2.INTER_LINEAR` 在恰好 2x 降采样时与 `INTER_AREA` **逐像素完全相同**(mean|Δ| = 0.00)
  → "内部 resize 路径"本身就是一次标准 2x 面积均值, 和 2x 层的构造方式**同类**;
* 2x 层读出的 patch 与"读 level0 再 resize"的差异: 随机坐标下 **mean|Δ| = 2.89 灰度级**(patch 256)/
  **3.55**(patch 512), 随纹理增强(平坦区 1.4, 复杂区 4.8–6.1); 而在 **TRIDENT 真实 patch 网格上
  (x=1024k, `_probe/check_phase_shift.py`) 只有 mean|Δ| ≈ 1.6 灰度级** —— 就是 Q85 JPEG
  重编码量级的噪声(对比: 转换后 level0 相对源图的误差也只有 0.4–0.9 灰度级 / PSNR 49–57dB)。
  另外确认 openslide 自己会把 level0 坐标正确映射到目标层, 直接传 `(x,y)` 即为最佳对齐, 无需手动补偿。

代价对比: 有 2x 层时 feat 每 patch 读 `512px@L1`; 没有则读 `1024px@L0` = **4 倍解码量**。

**⇒ 用 2x 层: 内容差异 ~3 灰度级(可忽略), I/O 省 4 倍。**
若日后想要 100% 复刻内部路径, 脚本一条参数即可切换(会掉到 4 倍 I/O):
`--levels 4,16,32`(即不写 2x 层)。

---

## ③ 工具方案

### 3.1 现状(已核查)

* `which vips / vipsheader / vipsthumbnail` → 无; `ldconfig -p | grep vips` → 无;
  conda pkgs 缓存里也没有 libvips。
* 12 个 conda env 全部没有 pyvips; 也无 bftools/ImageMagick。
* **但 trident env 里有现成的 `tifffile 2023.2.28 + imagecodecs 2025.3.30 + opencv 4.13 + zarr`**,
  base env 里还有 `tiffinfo/tiffcp/tiffsplit`。
* 网络可达 conda-forge(如需安装可行), 但**本次没有安装任何东西**。

### 3.2 采用方案(已端到端验证, 无需安装)

`convert_flat_to_pyramid.py` —— 纯 Python, 只用 trident env 现成依赖:

* 用 `tifffile.imread(..., maxworkers=16)` 并行解码源 LZW strips(17.6GB 约 23–47s);
* 用 `cv2.INTER_AREA` 逐级降采样(整数倍时数学上等于 block mean, 但直接输出 uint8、多线程、
  内存占用极小 —— numpy `.mean()` 在 17.6GB 输入上会 materialize ~35GB float64 中间量并卡住 1 分钟以上, 已弃用);
* 用 `tifffile.TiffWriter(..., bigtiff=True)` + `compression=('jpeg', 85)` 边建层边写盘(写入顺序=IFD 顺序),
  峰值内存 ≈ level0 + 一层(本次 ~22GB);
* 写完自动用 openslide 回读自检(层数/尺寸/mpp), 不一致返回非 0;
* 先写 `*.part` 再 `os.replace` 原子落盘; `--skip-existing` 支持断点续跑。

### 3.3 备选: 如果批准装 vips

```bash
conda create -y -n wsi_convert -c conda-forge libvips
conda run -n wsi_convert pip install pyvips
# 参考命令
vips tiffsave in.tif out_pyr.tif --tile --tile-width 256 --tile-height 256 \
     --pyramid --compression jpeg --Q 85 --bigtiff
```

理由/前提:
* vips 多线程、对超大 tif 流式处理, 速度可能更快(本次实测其实已经很快, 收益有限);
* **但两个坑必须先验证**: ① vips 的 `--pyramid` 会写全部 2x 层, 无法直接产出"无 2x 层"结构;
  ② 尺寸取整方式若为 ceil 会命中 ②.2 的 `int()` 地雷; ③ vips 是否写 `subfiletype=1` 需实测
  (不写则 openslide 只认 1 层)。**严禁动 SurvPGC 训练 env。**

---

## ④ 单张试转结果

对象(31 张里最小的): `/data/lizhe/Medteam_projects/test/eval_tiff/lung/2211834_20240105093257.tif`
源: 扁平单页 BigTIFF, **stripped(非 tiled)**, LZW(compression=5), RGB 8bit, 62755×93741 = 5.88 Gpx, 8.27 GB

命令与产物:

```bash
time /data/fangyuxuan/miniconda3/envs/trident/bin/python convert_flat_to_pyramid.py \
     /data/lizhe/.../lung/2211834_20240105093257.tif
# → .../SurvPGC_Workspace/external_fukun/2211834_20240105093257_pyr.tif
```

| 阶段 | 耗时 |
|---|---|
| 解码源 strips(maxworkers=16) | 23.1s(热缓存) / **46.9s(首次冷缓存)** |
| 写 level0(JPEG Q85, 17.6GB→0.88GB) | 17.0s |
| 写 level1..64x | ~7s |
| **整张 wall time** | **47.3s**(热) / **≈70s(冷, 按首次采样)** |

产物: **1.25 GB**(= 源体积的 15.1%, 7 层金字塔全含), 比源还小 6.6 倍。

openslide 验证(**SurvPGC env**, openslide lib 4.0.1):

```
dimensions   : (62755, 93741)          level_count : 7
L0:  62755 x 93741  downsample=1.000000  int()=1
L1:  31377 x 46870  downsample=2.000027  int()=2     ← = floor(62755/2)×floor(93741/2), 精确 2x(±1px)✓
L2:  15688 x 23435  downsample=4.000117  int()=4
L3:   7844 x 11717  downsample=8.000405  int()=8
L4:   3922 x  5858  downsample=16.001492 int()=16
L5:   1961 x  2929  downsample=32.002984 int()=32
L6:    980 x  1464  downsample=64.033226 int()=64
tiff.XResolution=40000   tiff.ResolutionUnit=centimeter   openslide.mpp-x=0.25
[2x step check] 6 处相邻层全部 exact-2x±1px = OK
```

TRIDENT 侧验证(**用真实 TRIDENT 类**, trident env):

```
TRIDENT sees: width=62755 height=93741 mpp=0.25 mag=40 levels=7
--mag 20 (patch 512): downsample=2.0 -> level=1 (ds=2.00003), read 512px@L1 -> 覆盖 level0 1024px (期望 1024) OK
seg hest (mag 10, input 512): downsample=4.0 -> level=2 (ds=4.00012), read 512px@L2 -> 覆盖 level0 2048px (期望 2048) OK
```

内容一致性:

* **转换是否忠实于源图**: 转换后 level0 直接与源 tif 同区域逐像素比对
  (`_probe/check_source_vs_converted.py`, 5 个区域含 (0,0)/(58000,88000) 边角) →
  mean|Δ| = 0.36–0.88 灰度级, PSNR 49–57 dB, 尺寸/朝向完全对齐(无转置/错位/丢内容),
  误差即 Q85 JPEG 本身的量级;
* **2x 层 patch vs 内部"level0+resize"路径 patch**: TRIDENT 真实 patch 网格上 mean|Δ| ≈ **1.6 灰度级**(见 ②.4)。

唯一瑕疵: libtiff 每次打开会打一行 `TIFFReadDirectoryCheckOrder: Warning, Invalid TIFF

唯一瑕疵: libtiff 每次打开会打一行 `TIFFReadDirectoryCheckOrder: Warning, Invalid TIFF
directory; tags are not sorted in ascending order.` —— tifffile 写 IFD 的 tag 顺序不是升序所致,
**纯提示, 读取结果完全正确**(层数/尺寸/mpp/像素全部验证过)。只是日志噪声, 可忽略。

---

## ⑤ 全量 31 张: 执行命令与 ETA

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/z_temp/external_cohort/conversion
nohup bash convert_all_external.sh > logs/_batch.log 2>&1 &
tail -f logs/_batch.log          # 看总进度; 单张细节在 logs/<stem>.log
```

脚本行为: 逐张串行 → 单张失败不中断(最后汇总 FAIL 名单) → 输出已存在则跳过
(⇒ **失败直接重跑同一条命令即可续传**) → 源目录只读。

**ETA**

| 项 | 估算 | 依据 |
|---|---|---|
| 时间 | **50–70 分钟**串行(2 并发约 30 分钟) | 实测 11.9 s/Gpx(冷); 全量约 260 Gpx(365.7GB 源 ÷ LZW 2.13 ≈ 779GB raw) |
| 单张峰值内存 | ~22–25 GB(最大 89055×120410 ≈ 32GB raw) | level0 + 一层 |
| 输出体积 | **60–100 GB**(预算按 150GB 留足) | 试转实测 15.1%(0.15×365.7=55GB); 致密区域采样 JPEG 比 ~10–12x 外推约 95GB |
| 磁盘 | /data 剩 1.2T, 够; 但整机使用率已 96%, 注意别写满 | `df` |

输出目录: `SurvPGC_Workspace/external_fukun/`(可用 `--outdir` 改), 命名 `<原stem>_pyr.tif`。
31 个 stem 已核查 **无重名**(肝/肺/肾三目录可直接平铺, 也满足 TRIDENT `--wsi_dir` 不允许嵌套的要求)。

---

## ⑥ 风险与注意事项

1. **floor 取整是硬约束, 不是风格问题**。ceil 会让 TRIDENT 在 L1 上读 4 倍面积,
   等效倍率从 20x 掉到 10x 且**完全静默**(②.2 已用真实代码复现)。改脚本/换工具后,
   务必先 `python verify_pyramid.py <out.tif>` 确认 `int_downsamples` 一行等于 `[1,2,4,8,16,32,64]`。
2. **mpp 必须写成 0.25**。源文件自带 Motic `Scale=0.263049`; 若照搬 0.263,
   seg 的 `downsample = 1/0.263 = 3.80` 落不进 4.0±0.1 的容差, 会退到 level0 读 2048px 大图
   (单 patch 读图量 ×16, feat 阶段同理受影响)。脚本已固定写 0.25(= 40x, 与内部 TCGA 同量级)。
   *代价*: 若这群片子真实光学分辨率确为 0.263, 则物理尺度比内部数据大约 5%;
   这是"管线行为一致性"与"物理尺度保真"之间的取舍, 当前按前者(0.25)。
3. **`--patch_size` 用 512 而不是 256**: 内部 P 的实际 config 是 `20.0x_512px_0px_overlap`
   (`_config_coords.json` 里 `patch_size: 512`), 外部要同参才能对齐特征分布。
4. **文件名后缀 `_pyr` 会影响下游按名字配对**:
   `datasets/dataset_survival.py:284-286` 用 `f"{slide_id}.pt"` 精确匹配特征文件, 且
   `_filter_label_data_by_available_wsi` 会把匹配不上的 case **直接剔除**。
   外部队列的 `slide_id` 列要么写成 `<stem>_pyr`, 要么用 `--output` 去掉后缀。**别到最后才发现整批被过滤。**
5. **失败重跑策略**: `convert_all_external.sh` 幂等 —— 已存在的 `_pyr.tif` 直接跳过;
   中断后重跑即可。单张失败只影响该张, 日志在 `logs/<stem>.log`。
   建议重跑前对失败张确认 `.part` 残file已清理(脚本用 `os.replace`, 失败时可能留 `.part`)。
6. **磁盘**: /data 已用 96%, 只剩 1.2T。本次输出 60–100GB 无压力, 但建议跑完顺手确认
   `df`; 若同时有别的大任务在写盘, 留足余量。
7. **内存**: 单张峰值 22–32GB(最大那张 level0 raw 就有 32GB)。若要并行 2–3 张,
   峰值 ~100GB, 本机 1TB 内存没问题, 但会与别的训练任务抢带宽。
8. **vips 路线(若批准)**: LZW BigTIFF 输入 vips 能读(libtiff 后端), 但必须验证
   ①`subfiletype=1` ②floor 取整 ③`--pyramid` 会连带写入全部 2x 层(拿不到"无 2x"结构)。
   本方案已验证可行, **没有必须装 vips 的理由**。
9. **zstd/其他压缩不可用**: openslide 4.0 只认 jpeg/lzw/deflate 等标准压缩(zstd(50000) 直接报
   `Unsupported TIFF compression`)。脚本默认 jpeg Q85; 若担心 JPEG 有损,
   可 `--compression lzw --quality` (体积约 4–5 倍, 且读写都更慢)。
10. **主仓库 CLAM 路线兼容性(备用, 已评估)**: 转换后的 tif 同样可用 ——
    `wsi_core/WholeSlideImage.py:32` `openslide.open_slide` 能开;
    `_assertLevelDownsamples`(`:361-369`)对非整数 downsample 走"按维度估计"分支, 行为与内部 svs 一致;
    `_getPatchGenerator`(`:282`)的 `int(level_downsamples[patch_level][0])` 同样是截断, floor 取整救了它;
    CLAM 通常用 `patch_level=0 + custom_downsample=2`(`:274-280`), 本质就是"读 level0 再缩", 不需要 2x 层。
    唯一要留意的是 `segmentTissue(seg_level=0)`(`:145`)会**把整个 level0 读进内存**
    (本批 17–32GB/张, 1TB 内存够但慢) —— 若走 CLAM 路线, 建议显式传 `seg_level=2` 之类。

---

## 附: 本次交付物

| 文件 | 说明 |
|---|---|
| `convert_flat_to_pyramid.py` | 转换主脚本(参数化: `--levels/--tile/--compression/--quality/--mpp/--outdir/--skip-existing/--dry-run`) |
| `verify_pyramid.py` | 转换后校验: openslide 层结构 + 精确 2x 检查 + **真实 TRIDENT 层选择复算** + 两设计 patch 差异 |
| `convert_all_external.sh` | 31 张串行批处理(幂等/失败汇总/单张日志) |
| `conversion_lung_2211834.log` | 单张试转完整日志 |
| `_probe/` | 调研期证据脚本: `test_openslide_pyramid.py`(subfiletype/压缩格式矩阵)、`test_int_truncation.py`(int 截断地雷)、`design_decompose2.py`(设计差异分解)、`density_est.py`(体积估计) |
| `../..../SurvPGC_Workspace/external_fukun/2211834_20240105093257_pyr.tif` | 试转产物(1.25GB, 已通过全部校验) |
