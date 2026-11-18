# `/data/lizhe/Medteam_projects` 中 6 个 TCGA 癌种目录结构整理

本文检查了以下 6 个目录：

- `TCGA-BRCA`
- `TCGA_LIHC`
- `TCGA-COAD`
- `TCGA-PRAD`
- `TCGA-READ`
- `TCGA-STAD`

目标是定位 3 类关键内容：

- `WSI` 数据：记录每位患者/样本的切片
- 基因数据：通常是 `Bulk_RNA`
- `Clinic` 数据：通常是 `clinical.cart.*.json`

---

## 1. 总体结论

这 6 个癌种目录**不是完全统一的组织方式**，主要差异有 4 类：

1. `Clinic` 文件放置位置不同  
   - 有的放在 `Bulk_RNA_cases/clinical...json`
   - 有的放在 `WSI_cases/clinical...json`
   - 有的放在单独的 `clinical/` 目录
   - BRCA 甚至每种模态各有一个 `*_case/clinical...json`

2. case 目录命名不统一  
   - BRCA 用 `Bulk_RNA_case`、`WSI_case`（单数）
   - COAD / PRAD / READ / STAD 用 `Bulk_RNA_cases`、`WSI_cases`（复数）
   - LIHC 不走 `*_cases` 结构，而是单独 `clinical/`

3. 基因目录内部结构大体相似，但 `annotations.txt` 不是每个样本目录都有  
   - 常见结构是：`UUID/*.rna_seq.augmented_star_gene_counts.tsv` + `logs/*.parcel`
   - 部分样本额外带 `annotations.txt`

4. `TCGA-STAD` 存在明显异常  
   - 目录名叫 `Bulk_RNA`
   - 但里面不是 RNA 的 `.tsv`
   - 而是和 `WSI` 一样的 `.svs` 切片文件
   - `Bulk_RNA` 与 `WSI` 的一级 UUID 子目录 **442/442 完全重合**
   - 因此 `TCGA-STAD/Bulk_RNA` 当前看起来像是 **误放/误复制的 WSI 数据**

---

## 2. 共同模式

### 2.1 WSI 目录的常见内部结构

6 个癌种里，`WSI/<uuid>/` 基本都类似：

```text
WSI/<uuid>/
├── TCGA-....svs
└── logs/
    └── TCGA-....svs.parcel
```

也就是说：

- 真正的切片文件通常是 `.svs`
- `logs/` 中常有对应的 `.parcel`

### 2.2 Bulk_RNA 目录的常见内部结构

正常情况下，`Bulk_RNA/<uuid>/` 一般类似：

```text
Bulk_RNA/<uuid>/
├── *.rna_seq.augmented_star_gene_counts.tsv
├── annotations.txt              # 不是每个样本都有
└── logs/
    └── *.rna_seq.augmented_star_gene_counts.tsv.parcel
```

其中：

- 真正的基因表达数据文件是 `*.rna_seq.augmented_star_gene_counts.tsv`
- `annotations.txt` 是可选项，不是所有目录都能看到

### 2.3 case / clinic 辅助目录常见内容

`*_case` / `*_cases` / `clinical/` 中常见这些文件：

- `biospecimen.cart.*.json`
- `clinical.cart.*.json`
- `gdc_manifest.*.txt`
- `gdc_sample_sheet.*.tsv`
- `metadata.cart.*.json`
- 一些统计图片，如 `case_num.png`、`COAD_Case.png`

---

## 3. 六个癌种逐个说明

## 3.1 `TCGA-BRCA`

### 顶层结构

```text
TCGA-BRCA/
├── Bulk_RNA/
├── Bulk_RNA_case/
├── Mutation/
├── Mutation_case/
├── Mythylation/
├── Mythylation_case/
├── RPPA/
├── RPPA_case/
├── WSI/
├── WSI_case/
├── TCGA_info
└── gdc-client
```

### 三类关键文件位置

- WSI 根目录：`TCGA-BRCA/WSI`
- Gene 根目录：`TCGA-BRCA/Bulk_RNA`
- Clinic 文件：
  - `TCGA-BRCA/Bulk_RNA_case/clinical.cart.2026-06-29.json`
  - `TCGA-BRCA/WSI_case/clinical.cart.2026-06-29.json`
  - 另外 `Mutation_case`、`Mythylation_case`、`RPPA_case` 中也各自有 `clinical...json`

### 代表性结构

Gene 样本：

```text
TCGA-BRCA/Bulk_RNA/0019c951-16c5-48d0-85c8-58d96b12d330/
├── ba295155-272e-43eb-9d6a-e4c9c392e68b.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── ba295155-272e-43eb-9d6a-e4c9c392e68b.rna_seq.augmented_star_gene_counts.tsv.parcel
```

WSI 样本：

```text
TCGA-BRCA/WSI/0018cc22-498a-45b8-bfd4-b0fe0c3a2f0a/
├── TCGA-S3-AA15-01Z-00-DX1.A2456A4A-E6E8-4429-8F09-B997AA497BB0.svs
└── logs/
    └── TCGA-S3-AA15-01Z-00-DX1.A2456A4A-E6E8-4429-8F09-B997AA497BB0.svs.parcel
```

### 备注

- BRCA 是 6 个目录里**最复杂**的一个，除了 WSI 和 Bulk_RNA，还带 `Mutation`、`Mythylation`、`RPPA`
- case 目录用的是单数：`*_case`

---

## 3.2 `TCGA_LIHC`

### 顶层结构

```text
TCGA_LIHC/
├── Bulk_RNA/
├── WSI/
├── clinical/
├── clinical.cart.2026-03-26.json
├── clinical.cart.2026-03-26.tar.gz
├── gdc_manifest.2026-03-26.152118_wsi.txt
├── gdc_manifest.2026-06-10.160348_rna.txt
├── metadata.cart.2026-03-26.json
├── TCGA_info
└── gdc-client
```

### 三类关键文件位置

- WSI 根目录：`TCGA_LIHC/WSI`
- Gene 根目录：`TCGA_LIHC/Bulk_RNA`
- Clinic 文件：
  - `TCGA_LIHC/clinical/clinical.cart.2026-06-01.json`
  - 根目录下还有一个旧版本：`TCGA_LIHC/clinical.cart.2026-03-26.json`

### 代表性结构

Gene 样本：

```text
TCGA_LIHC/Bulk_RNA/0036fcec-eaed-430b-9a23-5efb2d2cc7f2/
├── 32b682ec-8156-44ca-bff0-26155c7fdc12.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── 32b682ec-8156-44ca-bff0-26155c7fdc12.rna_seq.augmented_star_gene_counts.tsv.parcel
```

WSI 样本：

```text
TCGA_LIHC/WSI/0166427c-15c9-4b49-954c-eb3545e5699e/
├── TCGA-BC-A217-01Z-00-DX1.E15D9B32-F809-40D8-846B-2334447B80C0.svs
└── logs/
    └── TCGA-BC-A217-01Z-00-DX1.E15D9B32-F809-40D8-846B-2334447B80C0.svs.parcel
```

### 备注

- LIHC 是 6 个目录里**最不一样**的一个之一
- 它没有 `Bulk_RNA_cases` / `WSI_cases`
- `Clinic` 被单独放进 `clinical/`，并且根目录还保留了一份更早日期的 `clinical.cart`

---

## 3.3 `TCGA-COAD`

### 顶层结构

```text
TCGA-COAD/
├── Bulk_RNA/
├── Bulk_RNA_cases/
├── WSI/
├── WSI_cases/
├── TCGA_info
└── gdc-client
```

### 三类关键文件位置

- WSI 根目录：`TCGA-COAD/WSI`
- Gene 根目录：`TCGA-COAD/Bulk_RNA`
- Clinic 文件：
  - `TCGA-COAD/Bulk_RNA_cases/clinical.cart.2026-06-18.json`

### 代表性结构

你给出的示例目录：

```text
TCGA-COAD/Bulk_RNA/0c62b002-8912-4ce0-a2e5-41bc1e3f9996/
├── annotations.txt
├── f66a9fee-f63a-4398-b242-9c065bbe213d.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── f66a9fee-f63a-4398-b242-9c065bbe213d.rna_seq.augmented_star_gene_counts.tsv.parcel
```

另一个普通 Gene 样本：

```text
TCGA-COAD/Bulk_RNA/00ae9ab8-6eaa-4085-af72-26f96df97fa3/
├── 90c9f8cd-4c8c-4f07-af2f-e17db69bd561.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── 90c9f8cd-4c8c-4f07-af2f-e17db69bd561.rna_seq.augmented_star_gene_counts.tsv.parcel
```

WSI 样本：

```text
TCGA-COAD/WSI/001b7d97-9425-43c3-a9a3-a36cb3d2a591/
├── TCGA-A6-2686-01Z-00-DX1.0540a027-2a0c-46c7-9af0-7b8672631de7.svs
└── logs/
    └── TCGA-A6-2686-01Z-00-DX1.0540a027-2a0c-46c7-9af0-7b8672631de7.svs.parcel
```

### 备注

- COAD 的 `Clinic` 明确落在 `Bulk_RNA_cases`
- `WSI_cases` 中有 `biospecimen` / `manifest` / `sample_sheet` / `metadata`
- 但我没有在 `WSI_cases` 中看到 `clinical.cart...json`

---

## 3.4 `TCGA-PRAD`

### 顶层结构

```text
TCGA-PRAD/
├── Bulk_RNA/
├── Bulk_RNA_cases/
├── WSI/
├── WSI_cases/
├── TCGA_info
└── gdc-client
```

### 三类关键文件位置

- WSI 根目录：`TCGA-PRAD/WSI`
- Gene 根目录：`TCGA-PRAD/Bulk_RNA`
- Clinic 文件：
  - `TCGA-PRAD/Bulk_RNA_cases/clinical.cart.2026-06-18.json`
  - `TCGA-PRAD/WSI_cases/clinical.cart.2026-06-18.json`

### 代表性结构

Gene 样本：

```text
TCGA-PRAD/Bulk_RNA/0007888f-8d96-4c01-8251-7fef6cc71596/
├── 88215dd0-5841-44f1-9393-eefd8238cbb3.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── 88215dd0-5841-44f1-9393-eefd8238cbb3.rna_seq.augmented_star_gene_counts.tsv.parcel
```

WSI 样本：

```text
TCGA-PRAD/WSI/00784afd-6fc2-4f5e-b07e-0ebb38152339/
├── TCGA-YL-A8SL-01Z-00-DX2.F64D1539-E590-4B51-96B4-1F95CAE6E33A.svs
└── logs/
    └── TCGA-YL-A8SL-01Z-00-DX2.F64D1539-E590-4B51-96B4-1F95CAE6E33A.svs.parcel
```

### 备注

- PRAD 比较规整
- `Bulk_RNA_cases` 和 `WSI_cases` 都带 `clinical.cart...json`

---

## 3.5 `TCGA-READ`

### 顶层结构

```text
TCGA-READ/
├── Bulk_RNA/
├── Bulk_RNA_cases/
├── WSI/
├── WSI_cases/
├── TCGA_info
└── gdc-client
```

### 三类关键文件位置

- WSI 根目录：`TCGA-READ/WSI`
- Gene 根目录：`TCGA-READ/Bulk_RNA`
- Clinic 文件：
  - `TCGA-READ/Bulk_RNA_cases/clinical.cart.2026-06-21.json`
  - `TCGA-READ/WSI_cases/clinical.cart.2026-06-21.json`

### 代表性结构

Gene 样本：

```text
TCGA-READ/Bulk_RNA/00f55a16-0ee5-4939-8efb-de34e68d4ccd/
├── ff11a9e3-d32b-431b-9ebb-c5a3d9eb0e4f.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── ff11a9e3-d32b-431b-9ebb-c5a3d9eb0e4f.rna_seq.augmented_star_gene_counts.tsv.parcel
```

WSI 样本：

```text
TCGA-READ/WSI/01c43498-6495-4a01-a60f-dfdebbfc1629/
├── TCGA-F5-6861-01Z-00-DX1.011B771B-F52E-412E-9352-1578349BEAF1.svs
└── logs/
    └── TCGA-F5-6861-01Z-00-DX1.011B771B-F52E-412E-9352-1578349BEAF1.svs.parcel
```

### 备注

- READ 也比较规整
- `Bulk_RNA_cases` 与 `WSI_cases` 都有 `clinical.cart...json`

---

## 3.6 `TCGA-STAD`

### 顶层结构

```text
TCGA-STAD/
├── Bulk_RNA/
├── Bulk_RNA_cases/
├── WSI/
├── WSI_cases/
├── TCGA_info
├── gdc-client
└── pymp-pbjijnzr   # 当前权限受限，无法读取内部
```

### 三类关键文件位置

- WSI 根目录：`TCGA-STAD/WSI`
- 名义上的 Gene 根目录：`TCGA-STAD/Bulk_RNA`
- Clinic 文件：
  - `TCGA-STAD/Bulk_RNA_cases/clinical.cart.2026-06-21.json`
  - `TCGA-STAD/WSI_cases/clinical.cart.2026-06-21.json`

### 实际异常

我核查后发现：

- `TCGA-STAD/Bulk_RNA` 中 `rna_seq.augmented_star_gene_counts.tsv` 数量：`0`
- `TCGA-STAD/Bulk_RNA` 中 `.svs` 文件数量：`442`
- `TCGA-STAD/Bulk_RNA` 与 `TCGA-STAD/WSI` 的一级 UUID 子目录重合数：`442`

也就是说，`Bulk_RNA` 当前看上去并不是基因表达矩阵目录，而是 **WSI 目录的镜像/重复**。

### 代表性结构

名义上的 Gene 样本，实际内容却是 WSI：

```text
TCGA-STAD/Bulk_RNA/0006d12e-8e38-4f64-a339-12a1dfb26fa7/
├── TCGA-BR-A44T-01Z-00-DX1.46AA24E7-F2C9-418B-90AA-D6DA2896F5DE.svs
└── logs/
    └── TCGA-BR-A44T-01Z-00-DX1.46AA24E7-F2C9-418B-90AA-D6DA2896F5DE.svs.parcel
```

WSI 样本：

```text
TCGA-STAD/WSI/0006d12e-8e38-4f64-a339-12a1dfb26fa7/
├── TCGA-BR-A44T-01Z-00-DX1.46AA24E7-F2C9-418B-90AA-D6DA2896F5DE.svs
└── logs/
    └── TCGA-BR-A44T-01Z-00-DX1.46AA24E7-F2C9-418B-90AA-D6DA2896F5DE.svs.parcel
```

### 备注

- `TCGA-STAD/Bulk_RNA` 当前**不能直接当作 Gene 数据目录使用**
- 如果后续要做多模态配对，建议优先把 STAD 单独标记为“待人工复核”

---

## 4. 一个对照表

| 癌种 | Gene 目录 | WSI 目录 | Clinic 主要位置 | 组织特点 |
|---|---|---|---|---|
| `TCGA-BRCA` | `Bulk_RNA/` | `WSI/` | `Bulk_RNA_case/clinical...json`，`WSI_case/clinical...json` | 额外有 `Mutation` / `Mythylation` / `RPPA`，case 用单数 |
| `TCGA_LIHC` | `Bulk_RNA/` | `WSI/` | `clinical/clinical...json`，根目录还有旧版 `clinical...json` | 没有 `*_cases` 结构 |
| `TCGA-COAD` | `Bulk_RNA/` | `WSI/` | `Bulk_RNA_cases/clinical...json` | `WSI_cases` 未见 `clinical...json` |
| `TCGA-PRAD` | `Bulk_RNA/` | `WSI/` | `Bulk_RNA_cases/clinical...json`，`WSI_cases/clinical...json` | 结构规整 |
| `TCGA-READ` | `Bulk_RNA/` | `WSI/` | `Bulk_RNA_cases/clinical...json`，`WSI_cases/clinical...json` | 结构规整 |
| `TCGA-STAD` | `Bulk_RNA/` | `WSI/` | `Bulk_RNA_cases/clinical...json`，`WSI_cases/clinical...json` | `Bulk_RNA` 实际像 WSI，需复核 |

---

## 5. 最重要的结论

如果后续程序要统一读取这 6 个癌种，建议不要硬编码成同一种路径规则，而是至少分成下面 4 种情况：

1. `BRCA`：用 `*_case`，并且模态很多
2. `LIHC`：`Clinic` 在单独 `clinical/`
3. `COAD / PRAD / READ`：最接近统一模板
4. `STAD`：`Bulk_RNA` 当前疑似错误数据，必须单独排查

如果只关心 3 个关键内容，可以先按下面规则理解：

- **WSI**：优先看各目录下的 `WSI/`
- **Gene**：优先看各目录下的 `Bulk_RNA/`，但 **STAD 例外**
- **Clinic**：
  - BRCA 看 `Bulk_RNA_case/` 或 `WSI_case/`
  - LIHC 看 `clinical/`
  - COAD / PRAD / READ / STAD 看 `Bulk_RNA_cases/` 或 `WSI_cases/`
