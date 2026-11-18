# COAD → COADREAD 数据盘点（合并 READ）

> 结论：**公共 P/C/G 编码与全部表格、split，READ 已齐全**；**HGCN 图编码 READ 完全缺失**（目录在、文件全无）。合并评估前只需补 READ 的 HGCN 三步生成，再对合并队列做表格拼接 + workspace 建链 + 重新 5 折划分。
> 统计日期：2026-09-27。

## 1. 合格患者口径

| | COAD | READ | COADREAD |
|---|---|---|---|
| 患者总数 | 460 | 171 | 631 |
| P/C/G 全齐且可入 split | **429** | **153** | **582** |
| 事件（censorship=0） | 96 | 26 | **122** |
| 事件率 | 22.4% | 17.0% | 21.0% |
| survival/censorship 缺失 | 0 | 0 | 0 |

READ 18 人不合格：15 人缺 metadata+WSI，3 人缺 gene_fm+rna_csv。COAD 与 READ case_id 零重叠。

## 2. READ 已有 ✅ / 缺失 ❌ 一览

### 2.1 公共编码 `SurvPGC_Workspace/tcga_read/`

| 路径 | READ | 覆盖 | 状态 |
|---|---|---|---|
| `P/uni_v1/*.pt`（UNI 1024） | 158 | metadata 157 slide 全覆盖 | ✅ |
| `P/uni_v1_h5/*.h5` | 158 | 同上 | ✅ |
| `C/L0~L5/*.pt` | 各 171 | 全部患者 | ✅ |
| `C/D0~D5/*.pt` | 各 171 | 全部患者 | ✅ |
| `G/scFoundation_embedding_{cell,gene}_{norm,raw}/*.pt` | 各 166 | 与 rna_clean 一致 | ✅ |

### 2.2 HGCN 图编码 `SurvPGC_Workspace/hgcn data/tcga_read/`

| 路径 | READ | COAD（对照） | 状态 |
|---|---|---|---|
| `P/kimianet/t_img_fea.pkl` | — | 429 例 | ❌ 缺 |
| `G/msigdb_gsea_families/t_rna_fea.pkl` | — | 429 例 | ❌ 缺 |
| `C/L0~L5/{x_cli,t_cli_feas,ttt_cli_feas,edge_index_cli}.pkl` | — | L0~L5 全，各 458 例 | ❌ 缺 |
| pack（`patients/sur_and_time/all_data.pkl`） | — | COAD 也没有 | ⚪ 可选 |

> hgcn data 全局现状：brca/coad/kirc/kirp/lihc 齐全；**kich/prad/read/stad 为 0 文件**。
> 注意：hgcn pkl 是 **joblib 格式**（`joblib.load`），裸 `pickle.load` 会报 `invalid load key`。

### 2.3 表格数据

| 路径 | READ | 状态 |
|---|---|---|
| `datasets_csv/metadata/tcga_read.csv` | 157 行，生存字段无缺失 | ✅ |
| `datasets_csv/clinical_data/tcga_read_clinical.csv` | 171 行，列与 COAD 一致 | ✅ |
| `datasets_csv/raw_rna_data/combine/read/rna_clean.csv` | 166 行 | ✅ |
| `datasets_csv/feature_manifests/tcga_read.csv` | 166 行 | ✅ |
| `data_tcgal_stats/TCGA-READ/` | 171 行（has_P/C/G 标注） | ✅ |
| `splits/5foldcv/tcga_read/` | 153 人 5 折（按 censorship 分层） | ✅ |

## 3. READ 要补的：HGCN 三步生成

源数据均已就位，只需运行（命令见 `SurvPGC_Workspace/HGCN data gen.md`）：

| 步骤 | 命令 | 源数据 | 计算 |
|---|---|---|---|
| 1. WSI | `generate_native_hgcn_graph.py --study tcga_read` | `/data/lizhe/Medteam_projects/TCGA-READ/WSI/`（166 张 .svs ✅） | GPU（KimiaNet） |
| 2. RNA | `generate_hgcn_rna_graph.py --study tcga_read` | `/data/lizhe/Medteam_projects/TCGA-READ/Bulk_RNA/`（167 TSV ✅） | CPU |
| 3. Clinic | `generate_hgcn_clinic.py --study tcga_read` | A_pipeline 已产出 `projects/outputs/TCGA-READ/A_manual/HGCN_clinic/L0~L5` ✅ | 仅拷贝 |

## 4. 合并 COADREAD 清单

1. **表格拼接**：metadata / clinical / rna_clean / feature_manifest 各按行拼接（合格 582 人）；`data_tcgal_stats` 合并生成患者表。
2. **registry 注册**：`registry.yaml` + `registry.py` 新增 `tcga_coadread`（display_name `TCGA-COADREAD`）。
3. **workspace 建链**：`SurvPGC_Workspace/tcga_coadread/{P,C,G}` 链接 COAD+READ 现有文件。
4. **重新划分**：`generate_5fold_splits.py`（按 censorship 分层；约 24 事件/折）。
5. **HGCN 合并**（README 先补完 READ 后）：
   - `t_img_fea.pkl` / `t_rna_fea.pkl`：两 study dict 直接 merge（case_id 无重叠，**无需重算**）；
   - `C/L0~L5`：两 study 同 scheme pkl merge（字段顺序锁死，可合并；`edge_index_cli` 复用任意一侧）；
   - pack 可选。
6. **评估口径**：合并后以 `tcga_coadread` 一列替换现有 Table1 的 `tcga_coad`（需与现有 5 数据集结果对比口径保持一致）。
