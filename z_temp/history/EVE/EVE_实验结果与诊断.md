# 外部验证实验（EVE1/2/3）结果与诊断报告

结果与诊断日期：2026-10-14；文档口径核对：2026-11-18
现行实验设计见 `z_temp/EVE/EVE_design.md`；本报告保留原诊断范围，不代表后续代码的全面验证。
范围：CPTAC-COAD（100 例、8 事件）、CPTAC-BRCA（95 例、2 事件）外部验证 + TCGA 主线 7 数据集天然缺失评估（EVE3）
原诊断结论：当时核查未发现管线 bug；弱结果主要由低事件数统计噪声造成，叠加中等 domain shift。

---

## 1. 实验概览

| 实验 | 内容 | 参照 | checkpoint | 测试集 |
|---|---|---|---|---|
| EVE1 直接迁移 | TCGA 模型零微调直接预测外部队列，报 C-index（5 折 mean±std） | Table1/Table1b | TCGA-COAD / TCGA-BRCA 各 5 折（Table1 L0Test 官方批次，12 模型含 hgcn） | CPTAC-COAD 100 例 / CPTAC-BRCA 95 例（全部符合条件患者，无训练） |
| EVE2 风险分层 | 阈值 τ_k = 第 k 折源训练集风险中位数，risk_k > τ_k 记 High，5 折多数票（≥3）定组，KM + log-rank + Efron Cox HR | FigC（复用 coxph_efron/log-rank/绘图） | 同上 | 同上 |
| EVE3 真实缺失 | 第 k 折测试集完整患者 ∪ 该数据集全部天然缺失患者，混合配对 C-index（至少一方缺失） | Table2（缺失处理与 PCG 参照同口径） | Table2 树 4 个基线（concat_zero/concat_mean/mvae_poe/mopoe）+ Table1 L0Test mosaic；hgcn 仅参照 | TCGA 7 数据集 81 名天然缺失患者中 65 名有生存数据者 |

## 2. 关键配置决策

EVE3 的四个基线沿用 Table2 的 gene_raw，主模型沿用 Table1 的 cell_norm；输入差异按源 checkpoint 披露。旧正文“仅缺P 有生存46”已按 EVE3_missing_stats.csv 更正为45，与总计65一致。

- **splits 外部模式**：registry `split_mode="external_test"`（CPTAC 两 study）——5 个 splits 文件内容相同、train 列空、val==test=全部合格患者；文件 k 语义 = 第 k 个外部折模型的评估队列。
- **前向引擎**（`results_display/scripts/EVE_External_Forward.py`）：从模型目录 run.log `[CMD]` 行精确重建训练 args（模型构造与训练逐位一致）→ 路径重映射到目标队列 → **冻结源队列 bins**（仅首尾按目标时间范围放宽、受相邻内部边界约束保证单调）→ **源训练折 MinMaxScaler**（与内部验证口径一致）→ 复用 `_summary` 前向 → 同格式 pkl 落盘。
- **EVE2 阈值与投票**：τ_k 完全来自源队列训练折（外部队列零信息泄漏）；Cox 线性预测子跨折不可比问题由"每折独立投票 + 多数票"规避（与 FigC 折内中位数二分的动机一致）。
- **EVE3 自然缺失输入**：`allow_missing_modality_files`（opt-in，默认关）——工厂 WSI 过滤放行、三模态加载器对缺失文件零填充；每患者自然 avail 走既有 `unified_mask_csv` 机制。
- **EVE3 生存补齐**：`build_eve3_metadata.py` 从原始 GDC 临床 JSON 提取缺失患者生存（口径=pipeline.extract_survival：days_to_death→事件、否则 last_follow_up→删失，天÷30.44）。
- **hgcn**：CPTAC 双队列 hgcn pack 已构建（coad 100 例 / brca 95 例，P 侧 dcm 适配由用户完成）；前向走独立脚本 `EVE_HGCN_Forward.py`。

## 3. 实验结果

### 3.1 EVE1 直接迁移（C-index，5 折 mean±std：外部 / 内部参照）

| 模型 | CPTAC-COAD 外部 | TCGA-COAD 内部 | CPTAC-BRCA 外部 | TCGA-BRCA 内部 |
|---|---|---|---|---|
| MLP_clinic_flatten | 0.631±0.037 | 0.495±0.058 | 0.820±0.057 | 0.612±0.055 |
| SNN_clinic_flatten | 0.623±0.041 | 0.451±0.072 | 0.735±0.020 | 0.616±0.092 |
| MLP_gene_f | 0.524±0.122 | 0.483±0.058 | 0.404±0.015 | 0.609±0.074 |
| SNN_gene_f | 0.548±0.149 | 0.486±0.041 | 0.323±0.052 | 0.574±0.066 |
| SURVPC_f | 0.607±0.066 | 0.620±0.096 | 0.455±0.086 | 0.656±0.020 |
| MCAT | 0.509±0.106 | 0.599±0.085 | 0.484±0.214 | 0.586±0.055 |
| PORPOISE | 0.596±0.053 | 0.614±0.082 | 0.194±0.041 | 0.613±0.047 |
| SURVPATH | 0.570±0.049 | 0.645±0.051 | 0.280±0.099 | 0.605±0.047 |
| SURVGC_f | 0.601±0.078 | 0.485±0.020 | 0.638±0.257 | 0.570±0.066 |
| HGCN | 0.705±0.057 | 0.578±0.038 | 0.593±0.028 | 0.626±0.007 |
| SURVPGC_f | 0.617±0.062 | 0.575±0.094 | 0.461±0.115 | 0.622±0.035 |
| **Mosaic-Surv (Ours)** | **0.576±0.073** | **0.694±0.029** | **0.511±0.079** | **0.687±0.040** |

表注：COAD 可比对数 595、8 事件；BRCA 可比对数 179、2 事件。±std 为 5 折在同一批患者上的折间波动，**非患者抽样误差**（真实 ±0.25 量级，见 §4.3）。

### 3.2 EVE2 风险分层（log-rank p / HR，5 折多数票分组）

**CPTAC-COAD（100 例）**：

| 模型 | low/high | HR (95% CI) | log-rank p |
|---|---|---|---|
| SURVPGC_f | 58/42 | 3.74 (0.73–19.31) | 0.090 |
| SURVPATH | 80/20 | 2.81 (0.63–12.56) | 0.156 |
| Mosaic-Surv (Ours) | 77/23 | 2.54 (0.57–11.37) | 0.203 |
| MLP_clinic | 57/43 | 1.91 (0.43–8.53) | 0.389 |
| PORPOISE | 6/94 | 7.0e14（发散，单臂 0 事件） | 0.547 |
| HGCN | 1/99 | NaN（99/1 极端分组） | NaN |
| SNN_clinic | — | 全员投票一致，无法分组 | — |
| 其余（MLP_gene_f / SNN_gene_f / SURVPC_f / MCAT / SURVGC_f） | — | HR 1.15–1.86 | p 0.48–0.90 |

**CPTAC-BRCA（95 例，2 事件）**：

| 模型 | low/high | HR (95% CI) | log-rank p |
|---|---|---|---|
| MLP_clinic | 92/3 | NaN（2 事件落双臂） | 0.0000 |
| SNN_clinic | 90/5 | NaN | 0.0034 |
| HGCN | 64/31 | 2.18 (0.14–34.79) | 0.573 |
| MCAT | 31/64 | 0.52 (0.03–8.33) | 0.639 |
| Mosaic-Surv (Ours) | 53/42 | 1.21 (0.08–19.30) | 0.894 |
| 其余 | — | NaN（单臂 0 事件） | 0.18–0.64 |

注：**BRCA 仅 2 事件，多数 HR 不可估（记 NaN）、极端 log-rank p 不可信（卡方近似失效）；coad 的 PORPOISE/HGCN 分组极端（94/99 高组）。**

### 3.3 EVE3 真实缺失（混合配对 C-index，5 折 mean±std）

完整表（6 种缺失类型全列，无患者记 "-"）：

| 模型 | Mixed（全部） | 仅缺P | 仅缺C | 仅缺G | 缺PC(仅G) | 缺PG(仅C) | 缺CG(仅P) | PCG 参照 |
|---|---|---|---|---|---|---|---|---|
| concat_zero | 0.654±0.022 | 0.646±0.015 | - | 0.690±0.053 | - | - | - | 0.702±0.101 |
| concat_mean | 0.628±0.030 | 0.614±0.045 | - | 0.686±0.050 | - | - | - | 0.693±0.103 |
| MVAE_poe | 0.509±0.064 | 0.486±0.066 | - | 0.599±0.129 | - | - | - | 0.620±0.110 |
| MOPOE | 0.491±0.064 | 0.493±0.064 | - | 0.482±0.099 | - | - | - | 0.504±0.110 |
| **Mosaic-Surv** | **0.688±0.019** | **0.673±0.024** | - | **0.749±0.031** | - | - | - | **0.748±0.090** |
| HGCN | - | - | - | - | - | - | - | 0.696±0.105 |

- 仅缺P 47 人（有生存 45）、仅缺G 20 人（有生存 20）；仅缺C/缺PC(仅G)/缺PG(仅C)/缺CG(仅P) 四组无可用生存数据（LIHC 12 名仅G 不在临床 JSON、COAD 2 名仅P 无生存字段等）。
- 81 名天然缺失患者中 65 名可用（55 名从原始临床 JSON 新提取 + 10 名本在 metadata），16 名排除，明细见 `datasets_csv/eve3_metadata/提取报告.csv`。
- hgcn 缺图 pack 无法预测缺失患者，仅出参照列。

## 4. 独立诊断报告（历史快照：2026-10-14）

### 4.1 总体结论

**未发现管线 bug。** 没有 FigC 式（折间 offset 混入 pooled 切分）问题。EVE1/EVE2"普遍弱于内部"主要由**低事件数的统计噪声**（COAD 8 事件、BRCA 2 事件）造成，叠加**中等的真实 domain shift**。

### 4.2 七项核查结果

| # | 核查项 | 结论 |
|---|---|---|
| 1 | RNA 特征对齐 | ✅ rna_clean 两队列均为 5000 列（4999 基因），集合与顺序 diff=0；signatures 缺失基因两侧一致剔除，输入不错位 |
| 2 | 内部参照可信度 | ✅ EVE1 矩阵内部参照与官方 Table1a 逐模型吻合（mosaic 0.694 vs 0.6945 等） |
| 3 | 前向引擎 | ✅ pkl 100 例与 splits 逐一相等；生存值与 metadata 一致；路径重映射 7 字段全部核实且目标存在；risk 与 logits 自洽（nll 重算 diff 4e-7）；C/G 嵌入维度两侧一致（4×512 / 4×768） |
| 4 | 输入分布偏移 | ⚠️ domain shift（中等）：RNA 经源 scaler 变换越界 0.00%（完全在训练域内）；G 嵌入逐维偏移 max 0.99（相关 0.976）；WSI 逐维偏移 max 2.28；WSI patch 数 CPTAC 不低（median 9604 vs TCGA 7152） |
| 5 | 内外排序一致性 | ⚠️ 统计噪声：COAD ρ=−0.15、BRCA ρ=0.14（排序不迁移），但 8 事件下每模型外部 bootstrap SE≈0.10–0.15 完全可解释乱序 |
| 6 | EVE2 机制 | ✅ τ_k 与训练前向逐折一致（如 survpgc_f fold0 −3.3903）；同模型内二值比较无折间 offset 问题；risk 尺度不依赖 bins（nll 的 −ΣS 仅依赖 logits） |
| 7 | 事件数方差 | ⚠️ 统计噪声（主因）：见 §4.3 |

### 4.3 关键量化证据

- **bootstrap 95% CI（COAD，1000 次患者重采样）**：mosaic fold0 观测 0.556 → CI [0.316, 0.806]；survpgc 0.512 → [0.229, 0.819]；hgcn 0.786 → [0.559, 0.948]。单个外部 C-index 的真实不确定性约 ±0.25。
- **内部→外部"衰减"不显著**：P(mosaic 外部 ≥ 内部值 0.694) = 0.158。
- **模型间差异不显著**：mosaic vs hgcn 外部差 0.230 的 CI = [−0.122, 0.516]，含 0。
- **BRCA 几乎不含信息**：CI 宽达 [0.11, 0.72]–[0.48, 1.00]；13–14% 重采样无事件直接退化。
- EVE1 矩阵的 ±std（0.04–0.15）是折间波动，不代表患者抽样误差——展示层存在低估不确定性的陷阱。

### 4.4 归类汇总

| 项 | 归类 |
|---|---|
| RNA 对齐 / 内部参照 / 前向引擎 / EVE2 机制 | 未发现异常（管线无 bug） |
| G/WSI 逐维漂移 | domain shift（有证据，幅度中等；RNA 完全在域内） |
| 内外排序乱、内外差与模型间差均不显著 | 统计噪声（低事件数，有证据，主因） |

### 4.5 建议

1. **管线无需修改**。引擎"args 重建→scaler→前向"链路的闭环证明已做过：内部回归模式前向与历史 pkl 逐位一致（risk diff 0.0）。
2. 展示层：EVE1 外部列建议附注患者级 bootstrap CI（±0.25 量级），避免把折间 std 误当抽样误差。
3. 更强结论只能靠更多事件（纳入其他有生存的外部队列或合并多癌种）；COAD 8 事件、BRCA 2 事件下无法区分模型。
4. EVE2 退化情形（PORPOISE 94/6、HGCN 99/1 单组占优、BRCA HR 全 NaN）在表格中如实保留并在表注说明。

## 5. 工程资产清单

**脚本**（`results_display/scripts/` 除注明外）：
- `EVE_External_Forward.py` — 三模式前向引擎（外部/训练折/内部回归 + EVE3 混合队列）
- `EVE1_Table1_Cindex_Main.py`、`EVE2_FigC_KM_curves.py`、`EVE3_Table2_Modality_Missing_Natural.py` — 三个 display
- `EVE_HGCN_Forward.py` — hgcn 独立前向（双队列）
- `dataset_deployment/scripts/build_eve3_metadata.py` — 缺失患者生存提取

**结果树**：
- `results/EVE1_Table1_Cindex_Main/{cptac_coad,cptac_brca}/{model}/`（12 模型 × 5 折 pkl + test_result.csv）
- `results/EVE2_FigC_KM_curves/train_forward/{tcga_coad,tcga_brca}/{model}/`（τ_k 来源）
- `results/EVE3_Table2_Modality_Missing_Natural/{7 study}/{5 model}/` + `masks/`

**展示产物**（`results_display/`）：
- `EVE1_Table1_Cindex_Main/EVE1_Cindex_Matrix.csv`
- `EVE2_FigC_KM_curves/{cptac_coad,cptac_brca}/`（all_models.png|pdf、per_model/、km_stats.csv）
- `EVE3_Table2_Modality_Missing_Natural/EVE3_Mixed_Cindex_Matrix_full.csv`（完整表）+ `EVE3_Mixed_Cindex_Matrix.csv`（精简表）+ `EVE3_missing_stats.csv`

**数据产物**：
- `datasets_csv/eve3_metadata/{study}.csv` + `提取报告.csv`
- CPTAC hgcn pack：`SurvPGC_Workspace/hgcn data/{cptac_coad,cptac_brca}/pack/kimianet__msigdb_gsea_families__L0`
- splits：`splits/5foldcv/{cptac_coad,cptac_brca}` external_test 模式 + sidecar `split_mode.txt`

**核心代码改动**（全部 opt-in / 行为保真，内部回归验证 bit-exact）：`datasets/dataset_survival.py`（bins_override、allow_missing_files、slide 扩展名泛化、slide→case 映射）、`utils/core_utils.py`（参数透传、pkl key 查表）、splits 生成器 external_test 模式 + validate-only 误写修复、registry `split_mode` 字段。
