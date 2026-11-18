# 实验 pipeline（MosaicSurv）

本文件定义当前实验计划；详细分组、配置与指标见 `z_temp/实验配置清单.md`。历史讨论保留在 `z_temp/实验pipeline_历史归档.md`，不作为现行实验设置。

## 1. 研究问题

1. 完整模态下，MosaicSurv相对单模态、双模态及三模态方法的生存预测能力如何？
2. 推理期任意模态缺失、训练期不同模态缺失，以及真实天然缺失分别带来什么影响？
3. 条件化读出、子集训练、联合生存监督与分布正则分别贡献了什么？
4. 重构特征是否保持下游响应、群体分布及患者级身份信息，模型能否迁移到独立外部队列？

## 2. 公共实验条件

内部主线为TCGA-BRCA、COAD、KIRC、KIRP、LIHC、LUAD、LUSC七队列，使用患者级五折划分。P、G、C分别为病理、基因、临床模态；常规冻结表征为UNI-v1、scFoundation及L0临床prompt，HGCN保留原生图表征。

完整预测与缺失预测是不同实验条件。Table1与Table4完整模态使用cell_norm，Table2、Table3及Table4缺失模态使用gene_raw；不把不同批次或不同输入设置的主模型分数相互替换。

## 3. 内部实验

| 实验 | 比较设计 | 核心指标 | 说明 |
|---|---|---|---|
| Table1 | MosaicSurv对比13个单/双/三模态模型，完整模态输入 | C-index、IBS、2年/5年AUC、平均排名、配对检验 | `z_temp/Table1/Table1_design.md` |
| Table2 | MosaicSurv对比五个缺失模态基线，完整病例训练，七子集测试 | 各输入条件的C-index | `z_temp/Table2/Baselines_tests.md` |
| Table3 | 三方法共享七种训练缺失设置，各评七测试子集 | C-index与跨缺失条件变化 | `z_temp/Table3/Table3_design.md` |
| Table4 | A1读出头、A2训练增强、B训练范式、C损失项；分完整与缺失评估，完整模型作共同参照 | 相对同批次完整模型的C-index差异 | `z_temp/Table4/ablation_checklist.md` |
| FigC | 每个模型按折内风险中位数分高低风险组 | KM、log-rank、HR及95%CI | `z_temp/FigC/KM_curve_design.md` |
| FigD | 同患者真实与重构特征的配对对照，当前为第0折 | 注意力、降维分布/距离、患者级检索 | `z_temp/FigD/FigD_design.md` |

## 4. 外部迁移与天然缺失

实验说明独立放在 `z_temp/EVE/EVE_design.md`。

- EVE1：TCGA源模型零微调迁移到CPTAC-COAD/BRCA，比较外部C-index及内部参照。
- EVE2：以源训练折中位风险作阈值，五折多数票划分外部高低风险组，比较生存曲线、log-rank和HR/CI。
- EVE3：TCGA天然缺失患者与本折完整测试患者组成评估集，仅统计至少一方天然缺失的可比对；与四个缺失处理基线比较，并报告完整病例参照。

EVE1/2是独立外部验证，EVE3是内部来源的天然缺失评估。CPTAC-COAD为100例/8事件，BRCA为95例/2事件；EVE3共81名天然缺失患者，65名有生存数据。低事件数、重复评估相同患者和表征差异需分别披露。

投稿取舍保留EVE3正文评估；EVE1/2受外部事件数限制，是否展示需结合不确定性另行决定。尚未采用CPTAC与TCGA混合评估方案。

## 5. 数据与证据边界

数据构建资料统一放在 `z_temp/dataset_build/`，分别记录通用规范、TCGA、CPTAC、COADREAD和其他外部切片队列。COADREAD属于合并队列准备，不自动纳入当前七队列对比。

现有内部验证集与测试集重合；Table2部分主模型种子与对照不同，HGCN表征和优化器也不同。这些是现有结果的解释边界，不能通过文档表述成严格同配置实验。旧结果及诊断放在 `z_temp/history/`。

不确定性随缺失程度的单调性与校准曲线尚需独立实验支持，不能从VAE结构或IBS结果直接推定。
