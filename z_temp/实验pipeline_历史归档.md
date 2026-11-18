# 实验pipeline 历史归档

> 从旧版《实验pipeline》笔记(实验pipeline.md 2026-10-29 重写前版本)移出的历史遗留内容。
> 保留目的:过程取证/查旧口径用;论文写作与现行实验组织一律以重写后的 实验pipeline.md + z_temp/实验配置清单.md 为准。
> 更早的历史以飞书版本历史为准。

---

## 1. 调参剪枝算法(两层剪枝,论文不写)

现在是两层剪枝,比的东西不一样。

1. 数据集硬阈值
一个数据集 5 折跑完后,用每折 best val C-index(epoch >= 10 起存的最好点)取平均。低于 OPTUNA_STUDY_MIN_CINDEX 就立刻 TrialPruned,后面的数据集不再跑。
顺序是 kirp → coad → kirc → brca → lihc,所以 kirp 过不了 0.8115 就会直接砍掉。

2. MedianPruner
每个 epoch 上报的是当前 epoch 的 val C-index,不是 running best。
--optuna_pruner median 会拿这个值和已经完成的 trial 在同一个 global step 上的值的中位数比;方向是 maximize,比中位数差就剪。
前 5 个 trial 不剪(n_startup_trials=5),每个 trial 前 3 个 step 也不剪(n_warmup_steps=3)。
这个 step 是跨数据集、跨 fold 拼起来的,不是单折 epoch 号。
Objective 仍是 5 个数据集 mean 的再平均,这个值本身不做阈值。

---

## 2. Table2 历史设计讨论(已由现行 6 模型清单取代)

### 2.1 缺失模态适应基线实现

任何一个多模态方法都可以拆成两层:

- 输入适配器:把原始像素/表达量/临床字段变成特征。Flex-MoE 的 Custom3DCNN、PatchEmbeddings,HGCN 的 patch 切割,都属于这一层。
- 表示与缺失机制:模态怎么融合、缺模态怎么处理。这才是方法本身——PoE 的专家乘积、Flex-MoE 的 MoE 路由 + missing bank、HGCN 的在线 MAE 补全。

Table 2 比的是缺失机制,不是特征质量。所以公平协议只有一条:所有方法吃同一套冻结特征(WSI=UNI-v1 bag、Gene=scFoundation、Clinic=L4 prompt),把输入适配器统一替换成同一个轻量投影,机制层原样保留。

### 2.2 基线&改装说明

填在哪个模型上,决定了它是基线还是消融:

- 真正的朴素下界(放 Table 2、当地板)要填在一个普通 concat/late-fusion MLP 上,不带 VAE:三模态特征拼起来 → 两层 MLP → 生存头,测时缺失位填 0/均值。这才是读者看到表格底部期待的那个"最笨做法端到端"的数。

### 2.3 缺失回填 / 缺失专家 / FiLM 调制(早期改造方案讨论)

- 缺失回填:模型中 VAE 直接用缺失输入(例如 PC),生成缺失值(G'),最后缺失值放入输入(PCG')中经过 enc 与分类头完整的推理
- 缺失专家:7 个并行的专属投影头,专门处理 7 种不同类型的缺失情况(P C G PC PG CG PCG)
- FiLM 调制:单头替换为条件化 FiLM 调制层(也对 7 种不同类型的缺失情况)

### 2.4 B/C 分支的模态增强改造(已完成实验,未作为消融)

- B 分支的增强:
  - B+Single:基础 B +(阶段2改动)将 WSI 重建目标从池化特征改为 token 序列,并增加缺失回填机制。
  - B+Multi:基础 B +(阶段2改动)将单头替换为 7 个并行的专属投影头,一次编码,七种读法。
  - B+FiLM:基础 B +(阶段2改动)将单头替换为条件化 FiLM 调制层。
- C 分支的增强:
  - C+Single:与 C 没什么区别(实际上)。
  - C+Multi:基础 C + 端到端训练,将单头替换为 7 个并行的专属投影头。
  - C+FiLM:基础 C + 端到端训练,引入 FiLM 调制头。

> 现行消融变体(Table4 的 11 变体:single/single_enum/noenum/nodropout/multi/twostage/frozen/kl/nojeffreys/detached)由 z_temp/Table4 与 实验配置清单.md 定义,以上为早期方案讨论。

### 2.5 Table2 实验设置论证(仍有效,已并入新版笔记)

训练集 = 完整病例集,所有方法一致;缺失只发生在测试期。理由是 Table 2 要隔离的变量是"推理期鲁棒性",训练侧必须是同一起跑线。我方的 modality dropout 作为方法内置机制保留并披露,这个"便宜"由 Table 3 来还。

---

## 3. Table3 参考对象:EMMS 2026

**EMMS 2026: Evidential Fusion Network for Multimodal Survival Prediction under Missing Modalities**

- 跨模态一致性:无论是 Pathology-only(P)、Genomics-only(G)还是 Complete-modality(C)场景,EMMS 均保持稳定的高性能,表明其对单一模态缺失或混合缺失的适应能力。
- 数据缺失耐受性:在 60% 高缺失率下,EMMS 的性能衰减幅度小于基线方法。例如,BRCA 的 Complete-modality 场景中,EMMS 的 C-index 仅比 0% 缺失时降低 0.025(从 0.740 到 0.715),而 MUSE 降低 0.016(从 0.682 到 0.698,注:原文可能存在排版误差)。

(引用线索保留;Table3 现行口径 = 7 配比 × 7 子集 × 3 模型,见新版笔记)

---

## 4. Table4 零填充模型论述(早期设计讨论)

"ModelB 不做边缘化、改成喂零"(放 Table 4、当消融)才是"在我模型上改"。PoE 处理缺失的正确做法是把缺失那个专家从乘积里丢掉(边缘化);"喂零"是保留专家、灌零输入 → 它吐一个垃圾后验去污染整个乘积。这行的价值是证明"边缘化 > 填充",是漂亮的消融,但不是朴素基线。

---

## 5. FigC 旧定义:缺失率鲁棒性曲线(已改作 Table3 的 train_missing_rate.png)

**Fig C** 缺失率鲁棒性曲线,η ∈ {0,20,40,60,80,90,100}%。

现行 FigC = KM 生存曲线 + HR(折内中位数二分,coxph_efron,见新版笔记与 z_temp/km_check)。

---

## 6. 外部验证旧设计文本(已落地为 EVE1/2/3)

### 6.1 CPTAC 数据旧统计(旧口径,以 EVE 诊断报告为准)

- 本数据集来自 CPTAC-2(Phase 2,回顾性队列),GDC project_id = CPTAC-2;CPTAC-3(Phase 3 前瞻队列)样本不在本数据集内。
- 旧表格:CPTAC-COAD 178 病例目录 / 2437 个 .dcm,RNA 105 例,OS_MONTHS 102/109 有值、死亡 8 例;CPTAC-BRCA 198 病例目录,RNA 127 例,生存时间已有但死亡事件仅 2 例。

> 现行口径:CPTAC-COAD 100 例 / 8 事件、CPTAC-BRCA 95 例 / 2 事件(breast 无生存已退化)。见 z_temp/history/EVE/EVE_实验结果与诊断.md。

### 6.2 实验 1 直接迁移评估(旧文本,两版)

用 TCGA 训练好的模型,不做任何微调,直接在外部队列上预测,报告 C-index、IBS、2y/5y AUC。所有基线用完全相同的流程。
- 参照实验:Table1,指标和口径与 Table1b 一致。
- 模型:Table1 里的模型(12 模型含 hgcn)。
- checkpoint:TCGA-COAD 的 5 个折模型,不做任何微调。
- 测试集:CPTAC-colon 中全部符合条件的患者,没有训练集。
- 推理和指标:每折模型 k 都对全部 CPTAC 患者做预测,分别计算 C-index、IBS 和 2y AUC。IBS 和 AUC 所需的生存函数用该折 TCGA 训练集上拟合的 Breslow 基线,直接复用 Table1 的 Cox 缓存流程。5 折报告 mean±std。

### 6.3 实验 2 风险分层(旧文本)

- 参照实验:FigC,沿用 coxph_efron 和 log-rank 代码,只改分组方式。
- 分组:对每个折模型 k,计算它在第 k 折 TCGA 训练集上风险分数的中位数,作为阈值 τ_k。阈值完全来自 TCGA,不使用 CPTAC 的任何信息。用第 k 折模型预测每个 CPTAC 患者的风险,与 τ_k 比较,得到一个高/低风险判定。每个患者共有 5 票,按多数票(至少 3 票)决定最终分组。5 是奇数,所以不会平票。
- 动机:不同折的 Cox 线性预测子不可比,不能把 5 折的风险分数直接取平均。

### 6.4 实验 3 真实缺失场景评估(旧文本)

- 参照实验:Table2,模型、checkpoint 和缺失处理方式都与之一致。
- 模型(6 个):mosaic_surv、modality_concat_zero、modality_concat_mean、mvae_poe、mopoe、hgcn。
- checkpoint:直接用 Table2 已训练好的 7 个数据集 × 5 折三模态模型,mosaic_surv 用 Table1 的 L0Test 官方批次。
- 混合队列 C-index:对每个数据集 d、每一折 k:取出评估集 S = 第 k 折测试集中的完整患者(PCG)∪ 该数据集全部天然缺失患者 M_d。枚举 S 中的可比较对,只保留至少一方属于 M_d 的对。跨数据集合并:对每一折 k,计算 C_k = Σ_d conc(d,k) / Σ_d comp(d,k)。5 折报告 mean±std,与 Table2 的口径一致。
- 对照列:同样的第 k 折测试集上只算"完整-完整"对的 C-index(PCG 参照)。
- 按缺失类型拆分:缺 P(47 人)和缺 G(20 人)两组。

### 6.5 TCGA-COAD checkpoint 能否用于 CPTAC(论证文字)

可以,而且这正是外部验证的标准做法。先例:CLAM(Lu et al., Nature Biomedical Engineering 2021)在 TCGA 上训练、用 CPTAC 作为独立测试集;UNI 原论文大量下游评测直接使用 CPTAC 队列;CHIEF(Wang et al., Nature 2024)的预后部分。
成立原因:一是癌种相同(结肠腺癌);二是两边的特征提取器(UNI、scFoundation)都是冻结的,特征空间一致。
对齐项:1. 生存终点(TCGA 训练用 OS/DSS/PFI,CPTAC 基本只能得到 OS);2. 时间单位与离散化分箱边界;3. RNA 数据来源(GDC 下载、相同 workflow、同一套归一化);4. 肿瘤部位(剔除直肠癌样本)。

---

## 7. 附录旧清单(多数未实现)

- 完整单模态基线表(12 个单模态模型全放)
- 数据集统计表(样本数、事件数、事件率、中位随访、7 种组合各自 n)——(此项已在 data_tcgal_stats 有产出)
- 编码器敏感性(UNI-v1 vs v2,scF cell/gene × norm/raw × CSV-raw)——**未做**
- 双模态子协议(砍成 WSI+Gene 跟 LD-CVAE、DisPro、M2Surv 在原生设定下比)——**未做,信心不足**
- Adaptation of Missing-Modality Baselines 一节
- 计算成本(训练/推理的显存、时长、参数量、FLOPs)——**未做,可不加**

---

## 8. 下一步计划(已完成,留存取证)

1、调参:3 个参数 25 28;5 个参数正在训练(t028_alpha_learn、t025_alpha_learn、t028_orig、t015_alpha_beta_learn、t018_alpha_beta_learn(当时感觉效果最好)、t022_alpha_beta_learn)
2、将模型重命名 代码进行重构:Cfilm → Mosaic_Surv,之后将一个合适的参数代入主线运行全流程任务。
3、使用重构之后的模型与代码将所有实验完整的跑上一轮。

> 结果:t018_alpha_beta_learn 定为主模型超参(开源正式化规划 Phase 3);改名 Phase 2 完成;全流程实验 = 现行 Table1-4 + FigB/C/D + EVE。

---

## 9. 实现过程(第一~五阶段,历史)

### 第一阶段:数据准备与统计

- 数据集清单:共 8 个癌种:TCGA-LIHC、TCGA-BRCA、TCGA-COAD、TCGA-PRAD、TCGA-READ、TCGA-STAD、TCGA-LUAD、TCGA-LUSC(后两者原始合并,需按 project_id 字段拆分)。
- 模态对齐与缺失统计:以 submitter_id 为主键,三路特征(WSI / Clinic / Gene)做三表 outer join,统计每个病人的模态可用情况,每个数据集输出一张 7 行统计表(step 1)。
- 附属指标:中位随访时间、事件率(死亡比例)。当时决定:先不进行过滤(先验认为所有数据都可以正常使用),COX 自带高通滤波。
- 编码位置(旧路径):
  - Clinic 编码:CONCH-main/projects/outputs/TCGA_*
  - Gene 编码:projects/cancer_multimodal/outputs/embeddings/tcga_gene
  - WSI 编码:/data/wanshan/workspace/medical_cv/TRIDENT/tcga_processed/*/20.0x_256px_0px_overlap/features_uni_v1(h5)
- Q00:SurvPGC_data_tcgal_stats_癌种患者统计;Q01:多数据集完整代码可用;Q1:精简 models 单模态模型;Q2:元数据生成脚本整理到 Meta_Gen/

### 第二阶段:模型 Coding

编码顺序:model_utils.py 共享组件 → Model_A(无监督 VAE + 冻结 + 线性探针)→ Model_B / Model_C。
验证节点(TCGA-BRCA):shape check、前向验证(输入三模态 → z、重建、loss 无 NaN;缺 1/2 模态前向正常)、训练几步 loss 下降。
Model_A、B、C 均视为独立模型参与后续对比,不是消融,是递进的三个配置。
问题记录:卡死 OOM;是否用正规 MIL 作为 WSI Enc 之前预处理。

### 第三阶段:训练监控体系(WandB)

重建与生成(每 step):loss/total、loss/rec_*、loss/kl_or_jeffreys、logvar_obs_*
隐空间健康度(每 epoch):z/mean_norm、z/mean_std、poe/alpha_*(PoE 可学习权重 softmax 后)
下游生存(每 epoch):loss/surv

### 第四阶段:单模态基线

旧数字(已过期,现行以 Table1a L0Test 为准):
- Clinic: L4 + MLP flatten cindex = 0.6420(方差 0.001953)
- Gene: scF cell norm + MLP cindex = 0.6740(方差 0.003974)
- WSI: UNI v1 + ABMIL cindex = 0.6880(方差 0.000323)

### 第五阶段:完整 VAE 模型训练

- ~~数据划分:每数据集统一 5-fold CV,按事件率分层采样~~(已划掉,现行 splits 另定)
- 运行矩阵:Model_A/B/C × 8 数据集,24 个实验配置(step 7)
- 产出:数据集 × Model 的 C-index 汇总表
- 定位模型问题:先在单折上确认三个模型的 C-index 至少能接近(不需要超过)单模态基线,再进入 5 折验证和调参阶段。

---

## 10. 旧数据集取舍论证(已被主线 7 数据集取代)

决定性变量不是样本数,而是事件数(c-index 只由可比较的事件对决定):
- prad 和 kich 每折测试集只有约 2 个事件,c-index 基本是噪声(std 高达 ±0.15~0.32)。
- read(只有 25 个事件)全表所有模型都在 0.43~0.56 之间徘徊,随机水平。
- 旧建议:舍弃 kich、prad、read(stad 已因 gene FM 为 0 自然出局),主集保留 brca / coad / kirc / kirp / lihc;kirp 只有 40 个事件属于边缘,可留在主表并在稳健性分析里做 4 癌种版本。

> 现行主线 7 数据集 = brca/coad/kirc/kirp/lihc/luad/lusc(luad/lusc 为后来加入的肺癌两 study)。
