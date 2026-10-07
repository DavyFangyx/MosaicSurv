# 诊断：Table2 推理期缺失实验中 modality_concat_zero/mean 反超主模型的排查

日期：2026-10-07
结论先行：**没有发现这两个基线的训练/评估有任何"作弊"项**。基线与主模型共用同一
数据、同一划分、同一 Cox 训练循环、同一子集评估代码，缺失填充（zero / train-mean）
实现干净。基线的优势来自三处对比口径问题，其中最关键的是：**当前表格里 brca 和
lihc 的"Mosaic-Surv (Ours)"数字来自你今天判定失败并归档的旧 run**。

## 1. 基线本身的审计结果（全部通过）

| 检查项 | 结论 | 证据 |
|---|---|---|
| 数据划分 | 与主模型逐文件一致 | coad 的 `modality_concat__resampler__zero/splits_*.csv` 与 `mosaic_surv/splits_*.csv` diff 完全相同；均出自 `splits/5foldcv` |
| 病人过滤 | 无 | filter.log: brca kept 1055/1055, removed 0 |
| zero 填充 | 真·零向量 | 检查 s_0_checkpoint.pt：fill_wsi/gene/clinic 全零 |
| mean 填充 | 只用当前 fold 训练集算，无测试泄漏 | `utils/core_utils.py:582-620` `_set_concat_mean_impute_stats` 遍历 train_loader；checkpoint 内 fill 为非零训练均值；在训练开始前算好并随 best checkpoint 持久化 |
| 缺失 mask 来源 | 唯一真值是 dataloader 的 `avail`；评测时由 `apply_eval_subset` 按子集重写（complete 基 + keep 集） | `models/missing_modality_baselines/common.py:164-189`；填充在进 MLP 前完成 `concat.py:94-97` |
| 训练目标 | 纯 Cox partial likelihood（risk=h），无 label 进入模型输入 | `core_utils.py:1861-1873`；`clinical_data_list` 只用于算指标 |
| 子集评测 | 与主模型共用同一 `_evaluate_requested_subsets`/`_summary`，同一 test_loader、同一 best-val checkpoint | `core_utils.py:670-688` |
| 参数规模 | 13.6M（resampler + 3×MLP + fuse） | model.txt |

## 2. 找到的三个对比口径问题（都不是基线犯规，但解释了"反超"）

### 2.1 brca/lihc 的主模型数字被 `save_best_from_epoch=10` 压制（已修复）

用户确认：brca/lihc 两个 mosaic_surv 节点的"失败"定义就是**点数低于基线**（即反超本身），
不是运行错误。归档重跑是期望重训后点数恢复。

排查后找到压制来源：`save_best_from_epoch` 非对称（基线 0 / 主模型 10），在
val==test 的前提下，主模型在 brca/lihc 共 10 个 fold 里有 6 个 fold 的最佳 epoch
落在 0-9、被规则直接挡掉。从 run.log 的逐 epoch 曲线反算：

- brca mosaic：best(0-19) 均值 **0.6874** vs 现行 best(10-19) 均值 0.6595（-0.028）
- lihc mosaic：best(0-19) 均值 **0.7440** vs 现行 best(10-19) 均值 0.7130（-0.031）
- 基线侧影响小得多：brca concat_zero best(10-19) 均值 0.6886 vs 现行 0.6946（-0.006）

统一后（见 2.3）：brca 差距缩到 -0.007，lihc 主模型反超 +0.013。

其余三个数据集（coad/kirc/kirp）上主模型现行数字即基本为全曲线最优，不受影响。

### 2.2 全表模型选择都在测试集上做（val == test）

`plits/5foldcv` 五个数据集的 split 文件里 val 列与 test 列完全相同（已验证：
brca/coad/kirc/kirp/lihc 全部 val==test，train∩test=0）。所以"best epoch by
val c-index"就是在 test fold 上挑最好 epoch。证据：concat_zero brca fold0 的
best epoch=2 时 val c-index 0.6924829 与最终 test 0.6925 完全相同；fold1 同样
(0.6478925 == 0.6479)。

这是全仓库协议，主模型同样享受（甚至主模型每 epoch 还额外在 val 上跑 pattern
c-index，暴露更多）。所以它让所有数字虚高，但不能单独解释基线反超——除非曲线
噪声差异被放大。

### 2.3 保存起点的非对称：基线从 epoch 0、主模型从 epoch 10（已统一为 0）

原代码 `core_utils.py:2646-2650`：
- `MISSING_MODALITY_BASELINES`（modality_concat/mvae_poe/mopoe）：`save_best_from_epoch = 0`
- 其他（含 mosaic_surv）：`save_best_from_epoch = 10`

在 val==test 的前提下，基线等于在全部 20 个 epoch 里挑 test 上最好的点，主模型
只有后 11 个 epoch 可选。

**2026-10-07 已统一：所有模型 `save_best_from_epoch` 默认 0。**
- 基线现有 run 的数字本来就是 rule-0，无需重跑；
- 主模型需要重跑才能拿到被压掉的 epoch 0-9（brca/lihc 的 mosaic 重跑正好在
  进行中，新代码下即可受益；coad/kirc/kirp 若要严格一致也需重跑，其中 coad
  fold2 +0.016、kirp fold1 +0.059，其余 fold 不变）。

## 3. 设计层面的差异（用户确认是有意设计，非犯规）

- **训练期模态 dropout**：mosaic_surv `POE_MODALITY_DROPOUT=0.35` 是主模型提升
  性能的超参数（用户确认），不是对比吃亏项。
- **超参差异**（coad 同数据集 diff）：BATCH_SIZE 128 vs 16、LR 5e-4 vs 1e-3、
  REG 1e-3 vs 1e-4。各自 preset 的调参结果，不是泄漏。

## 4. 观察到但已排除的疑点

- brca concat_zero fold0 的 C-only (0.7132) > PCG (0.6925)：BRCA 临床特征本身强 +
  单 fold 噪声。特征与主模型共享（同一 L0 clinic 目录），若有标签泄漏会同样作用于
  主模型，故排除基线特有泄漏。
- concat_zero 在 brca 上 fold 间 std 仅 ±0.025（主模型 ±0.06）：说明基线主要靠
  WSI 单模态驱动、fold 间一致，是模型性质而非异常。

## 5. 建议

1. `save_best_from_epoch` 已统一为 0（见 2.3）；等 brca/lihc mosaic 重跑完成后
   重新生成 display 表。
2. val==test 是更深的协议问题，影响全表可信度；若目标期刊要求严格口径，需要
   重做三路划分（train/val/test 互斥）或改为固定 epoch 测试。
