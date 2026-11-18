> 历史归档：本文件保留原排查时的命令、参数与结论，不作为当前实验协议；现行说明见 `z_temp/实验配置清单.md`。

# ModelB Stage1 Scan 与单模态 Dropout 排查报告

**排查日期：** 2026-08-15  
**结论范围：** 当前工作区代码、已有实验配置/结果，以及单元级 mask 实测。

## 结论摘要

1. **ModelB stage1 scan 与 non-scan 的 stage1 训练动力学代码一致。** Scan 只增加固定间隔 checkpoint 保存，并不会改变 stage1 的 forward、loss、optimizer、学习率 scheduler、数据 loader 或 beta warmup。
2. **不能说 scan 与 non-scan 的完整训练过程“随机化完全一样”。** Stage1 在相同 seed 和完全相同配置下可保持同一条随机轨迹；但 scan 后的每个 stage2 子运行没有重新设 seed，会继续消耗全局 RNG，因此各 scan 子运行与 non-scan 不会严格使用同一随机序列。
3. **`--selected_modalities clinic` 不会把 clinic dropout 掉。** 代码只有在选中模态数大于 1 时才启用 modality dropout；单模态 mask 保持不变。已有单元级实测 10,000 个样本：单模态 active mask 改变数为 0，空模态样本数为 0。
4. **已有 scan/non-scan 汇总结果不是严格公平的开关对照。** 现有 `model_B` non-scan 与 `model_B_STAGE1test` scan 的历史配置不同，至少包括 stage1 epoch、stage1 batch size、stage2 epoch 等，不能把指标差异直接归因于 scan。

## 1. ModelB Stage1 Scan

### 1.1 Stage1 是否走同一条训练路径

是。`utils/core_utils.py:_step_stage1_poe()` 中 scan 分支只做以下事情：

- `poe_variant == "B"` 且开启 `poe_scan_stage1_ckpts` 时创建 `stage1_scan/fold_*` 目录；
- 每隔 `poe_stage1_ckpt_interval` 个 epoch 保存一个 checkpoint；
- 最后一个 epoch 即使不整除 interval 也保存；
- 正常的最佳 checkpoint 仍按 validation VAE loss 保存到 `s_<fold>_stage1_checkpoint.pt`。

stage1 的实际训练仍统一调用 `_train_loop_stage1_poe()`。因此以下内容在 scan/non-scan 之间没有代码分支差异：

| 项目 | 当前实现 |
|---|---|
| Stage1 loss | `reconstruction + beta * Jeffreys` |
| Stage1 optimizer | `AdamW` |
| Stage1 learning rate | `lr_stage1` |
| Weight decay | `reg` |
| Stage1 LR scheduler | cosine with warmup |
| Stage1 LR warmup | 固定 `len(train_loader)` steps，即 1 个 stage1 epoch |
| Stage1 beta warmup | 使用 `warmup_epochs`；例如 `warmup_epochs=3` 时 epoch 0/1/2 的 beta 为 `0/0.5/1.0`，之后保持 1.0 |
| Stage1 modality dropout | 训练阶段启用；多模态时按 `poe_modality_dropout` 随机丢弃 |
| ModelB latent sampling | stage1 强制 reparameterization sampling |
| train loader | 同一 `_get_split_loader(..., batch_size=args.batch_size_stage1)` 路径 |

因此，**如果 scan 和 non-scan 在独立进程中使用同一个 seed、同一数据、同一 batch size、同一 stage1 epoch 和同一其他参数，stage1 本身应当走同一随机化轨迹；checkpoint 保存本身不改变训练逻辑。**

### 1.2 Stage2 的实际区别

区别发生在 stage2：

- non-scan：加载 `s_<fold>_stage1_checkpoint.pt`，即 stage1 validation VAE loss 最优 checkpoint；
- scan：依次加载 `ckpt_epoch_*.pt`，每个 checkpoint 启动一个独立 stage2 子运行；
- 每个 stage2 子运行重新初始化 optimizer 和 scheduler，使用相同的 args 配置；
- stage2 使用 `args.warmup_epochs * len(train_loader)` 个 LR warmup steps，默认 scheduler 为 cosine；
- ModelB stage2 仍会 sampling latent，但 B 的 modality dropout 在 stage2 被关闭。

**随机性结论：**

- Scan/non-scan 的 stage1：同配置、同 seed 时可一致。
- Scan 的不同 stage2 子运行之间：**不保证同随机序列**。代码没有在 `for stage1_epoch, ckpt_path in stage1_ckpts` 循环内重新调用 `_seed_torch()`；DataLoader 的 `RandomSampler` 和 ModelB 的 reparameterization 会继续消耗全局 RNG。
- 因此 scan 比较的核心变量是 stage1 初始化 checkpoint，但每个 stage2 分支还包含随机训练噪声。

### 1.3 现有实验结果是否可作为公平 scan/non-scan 对照

不能直接作为严格因果对照。已有历史 WandB 配置显示：

| 配置 | non-scan `model_B` | scan `model_B_STAGE1test` |
|---|---:|---:|
| `seed` | 1 | 1 |
| `batch_size_stage1` | 1 | 128 |
| `max_epochs_stage1` | 5 | 32 |
| `batch_size` | 1 | 128 |
| `max_epochs` | 12 | 25 |
| `warmup_epochs` | 3 | 3 |
| `lr_stage1` | 0.0001 | 0.0001 |
| `poe_modality_dropout` | 0.2 | 0.2 |
| `poe_scan_stage1_ckpts` | false/未设置 | true |
| checkpoint interval | 不适用 | 4 |

所以 `results_display/ModelB_Stage1_scan/summary/` 中的 non-scan 均值与 scan 各 epoch 均值，**只能作为已有实验结果汇总，不能证明 scan 开关本身造成了性能差异**。

## 2. `--selected_modalities clinic` 的 Modality Dropout

### 2.1 代码路径

`models/model_SurvTriPoEVAE.py:_build_available_mask()` 先构造 selected modality mask：

```text
selected_modalities=clinic
available_mask=[False, False, True]
```

随后只有满足以下条件才调用真正的随机 dropout：

```text
self.training
and len(self.selected_modalities) > 1
and (self.training_stage == "stage1" or self.poe_variant == "C")
```

对于 `clinic`：

- `len(self.selected_modalities) == 1`；
- `use_dropout == False`；
- `modality_dropout()` 直接返回原始 mask；
- clinic 不会被以 0.2 概率丢弃。

此外，单模态配置生成脚本 `_single_modal_common.bash` 还显式设置：

```text
POE_MODALITY_DROPOUT=0
```

因此从模型代码和单模态配置两层都不会出现“约 20% 样本唯一模态被丢弃”的 bug。

### 2.2 多模态约束

对多模态输入，`models/model_utils.py:modality_dropout()` 使用独立 Bernoulli mask，并对全空样本做 fallback：

- 每个已 available 模态独立按 `drop_prob` 尝试丢弃；
- 如果某行最终全为 false，则恢复该样本的一个原始 available 模态；
- 因此不会出现三个模态同时丢失；
- 对单模态输入，由于上层根本不启用 dropout，fallback 不是依赖的保护机制。

### 2.3 实测

在项目 `SurvPGC` Python 环境中，对 `modality_dropout()` 进行 10,000 行 mask 实测：

| 输入 available 模态数 | 输出最少 active 模态数 | 空模态样本数 | mask 改变样本数 |
|---:|---:|---:|---:|
| 1 | 1 | 0 | 0 |
| 2 | 1 | 0 | 3,634 |
| 3 | 1 | 0 | 4,819 |

实测与代码结论一致：**单模态不会被 dropout；多模态会随机 dropout，但至少保留一个模态。**

## 最终判断

- **问题 1：** ModelB scan/non-scan 的 stage1 训练动力学实现相同，warmup 和主要 stage1 配置逻辑相同；但完整训练不是严格同随机化，尤其是 scan 的多个 stage2 子运行没有重新设 seed。现有历史 scan/non-scan 结果还存在明显配置不一致，不能直接做公平归因。
- **问题 2：** `--selected_modalities clinic` 已有单模态豁免，不会把唯一 clinic 模态丢掉；未发现约 20% 样本变成空模态的 bug。
- **建议：** 若需要严格比较 scan/non-scan，应固定所有参数并在每个 stage2 scan 子运行开始前按 `base_seed + stage1_epoch` 显式重置 Python/NumPy/PyTorch RNG，同时将完整 effective config 写入每个 stage2 子目录。
