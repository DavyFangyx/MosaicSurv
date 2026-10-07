# Table4: MosaicSurv 消融实验清单（2026-10-07 规范化版）

主模型：`mosaic_surv`（C 联合训练 + FiLM 条件化生存头，类 `MosaicSurv`）。
统一超参来自 `configs/z_exp_gen/mosaic_hparams.sh`（t018 十键），全部消融 preset 共享。

## A 组：读出层 × 训练方式

| 实验 | preset / modality | 实现 | 类 |
|---|---|---|---|
| single | `mosaic_surv_single` | C 联合训练，单一生存头 | `MosaicSurvSingle` |
| single_enum | `mosaic_surv_single_enum` | C 联合训练，7-pattern PoE 枚举，共享单头 | `MosaicSurvSingleEnum` |
| noenum | `mosaic_surv_noenum` | C 联合训练，按当前 `avail` 单次 PoE，不做 pattern 枚举 | `MosaicSurvNoEnum` |
| **main** | `mosaic_surv` | C 联合训练，FiLM 条件化头，主模型 | `MosaicSurv` |
| nodropout | `mosaic_surv_nodropout` | 主模型同构，**训练期关闭模态 dropout（`POE_MODALITY_DROPOUT=0`）** | `MosaicSurv` |
| multi | `mosaic_surv_multi` | C 联合训练，7 个 pattern-specific fuse head | `MosaicSurvMulti` |

## B 组：训练范式

| 实验 | preset / modality | 实现 | 类 |
|---|---|---|---|
| twostage | `mosaic_surv_twostage` | B 两阶段训练，冻结 Enc/PoE/Dec，仅训练 FiLM 头 | `MosaicSurvTwostage` |
| frozen | `mosaic_surv_frozen` | A 训练范式，冻结 backbone，仅训练 FiLM 读出头 | `MosaicSurvFrozen` |

## C 组：损失项

| 实验 | preset / modality | 实现 | 类 |
|---|---|---|---|
| kl | `mosaic_surv_kl` | Jeffreys 替换为标准 KL | `MosaicSurvKL` |
| nojeffreys | `mosaic_surv_nojeffreys` | 去除联合损失中的 J 项（`POE_BETA_TARGET=0`） | `MosaicSurv` |
| detached | `mosaic_surv_detached` | 去除联合损失中的 L_surv 回传（FiLM 头保留 λL_surv；L_surv 不回传 Encoder/PoE alpha） | `MosaicSurvDetached` |

detached（原 surv0）实现语义：`surv_detach_poe=True` → multi_pattern_surv_step 里 PoE
计算包在 `torch.no_grad()` 中。实测梯度：encoder_grad=None，head_grad=有。
即 L_surv 仍保留在损失里、仍训练 FiLM 头，只是不回传 encoder/PoE alpha。

## 生成与汇总

默认生成 BRCA、COAD、KIRC、KIRP、LIHC 五个数据集：

```bash
# _Cindex_Main（A/B/C 三组完整模态 C-index）
bash configs/z_exp_gen/gen_Table4_Abaltion_Test_Cindex_Main.sh
# _Modality_Missing（仅 A 组，Table2 式缺失模态评测）
bash configs/z_exp_gen/gen_Table4_Abaltion_Test_Modality_Missing.sh
```

结果目录：

- `results/Table4_Abaltion_Test/_Cindex_Main/{study}__cell_norm__uni_v1/{modality}/`
- `results/Table4_Abaltion_Test/_Modality_Missing/{study}__L0__gene_raw__uni_v1/{modality}/`

汇总脚本：

```bash
python3 results_display/scripts/Table4_Abaltion_Test_Cindex_Main.py
python3 results_display/scripts/Table4_Abaltion_Test_Modality_Missing.py
```
