# Test1-4 各模型超参数使用说明（MosaicSurv 主模型）

主模型为 `mosaic_surv`（MosaicSurv）。本文档（执行版）说明 Test1-4 四个实验中
各模型超参数的使用规则：哪些模型随 MosaicSurv 公共超参一起变化、哪些保持独立、
哪些绑定关系是代码里已有的。配套的算法视角说明见同目录
`Test1-4_MosaicSurv_Hparams_Algo.md`。

## 模型分类总览

| 类别 | 模型（preset） | 出现位置 | 超参使用 |
|---|---|---|---|
| **主模型** | `mosaic_surv`（MosaicSurv） | Test1 / Test2 / Test3 / Test4（消融参照行） | `mosaic_hparams.sh` 全套 10 键 |
| **基线**（Test1） | `abmil_wsi`、`mlp_wsi`、`transmil_wsi`、`mlp/snn_clinic_mean/flatten`、`clinic_cox`、`mlp_gene`、`snn_gene`、`survpc_f`、`porpoise`、`survpath`、`mcat`、`survgc_f`、`survpgc_f` | 仅 Test1 | defaults + 脚本默认，与 MosaicSurv 无关 |
| **基线**（Test2） | `modality_concat_zero`、`modality_concat_mean`、`mvae_poe`、`mopoe`、`hgcn` | 仅 Test2 | 同上（hgcn 的 batch 由 presets.sh 硬绑 32） |
| **基线**（Test3） | `hgcn` | 仅 Test3 | 同上 |
| **消融**（仅 Table4） | `mosaic_surv_single`、`mosaic_surv_single_enum`、`mosaic_surv_noenum`、`mosaic_surv_multi`、`mosaic_surv_twostage`、`mosaic_surv_frozen`、`mosaic_surv_kl`、`mosaic_surv_nojeffreys`、`mosaic_surv_detached` | 仅 Test4 | `mosaic_hparams.sh` 全套 10 键（消融维度例外见 Test4 节） |

POE 家族其余成员（`survtri_poe_vae_A`、`B`、`C`、`B_single`、`B_multi` 等）
在主表 Test1-3 中全部注释不用，不作为基线出现在任何主表中。

## 1. MosaicSurv 公共超参的唯一来源

`configs/z_exp_gen/mosaic_hparams.sh` 是四个实验 MosaicSurv 公共超参的**唯一来源与入口**，
修改该文件即可同时改变四个实验。当前取值来自 optuna 点 `t018_alpha_beta_learn`
（`Cfilm_Hparam_Eval/cfilm_table1_hparams.csv` 的 t018 行）：

| 键（.conf 键名） | 值 |
|---|---|
| `LR` | 0.001 |
| `REG` | 0.0001 |
| `POE_SURV_LAMBDA` | 1.0 |
| `POE_BETA_TARGET` | 0.1 |
| `POE_MODALITY_DROPOUT` | 0.35 |
| `POE_MMHID` | 128 |
| `BATCH_SIZE` | 16 |
| `ALPHAFIX` | false（可学习 PoE alpha） |
| `ALPHAPGC` | 空（None，不用固定 alpha） |
| `BETAFIX` | false（Jeffreys β warmup 生效，warmup 目标为 `POE_BETA_TARGET=0.1`） |

`BETAFIX` 与 `ALPHAFIX` 一样是可控开关，只是历史原因它们的控制入口有两套：

- **主 Test1-4 生成脚本**：都从 `mosaic_hparams.sh` 取这 10 键，把
  `ALPHAFIX=...`、`BETAFIX=...` 显式写进生成的 .conf；
- **`Cfilm_Hparam_Eval/gen_Table1/2/3_cfilm_hparams.sh`**（历史多组超参评估，遗留区文件名未改）：
  从 CSV 清单的 `alphafix`/`betafix`/`alphapgc` 列读取，逐行写入各 run_id 的
  conf（这套机制保持不变）；
- `defaults.conf` 里的 `ALPHAFIX=false`/`BETAFIX=false` 只是兜底：只有当某个
  conf 没有显式写这两个键时才会生效。写了的 conf 永远以 conf 行为准。

`BATCH_SIZE_STAGE1`（stage1 预训练 batch）保持历史默认 128，不随 MosaicSurv 的
stage2 `BATCH_SIZE=16` 变化。

**参数生效层级**（后者覆盖前者，不做环境变量覆盖，避免同名变量打架）：

```
configs/defaults.conf  <  生成脚本写出的 .conf  <  configs/presets.sh 个别硬编码
```

生成器只从 `mosaic_hparams.sh` 读值（`MOSAIC_*` 变量 / `mosaic_conf_lines()`），
把值显式写入生成的 .conf；`presets.sh` 中个别硬编码（如 hgcn 的
`BATCH_SIZE=32`）最后覆盖。

## 2. 各实验模型分组与超参使用

**总规则**：MosaicSurv（`mosaic_surv`）是主模型。POE 家族其余成员
（A / B / C 系全部变体）在主表 Test1-3 中**全部注释不用**，它们的位置是
Test4 消融（或弃用）。Test4 中的全部消融 preset 都与 MosaicSurv 共享公共超参。

### Test1 — Table1 cindex 主表
生成入口：`gen_Table1_cindex_main_test.sh`
→ `Table1_cindex_main/gen_baselines/gen_baselines.sh` + `Table1_cindex_main/gen_ours/gen_ours.sh`

| 分组 | 模型（preset） | 超参使用 |
|---|---|---|
| 主模型（ours） | `mosaic_surv` | **是**：全套 10 键来自 `mosaic_hparams.sh` |
| POE 家族其余（ours） | `survtri_poe_vae_A`、`B`、`C`、`B_single`、`B_multi`、`mosaic_surv_twostage`、`mosaic_surv_single`、`mosaic_surv_multi` | **已注释不用**（降级为消融对象，见 Test4） |
| 基线 | abmil/mlp/snn/transmil/clinic_cox、survpc_f/porpoise/survpath/mcat/survgc_f/survpgc_f | 与 MosaicSurv 架构无关，不变 |

### Test2 — Table2 缺失模态基线
生成入口：`gen_Table2_missing_modality_baselines.sh`

| 分组 | 模型（preset） | 超参使用 |
|---|---|---|
| 主模型 | `mosaic_surv` | **是**：全套 10 键来自 `mosaic_hparams.sh` |
| POE 家族其余 | `survtri_poe_vae_B`、`B_single`、`B_multi`、`mosaic_surv_twostage`、`mosaic_surv_single`、`mosaic_surv_multi` | **已注释不用**（降级到 Test4 消融） |
| 基线 | `modality_concat_zero/mean`、`mvae_poe`、`mopoe`、`hgcn` | 不变（hgcn 的 batch 由 presets.sh 硬绑 32） |

### Test3 — Table3 训练缺失配比
生成入口：`gen_Table3_missing_rate_test.sh`

| 分组 | 模型（preset） | 超参使用 |
|---|---|---|
| 主模型 | `mosaic_surv` | **是**：全套 10 键，但有一个例外 |
| POE 家族其余 | `survtri_poe_vae_B` 及 B/C 系变体 | **已注释不用**（降级到 Test4 消融） |
| 基线 | `hgcn` | 不变（batch 硬绑 32） |

**例外**：`MISSING_MODE=unified_mask_csv` 下缺失完全由共享 mask csv 决定，
`poe_modality_dropout`（drop_prob）不生效（`utils/missing_mask_protocol.py` 的
`resolve_case_availability` 优先走 mask 分支），所以 mosaic_surv 的 conf 里
`POE_MODALITY_DROPOUT` 固定写 0（与 `Cfilm_Hparam_Eval/gen_Table3` 惯例一致）。

### Test4 — Table4 消融
生成入口：`gen_Table4_Abaltion_Test.sh` → `Table4_Abaltion_Test/gen_ablation_test.sh`

POE 家族降级的 A/B/C 系都在这里做消融。**全部 10 个消融 preset 都随 MosaicSurv
公共超参变化**（它们与主模型的差异只在消融维度本身，公共超参必须一致才是
公平消融）：

| 组 | preset | 差异维度 | 公共超参 |
|---|---|---|---|
| A 读出层 | `mosaic_surv_single`、`mosaic_surv_single_enum`、`mosaic_surv_noenum`、`mosaic_surv`（主模型）、`mosaic_surv_multi` | 读出层设计 | 全套 10 键 |
| B 训练范式 | `mosaic_surv_twostage`、`mosaic_surv_frozen` | 训练范式 | 全套 10 键 |
| C 损失项 | `mosaic_surv_kl`、`mosaic_surv_nojeffreys`、`mosaic_surv_detached` | 损失项 | 全套 10 键 |

**例外**：`mosaic_surv_nojeffreys` 的 conf 中 `POE_BETA_TARGET=0`（由 `_common.bash` 的
preset 覆盖逻辑写回 0，10 键中其余 9 键与主模型一致）；`mosaic_surv_detached`/`kl`/`noenum`
的消融在模型内部实现，公共超参保持不变。

## 3. 已有的绑定机制（代码里一直存在的）

1. **`configs/presets.sh`（架构级绑定）**：所有 `survtri_poe_vae_*` preset 的
   `EXTRA_ARGS` 都从环境读取 `POE_SURV_LAMBDA`、`POE_MODALITY_DROPOUT`、
   `POE_MMHID`、`POE_BETA_TARGET`、`POE_DECODER_HIDDEN_DIM`、
   `POE_TRANSFORMER_LAYERS` 及 stage1 参数。因此 .conf 里写这些键会自动作用于
   全部 poe 系模型——这也是本方案"写 conf 即可传播"的基础。
2. **`run.sh`（全局传参）**：`LR`/`REG`/`BATCH_SIZE`/`ALPHAFIX`/`ALPHAPGC`/
   `BETAFIX` 全局传给所有模型，但只有 poe 系模型真正消费；`ALPHAPGC` 为空时
   不传 `--alphapgc`，`ALPHAFIX=false`/`BETAFIX=false` 时分别不传
   `--alphafix`/`--betafix`。
3. **`presets.sh` 的 hgcn 硬绑**：`apply_preset` 对 hgcn 强制 `BATCH_SIZE=32`，
   覆盖 conf 值（本次未改动 hgcn，仍生效）。
4. **旧 Table4 CSV 绑定（本次已移除）**：原 `Table4_Abaltion_Test/_common.bash`
   按 `CFILM_RUN_ID` 从 `Cfilm_Hparam_Eval/cfilm_table1_hparams.csv` 导入
   lr/reg/lambda/beta/dropout/mmhid 应用于全部 preset，但 **batch_size 从未导入**
   （一直用默认 128），alphafix/alphapgc/betafix 也未导入。本次改为从
   `mosaic_hparams.sh` 取全套 10 键（含 batch_size=16、betafix=false）。

## 4. 本次修改的文件

- `configs/z_exp_gen/mosaic_hparams.sh`（新增）：MosaicSurv 公共超参唯一来源。
- `configs/z_exp_gen/gen_Table4_Abaltion_Test.sh`：从 CSV 多行 T4 逐组生成
  改为单组超参一次生成。
- `configs/z_exp_gen/Table4_Abaltion_Test/_common.bash`：移除 CSV 导入，
  全部 preset 从 `mosaic_hparams.sh` 取值；beta0 覆盖保留。
- `configs/z_exp_gen/Table1_cindex_main/gen_ours/_common.bash`：C 系 preset
  写入 MosaicSurv 公共超参；B 系不变。
- `configs/z_exp_gen/gen_Table2_missing_modality_baselines.sh`、
  `configs/z_exp_gen/gen_Table3_missing_rate_test.sh`：C 系 preset 追加
  MosaicSurv 公共超参行（Table3 的 dropout 例外写 0）。
- 文档同步：`Cfilm_Hparam_Eval/README.md`、`z_temp/Table4/ablation_checklist.md`。

## 5. 已知影响 / 后续事项

1. **Table1-3 重新生成后会覆盖旧结果目录**：C 系模型的 `RUN_NAME` 未变，重新
   生成并训练会把新超参结果写进旧的 results 目录（旧结果用的是 defaults 超参）。
   重新跑之前请先备份/移走旧结果，或按需改用 `RESULTS_BASE`
   （参考 `Cfilm_Hparam_Eval/gen_Table2/3` 的做法）。
2. **Table4 汇总脚本需适配**：新结果目录没有 run_id 后缀
   （`results/Table4_Abaltion_Test/<study>__cell_norm__uni_v1/<model>/`），
   而 `results_display/scripts/Table4_Abaltion_Test.py` 目前按 CSV 的 T4 列
   run_id glob `{study}__*__{run_id}`，需要小改（例如去掉 run_id 后缀匹配）。
3. **`Cfilm_Hparam_Eval` 子系统保持不变**：其 CSV 清单与 T1/T2/T3 生成器是
   历史多组超参评估流程，未纳入本次修改；`mosaic_hparams.sh` 是四个主实验
   生成器的新入口。
