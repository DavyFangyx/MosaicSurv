对照范围：`33ce8ea6e6f6437ed781bcdc0aebd1e908e89a0c`（Table1 L0 结果整理）之后，到当前工作区。只看主模型 `survtri_poe_vae` B/C，以及 `configs/z_exp_gen/Table1_cindex_main` 生成的最终训练 conf。

结论先说：**和 `33ce8ea` 当时的默认 B/C 协议比，当前 queue 里这两份 conf 实际会跑的训练协议，只剩 dropout 两处不同。**

1. dropout 从模型里逐 batch 抽样，改成 dataloader 按 case 缓存。
2. B 的 stage2 以前不 dropout，现在 train split 也会 dropout。

除此之外，alpha / beta、超参、encoder-decoder、Cox head、B 两阶段 / C 联合训练、选点规则，都和 `33ce8ea` 相同。代码里后来加的 `--alphafix` / `--betafix` 这两份 conf 都没打开，不会走到固定 α / 固定 β。

这句不要和 2026-08-26 的 `L0Test` 调参训练混在一起。那一次超参已经改过，和现在这两份 conf 不同。

历史结果分三套，不要混：

| 结果目录 | 时间 | 含义 |
|---|---|---|
| `results/Table1_Cindex_Main/L0Test_BeforeTune` | 2026-08-17 | 提交 `33ce8ea` 时那套默认协议：B/C 共用 `batch=128, lr=5e-4, beta=1.0`，beta 走 warmup |
| `results/Table1_Cindex_Main/L0Test` | 2026-08-26 | 调参后真正训练的 Table1 L0：B/C 各自写死一组超参，`BETAFIX=true` |
| `results/Table1_Cindex_Main/L0Test After Fix some err` | 2026-08-27 | 和当前 gen 一致：调参超参被注释掉，B/C 都退回 defaults |

下面“以前”默认指 **L0Test 那次实际训练**，不是 `33ce8ea` 当天的 BeforeTune。

## 1. 模型结构

Encoder / Decoder / 生存 head 没有改：

- Gene / Clinic：`TokenSetEncoder`
- WSI：MIL Resampler -> 同一套 `TokenSetEncoder`
- 融合：3 个模态专家 + 固定先验专家的 Generalized PoE
- 重建：每模态独立 decoder + 可学习观测方差
- 散度：默认仍是 Jeffreys（`SurvTriPoEVAE_KL` 只给消融用）
- B/C 下游 head：`z -> fuse_fc -> classifier -> Cox risk`

真正改到主模型结构的只有 PoE alpha：

| 项 | 以前（`33ce8ea` / L0Test） | 现在 |
|---|---|---|
| 默认 alpha | 可学习 logit `a_i`，只在当前可用模态上 softmax | 同左，**默认仍可学习** |
| 新开关 | 无 | `--alphafix` + `--alphapgc pathology,gene,clinic`。打开后不再学 `a_i`，缺失/dropout 后在可用子集上重新归一化 |
| Table1 是否打开 | L0Test 的 B/C 都没写 `ALPHAFIX` | 当前 gen 也没写。defaults 是 `ALPHAFIX=false` |

所以：**Table1 主模型默认结构仍是可学习 alpha。** alphafix 已经进代码和设计说明，但还没进当前 B/C conf。

## 2. 训练流程

B/C 的大框架没变：

- B：stage1 无监督 `L_rec + βJ` -> 加载 val VAE loss 最优 ckpt -> stage2 只训 Cox，不冻 backbone
- C：从头联合训练 `L_rec + βJ + λ L_surv`
- Stage2 选点：仍从 `epoch >= 10` 起按 val C-index 存
- B 的 encoder 学习率并没有做成文档第 8.3 节那种 `0.1 × head lr`，以前就没有，现在也没有

流程上真正会改训练动力学的有三处。

### 2.1 Beta warmup

`--betafix` 不是 `33ce8ea` 当时就有的参数。当时 `_get_poe_beta()` 没有这个开关：只要 `WARMUP_EPOCHS > 1`，Jeffreys 的 β 就会从 0 线性升到 `POE_BETA_TARGET`。`L0Test_BeforeTune`（2026-08-17）的命令行也没有 `--betafix`。

这个开关是 2026-08-22 的 `7155f0b` 才加进来的。之后 2026-08-26 的 `L0Test` 实际训练命令里已经显式带了 `--betafix`，`config.snapshot` / `effective_config.txt` 也写成了 `BETAFIX=true`。所以对 **L0Test 那次实际训练** 来说，β 是固定值，不是 warmup。

| | `33ce8ea` / BeforeTune | L0Test 实际训练（2026-08-26） | 当前 queue conf |
|---|---|---|---|
| 有没有 `BETAFIX` 这个参数 | 代码里还没有 | 有，命令行带了 `--betafix` | 有这个参数，但 conf 没写 |
| B 的 β | warmup 到 `1.0` | 固定 `0.0164` | conf 不写，落到 defaults：warmup 到 `1.0` |
| C 的 β | warmup 到 `1.0` | 固定 `0.268` | 同上，warmup 到 `1.0` |

`--betafix` 只固定 Jeffreys 的 β，不影响 cosine LR warmup。`WARMUP_EPOCHS=3` 在 L0Test 里仍然存在，但当时因为 `--betafix` 已经打开，它只作用在学习率 scheduler 上。

### 2.2 模态 dropout 从“逐 batch”改成“按 case 缓存”

以前：模型 forward 里对每个 batch 独立 Bernoulli dropout。B 只在 stage1 开，C 全程开；B 的 stage2 关。

现在：dropout 挪到 dataloader。`SurvivalDataset._get_case_avail()` 按 `seed + fold + case_id + drop_prob` 生成一次，然后缓存。模型只消费 `avail`。

这会带来两个协议变化：

1. 同一 case 在整个 epoch、甚至整个 split 里 mask 固定，不再每个 batch 重抽。
2. **B 的 stage2 现在也会 dropout。** 因为 train loader 不区分 stage1/stage2，只要 `poe_modality_dropout > 0` 就会给 train split 生成缺失 mask。

Val/test 在 `missing_mode=model_gen` 下仍是全模态，这一点没变。

### 2.3 其他训练入口

这些对 Table1 B/C 默认 conf **不生效**，但代码已经在：

- `MISSING_MODE=model_gen | unified_mask_csv`
- `EVAL_MODALITIES` 子集评测
- Model B 消融：`B_kl` / `B_crossstage1` / `B_nopretrain`

当前 Table1 gen 不生成这些。

## 3. 当前 gen 写出的 conf，和以前最后训练是否相同

当前入口：`configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_ours.sh`
实际生成逻辑：`gen_ours/_common.bash`

我按当前脚本 dry-run 过：每个 study 只出 B、C 两个 conf，字段完全一样。以 BRCA 为例，现在写出的是：

```text
EXP_GROUP=L0_BRCA_poe_model_val
RUN_NAME=tcga_brca__L0__cell_norm__uni_v1__survtri_poe_vae_B
PRESET=survtri_poe_vae_B
STUDY=tcga_brca
CLINIC_EXPERIMENT=L0
GENE_EXPERIMENT=scFoundation_embedding_cell_norm
WSI_EXPERIMENT=uni_v1
BAG_LOSS=cox_surv
BATCH_SIZE=128
BATCH_SIZE_STAGE1=128
MAX_EPOCHS=20
MAX_EPOCHS_STAGE1=10
WARMUP_EPOCHS=3
```

C 除了 `PRESET/RUN_NAME` 以外，字段相同。调参超参都在 `_common.bash` 里被注释掉了，因此会落到 `configs/defaults.conf`。

### 3.1 和 L0Test（2026-08-26 实际训练）比

**不相同。**

Model B

| 字段 | L0Test 实际训练 | 当前 gen + defaults |
|---|---|---|
| `LR` | `6e-4` | `5e-4` |
| `LR_STAGE1` | `8e-5` | `1e-4` |
| `REG` | `7e-5` | `1e-3` |
| `POE_BETA_TARGET` | `0.0164` | `1.0` |
| `BETAFIX` | `true` | `false` |
| `POE_MODALITY_DROPOUT` | `0.25` | `0.2` |
| `POE_MMHID` | `128` | `256` |
| `POE_DECODER_HIDDEN_DIM` | `512` | `512` |
| `BATCH_SIZE` | `128` | `128` |
| `ALPHAFIX` | 未写，等价 false | false |

Model C

| 字段 | L0Test 实际训练 | 当前 gen + defaults |
|---|---|---|
| `BATCH_SIZE` | `16` | `128` |
| `LR` | `1.49e-4` | `5e-4` |
| `REG` | `1.25e-5` | `1e-3` |
| `POE_BETA_TARGET` | `0.268` | `1.0` |
| `BETAFIX` | `true` | `false` |
| `POE_SURV_LAMBDA` | `0.361` | `1.0` |
| `POE_MODALITY_DROPOUT` | `0.2` | `0.2` |
| `POE_MMHID` | `256` | `256` |
| `ALPHAFIX` | 未写，等价 false | false |

conf 文本层面，当前 B/C 已经不是 L0Test 那套。
再加上 2.2 的 dropout 位置变化，即使把上面字段手工填回去，B 的 stage2 也不会和当时完全同协议。

### 3.2 和 `33ce8ea` 当时的 gen / BeforeTune 比

**conf 字段基本相同，训练协议仍不完全相同。**

`33ce8ea` 的 `_common.bash` 对 A/B/C 写同一组：

```text
BATCH_SIZE=128
BATCH_SIZE_STAGE1=128
MAX_EPOCHS=20
MAX_EPOCHS_STAGE1=10
WARMUP_EPOCHS=3
```

这和当前 B/C conf 一致，也和 `L0Test After Fix some err` 的 snapshot 一致。未覆盖项都走 defaults：`lr=5e-4, reg=1e-3, beta=1.0, dropout=0.2, mmhid=256, lambda=1.0`。

仍有这些差异：

| 项 | `33ce8ea` gen | 当前 gen |
|---|---|---|
| 生成模型 | A、B、C | 只生成 B、C，A 被注释 |
| 数据集 | 8 个 study | 只剩 BRCA/COAD/KIRC/KIRP/LIHC |
| 文件名序号 | B=`002`，C=`003` | B=`001`，C=`002` |
| 单模态 B/C 脚本 | 和主 gen 放一起 | 挪到 `gen_ours/gen_BC_single_modal/`，`gen_ours.sh` 不再调用 |
| beta | 无 `BETAFIX`，warmup | defaults 显式 `BETAFIX=false`，行为相同 |
| alpha | 只能学 | 代码支持 alphafix，但 conf 默认关闭 |
| dropout | 模型内逐 batch；B stage2 关闭 | dataloader 按 case 缓存；B stage2 也 dropout |

### 3.3 当前 gen 自己的状态

`_common.bash` 里已经写好了 L0Test 那组固定超参，但赋值和写入都被注释了：

- B：`LR=6e-4, LR_STAGE1=8e-5, REG=7e-5, BETA=0.0164, DROPOUT=0.25, MMHID=128, DECODER=512`
- C：`BATCH=16, LR=1.49e-4, REG=1.25e-5, BETA=0.268, BETAFIX=true, LAMBDA=0.361`
- `ALPHAFIX/ALPHAPGC` 也注释着，默认继续学 alpha

所以当前脚本生成的是 **After Fix** 那套，不是 **L0Test** 那套。

## 4. 差异明细

| ID | 层级 | 是否影响 Table1 B/C 复现 | 差异 |
|---|---|---|---|
| D1 | conf | 是 | 当前 gen 没有写出 L0Test 的 B/C 调参超参，全部退回 defaults |
| D2 | conf | 是 | C 的 batch 从 16 变回 128 |
| D3 | conf | 是 | B/C 都不再 `BETAFIX`，beta 从固定值改成 warmup 到 1.0 |
| D4 | conf | 弱 | 不再生成 Model A；KICH/PRAD/READ 不再进 `gen_ours.sh` |
| D5 | 结构 | 否（默认没开） | PoE 增加 alphafix；Table1 conf 仍是可学习 alpha |
| D6 | 流程 | 是 | dropout 从逐 batch 改为按 case 缓存 |
| D7 | 流程 | 是 | B stage2 以前不 dropout，现在 train split 会 dropout |
| D8 | 流程 | 否 | Stage2 选点仍是 `epoch >= 10` 的 val C-index |
| D9 | 结构 | 否 | Encoder / Decoder / Jeffreys / Cox head 未改 |
| D10 | 文档 | 否 | 设计说明补了 alphafix、betafix、clinic 从 L4 改 L0、Optuna 五数据集扫描；第 8 节 B/C 流程描述仍是旧口径 |

**直接回答当前 B/C conf 和 `33ce8ea` 是否相同：训练协议基本相同，只差 dropout。**
差的就是：按 case 缓存，以及 B stage2 现在也会 dropout。alpha / beta 实际配置相同。和 8 月 26 日 `L0Test` 那次调参训练比，仍然不相同。
