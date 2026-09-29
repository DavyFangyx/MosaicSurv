# Test1-4 各模型超参数：算法视角说明（Cfilm 主模型）

主模型为 `survtri_poe_vae_C_film`（Cfilm）。本文档说明 Test1-4 四个实验中
各模型超参数的使用规则：哪些模型随 Cfilm 公共超参一起变化、哪些保持独立、
哪些绑定关系是代码里已有的。

## 模型分类总览

| 类别 | 模型（preset） | 出现位置 | 超参使用 |
|---|---|---|---|
| **主模型** | `survtri_poe_vae_C_film`（Cfilm） | Test1 / Test2 / Test3 / Test4（消融参照行） | `cfilm_hparams.sh` 全套 9 键 |
| **基线**（Test1） | `abmil_wsi`、`mlp_wsi`、`transmil_wsi`、`mlp/snn_clinic_mean/flatten`、`clinic_cox`、`mlp_gene`、`snn_gene`、`survpc_f`、`porpoise`、`survpath`、`mcat`、`survgc_f`、`survpgc_f` | 仅 Test1 | defaults + 脚本默认，与 Cfilm 无关 |
| **基线**（Test2） | `modality_concat_zero`、`modality_concat_mean`、`mvae_poe`、`mopoe`、`hgcn` | 仅 Test2 | 同上（hgcn 的 batch 由 presets.sh 硬绑 32） |
| **基线**（Test3） | `hgcn` | 仅 Test3 | 同上 |
| **消融**（仅 Table4） | `survtri_poe_vae_C_single`、`C_single_enum`、`C_film_noenum`、`C_multi`、`B_film`、`A_film`、`C_film_kl`、`C_film_beta0`、`C_film_surv0` | 仅 Test4 | `cfilm_hparams.sh` 全套 9 键（消融维度例外见 Test4 节） |

POE 家族其余成员（`survtri_poe_vae_A`、`B`、`C`、`B_single`、`B_multi` 等）
在主表 Test1-3 中全部注释不用，不作为基线出现在任何主表中。

## 0. 参数归属一句话结论

- **Cfilm 公共超参 9 键**：`LR`、`REG`、`POE_SURV_LAMBDA`、`POE_BETA_TARGET`、
  `POE_MODALITY_DROPOUT`、`POE_MMHID`、`BATCH_SIZE`、`ALPHAFIX`、`ALPHAPGC`。
- **与 Cfilm 相同**：Test1/2/3 的主模型 Cfilm 本身；Test4 全部 10 个消融 preset
  （公平消融：除消融维度外超参必须一致）。
- **独立**：Test1-3 的基线模型；9 键之外的公共参数（stage1 三键、`BETAFIX`、
  架构参数、训练计划）。
- POE 家族其余成员（A/B/C 系变体）已全部退出主表，其中消融相关的降级到 Test4，
  共享 9 键。

## 1. Cfilm 算法结构与数学形式

### 1.1 前向链路

```
WSI(4096 patches, 1024-d) ──WSIMILResampler(16 tokens)→ 768×16 tokens ──┐
gene(4 tokens, 768-d) ──────────────────────────────────────────────────┼─ TokenSetEncoder 各模态
clinic(n tokens, 512-d) ────────────────────────────────────────────────┘  → (μ_m, logσ²_m)
        │
        ▼ GeneralizedPoE(α) —— 缺失/被丢模态 α 置 0，其余重新归一化
        │
   q(z|x)=N(μ_joint, σ²_joint) ──reparameterize──→ z
        │
        ├─→ ModalityDecoder×3 → 重建 P/G/C（缺失/被丢模态也重建，见 1.5）
        └─→ 7-pattern 枚举 PoE → FiLM 条件化头 → risk
```

- `TokenSetEncoder`（`models/model_utils.py:255`）：把可学习的 `μ_query`、
  `logvar_query` 两个 query 拼在 token 序列前，过 transformer block（层数 =
  `POE_TRANSFORMER_LAYERS`），取前两个位置读出 μ、logσ²。set 编码使任意
  数量/顺序的 token（patch 数变化、clinic 字段数变化）都能编码成定长潜变量。
- WSI 侧先把 4096 个 1024-d patch 经 resampler（16 个可学习 query，cross-attn）
  压缩为 16 个 768-d token；gene/clinic 直接以 token 形式进入。

### 1.2 GeneralizedPoE 融合（α 的作用点）

`models/model_utils.py:398`，精度加权融合：

```
τ_m = exp(−logσ²_m)                      # 各模态专家精度
α = softmax(logits, mask=avail)          # 可学习：缺失模态 logit→−∞，可用模态重新归一化
   或 α = fixed_alpha（ALPHAFIX=true 时，来自 ALPHAPGC，归一化到和为 1）
τ_joint = 1 + Σ_m α_m·τ_m                # 1 = 先验 N(0,I) 的精度
μ_joint = Σ_m α_m·τ_m·μ_m / τ_joint
logσ²_joint = log(1/τ_joint)，截断 [−4, 2]
```

- 标准 PoE 是 α_m≡1 的等权精度加权；**GeneralizedPoE 让每个模态专家有一个权重**。
  可学习时每个模态一个 logit（初始 0 → 初始等权），由训练数据决定专家重要性；
  `ALPHAFIX=true` 时退化为固定权重（`ALPHAPGC` 归一化）。
- 缺失模态的 μ、σ² 仍在计算，但 α 被 mask 掉、完全不进融合。

### 1.3 联合损失

```
L = L_rec  +  β·J(q(z|x) ‖ N(0,I))  +  λ·L_surv(Cox)
```

三项各自的形式：

**① 重建损失 L_rec（σ² 加权，`models/model_utils.py:456`）**

```
L_rec = Σ_m [ MSE(recon_m, target_m) / (2σ_m²) + 0.5·log σ_m² ]
```

每个模态的 σ_m² 是可学习参数（logvar 截断 [−4,4]），等价于自动按各模态重建
难度加权（难重建的模态 σ 学大、权重自动变小），是常见的 uncertainty weighting。

**② Jeffreys 散度（`models/model_utils.py:494`）**

```
J = 0.5·(σ² + σ⁻² − 2 + μ²(1+σ⁻²))，对批次求均值
```

Jeffreys 是 KL(q‖p)+KL(p‖q) 的对称化：正向 KL 防后验塌缩、反向 KL 促模态对齐，
比单向 KL 更温和。`c_film_kl` 消融验证这个选择（换成标准 KL，
`models/model_SurvTriPoEVAE.py:446`）。

**③ 生存损失 L_surv（Cox + 7-pattern 枚举，`models/model_utils.py:620`）**

7 个可用性 pattern（P/C/G/PC/PG/CG/PCG）。每个 pattern s：

```
取 avail ≥ pattern 的样本子集 → 在该子集上用其 mask 重算 PoE → risk_s = head(μ_s, s)
L_surv = (1/7)·Σ_s CoxLoss(risk_s, t_s, c_s)        # 子集样本数 < 8 的 pattern 跳过
```

即每次前向把 batch 按 7 种缺失形态各"模拟"一次，让生存头在任何可用模态子集上
都被训练；推理期完整三模态的风险用 `pattern_id=7`（PCG）计算。

### 1.4 FiLM 条件化生存头

`models/model_utils.py:599`：

```
e = Emb[pattern_id] ∈ R³²          # 8 个入口（0~7），pattern 嵌入
(γ, β) = Linear(e) 对半切开
z = (1+γ) ⊙ μ_joint + β            # FiLM 调制
risk = classifier( fuse_fc(z) )    # 共享 MLP（隐层宽 = POE_MMHID）
```

初始化 γ=0、β=0 → 起步等价于无调制。与 `C_multi`（每个 pattern 一个独立 MLP，
`model_utils.py:578`）相比，FiLM 用一个共享 MLP + 32 维 pattern 条件，参数少、
pattern 间共享表示；`C_single_enum`（共享单头 + 枚举，`pattern_id` 被忽略）则
完全不区分 pattern。三个是 A 组消融的对比轴。

### 1.5 模态 dropout 与"缺失仍重建"

- 训练期（`model_gen` 模式）每个 case 每个模态独立以 `POE_MODALITY_DROPOUT`
  概率从 `available_mask` 中丢弃：`rng(random)` 按 (missing_seed, fold, case,
  drop_prob) 播种，**重跑结果确定**；全丢的行强制保留一个模态
  （`utils/missing_mask_protocol.py:360`，入口 `datasets/dataset_survival.py:982`）。
- 被丢模态不参与 PoE 融合（α 置 0），但**解码器仍重建它**——因为重建只依赖
  z_joint。这一设计的含义：z_joint 必须是对完整三模态信息的一个"压缩视图"，
  输入缺什么、输出都要补出来，这是对缺失模态鲁棒性的正则，也是与
  modality_concat 类插补基线在机制上的根本区别。
- 推理期不丢；Test3 的 `unified_mask_csv` 模式由共享 mask csv 决定缺失，
  随机 drop 被强制置 0（`utils/missing_mask_protocol.py:338`）。

### 1.6 β warmup 调度

`utils/core_utils.py:1732`（`_get_poe_beta`），stage1/stage2 训练循环都用：

```
BETAFIX=true   →  β ≡ POE_BETA_TARGET
BETAFIX=false  →  β = POE_BETA_TARGET · min(1, epoch/(WARMUP_EPOCHS−1))
```

`WARMUP_EPOCHS=3`、`POE_BETA_TARGET=0.1` 时：epoch 0 → 0，epoch 1 → 0.05，
epoch ≥ 2 → 0.1。**先纯学重建、再逐步压先验对齐**，避免训练初期 Jeffreys 项
把潜空间过早压向先验导致重建塌缩。

### 1.7 训练范式 A/B/C（Test4 B 组消融的对比轴）

| 范式 | stage1 | stage2 | 损失 |
|---|---|---|---|
| A 线性探针 | VAE 预训练（`L_rec+βJ`，`LR_STAGE1`/`MAX_EPOCHS_STAGE1`/`BATCH_SIZE_STAGE1`） | 冻结主干，只训 linear probe + FiLM 头 | 只有 L_surv |
| B 两阶段 | 同上 | 冻结主干只训头（B_single/multi/film）；plain B 是分组 lr 微调（编码器 = `LR`×`POE_ENCODER_LR_RATIO`） | 只有 L_surv |
| C 联合（Cfilm） | 无 | 单阶段，全部参数一个 lr = `LR` | `L_rec+βJ+λL_surv` |

C 系 stage2 直接进入联合训练（`models/model_SurvTriPoEVAE.py:65` 把 C 的
`training_stage` 初始为 stage2）；A/B 系先走 stage1（`utils/core_utils.py:2738`）。

## 2. Cfilm 公共超参 9 键：逐项算法含义

| # | 键 | 值 | 算法作用 | 变化方向的影响（由数学形式推出） |
|---|---|---|---|---|
| 1 | `LR` | 0.001 | C 系单参数组学习率：编码器/PoE/解码器/生存头同一 lr | —（通用优化器参数） |
| 2 | `REG` | 1e-05 | L2 weight decay | —（通用正则） |
| 3 | `POE_SURV_LAMBDA` | 2.0 | λ：L_surv 在联合损失中的权重 | λ↑ → 表示学习偏向判别性（Cox）；λ↓ → 偏向生成性（VAE 重建）。A/B 范式 stage2 无 VAE 项，此键不生效 |
| 4 | `POE_BETA_TARGET` | 0.1 | β：Jeffreys 项权重（warmup 目标值，见 1.6） | β↑ → 潜空间更贴近先验（更规则、模态间更对齐），重建压力相对下降 |
| 5 | `POE_MODALITY_DROPOUT` | 0.35 | 训练期每模态独立丢弃概率（见 1.5） | ↑ → 缺失鲁棒性强、但每模态可见信息变少；0 = 关闭随机缺失 |
| 6 | `POE_MMHID` | 128 | 生存读出 MLP（fuse_fc / FiLMHead / MultiPatternHead）隐层宽。注意潜空间 `latent_dim=128` 是独立架构参数，不在此键 | ↑ → 读出层容量增加 |
| 7 | `BATCH_SIZE` | 16 | stage2 训练 batch（val/test 恒为 1） | ↑ → Cox 损失在更大 risk 集合上排序，但 pattern 枚举的子集拆分更细 |
| 8 | `ALPHAFIX` | false | PoE 专家权重 α 可学习（softmax logits，初始等权）；true = 用 `ALPHAPGC` 固定值 | false → 数据决定专家权重；true → 人为设定（可作消融对照） |
| 9 | `ALPHAPGC` | 空 | 固定 α=(α_P,α_G,α_C)，归一化到和为 1，仅 `ALPHAFIX=true` 时生效；空串 = 不传该 flag | 只在 ALPHAFIX=true 时有意义 |

各键在代码中的位置：λ、β 组合见 `models/model_SurvTriPoEVAE.py:287-290`；
α 见 `model_utils.py:398`；dropout 见 `missing_mask_protocol.py:360`；
mmhid 见 `model_utils.py:578,599`。

## 3. 独立超参（不随 Cfilm 变化的键）

测
| `OPT` / `LR_SCHEDULER` / `REG_TYPE` | radam / cosine / L2 | `defaults.conf` | 优化器与调度器 |
| `BATCH_SIZE_STAGE1` | 128 | Table 脚本 | stage1 VAE 预训练 batch（只 A_film/B_film 用）；保持历史配置，不随 stage2 `BATCH_SIZE=16` 联动 |
| `LR_STAGE1` | 1e-4 | `defaults.conf` | stage1 AdamW 学习率（只 A_film/B_film 用） |
| `MAX_EPOCHS_STAGE1` | 10 | Table 脚本 | stage1 预训练 epoch |
| `MAX_EPOCHS` / `WARMUP_EPOCHS` | 20 / 3 | Table 脚本 | 总 epoch / β warmup 长度（β 具体值由 9 键决定，长度独立） |
| `POE_DECODER_HIDDEN_DIM` | 512 | `defaults.conf` | 三模态解码器隐层宽（架构参数，不属于调参集） |
| `POE_TRANSFORMER_LAYERS` | 1 | `defaults.conf` | 各模态编码器 transformer 层数（架构参数） |
| `latent_dim` | 128 | 代码默认 | 联合潜空间维度（架构参数，无 .conf 键） |
| `POE_ENCODER_LR_RATIO` | 0.1 | `defaults.conf` | plain B 专属：stage2 编码器 lr = 头 lr × 0.1。plain B 已退出全部实验，**当前无生效路径** |
| hgcn 的 `BATCH_SIZE` | 32 | `presets.sh:582` | HGCN 历史固定 batch，硬绑覆盖 conf |

## 4. 各实验的参数归属

| 实验 | 与 Cfilm 相同的键 | 例外（算法原因） | 独立（不随 Cfilm 变化） |
|---|---|---|---|
| Test1（Table1 主表） | 主模型 Cfilm：全套 9 键 | 无 | 基线（14 个，与 POE 家族无关）用 defaults + 脚本默认，batch=1 |
| Test2（Table2 缺失模态） | Cfilm：全套 9 键 | 无 | 5 个基线（modality_concat×2、mvae_poe、mopoe、hgcn）；hgcn batch=32 硬绑 |
| Test3（Table3 缺失配比） | Cfilm：8 键 | `POE_MODALITY_DROPOUT=0`：unified_mask_csv 下缺失由共享 mask csv 决定，随机 drop 被强制归零（`missing_mask_protocol.py:338`），写 0 是显式化 | hgcn（batch=32 硬绑） |
| Test4（Table4 消融） | 全部 10 个 preset：全套 9 键 | `C_film_beta0` 的 `POE_BETA_TARGET=0`——改超参即消融本身（去掉 J 项） | stage1 三键（`LR_STAGE1`/`MAX_EPOCHS_STAGE1`/`BATCH_SIZE_STAGE1`）只作用于 A_film/B_film；其余独立键同 §3 |

（各实验启用哪些模型、哪些注释，见执行版文档的模型分类总览。）

## 5. Test4 消融：差异在计算图，不在超参

超参全部一致（§4），10 个 preset 与主模型的差异只在算法维度：

**A 组 — 生存读出层**（都走 C 联合训练）：

| preset | 读出层 | 与 Cfilm 的算法差异 |
|---|---|---|
| `C_single` | 单头 | 共享 fuse_fc，按当前可用 mask 一次前向，无 pattern 枚举 |
| `C_single_enum` | 单头 + 枚举 | 共享单头但 7-pattern 枚举训练（pattern_id 被忽略） |
| `C_film_noenum` | FiLM 头 | 有 FiLM 条件化，但去掉 7-pattern 枚举（只用当前 pattern 一次） |
| `C_film`（主模型） | FiLM 头 + 枚举 | 参照 |
| `C_multi` | 7 专属头 | FiLM 调制换成每个 pattern 一个独立 MLP（`MultiPatternHead`） |

**B 组 — 训练范式**（读出层都是 FiLM 头）：

| preset | 范式差异（见 1.7） |
|---|---|
| `A_film` | 线性探针：冻结主干，只训 FiLM 头 + linear probe |
| `B_film` | 两阶段：stage1 预训练，stage2 冻结主干只训 FiLM 头（`core_utils.py:2814`） |
| `C_film` | 联合训练（对照） |

**C 组 — 损失项**（架构不变，只动损失）：

| preset | 损失差异 |
|---|---|
| `C_film_kl` | Jeffreys（对称 KL）→ 标准 KL（`SurvTriPoEVAE_KL`，`model_SurvTriPoEVAE.py:446`） |
| `C_film_beta0` | β≡0 去掉 J 项：L = L_rec + λ·L_surv |
| `C_film_surv0` | L_surv 不回传主干：编码器/PoE α 只收 L_rec+βJ 梯度，FiLM 头仍收 λ·L_surv 梯度（`surv_detach_poe`：PoE 计算包在 `torch.no_grad()` 里） |

## 6. 超参 → 代码位置速查

| 机制 | 位置 |
|---|---|
| 联合损失组合（λ·L_surv + β·J + L_rec） | `models/model_SurvTriPoEVAE.py:287-290` |
| β warmup 调度（BETAFIX 开关） | `utils/core_utils.py:1732`（`_get_poe_beta`）；stage2 训练循环使用处 `:1768` 起 |
| GeneralizedPoE（α 可学习 / 固定） | `models/model_utils.py:398` |
| Jeffreys / 标准 KL | `models/model_utils.py:494` / `:502` |
| σ² 加权重建损失 | `models/model_utils.py:456` |
| 模态 dropout（按 case 播种、缺失仍重建） | `utils/missing_mask_protocol.py:360`；入口 `datasets/dataset_survival.py:982` |
| unified_mask_csv 强制 dropout=0 | `utils/missing_mask_protocol.py:338` |
| 7-pattern 枚举与 surv0 detach | `models/model_utils.py:620` |
| FiLMHead / MultiPatternHead（mmhid 作用点） | `models/model_utils.py:599` / `:578` |
| TokenSetEncoder（query + transformer 读出） | `models/model_utils.py:255` |
| A/B/C 范式分派（stage1 门槛） | `utils/core_utils.py:2738` |
| plain B 分组学习率（当前无实验使用） | `utils/core_utils.py:761` |
| B_single/multi/film 冻结主干只训头 | `utils/core_utils.py:2814` |
| A 冻结主干线性探针 | `models/model_SurvTriPoEVAE.py:212` |
| hgcn batch=32 硬绑 | `configs/presets.sh:582` |
| ALPHAFIX/ALPHAPGC/BETAFIX 传参 | `run.sh:382-386` |

## 附：数据设置（非超参，跨实验不同，备查）

| 实验 | GENE_EXPERIMENT | CLINIC | WSI |
|---|---|---|---|
| Test1 / Test4 | `scFoundation_embedding_cell_norm` | L0 | uni_v1 |
| Test2 / Test3 | `scFoundation_embedding_gene_raw` | L0 | uni_v1 |
