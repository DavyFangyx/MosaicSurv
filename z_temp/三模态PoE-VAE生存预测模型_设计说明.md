# 三模态 PoE-VAE 生存预测模型 — 设计说明

本文档整合此前讨论中确定的所有设计决策，作为后续撰写 Codex 执行 prompt 的直接依据。凡本文档未覆盖的实现细节，均属于遗留待定项，需在写执行 prompt 前单独确认。

---

## 0. 总体结构

三种模态（WSI / Clinic / Gene）各自通过一个 Encoder 输出对角高斯 (μᵢ, logσ²ᵢ)，经 Generalized PoE 融合为单一联合高斯 (μ_joint, σ_joint)，重参数化采样得到共享隐变量 z（训练时采样，推理时取 μ_joint）。三个 Decoder 从 z 分别重建各模态对应的定长目标向量。下游生存预测 head 复用已有工作实现，不在本文档中设计。

代码组织：共享组件全部放在 `model_utils.py`（Encoder、PoE、重参数化、Decoder、Loss 等基础模块），Model_A / Model_B / Model_C 三个训练模式各自在独立文件中，通过拼装 `model_utils.py` 里的组件构建，不重复实现底层逻辑。

---

c
| 模态 | 特征提取器 | 输入维度 | 备注 |
|---|---|---|---|
| WSI | UNI 1 | `(n_patch, 1024)` | `n_patch` 逐病人变化 |
| Clinic | COACH | `(n_c, 512)` | `n_c` 同一实验组内固定，初始化时按 `.pt` 文件检测 |
| Gene | scFoundation | `(4, 768)` | 固定 |

**隐变量维度 d_z = 128，硬编码，不做成超参。**

---

## 2. Encoder 设计

### 2.1 Gene / Clinic — 共享结构定义的 TokenSetEncoder

Gene 与 Clinic 走同一种 `TokenSetEncoder` 结构定义，但分别实例化、分别训练，不共享权重。当前实现里：

- Gene 输入 token dim = `768`
- Clinic 输入 token dim = `512`

#### Gene Encoder

```
输入: gene_tokens (B, 4, 768)

1. 直接送入 TokenSetEncoder(input_dim=768)
2. 内部通过 query-style 汇聚得到一组全局表示
3. 输出 mu_gene, logvar_gene
4. logvar_gene.clamp(min=-4, max=2)

输出: mu_gene, logvar_gene (B, 128)
```

#### Clinic Encoder

```
输入: clinic_tokens (B, n_c, 512)

1. 直接送入 TokenSetEncoder(input_dim=512)
2. 不做额外 512 -> 768 投影
3. 输出 mu_clinic, logvar_clinic
4. logvar_clinic.clamp(min=-4, max=2)

输出: mu_clinic, logvar_clinic (B, 128)
```

说明：

- 当前实现只会动态检测 `n_c`
- 当前实现不会自动把 Clinic 的最后一维从别的维度投影到 512 或 768
- 因此 Clinic 输入默认假设为 `(n_c, 512)`

### 2.2 WSI — MIL Resampler + 复用 Encoder_TRANSFORMER

WSI 编码器改为两段级联结构。第一段使用 WSI 专属的 MIL Resampler 将变长 patch 序列压缩为定长 token set；第二段直接复用 2.1 节 Gene / Clinic 的 Encoder_TRANSFORMER 定义，对该定长 token set 输出 `(mu_wsi, logvar_wsi)`。

#### 第一段：MIL Resampler（WSI 专属，不与 Gene / Clinic 共享权重）

```
输入: patches (B, n_patch, 1024), padding_mask

0. 显式输入投影:
   Linear(1024, 768)
   - 将原始 UNI patch 特征从 1024 维投到 768 维

1. K 个可学习 query token: (K, 768)
   - K 硬编码为超参，建议默认 K = 16

2. query 对投影后的 patch tokens 做 cross-attention（Perceiver-resampler 式）
   - attention 计算时必须传入 padding_mask，屏蔽 padding patch
   - 可堆叠 1~2 层 cross-v block

3. （当前默认已启用）K 个输出 token 之间再加 1 层轻量 self-attention
   - 因为 K 远小于 n_patch，这层计算量可忽略
   - 用于补回 resampler 阶段丢掉的 token-token 交互

输出: wsi_tokens (B, K, 768)   # 定长，不再需要 padding_mask
```

#### 第二段：直接复用 2.1 节 Gene / Clinic Encoder_TRANSFORMER，nfeats = K

```
输入: wsi_tokens (B, K, 768)

1. muQuery, sigmaQuery 拼接 -> (K+2, B, 768)
2. TransformerEncoderLayer（与 Gene/Clinic 同一套结构定义，权重不共享，单独实例化）
3. 取 mu_token, logvar_token -> Linear(768, 128) -> mu_wsi, logvar_wsi
4. logvar_wsi.clamp(min=-4, max=2)

输出: mu_wsi, logvar_wsi (B, 128)
```

### 2.3 WSI 重建目标（防止表征塌缩，独立池化头）

**不使用 mu_wsi 作为重建目标。** 单独设计一个与 mu 分支不共享参数的池化头：

```
输入: wsi_tokens (B, K, 768)   # 来自 2.2 节第一段 MIL Resampler 的输出

1. 均值池化（K 个 token 已定长且无 padding，直接 mean，不需要 mask）:
   pooled = mean(wsi_tokens, dim=1)   # (B, 768)

2. 独立线性层（参数与 mu/logvar 分支完全分离）:
   wsi_target = Linear(768, 768)(pooled)

输出: wsi_target (B, 768)   # 作为 Decoder 重建的 ground truth
```

该目标向量与 `mu_wsi / logvar_wsi` 来自同一条 WSI backbone，但走独立投影头，参数不共享，避免 decoder 直接照抄 posterior 分支导致平凡解。

---

## 3. Generalized PoE 融合

三个模态专家 + 1 个固定先验专家：

```
先验专家: μ0 = 0, τ0 = 1（固定，不参与 softmax，始终存在）

模态专家权重: 默认每个模态 i 有一个可学习 logit a_i
α_i = softmax(a_1, a_2, a_3)_i   # 仅在该模态可用时参与 softmax

`--alphafix` 打开后不再学习 a_i，改为使用 `--alphapgc pathology,gene,clinic` 给出的固定权重。
该三元组会先归一化到和为 1；缺失 / dropout 掉的模态权重置 0 后，再在当前可用子集上重新归一化。

τ_i = 1 / exp(logσ²_i)  (即 1/σ_i²)

τ_joint  = τ0 + Σ_{i ∈ available} α_i * τ_i
μ_joint  = (τ0*μ0 + Σ_{i ∈ available} α_i * τ_i * μ_i) / τ_joint
σ_joint² = 1 / τ_joint
```

**缺失模态处理（训练时的模态 dropout 和推理/验证时的真实缺失，逻辑一致）：**
softmax 只在当前可用（未被 dropout / 未缺失）的模态子集上计算，即分母重新归一化，而不是对缺失模态的 α 置零后不归一。

---

## 4. 重参数化

```
训练: z = μ_joint + σ_joint ⊙ ε,  ε ~ N(0, I)
推理: z = μ_joint   （不采样，避免下游 c-index 抖动）
```

---

## 5. Decoder 设计

Decoder 输入仅为 z，**不拼接任何 condition**：

```
输入: z (B, 128)

每个模态一个独立 decoder_i:
  Linear(128, hidden) -> ELU -> Linear(hidden, output_dim_i)

- decoder_wsi:    output_dim = 768                → 对应 wsi_target (2.3节)
- decoder_gene:   output_dim = 4 * 768，reshape 为 (4, 768)   → 对应原始 Gene token 特征
- decoder_clinic: output_dim = n_c * 512，reshape 为 (n_c, 512) → 对应原始 Clinic token 特征
                  （n_c 在模型初始化时按当前实验组检测值写死到 decoder 里）
```

---

## 6. 损失函数

### 6.1 重建损失（每模态独立学习观测方差）

对每个模态 m 学一个标量可学习参数 logσ²_m：

```
L_rec_m = mean_elementwise_MSE_m / (2 * σ²_m) + (1 / 2) * log(σ²_m)
L_rec   = Σ_m L_rec_m       # m ∈ {wsi, gene, clinic}
```

其中 `mean_elementwise_MSE_m` 表示该模态重建目标展平后，对所有元素先求逐元素平方误差，再按元素取平均，同时对 batch 取 mean。

dim_m 取该模态重建目标的展平总维度（wsi=768，gene=4*768，clinic=n_c*768），当前实现按元素平均，因此不再显式乘 `dim_m`。

### 6.2 KL / Jeffreys 散度

第一阶段训练全程使用 Jeffreys 散度（对称化 KL），作用于 joint 后验 q(z|m1,m2,m3) 与先验 N(0,I) 之间：

```
J = (1/2) * mean_j ( σ_j² + 1/σ_j² - 2 + μ_j² * (1 + 1/σ_j²) )
```

logσ²（各专家及 joint）均需 clamp 到 [-4, 2]。

### 6.3 总损失与 warm-up

```
L = L_rec + β(t) * J

默认:
β(t): 前 N_warmup 个 epoch 从 0 线性升到目标值 β_target，之后保持不变

--betafix 打开后:
β(t) = β_target，全程固定，不再做 warm-up
```

（β_target 和 N_warmup 作为超参，具体数值留待调参，不在本设计文档中固定。）
需要固定参数 β 时，打开参数开关 `--betafix` 只固定 Jeffreys 的 β，不影响学习率 scheduler 的 `--warmup_epochs`。
需要固定 PoE 模态权重 α 时，打开 `--alphafix` 并同时给出 `--alphapgc pathology,gene,clinic`，例如 `--alphapgc 0.5,0.3,0.2`。`--alphafix` 只固定 α，不影响 β warmup，也不影响学习率 scheduler。

当前这套重建损失与 Jeffreys 缩放是共享实现，因此会自动作用于所有会计算 VAE 损失的训练阶段：Model_A 的 stage1、Model_B 的 stage1，以及 Model_C 的联合训练中的 VAE 部分。

---

## 7. 模态 dropout

- 概率 p = 0.2，每个模态独立 Bernoulli 采样是否丢弃（每个 batch 内每个样本独立）
- 约束：任意样本不能三个模态同时被丢弃，至少保留一个
- 实现：在 dataloader 中，avail 先从 dataloader 进入，再按原来的 selected_modalities 规则做同一个 modality_dropout()
---

## 8. 训练流程与代码组织

### 8.1 共享组件（model_utils.py）
- `TokenSetEncoder`（原 GeneClinicEncoder 改名；Gene / Clinic / WSI 共用同一类定义，各自单独实例化权重，2.1 / 2.2）
- `WSIMILResampler`（2.2 第一段）
- `WSITargetPoolingHead`（2.3，内部实现为直接 mean，不再使用 mask）
- `GeneralizedPoE`（3）
- `reparameterize`（4）
- `Decoder_Share`（5，三个模态各一份权重）
- `ReconstructionLoss`（6.1，含每模态可学习 logσ²_m）
- `JeffreysDivergence`（6.2）
- `modality_dropout`（7）

### 8.2 Model_A（当前优先实现）
1. 用上述组件构建完整 VAE（Enc + PoE + Dec），无监督训练：`L = L_rec + β(t)*J`，训练期启用模态 dropout。
2. 训练完成后，**冻结 Encoder 与 PoE 的全部参数**。
3. 对每个病人（含验证集中模态缺失的样本），用其可用模态过一遍冻结的 Enc + PoE，取 `z = μ_joint`（不采样）。
4. 在冻结的 z 上训练一个线性探针 head（线性 Cox 或线性 bin-hazard，用于诊断）。
5. 报告 c-index：若 ≈ 0.5，说明重建目标学到的表示对预后无信息，需直接转 Model_C。

### 8.3 Model_B（后续）
- 复用 Model_A 训练好的 Enc/PoE 权重作为初始化，不冻结
- Encoder 学习率 = head 学习率 × `poe_encoder_lr_ratio`（默认 0.1）
  - `--lr` 为 survival head（`fuse_fc + classifier`）学习率
  - encoder / PoE / decoder 使用 `--lr * poe_encoder_lr_ratio`
- 下游 head 换成已有工作的生存预测 head（复用实现，接口见第 9 节）

### 8.4 Model_C（后续）
- Enc + Head 从头联合训练：`L = L_rec + β*J + λ*L_surv`
- λ 需扫 {0.1, 1, 10}

---

## 9. 生存预测 head（不在本文档中设计）

Model_B / Model_C 使用的生存预测 head 直接复用已有工作的实现，届时以"参考 xxxx.py"的形式在 Codex 执行 prompt 中给出源码引用，本设计文档只约定接口：

```
输入: z  (B, 128)
输出: 与参考实现一致（如离散时间 bin 的 hazard logits）
```

Model_A 的线性探针 head 除外——那是一个单独的、极简的线性层。

---

## 10. 遗留待确认项

- β_target、N_warmup、各 decoder hidden 维度、TransformerEncoderLayer 的 nhead/层数、MIL Resampler 的 K 值与层数等具体超参数值，尚未固定，写执行 prompt 前需给出默认值。
- Model_A 线性探针使用用线性 Cox 。
- Model_B / Model_C 当前直接使用本项目中已实现的生存 head：`z -> fuse_fc -> classifier -> Cox risk score`，不再额外等待外部参考实现文件。

---

## 11. 各组的训练监控体系

当前 `survtri_poe_vae` 已接入 WandB 监控，覆盖 Model_A / Model_B / Model_C。

- Model_A：
  stage1 为 VAE 预训练；
  stage2 为冻结 backbone 后的线性 Cox probe。
- Model_B：
  stage1 为 VAE 预训练；
  stage2 为 `fuse_fc + classifier + Cox` 微调。
- Model_C：
  直接进行联合训练，优化 `VAE + survival`。

### 11.1 Step 级监控

在 TriPoEVAE 的训练 step 中记录：

- `loss/total`
- `loss/rec_wsi`
- `loss/rec_gene`
- `loss/rec_clinic`
- `loss/kl_or_jeffreys`
- `logvar_obs_wsi`
- `logvar_obs_gene`
- `logvar_obs_clinic`

其中：

- `loss/rec_*` 为三路重建损失的加权值
- `loss/kl_or_jeffreys` 当前对应 Jeffreys 散度
- `logvar_obs_*` 为三路可学习观测方差

### 11.2 Epoch 级监控

每个 epoch 记录潜空间与 PoE 健康度：

- `z/mean_norm`
- `z/mean_std`
- `poe/alpha_wsi`
- `poe/alpha_gene`
- `poe/alpha_clinic`

其中：

- `z/mean_norm` 为 batch 内 `mu_joint` 的平均 L2 norm
- `z/mean_std` 为 batch 内 `z_joint` 各维度方差的均值
- `poe/alpha_*` 为 PoE 权重 softmax 后的 batch 平均值；`--alphafix` 打开后则为 `--alphapgc` 在当前可用模态上重新归一化后的值

### 11.3 生存监控

在包含 Cox 优化的训练阶段按 epoch 记录：

- `loss/surv`

说明：

- Model_A / Model_B 的 stage1 纯 VAE 训练阶段不包含 survival loss
- Model_A / Model_B 的 stage2，以及 Model_C 的联合训练阶段会记录 `loss/surv`

### 11.4 WandB 开启方式

当前 WandB 由命令行参数控制：

- `--wandb_mode disabled`
- `--wandb_mode offline`
- `--wandb_mode online`
- `--wandb_project <project_name>`
- `--wandb_entity <entity_name>`

示例：

```bash
CUDA_VISIBLE_DEVICES=0 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_lihc \
  --modality survtri_poe_vae \
  --poe_variant C \
  --bag_loss cox_surv \
  --wandb_mode online \
  --wandb_project SurvPGC
```

---

## 12. 当前可直接运行的测试命令（集成了wandb后的）

以下命令针对当前仓库的真实目录结构编写，默认：

- 项目根目录：
  `/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init`
- Python 环境：
  `/data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python`
- Workspace 结构：
  `SurvPGC_Workspace/<study>/P/...`
  `SurvPGC_Workspace/<study>/C/...`
  `SurvPGC_Workspace/<study>/G/...`
- split 结构：
  `splits/5foldcv/<study>/splits_*.csv`


下面示例使用 `tcga_lihc`、`P/uni_v1`、`C/L0`、`G/scFoundation_embedding_cell_norm`，并只跑 `fold 0` 进行单折测试。

### 12.1 Model_A：VAE 预训练 + 冻结 backbone + 线性 Cox probe

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init
conda activate SurvPGC

CUDA_VISIBLE_DEVICES=3 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_lihc \
  --modality survtri_poe_vae \
  --poe_variant A \
  --selected_modalities clinic \
  --bag_loss cox_surv \
  --label_dim 1 \
  --encoding_dim 1024 \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/G/scFoundation_embedding_cell_norm \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/C/L0 \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_lihc \
  --k 5 --k_start 0 --k_end 1 \
  --batch_size 128 \
  --batch_size_stage1 128 \
  --max_epochs_stage1 20 \
  --max_epochs 20 \
  --warmup_epochs 3 \
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
  --exp_group poe_vae_test \
  --run_name model_A_DropTest
```

### 12.2 Model_B：VAE 预训练 + `fuse_fc + classifier + Cox`

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init

CUDA_VISIBLE_DEVICES=2 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_lihc \
  --modality survtri_poe_vae \
  --poe_variant B \
  --selected_modalities wsi,gene,clinic \
  --bag_loss cox_surv \
  --label_dim 1 \
  --encoding_dim 1024 \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/G/scFoundation_embedding_cell_norm \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/C/L0 \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_lihc \
  --k 5 \
  --batch_size 128 \
  --batch_size_stage1 128 \
  --max_epochs_stage1 20 \
  --max_epochs 25 \
  --warmup_epochs 3 \
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
  --exp_group poe_vae_test \
  --alphafix \
  --alphapgc 1,1,1 \
  --run_name model_B_alphafixed
```

### 12.3 Model_C：联合训练 `L_rec + βJ + λL_surv`

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init

CUDA_VISIBLE_DEVICES=5 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_lihc \
  --modality survtri_poe_vae \
  --poe_variant C \
  --selected_modalities wsi,gene,clinic \
  --bag_loss cox_surv \
  --label_dim 1 \
  --encoding_dim 1024 \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/G/scFoundation_embedding_cell_norm \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/C/L0 \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_lihc \
  --k 5 --k_start 0 --k_end 1 \
  --batch_size 128 \
  --batch_size_stage1 128 \
  --max_epochs 25 \
  --warmup_epochs 3 \
  --poe_surv_lambda 1.0 \
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
  --exp_group poe_vae_test \
  --run_name model_C_DropTest
```

### 12.4 说明

- 若已 `conda activate SurvPGC`，也可以将上面命令中的 Python 路径替换为：
  `python`
- 当前代码里 best checkpoint 从 `epoch >= 10` 才开始保存，因此 `--max_epochs` 不应低于 `11`。
- batch 参数：
  `--batch_size` 控制主训练阶段
  `--batch_size_stage1` 控制 Model_A / Model_B 的 stage1 VAE 预训练
  `val / test` 固定为 `batch_size = 1`
- 若要切换数据集，只需同步替换：
  `--study`
  `--data_root_dir`
  `--gene_dir`
  `--clinic_dir`
  `--split_dir`
- 训练折数：
  --k 5 \
  --k 5 --k_start 0 --k_end 1 \
- 模态选择：
  `--selected_modalities`
  wsi
  gene
  clinic
  wsi,gene
  wsi,clinic
  gene,clinic
  wsi,gene,clinic
- ModelB stage1 扫描：
--poe_scan_stage1_ckpts
--poe_stage1_ckpt_interval 5
- wandb模式：
离线模式
  --wandb_mode offline \
在线模式
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
第一次用这个环境时执行：
/data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python -m wandb login
- 固定参数 beta ：
--betafix \
--poe_beta_target 0.25 \
- 固定 PoE alpha ：
--alphafix \
--alphapgc 0.5,0.3,0.2 \

## 13. Optuna 调参体系

当前 `survtri_poe_vae` 已接入 Optuna 调参入口 `main_tune_optuna.py`，用于负责：

- 超参数采样
- pruning
- 记录 trial 级最优验证指标

WandB 只负责过程记录与可视化。

当前实现范围：

- 支持 `Model_A / Model_B / Model_C`
- 当前已支持：
  `single`
  `5-fold mean val c-index`
- 当前推荐采用 **5-fold mean val c-index 调参**
- 目标值统一为：
  `single` 模式：单折 `best val c-index`
  `mean_cv` 模式：多折 `mean(best val c-index)`
- Stage2 选点与主训练相同：`survtri_poe_vae` 默认从 `epoch >= 10` 起按验证 C-index 保存 checkpoint，Optuna 不再把 `save_best_from_epoch` 强制设为 `0`
- 每个 epoch 结束后执行：
  `wandb.log(...) -> trial.report(val_cindex, step=global_epoch_step) -> pruning 判断`

### 13.1 各组调参空间

- Model_A：
  `lr`、`lr_stage1`、`reg`、`poe_beta_target`、`poe_modality_dropout`、`alphapgc`
- Model_B：
  `lr ∈ [1e-5, 1e-3]`、`lr_stage1 ∈ [5e-5, 5e-4]`、`reg ∈ [1e-5, 1e-3]`、`poe_beta_target ∈ [0.01, 0.5]`、`poe_modality_dropout ∈ [0.0, 0.4]`、`poe_mmhid ∈ {128, 256}`、`poe_decoder_hidden_dim ∈ {256, 512}`、`poe_encoder_lr_ratio ∈ {0.01, 0.1, 1, 10}`、`alphapgc`
- Model_C：
  `lr ∈ [5e-5, 7e-4]`、`reg ∈ [1e-5, 1e-3]`、`poe_surv_lambda ∈ [0.05, 2.0]`、`poe_beta_target ∈ [0.02, 1.0]`、`poe_modality_dropout ∈ [0.0, 0.4]`、`alphapgc`
  `batch_size` 不在搜索空间内，继承命令行 `--batch_size`（Table1 Model C 固定为 16）

`--betafix` 不是搜索变量，而是从命令行继承到每个 trial。打开后，Optuna 采样到的 `poe_beta_target` 会作为固定 β 使用，不再走 warm-up。默认 study 名会带 `_betafix` 后缀，避免和旧的 warm-up study 共用同一个 sqlite 库。
A/B/C 都会搜索固定 PoE alpha：每个 trial 采样 `alphapgc_pathology ∈ [0.45, 0.75]`、`alphapgc_gene ∈ [0.05, 0.20]`、`alphapgc_clinic ∈ [0.20, 0.40]`，L1 归一化后写成 `--alphapgc p,g,c`，并强制 `--alphafix`。CLI 仍接受原始权重（如 `1,1,1`），不改成必须百分比。默认 study 名带 `_alphafix_search` 后缀，避免和学习 α、或 CLI 固定某一组 α 的旧 study 共用同一个 sqlite 库。

### 13.2 采样与剪枝策略

- sampler：
  当前支持 `TPE` 与 `Random`，默认 `TPE`
- pruner：
  当前支持 `MedianPruner` 与关闭剪枝，默认 `MedianPruner`
- 多数据集扫描：
  `--optuna_studies kirp,lihc,coad,kirc,brca` 决定一个 trial 的数据集顺序。不传则退回单库 `--study`
  每个数据集仍按 `mean_cv` 跑 5 折，共 25 组。路径由 `infer_standard_paths()` 绑定为 `P/uni_v1`、`C/L0`、`G/scFoundation_embedding_cell_norm`
- objective：
  `mean(5 个数据集的 5-fold mean val C-index)`。这个总均值只用于选参，不做阈值校验
- 硬阈值剪枝：
  某个数据集 5 折结束后，若该数据集 mean val C-index 低于 `utils/optuna_utils.py` 里的 `OPTUNA_STUDY_MIN_CINDEX[study]`，立刻 `TrialPruned`，后续数据集不再跑。改阈值只改这个字典，没有 CLI flag
- epoch 级 MedianPruner 仍保留：
  每个 epoch 验证后 `trial.report(val_cindex, step)`。`step = study_idx * k * max_epochs + fold_idx * max_epochs + epoch`
- `single` 模式：
  使用 `--optuna_fold` 指定单折
- `mean_cv` 模式：
  使用：
  `--k`
  `--k_start`
  `--k_end`
  推导 folds，并返回其验证集 `c-index` 均值
- 当前推荐设置：
  `--optuna_trials 10`
  `--optuna_studies kirp,lihc,coad,kirc,brca`
  `--optuna_fold_mode mean_cv`
  `--optuna_sampler tpe`
  `--optuna_pruner median`
  `--optuna_n_startup_trials 5`
  `--optuna_n_warmup_steps 3`

### 13.3 输出产物

每个 study 会在结果目录下输出：

- `optuna_trials.csv`
- `best_trial.txt`
- sqlite study 数据库（若使用默认 storage）

同时每个 trial 仍会保留各 fold 的训练结果目录与 WandB run。

### 13.4 Model_A Optuna 运行示例

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init

CUDA_VISIBLE_DEVICES=1 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main_tune_optuna.py \
  --study tcga_lihc \
  --modality survtri_poe_vae \
  --poe_variant A \
  --bag_loss cox_surv \
  --label_dim 1 \
  --encoding_dim 1024 \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/G/scFoundation_embedding_cell_norm \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/C/L0 \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_lihc \
  --optuna_fold_mode mean_cv \
  --k 5 \
  --optuna_trials 10 \
  --max_epochs_stage1 5 \
  --max_epochs 20 \
  --warmup_epochs 3 \
  --batch_size 128 \
  --batch_size_stage1 128 \
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
  --wandb_entity davyfangyuxuan-nanjing-university-of-aeronautics-and-ast \
  --exp_group poe_vae_optuna \
  --run_name lihc_poeA
```

### 13.5 五数据集 Optuna 运行示例

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init

CUDA_VISIBLE_DEVICES=6 bash bg_tune.sh gpu6_optunaB.log \
  --optuna_studies kirp,lihc,coad,kirc,brca \
  --modality survtri_poe_vae \
  --poe_variant B \
  --betafix \
  --optuna_fold_mode mean_cv \
  --optuna_pruner median \
  --optuna_n_startup_trials 5 \
  --optuna_trials 15 \
  --bag_loss cox_surv \
  --label_dim 1 \
  --max_epochs_stage1 10 \
  --max_epochs 25 \
  --warmup_epochs 3 \
  --k 5 \
  --batch_size 128 \
  --wandb_mode online \
  --wandb_project POE_MODELB_TUNE

```

### 13.6 Model_B Optuna 运行示例

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init

CUDA_VISIBLE_DEVICES=6 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main_tune_optuna.py \
  --study tcga_lihc \
  --modality survtri_poe_vae \
  --poe_variant B \
  --bag_loss cox_surv \
  --label_dim 1 \
  --encoding_dim 1024 \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/G/scFoundation_embedding_cell_norm \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/C/L0 \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_lihc \
  --optuna_fold_mode mean_cv \
  --k 5 \
  --optuna_trials 10 \
  --max_epochs_stage1 5 \
  --max_epochs 20 \
  --warmup_epochs 3 \
  --betafix \
  --batch_size 128 \
  --batch_size_stage1 128 \
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
  --wandb_entity davyfangyuxuan-nanjing-university-of-aeronautics-and-ast \
  --exp_group poe_vae_optuna \
  --run_name lihc_poeB
```

### 13.7 Model_C Optuna 运行示例

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init

CUDA_VISIBLE_DEVICES=3 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main_tune_optuna.py \
  --study tcga_lihc \
  --modality survtri_poe_vae \
  --poe_variant C \
  --bag_loss cox_surv \
  --label_dim 1 \
  --encoding_dim 1024 \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/G/scFoundation_embedding_cell_norm \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_lihc/C/L0 \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_lihc \
  --optuna_fold_mode mean_cv \
  --k 5 \
  --optuna_trials 10 \
  --max_epochs 20 \
  --warmup_epochs 3 \
  --betafix \
  --batch_size 128 \
  --batch_size_stage1 128 \
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
  --wandb_entity davyfangyuxuan-nanjing-university-of-aeronautics-and-ast \
  --exp_group poe_vae_optuna \
  --run_name lihc_poeC
```

使用单折

```
--optuna_fold_mode single \
--optuna_fold 1
```

后台启动不走 `configs/queue`，直接用仓库根目录的 `bg_tune.sh`：

```bash
CUDA_VISIBLE_DEVICES=6 bash bg_tune.sh gpu6_optunaB.log \
  --study tcga_lihc \
  --modality survtri_poe_vae \
  --poe_variant B \
  --betafix \
  --bag_loss cox_surv \
  --label_dim 1 \
  --optuna_fold_mode mean_cv \
  --optuna_trials 10
```

### 13.8 当前限制

- 当前 `mean_cv` 模式按 fold 顺序串行训练，计算开销显著高于 `single` 模式
- 当前 Optuna 入口仅面向 `survtri_poe_vae`
- 目标函数固定使用验证集 `c-index`，不直接使用测试集指标
- Model C 的 stage2 `batch_size` 不搜索；旧 study 若仍采样过 `batch_size`，不能直接与当前协议混比
