# Model C + FiLM 完整结构

> 正式命名：MosaicSurv（原 Cfilm / survtri_poe_vae_C_film）

对应实现：`SurvTriPoEVAE_CFiLM`（[model_C_film.py](/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/ablation_models/model_C_film.py)）。
入口类只做三件事：强制 `poe_variant="C"`、挂上 `FiLMHead`、冻结旧的 `fuse_fc / classifier / linear_probe`。骨干与设计说明中的三模态 PoE-VAE 相同，来自 [model_SurvTriPoEVAE.py](/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/model_SurvTriPoEVAE.py)。

```
WSI / Gene / Clinic
        |
   各模态 Encoder
        |  (mu_i, logvar_i)
        v
  Generalized PoE  ->  (mu_joint, logvar_joint)
        |
        |- 训练采样 z，推理取 mu_joint
        |
        |- 三个 Decoder 重建各模态
        |
        +- FiLMHead(mu, pattern_id) -> Cox risk
```

---

## 1. 定位

| 项 | 实现 |
|---|---|
| 训练模式 | Model C：Enc + PoE + Dec + Head 从头联合训练，无 stage1 预训练 |
| 生存头 | 共享 MLP，用 7 种缺失模式 embedding 做 FiLM |
| 总损失 | `L = L_rec + beta * J + lambda * L_surv` |
| 代码 | `SurvTriPoEVAE_CFiLM` = `SurvTriPoEVAE` + `FiLMHead` |

和相邻消融的差别：

- `C_single`：同一套 VAE，单头 `fuse_fc + classifier`
- `C_multi`：同一套 VAE，7 个独立 MLP 头
- `C_film`：同一套 VAE，一个共享 MLP + FiLM 调制

---

## 2. 输入

| 模态 | 特征 | 形状 |
|---|---|---|
| WSI | UNI v1 | `(B, n_patch, 1024)`，`n_patch` 可变，带 padding mask |
| Gene | scFoundation | `(B, 4, 768)` |
| Clinic | COACH | `(B, n_c, 512)`，`n_c` 初始化时按 `.pt` 检测 |

另需 dataloader 提供 `avail = {wsi, gene, clinic}`。训练时再叠加模态 dropout（默认 `p=0.2`，禁止三模态全丢）。PoE 只融合当前可用模态。

隐变量维度硬编码 `d_z = 128`。

---

## 3. 编码器

三路专家结构相同、权重独立，各自输出对角高斯 `(mu_i, logvar_i)`，形状 `(B, 128)`，`logvar` clamp 到 `[-4, 2]`。

### 3.1 WSI：MIL Resampler + TokenSetEncoder

```
patches (B, n_patch, 1024)
  Linear(1024 -> 768)
  K=16 个可学习 query，对 patch 做 2 层 cross-attention（使用 padding mask）
  再 1 层 query 间 self-attention
-> wsi_tokens (B, 16, 768)
-> TokenSetEncoder -> (mu_wsi, logvar_wsi)
```

WSI 重建目标不走 posterior 分支，而是独立池化头：`mean(wsi_tokens) -> Linear(768, 768)`。

### 3.2 Gene / Clinic：TokenSetEncoder

```
[mu_query, logvar_query, tokens]
  -> 1 层 Transformer（LN -> MHA -> residual -> LN -> FFN）
  -> 取两个 query，分别 Linear 到 128 维
```

Gene 输入 dim=768，Clinic 输入 dim=512，不先投影到同一维。

---

## 4. Generalized PoE

三个模态专家 + 一个固定先验 `N(0, I)`：

```
alpha = softmax(a)            # 只在当前可用模态上归一化
tau_i = 1 / sigma_i^2
tau_joint = 1 + sum_{i in avail} alpha_i * tau_i
mu_joint = (sum_{i in avail} alpha_i * tau_i * mu_i) / tau_joint
```

`--alphafix` 打开后，`alpha` 改为 `--alphapgc pathology,gene,clinic` 的固定权重，缺失后再在可用子集上重归一化。

重参数化：训练 `z = mu + sigma * eps`，推理 `z = mu_joint`。Decoder 用 `z`；FiLM 生存头用 `mu`。

---

## 5. Decoder 与 VAE 损失

每个模态一个 decoder，输入只有 `z`，不拼接 condition：

```
Linear(128, 512) -> ELU -> Linear(512, out_dim)
```

| Decoder | 输出 | 重建目标 |
|---|---|---|
| WSI | 768 | 独立池化头 `wsi_target` |
| Gene | `4 x 768` | 原始 gene tokens |
| Clinic | `n_c x 512` | 原始 clinic tokens |

当前实现对缺失模态也会算重建（`available_mask` 不参与 reconstruction）。每模态学一个观测方差 `logvar_m`：

```
L_rec_m = MSE_m / (2 * sigma_m^2) + 0.5 * log(sigma_m^2)
L_rec   = sum_m L_rec_m
```

正则项是 joint 后验与 `N(0,I)` 的 Jeffreys 散度：

```
J = 0.5 * mean_j (sigma_j^2 + 1/sigma_j^2 - 2 + mu_j^2 * (1 + 1/sigma_j^2))
```

---

## 6. FiLM 生存头

真正的预后头是 `FiLMHead`（[model_utils.py](/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/model_utils.py)）：

```
pattern_id in {1..7}          # P=1, G=2, PG=3, C=4, PC=5, GC=6, PGC=7
                              # bit0=WSI, bit1=Gene, bit2=Clinic

e = Embedding(8, 32)[pattern_id]
(gamma, beta) = Linear(32 -> 256)(e)  # 2 * d_z；权重偏置初始化为 0
z_tilde = (1 + gamma) * mu + beta     # 起步等价于恒等变换

h = Linear(128 -> mmhid) -> ReLU -> Dropout(0.1)
  -> Linear(mmhid -> mmhid) -> ReLU -> Dropout(0.1)
risk = Linear(mmhid -> 1)(h)
```

`mmhid` 默认 256。7 种缺失模式共享同一套 MLP，只通过 FiLM 的 `gamma, beta` 区分。

---

## 7. 7-pattern Cox 与联合训练

Model C 没有预训练阶段，直接优化：

```
L = L_rec + beta * J + lambda * L_surv
```

`lambda` 默认 1.0（`--poe_surv_lambda`），`beta` 可 warmup 或 `--betafix` 固定。

`L_surv` 不是只对当前 `avail` 算一次 Cox，而是对 7 种模式分别重做 PoE 再加权平均：

1. 用已经算好的三路 `(mu_i, logvar_i)`
2. 对模式 `s in {P, G, C, PG, PC, GC, PGC}`，只取真实拥有该模式全部模态的样本
3. 按该模式的 avail mask 再融合一次 PoE
4. `FiLMHead(mu_s, s)` 得到该模式的 risk
5. 子集人数 `>= 8` 才计入 Cox，否则跳过
6. `L_surv = sum_s (1/7) * Cox_s`

前向里的 VAE 重建仍按样本真实 `avail` 融合；生存损失则在同一组 encoder 后验上枚举 7 种融合模式。推理时不再枚举，只用当前真实缺失模式对应的 `pattern_id`，取 `mu_joint` 过 FiLM 得到 risk。

---

## 8. 数据流

**训练**

```
三模态输入 + avail
  -> Encoder 得到 (mu_wsi, mu_gene, mu_clinic)
  -> PoE(真实 avail) -> 采样 z -> Decoder -> L_rec + beta * J
  -> 对 7 个 pattern 再 PoE -> FiLMHead -> lambda * L_surv
```

**推理**

```
可用模态 Encoder
  -> PoE(真实 avail) -> mu_joint
  -> pattern_id = bits_to_id(avail)
  -> FiLMHead(mu_joint, pattern_id) -> risk
```
