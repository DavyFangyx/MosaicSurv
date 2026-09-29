# MultiVae / MVAE / MoPoE 结构说明（给执行 AI）

本文只说明这两个模型的真实结构，不写代码对接。

---

## 1. 共同输入格式

这两者都属于 `MultiVae` 框架，输入不是“一个大张量”，而是：

- `data: Dict[str, Tensor]`
- 可选 `masks: Dict[str, BoolTensor]`

约定：
- `data[m]` 是第 `m` 个模态的样本张量
- `masks[m]` 中 `True` 表示该样本该模态可用
- 缺失模态在 `data[m]` 里必须已经填成同 shape 的占位值，通常是 0

原生支持：
- `MultimodalBaseDataset`
- `IncompleteDataset`

### 1.1 Shape contract

每个模态 `m` 的基本形状约定是：

- 输入 `x_m: (B, D_m)` 或 `(B, T_m, D_m)`，取决于 encoder 实现
- encoder 输出 `mu_m, logvar_m: (B, d_z)`
- decoder 输出 `x_hat_m`，shape 与该模态的重建目标一致

本任务里可理解为：

- `WSI` 先被压成定长特征再进 encoder
- `Clinic` / `Gene` 直接用定长冻结特征
- `d_z = 128`

---

## 2. MVAE

### 2.1 结构

MVAE 是标准的 `Product-of-Experts VAE`。

流程：
1. 每个模态一个独立 encoder
2. encoder 输出该模态后验参数 `(mu_m, logvar_m)`
3. 把可用模态后验和先验一起做 PoE
4. 得到联合后验 `q(z | x_available)`
5. 从联合后验采样共享隐变量 `z`
6. 每个模态一个 decoder，从 `z` 重建该模态输入

### 2.1 Shape flow

- `x_m -> enc_m(x_m) -> (mu_m, logvar_m)`, 形状都是 `(B, d_z)`
- `PoE({mu_m, logvar_m}) -> (mu_joint, logvar_joint)`, 形状都是 `(B, d_z)`
- `z ~ q(z|x_available)`, 形状 `(B, d_z)`
- `dec_m(z) -> x_hat_m`, 形状回到该模态的重建目标 shape

### 2.2 输入输出

- 输入：任意多个模态的 `data`
- 缺失输入：通过 `masks` 指定，只对可用模态编码
- 输出：共享 latent `z`，以及各模态重建

### 2.3 缺失模态处理

MVAE 可以直接训练不完整数据。

- 对缺失模态，encoder 不参与 PoE
- 对单个样本，如果某模态缺失，它在该模态重建项里不计入 loss
- 若样本在当前 subset 下完全没有可用模态，则该 subset 被跳过

### 2.4 训练目标

MVAE 的目标是 subset ELBO 的和：

- joint ELBO
- 可选 unimodal ELBO
- 可选若干随机 subset ELBO

总损失结构：
- `reconstruction + beta * KL`

其中：
- `beta` 由 warmup 线性 anneal
- `use_subsampling=True` 时，训练不仅看 joint，还会采样额外 subset

### 2.5 结构要点

- MVAE 只有一个共享 latent space
- 不默认带模态私有 latent
- PoE 是核心机制，不能换成 MoE
- 它本身不是分类/生存模型，外接任务头是下游改造，不属于原模型

---

## 3. MoPoE

### 3.1 结构

MoPoE 是 `Mixture of Product-of-Experts`。

它比 MVAE 多了一层“子集级融合”：

1. 先枚举所有非空模态子集
2. 对每个子集内部先做 PoE
3. 再把所有子集 posterior 做 MoE
4. 从混合后的联合 posterior 采样 `z`
5. 每个模态 decoder 从 `z` 重建输入

### 3.1 Shape flow

- 每个模态 encoder 先把 `x_m` 映射到 `(B, d_z)`
- 每个子集 PoE 仍输出 `(B, d_z)`
- 子集 MoE 的最终联合后验还是 `(B, d_z)`
- decoder 重建输出与 MVAE 一样，回到各自模态目标 shape

### 3.2 输入输出

- 输入：`data` + 可选 `masks`
- 输出：共享 latent `z`，以及每模态重建

### 3.3 缺失模态处理

MoPoE 对缺失更自然，逻辑是：

- 只枚举当前样本实际可用模态组成的 subset
- subset 内部只对可用模态做 PoE
- 所有可用 subset 再做 MoE
- 若某个 subset 对当前 batch 全部样本都不可用，则跳过

### 3.4 训练目标

MoPoE 的核心目标是 `joint_elbo`：

- 每个可用模态 subset 都会贡献重建项
- 所有 subset 的 KL 按 mixture 权重聚合
- 若配置了 `modalities_specific_dim`，还会额外加模态私有 latent 的 KLD 项

总体形式：
- `reconstruction + beta * joint_divergence (+ beta_style * private_kld)`

### 3.5 结构要点

- MoPoE = “subset PoE” + “subset MoE”
- 它比 MVAE 多了 subset mixture 层
- 默认也只有共享 latent；私有 latent 只是可选扩展
- 它不是把 MVAE 改个名字，而是更高一层的聚合机制

---