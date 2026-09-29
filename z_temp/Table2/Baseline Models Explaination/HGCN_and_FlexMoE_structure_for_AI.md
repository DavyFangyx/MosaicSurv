# HGCN / Flex-MoE 结构说明（给执行 AI）

本文只说明这两个模型的真实结构，不写代码对接。
规则只有两条：
1. 机制本体不许改。
2. 改造时必须把输入、图/序列构造、融合、输出、loss 写清楚并显式记录限制。

---

## 1. HGCN

### 1.1 输入与图构建

HGCN 不是端到端图生成模型。它先把每个病人的三模态特征离线封成图，再做图编码。

| 模态 | 节点含义 | 节点特征 shape | 边规则 |
|---|---|---|---|
| WSI | patch 节点 | `x_img: (N_img, 1024)` | 8 邻域空间连边 |
| RNA | GSEA 通路节点 | `x_rna: (5, 1024)` | 全连接去自环 |
| Clinic | 临床变量节点 | `x_cli: (10, 1024)` | 全连接去自环 |

补充：
- 原始代码里每个病人是一个 `Data(...)`，并存 `edge_index_image / edge_index_rna / edge_index_cli`。
- `edge_index_model` 也被写进对象里，但当前 forward 不用它。
- WSI 必须先确认 patch 坐标还在不在。若 `.h5/.pt` 里没有 `coords`，就不能假装保留原版 8 邻域，只能显式改成别的建图方式并写 limitation。

### 1.2 图编码

当前代码是三套独立的 `SAGEConv`，**不是共享权重**：

- `img_gnn_2: SAGEConv(1024 -> out_classes)`
- `rna_gnn_2: SAGEConv(1024 -> out_classes)`
- `cli_gnn_2: SAGEConv(1024 -> out_classes)`

默认 `out_classes = 512`，所以：

- 图卷积后：`(N_img, 512) / (5, 512) / (10, 512)`
- `ReLU + LayerNorm + Dropout` 不改 shape
- `my_GlobalAttention` 池化后：每个模态变成 `[(1), 512]`

### 1.3 在线 MAE

HGCN 的 MAE 是在“模态 token”上做的，不是在 patch 节点上做的。

流程：
- 先把每个模态池化成一个 token，得到 `pool_x: (T, 512)`，`T` 是当前样本可用模态数
- `PretrainVisionTransformer` 对这些 token 做遮盖重建
- `mae_out` 的 shape 与 `mae_labels` 相同，都是 `(T, 512)`
- 重建出来的 token 再 broadcast 加回各自模态的节点特征图

所以 HGCN 是：
`node graph -> modality token -> MAE -> broadcast back -> node graph -> final risk`

### 1.4 输出与 loss

最终输出不是单一路径。

- `multi_x: (T, 1)`，每个模态一路风险
- `one_x: (1,)`，`multi_x` 的均值融合

原始训练里：
- `format_of_coxloss = one` 时，只对 `one_x` 算 Cox
- `format_of_coxloss = multi` 时，对各模态风险分别算 Cox，再按权重相加
- 可选再加 `mse_loss_of_mae * MSE(mae_out, mae_labels)`

改造时不要把它压成单分支。只能统一末端 risk head 的形式，不能删掉 `multi_x` 这条机制。

### 1.5 必须写死的限制

- 若 patch 坐标存在，必须沿用 8 邻域边。
- 若 patch 坐标不存在，必须显式退化，不可静默假装原版图结构还在。
- 三个 `SAGEConv` 当前是独立参数，不共享权重。
- 原始语义节点很重要：RNA 是 5 个 GSEA 通路，Clinic 是 10 个临床变量。换成别的 token 后，图还能跑，但语义已经漂移，必须写 limitation。
- 这个模型很吃显存。`N_img` 很大时要提前评估，不要跑到一半再临时改采样策略。

---

## 2. Flex-MoE

### 2.1 输入与 token 化

Flex-MoE 不是图模型，是“模态 token + Sparse MoE Transformer”模型。

ADNI 默认 4 模态：`image / genomic / clinical / biospecimen`。
MIMIC 默认 3 模态：`lab / note / code`。

每个模态先变成固定 token 序列：

- 原始/预处理后的模态向量 `x_m: (B, D_m)`
- 经 `PatchEmbeddings(D_m, num_patches, hidden_dim)` 变成 `t_m: (B, P, H)`
- 默认 `P = 16`, `H = 128`

`image` 分支有两种：
- 预处理图像：直接走 `PatchEmbeddings`
- 原始 3D 图像：`Custom3DCNN -> (B, H) -> PatchEmbeddings(H, P, H) -> (B, P, H)`

缺失模态不是在模型里“补值”，而是用可学习的 `missing_embeds`：

- shape: `((2^M)-1, M, P, H)`
- 其中 `M` 是模态数
- `batch_mcs` 给出当前样本属于哪个模态组合

### 2.2 融合与 MoE

融合主干是若干层 `TransformerEncoderLayer`：

1. 各模态 token 先在序列维拼接成 `(B, M*P, H)`
2. 加位置编码
3. 再 split 回各模态 chunk
4. 每层先做 self-attention
5. 每层的 MLP 子层要么是普通 MLP，要么是 `FMoETransformerMLP`
6. `FMoETransformerMLP` 内部用 `AddtionalNoisyGate` 做 top-k 路由

要点：
- `top_k` 决定每个 token 选几个 expert
- `expert_indices` 来自模态组合 id
- `full_modality_index = 0`
- `set_full_modality` 和 `gate_loss` 都是原机制的一部分，不能删

### 2.3 输出与 loss

每层融合后，模型再把每个模态 chunk 做 mean pooling：

- 每个模态得到 `h_m: (B, H)`
- 拼接后得到 `h: (B, M*H)`
- 最后送入预测头，原始任务输出分类 logits `(B, C)`

在本项目改造版里：
- 末端分类头要换成统一 Cox risk head
- 但 router / gate loss 必须保留

原始 loss：
- `CrossEntropy + lambda * gate_loss`

改造后 loss：
- `Cox + lambda * gate_loss`

`gate_loss` 本身由两部分组成：
- expert index 约束
- load balance / importance 约束

### 2.4 必须写清楚的限制

- 这是 token Transformer，不是图。
- `missing_embeds` 是原始机制，不要改成后验补全。
- `PatchEmbeddings` 是按特征维切块，不是学习型 tokenizer。
- `FlexMoE` 当前实现对每个模态 token 序列长度一致，默认 `num_patches=16`。
- 改造时不要破坏 `expert_indices -> router -> gate_loss` 这条链。

---

