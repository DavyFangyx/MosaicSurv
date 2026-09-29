# HGCN / Flex-MoE / MVAE / MoPoE 补充说明（给执行 AI）

这份补充只回答实现时最容易误读的点。

---

## 1. HGCN

### 1.1 `pool_x` 的 `T`

默认训练/评估路径里，`T = len(train_use_type)`，当前配置就是 `3`。

原因：
- `train.py` 在多模态模式下把 `use_type` 传成 `args.train_use_type`
- `forward()` 里 `data_type = use_type`
- 所以 `pool_x` 是按训练模态集合而不是按样本当前可用模态数来组的

但代码里保留了另一条分支：
- 当 `use_type != train_use_type` 时，会把缺失模态位置补成 0 的 `tmp_x`
- 这时 `pool_x` 仍然落在 `train_use_type` 这 3 个位置上

结论：
- 原始主路径是“3 个模态槽位”
- 不是“当前可用模态数”
- 如果要写成补全式基线，必须显式改代码并写明；否则它只是对可用分支做 risk 聚合

### 1.2 patch 坐标

当前仓库里没有可验真的 `.h5/.h5ad` patch 资产，也没有 `coords` 字段可直接读取。

结论：
- 不能在本地证明 WSI patch 坐标还在
- 若 upstream 数据里没有 coords，就不能宣称保留原版 8 邻域图
- 这必须写成 limitation 或 fallback

### 1.3 `N_img` 和完整样本数

当前仓库里可以确认的只有“完整样本数”（PCG）：

- BRCA: 1059 / 1098
- COAD: 448 / 460
- KIRC: 509 / 513
- KIRP: 274 / 275
- LIHC: 358 / 377

来源：`data_tcgal_stats/汇总.csv`

`N_img` 的 patch-level 分布：
- 仓库里没有 patch 级明细
- 所以不能在本地给出中位数 / 95 分位

### 1.4 `n_c`

当前仓库没有提供 prompt clinic 的 `.pt` 样本文件，所以 `n_c` 不能静态确认。

代码里的实际策略是：
- 运行时从 `clinic_dir` 的第一个 `.pt` 里读 `shape[0]`
- `n_c` 就是这个 token 数

原版 HGCN 里临床节点数是 10，但那不是当前 prompt clinic 的静态保证。

### 1.5 默认超参

- `format_of_coxloss = "multi"`
- `img_cox_loss_factor = 5`
- `rna_cox_loss_factor = 1`
- `cli_cox_loss_factor = 5`
- `add_mse_loss_of_mae = True`
- `mse_loss_of_mae_factor = 5`
- `out_classes = 512`
- `drop_out_ratio = 0.5`
- attention gate 网络：`512 -> 128 -> 1`

### 1.6 训练语义

HGCN 不是把缺失模态真正补成完整图，而是：
- 先做模态图编码
- 再用 MAE 对模态级 token 做重建
- 再把 token 加回去

所以不要把它写成 PoE 式补全模型。

---

## 2. Flex-MoE

### 2.1 默认超参

- `num_experts = 16`
- `top_k = 4`
- `num_layers_fus = 1`
- `num_layers_pred = 1`
- `num_heads = 4`
- `hidden_dim = 128`
- `num_patches = 16`
- `gate_loss_weight = 1e-2`
- `warm_up_epochs = 5`
- `dropout = 0.5`
- `initial_filling = "mean"`

### 2.2 调度

原始 `main.py` 有分阶段训练：
- 前 `warm_up_epochs` 用按模态组合排序的数据加载器
- 之后切到打乱顺序的数据加载器

所以它不是完全一把梭的端到端训练。

### 2.3 结构提醒

- 不是图模型
- 缺失模态靠 `missing_embeds`，不是补图
- `gate_loss` 是原机制的一部分，不能删

---

## 3. MVAE

### 3.1 默认值

- `use_subsampling = True`
- `k = 0`
- `warmup = 10`
- `beta = 1`

### 3.2 shape

对每个模态：
- 输入 `x_m -> encoder`
- encoder 输出 `mu_m, logvar_m : (B, 128)`
- PoE 后联合后验还是 `(B, 128)`
- decoder 输出回到该模态重建目标 shape

### 3.3 缺失策略

在本项目里，推荐写法是：
- 统一用外部 `masks` 决定本样本观测到哪些模态
- MVAE 只在这些观测模态组成的 subset 上枚举

这样不会额外制造缺失来源。

---

## 4. MoPoE

### 4.1 默认值

- `modalities_specific_dim = None`
- `beta_style = 1.0`
- `beta = 1.0`

### 4.2 推荐设置

建议 `modalities_specific_dim = None`。

原因：
- 不引入额外私有 latent 容量
- 和主模型更可比

### 4.3 shape

和 MVAE 一样：
- encoder 输出 `(B, 128)`
- subset PoE 输出 `(B, 128)`
- subset MoE 后联合 posterior 仍是 `(B, 128)`

### 4.4 缺失策略

只枚举当前样本实际可用模态的 subset。

---

## 5. 一句话结论

- HGCN 默认是 3 个模态槽位，不是动态 `T`
- patch 坐标和 `N_img` 分布当前仓库都无法本地验真
- `n_c` 运行时推断，不应写死
- Flex-MoE 有 warm-up 调度
- MVAE 默认 `use_subsampling=True`
- MoPoE 默认不加 private latent
