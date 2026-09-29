# 对比模型实现明细

这份文件只记录目前已经实际落地的内容。
每一条都按“原来是什么 / 现在改成什么 / 为什么这么改”来写。
Flex-MoE 不在本文件范围内。

## 1. Concat 地板组（`--modality modality_concat`）

### 1.1 接入本项目训练入口
- 文件：
  - `models/missing_modality_baselines/concat.py`
  - `utils/core_utils.py`
  - `datasets/dataset_survival.py`
  - `main.py`
- 原来：
  - 仓库里没有 concat 地板组
  - 也没有 `--concat_wsi` / `--concat_impute`
- 现在：
  - 新增 `ConcatMissingModalityBaseline`
  - 通过 `--modality modality_concat` 走本项目 `main.py` / `_train_val_test`
  - 和主模型共用同一份 patient-level 数据、同一份 `avail` mask、同一套 Cox 训练循环
  - `--concat_wsi` 默认 `resampler`，再和 `--concat_impute {zero, mean}` 组合
  - `meanpool` 代码还在，但不作为当前实验入口
- 原因：
  - 需要一个没有融合机制、只做填充再拼接的地板对照

### 1.2 WSI 前端拆成 meanpool / resampler 两路
- 文件：`models/missing_modality_baselines/concat.py`
- 原来：
  - 没有统一的 concat 地板组 WSI 前端
- 现在：
  - 当前默认 / 实际调用的是 `resampler`
  - `resampler`：复用主模型同结构的 `WSIMILResampler`（独立权重，不加载主模型 ckpt）
    - `(n_patch, 1024) -> (B, 16, 768) -> mean -> (B, 768)`
  - `meanpool`：对有效 patch 做 masked mean，得到 `(B, 1024)`。（已实现，但不调用）
  - Gene：flatten 成 `(B, 3072)`
  - Clinic：flatten 成 `(B, n_c * 512)`
- 原因：
  - 当前实验只跑 `concat_rs`，用来和主模型只差融合与填充
  - `concat_mp` 作为最朴素参考值已经写好，但当前不调用

### 1.3 缺失填充发生在 MLP 之前
- 文件：
  - `models/missing_modality_baselines/concat.py`
  - `utils/core_utils.py`
- 原来：
  - 没有本项目口径的缺失填充
- 现在：
  - dataloader 下发的 `avail` 是唯一缺失真值
  - 缺失模态在进入 `MLP_m` 之前，整条向量替换为 `fill_m`
  - `--concat_impute zero`：`fill_m = 0`
  - `--concat_impute mean`：每个 fold 训练开始前，只用当前 fold 训练集上该模态可用样本算一次池化特征均值，写入 `register_buffer`
  - val / test 直接读 buffer，不重算
  - `MLP_m = Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> ReLU`
  - LayerNorm 放在第一个 Linear 之后
  - 不拼接缺失指示位、mask embedding、模态 ID
  - 不对输入特征做逐维 z-score
- 原因：
  - 填充如果放到 MLP 之后，带 bias 的线性层会把零输入映射成常数向量，等价于隐式 missing token
  - LayerNorm 如果放到填充和 Linear 之间，会把 `zero` 和 `mean` 抹平

### 1.4 生存头改成 fuse_fc + classifier
- 文件：`models/missing_modality_baselines/concat.py`
- 原来：
  - 初版用 `CoxHead(mmhid -> mmhid -> 1)`，等于在一层 `fuse_fc` 后再叠一层隐藏层
- 现在：
  - `h = concat(h_w, h_g, h_c)`，形状 `(B, 384)`
  - `fuse_fc`：`Linear(384, 256) -> ReLU -> Dropout -> Linear(256, 256) -> ReLU -> Dropout`
  - `classifier`：`Linear(256, 1)` 出 Cox risk
- 原因：
  - 对齐主模型的生存头形式
  - concat 先拼三路 `h_m`，所以 `fuse_fc` 输入维是 `3 * 128`，不是主模型的 `128`

## 2. MVAE（`--modality mvae_poe`）

### 2.1 保留 MultiVae 的 MVAE 机制，改走本项目训练循环
- 文件：
  - `models/missing_modality_baselines/vae_family.py`
  - `models/missing_modality_baselines/third_party/MultiVae/src/multivae/models/mvae/mvae_model.py`
  - `utils/core_utils.py`
- 原来：
  - 原始 MVAE 用 MultiVae 自带 trainer
  - encoder / decoder 面向像素或原始 token
  - 没有 Cox 生存头
- 现在：
  - 直接实例化第三方 `MVAE`
  - 不使用 MultiVae trainer，改走本项目 `_train_val_test`
  - 保留 Product-of-Experts 联合后验
  - 保留 subset ELBO：joint + unimodal + 随机 subset
  - `use_subsampling=True`
  - `k_subsets=2`
  - 保留 beta warmup，`warmup=10`，`beta=1.0`
  - 总 loss = ELBO + `1.0 * L_cox`
- 原因：
  - 需要把原生 MVAE 接到本项目的冻结特征和生存任务上
  - 聚合规则仍用原实现，不改写成主模型 Generalized PoE

### 2.2 输入预处理和缺失 mask
- 文件：`models/missing_modality_baselines/vae_family.py`
- 原来：
  - MultiVae 要求定长 `data[m]`，并用 `masks` 表示缺失
- 现在：
  - WSI：
    - `(n_patch, 1024)` 先过主模型同结构的 `WSIMILResampler`，得到 `(B, 16, 768)`
    - 再过 `WSITargetPoolingHead`，得到 `(B, 768)`
    - 这不是 masked mean pooling
  - Gene：flatten 成 `(B, 4 * 768)`
  - Clinic：flatten 成 `(B, n_c * 512)`
  - `avail` 转成 MultiVae 的 `masks`
  - 缺失模态在 `data[m]` 里填同 shape 的零占位，只满足 shape contract，不是 concat 那种补值语义
- 原因：
  - MultiVae 的 MLP encoder 吃不了变长 patch
  - WSI 压成定长的方式和主模型一致：resampler + 独立 `wsi_target` 头，权重独立训练

### 2.3 encoder / decoder 仍用 MultiVae 默认 MLP
- 文件：
  - `models/missing_modality_baselines/vae_family.py`
  - `models/missing_modality_baselines/third_party/MultiVae/src/multivae/models/nn/default_architectures.py`
- 原来：
  - MultiVae 默认 `Encoder_VAE_MLP` / `Decoder_AE_MLP`
- 现在：
  - 仍然用这套默认 MLP
  - encoder：冻结特征 / 压扁后的 WSI 向量 -> `(B, 128)` 对角高斯
  - decoder 重建目标：
    - WSI：`(B, 768)`，也就是 `WSITargetPoolingHead` 的输出，不是原始 `(n_patch, 1024)`
    - Gene / Clinic：flatten 后的冻结特征
  - decoder 末端仍带 Sigmoid
- 原因：
  - 当前实现先把原生 MVAE 接到冻结特征和 Cox 头上
  - 没有把 encoder / decoder 替换成主模型的 `TokenSetEncoder` / `ModalityDecoder`

### 2.4 生存头
- 文件：`models/missing_modality_baselines/vae_family.py`
- 原来：
  - 原生 MVAE 没有生存头
- 现在：
  - 从 MVAE 的 joint posterior 采样 `z`
  - `CoxHead`：`Linear(128, 256) -> ReLU -> Dropout -> Linear(256, 1)`
  - 这和主模型 `fuse_fc + classifier` 不是同一套结构
- 原因：
  - 只在 joint latent 上挂一个 Cox 头，让生成式目标和生存目标一起训练

## 3. MoPoE（`--modality mopoe`）

### 3.1 与 MVAE 共用同一套 wrapper，只换模型类
- 文件：`models/missing_modality_baselines/vae_family.py`
- 原来：
  - 原始 MoPoE 也是独立的 MultiVae 模型 + 自带 trainer
- 现在：
  - `MVAEBaseline` 和 `MoPoEBaseline` 共用 `_SurvivalMultiVAEBase`
  - 同一套 WSI resampler / `wsi_target` 头
  - 同一套 MultiVae 默认 encoder / decoder
  - 同一套 `avail -> masks` 转换
  - 同一套 Cox 头和 `lambda_surv = 1.0`
  - 唯一差别是模型类：`MVAE` vs `MoPoE`
  - MoPoE 设置 `modalities_specific_dim=None`，不启用模态私有 latent
- 原因：
  - 对照应只落在聚合规则上：vanilla PoE vs 先 subset PoE 再 MoE

### 3.2 保留 MoPoE 原生两级聚合
- 文件：`models/missing_modality_baselines/third_party/MultiVae/src/multivae/models/mopoe/mopoe_model.py`
- 原来：
  - 先对每个非空模态子集做 PoE，再对所有子集后验做 MoE
  - `joint_elbo` 按 mixture 权重聚合
- 现在：
  - 这部分原实现未改
  - incomplete 数据时，subset 可用性由 `masks` 过滤
  - `masks` 来自本项目 dataloader 的 `avail`
- 原因：
  - L3 聚合机制保持原样，只替换输入特征和生存头

## 4. HGCN

### 4.1 输入维度改造
- 文件：`models/missing_modality_baselines/third_party/HGCN/HGCN_code/mae_model.py`
- 原来：
  - `fusion_model_mae_2.__init__(in_feats, ...)`
  - 三个 `SAGEConv` 都共用同一个 `in_feats`
  - 默认等价于 `x_img / x_rna / x_cli` 都按 1024 维输入
- 现在：
  - 改成 `fusion_model_mae_2(img_in_feats, rna_in_feats, cli_in_feats, ...)`
  - `img_gnn_2` 输入维度是 `1024`
  - `rna_gnn_2` 输入维度是 `768`
  - `cli_gnn_2` 输入维度是 `512`
- 原因：
  - `z_temp/Table2/部署指南.md` 明确要求 HGCN 三路输入维度改为 `1024 / 768 / 512`
  - 这里只改输入维度，不改图卷积、池化、MAE、Mixer、输出结构

### 4.2 HGCN 训练入口参数同步
- 文件：`models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py`
- 原来：
  - 实例化模型时传 `in_feats=1024`
- 现在：
  - 改成显式传
    - `img_in_feats=1024`
    - `rna_in_feats=768`
    - `cli_in_feats=512`
- 原因：
  - 与 1.1 的模型签名保持一致
  - 只改实例化参数，不改训练流程

### 4.3 HGCN split 读取方式改造
- 文件：`models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py`
- 原来：
  - 固定 split 来自 `seed_fit_split.pkl`
- 现在：
  - 新增 `_resolve_split_dir()`
  - 新增 `_load_split_csv()`
  - `if_fit_split=True` 时改为读取 `splits_{fold}.csv`
  - 新增参数：
    - `--split_root`
    - `--split_dir`
- 原因：
  - 你上一轮明确要求这两个模型只用 csv split
  - 这里只改 split 来源，不改 fold 训练逻辑

### 4.4 HGCN 图构造改成原生三模态 pkl
- 文件：`SurvPGC_Workspace/HGCN data gen.md`
- 原来：把 UNI / scFoundation / CONCH embedding 转成按病人 `.pt` 图
- 现在：三个模态从原始数据写成 `hgcn data/` 下的目录级 pkl

### 4.5 HGCN GraphData 不再把缓存图钉在 GPU 上
- 文件：
  - `simple_graph.py`
  - `models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py`
- 原来：
  - `GraphData.to(device)` 原地改 `all_data[id]` 里的 tensor
  - 训练/验证/测试每扫过一个病人，这张图就留在 GPU 上直到进程退出
- 现在：
  - `to(device)` 返回一份拷贝，CPU 缓存图保持不动
  - 训练循环继续写 `graph = all_data[id].to(device)`
- 原因：
  - 原实现会把整个队列的图逐渐堆进显存，batch 还没到 32 就先 OOM

### 4.6 HGCN 使用独立 batch size=32
- 文件：
  - `configs/presets.sh`
  - `configs/z_exp_gen/gen_Table2_missing_modality_baselines.sh`
  - `configs/z_exp_gen/gen_Table3_missing_rate_test.sh`
- 原来：
  - Table2 / Table3 生成脚本默认 `BATCH_SIZE=128`
  - HGCN 也吃到同一个值
- 现在：
  - `apply_preset hgcn` 把 `BATCH_SIZE` 固定成 `32`
  - 生成脚本给 HGCN 再写一行 `BATCH_SIZE=32`
- 原因：
  - 原始 HGCN 论文代码默认 `batch_size=32`
  - 仓库里的 `4096` 是 MIL patch 采样数 `NUM_PATCHES`，不是 HGCN 的病人级 batch
  - `128` 是本仓库 MIL / concat / VAE 对照的默认值，不能直接套到 HGCN 图训练上

### 4.7 HGCN event 编码改成 1=event
- 文件：
  - `models/missing_modality_baselines/native_hgcn/pack.py`
  - `models/missing_modality_baselines/hgcn_graph_build.py`
- 原来：
  - `sur_type` / `sur_and_time[0]` 直接写 metadata 的 `censorship`
  - TCGA 口径是 `1=censored`，HGCN Cox / C-index 却按 `1=event` 用
- 现在：
  - 统一写成 `event = 1 - censorship`
  - pack summary 明确记录 `1=event, 0=censored`
- 原因：
  - 原来所有删失病人都被当成事件，C-index 和 Cox 权重都是反的

### 4.8 HGCN 空图前向不再走 SAGEConv
- 文件：`models/missing_modality_baselines/third_party/HGCN/HGCN_code/mae_model.py`
- 原来：
  - `use_type` 里有 `img` 时，即使 `x_img` 是 `(0, 1024)` 也会进 `SAGEConv`
  - 空 `batch` 再进 `GlobalAttention` 会直接崩
- 现在：
  - 0 节点模态改成 1 行零向量占位，跳过 GNN / pooling
  - MAE 仍按 `train_use_type` 对齐三路 token
- 原因：
  - 缺失 WSI 的病人现在是合法输入，不能因为空图把训练打掉

### 4.9 HGCN split 只保留组装到的病人
- 文件：`models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py`
- 原来：
  - `if_fit_split=True` 时直接读 `splits_{fold}.csv`
  - csv 里有、图没装出来的 id 会在 `all_data[id]` 处 KeyError
- 现在：
  - train / val / test 都和组装结果做交集
  - 某一 split 交完是空集才报错
- 原因：
  - 原生 HGCN pkl 覆盖率和 csv split 并不完全重合
