# SurvPGC
Code of "Multimodal Deep Learning for Cancer Prognosis Prediction with Clinical Information Prompts Integration"

Thanks for code from: 
https://github.com/mahmoodlab/CLAM
https://github.com/mahmoodlab/PORPOISE
https://github.com/mahmoodlab/MCAT
https://github.com/mahmoodlab/SurvPath

More visualization results, including heatmaps and top patches, are available at: https://pan.baidu.com/s/1b7syjwdz-uCo0MxQXeGY8Q?pwd=2w9j

## 12. 各组的训练监控体系

当前 `survtri_poe_vae` 已接入 WandB 监控，覆盖 Model_A / Model_B / Model_C。

- Model_A：
  stage1 为 VAE 预训练；
  stage2 为冻结 backbone 后的线性 Cox probe。
- Model_B：
  stage1 为 VAE 预训练；
  stage2 为 `fuse_fc + classifier + Cox` 微调。
- Model_C：
  直接进行联合训练，优化 `VAE + survival`。

### 12.1 Step 级监控

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

### 12.2 Epoch 级监控

每个 epoch 记录潜空间与 PoE 健康度：

- `z/mean_norm`
- `z/mean_std`
- `poe/alpha_wsi`
- `poe/alpha_gene`
- `poe/alpha_clinic`

其中：

- `z/mean_norm` 为 batch 内 `mu_joint` 的平均 L2 norm
- `z/mean_std` 当前实现为 batch 内跨样本、对 latent 各维度求方差后再取平均
- `poe/alpha_*` 为 PoE 权重 softmax 后的 batch 平均值；`--alphafix` 打开后则为 `--alphapgc` 在当前可用模态上重新归一化后的值

### 12.3 生存监控

在包含 Cox 优化的训练阶段按 epoch 记录：

- `loss/surv`

说明：

- Model_A / Model_B 的 stage1 纯 VAE 训练阶段不包含 survival loss
- Model_A / Model_B 的 stage2，以及 Model_C 的联合训练阶段会记录 `loss/surv`

### 12.4 WandB 开启方式

当前 WandB 由命令行参数控制：

- `--wandb_mode disabled`
- `--wandb_mode offline`
- `--wandb_mode online`
- `--wandb_project <project_name>`
- `--wandb_entity <entity_name>`
1
