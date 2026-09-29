# Table4: Model_B_kl

## 设计
- 目标: 保留 `Model_B` 的两阶段训练流程，仅将 VAE 正则项从 Jeffreys 散度替换为标准 KL 散度。
- 结构: 与 `Model_B` 完全一致，仍执行 stage1 VAE 预训练 + stage2 survival finetune。
- 差异: `loss = reconstruction + beta * KL`，代码层面直接将原 `JeffreysDivergence` 替换为 `KLDivergence`。
- 接入: 新增 modality `survtri_poe_vae_b_kl`，内部固定按 B 路线跑 stage1/stage2。

## 测试命令

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init
conda activate SurvPGC

CUDA_VISIBLE_DEVICES=2 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_lihc \
  --modality survtri_poe_vae_b_kl \
  --poe_variant B \
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
  --max_epochs 20 \
  --max_epochs_stage1 10 \
  --warmup_epochs 3 \
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
  --exp_group poe_vae_ablation \
  --run_name model_B_kl
```
