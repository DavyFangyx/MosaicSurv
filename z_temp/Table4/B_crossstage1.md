# Table4: Model_B_crossstage1

## 设计
- 目标: 将 `Model_B` 的“多数据集 stage1 预训练 + 单数据集 stage2 finetune”拆成独立模型。
- 结构: 与 `Model_B` 相同，仍执行 stage1 VAE 预训练 + stage2 survival finetune。
- 差异: stage1 study 池由 `--poe_stage1_studies` 指定，不再暴露 `poe_cross_stage1` 命令行参数。
- 接入: 新增 modality `survtri_poe_vae_b_crossstage1`，固定按 B 路线跑 stage1/stage2，并复用共享 stage1 checkpoint。

## 测试命令

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init
conda activate SurvPGC

CUDA_VISIBLE_DEVICES=5 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_lihc \
  --modality survtri_poe_vae_b_crossstage1 \
  --selected_modalities wsi,gene,clinic \
  --poe_stage1_studies brca,coad,kirc,kirp,lihc \
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
  --run_name model_B_crossstage1
```

## 说明
- `--poe_stage1_studies` 支持 `brca,coad,...` 或 `tcga_brca,tcga_coad,...`。
  --poe_stage1_studies brca,coad,kirc,kirp,lihc \

- stage1 仍按 fold 对齐：例如 `fold=0` 会拼接参数指定的多个 study 的 `splits_0.csv` 做联合预训练，然后只在当前 `--study` 的 `fold=0` 上做 stage2 与测试。
