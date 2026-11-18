> 历史归档：本文件保留原排查时的命令、参数与结论，不作为当前实验协议；现行说明见 `z_temp/实验配置清单.md`。

# Table4: Model_B_nopretrain

## 设计
- 目标: 跳过 stage1，随机初始化后直接进入 stage2。
- 结构: 与 `Model_B` 的 stage2 完全一致。
- 差异: 不加载/不训练 stage1，不保存 stage1 checkpoint，不做 stage1 scan。
- 接入: 新增 modality `survtri_poe_vae_b_nopretrain`，内部固定按 B 路线跑 stage2。

## 测试命令

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init
conda activate SurvPGC

CUDA_VISIBLE_DEVICES=5 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_lihc \
  --modality survtri_poe_vae_b_nopretrain \
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
  --max_epochs 20 \
  --warmup_epochs 3 \
  --wandb_mode online \
  --wandb_project SurvPGC_MultiVAE \
  --exp_group poe_vae_ablation \
  --run_name model_B_nopretrain
```
