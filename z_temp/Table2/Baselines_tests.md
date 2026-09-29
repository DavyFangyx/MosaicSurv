# Table2 Baselines Test Commands

```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init
conda activate SurvPGC
```

训练缺失和测试缺失是两套独立开关。Table2 只开测试 7 子集；训练配比留给 Table3。

- 测试开关：`--eval_modalities` / `EVAL_MODALITIES`
- 训练开关：`--missing_mode` / `MISSING_MODE`，配比是 `--missing_pattern` / `MISSING_PATTERN`

`configs/z_exp_gen/gen_Table2_missing_modality_baselines.sh` 的结果矩阵是 7 个测试子集 x 6 个方法 x 5 个数据集。训练不是 210 次：6 个方法 x 5 个数据集只生成 30 份 config，每个模型只训一次三模态，推理期 7 个子集共用这次训练。

## 开关语义

### 测试 7 子集

`main.py`：`--eval_modalities`
config：`EVAL_MODALITIES`

- 关：`--eval_modalities off`，或不传。只评完整三模态 PCG，结果写模型目录 `test_result.csv`，不写 `eval_subsets/`。
- 开全部：`--eval_modalities all`。这是 Table2 批量写法。PCG 写 `test_result.csv`，7 个子集写到 `eval_subsets/{P,C,G,PC,PG,CG,PCG}/test_result.csv`，共用同一次训练。
- 也可只开部分：`--eval_modalities P,C,G` 或 `--eval_modalities wsi,clinic`。

开/关后各模型：

- 主模型 / concat / mvae / mopoe：关则测试 `avail` 保持 dataloader 原样；开则从 splits 的完整三模态再按子集盖 mask，不和训练 csv 做 AND。
- HGCN：关则只评原生三模态 `use_type=[img,rna,cli]`；开则按子集改 `use_type`，缺的槽位仍在图里，补 0 再交给原来的 MAE。原生评估只有 lifelines c-index，IPCW/IBS/IAUC 列保持 0。

### 训练 mask

`main.py`：`--missing_mode` `--missing_pattern` `--missing_seed` `--poe_modality_dropout`
config：`MISSING_MODE` `MISSING_PATTERN` `MISSING_SEED` `POE_MODALITY_DROPOUT`

- 关：`--missing_mode model_gen`。这是 Table2 默认，不启用统一训练配比。
- 开：`--missing_mode unified_mask_csv --missing_pattern 60,0,0`。同一 experiment group 的模型读同一份 fold csv；`--poe_modality_dropout` 会被强制打成 0。

关着时各模型：

- 主模型：只有训练集按 `--poe_modality_dropout` 做 case 级 dropout，默认 0.2；val/test 不丢。
- concat / mvae / mopoe：训练不额外丢模态，三模态都按完整样本走。
- HGCN：训练仍用原生 `generate_mask()`；val/test 空 mask，不套统一配比。

开着时各模型：

- 所有模型，包括 HGCN，都读同一份 `results/<exp_group>/masks/<study>/pW_G_C/seed_*/fold_*.csv`。
- 主模型不再叠加 `poe_modality_dropout`。
- HGCN 不再自己随机丢模态，改把 csv 转成原生 `in_mask`。
- 当前 csv 会盖到 train/val/test。`--eval_modalities` 的 7 个测试子集不再读这份 csv，而是从 splits 的完整三模态再盖评测子集。两套开关去耦合，可以同时开。

配比格式是 `wsi,gene,clinic` 百分数，各模态独立采样，例如 `60,0,0` / `0,60,0` / `0,0,60` / `20,20,20` / `30,30,0`。全缺样本会随机留一个模态。

## 1. HGCN
先生成镜像图树：
```bash
CUDA_VISIBLE_DEVICES=0 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python SurvPGC_Workspace/generate_hgcn_graph.py
```

再训练：
```bash
CUDA_VISIBLE_DEVICES=2 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_kirp \
  --modality hgcn \
  --eval_modalities all \
  --exp_group Table2_Baselines \
  --run_name tcga_kirp__hgcn \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_kirp \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/G/scFoundation_embedding_gene_raw \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/C/L4 \
  --max_epochs 3
```

## 2. Concat - zero
```bash
CUDA_VISIBLE_DEVICES=3 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_kirp \
  --modality modality_concat \
  --concat_impute zero \
  --bag_loss cox_surv \
  --eval_modalities all \
  --exp_group Table2_Baselines \
  --run_name tcga_kirp__concat_zero \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_kirp \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/G/scFoundation_embedding_gene_raw \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/C/L4 \
  --k 1 --k_start 0 --k_end 1 \
  --max_epochs 1 \
  --batch_size 32
```

## 3. Concat - mean
```bash
CUDA_VISIBLE_DEVICES=4 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_kirp \
  --modality modality_concat \
  --concat_impute mean \
  --bag_loss cox_surv \
  --eval_modalities all \
  --exp_group Table2_Baselines \
  --run_name tcga_kirp__concat_mean \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_kirp \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/G/scFoundation_embedding_gene_raw \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/C/L4 \
  --k 1 --k_start 0 --k_end 1 \
  --max_epochs 1 \
  --batch_size 32
```

## 4. MVAE
```bash
CUDA_VISIBLE_DEVICES=2 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_kirp \
  --modality mvae_poe \
  --bag_loss cox_surv \
  --eval_modalities all \
  --exp_group Table2_Baselines \
  --run_name tcga_kirp__mvae \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_kirp \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/G/scFoundation_embedding_gene_raw \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/C/L4 \
  --k 1 --k_start 0 --k_end 1 \
  --max_epochs 1 \
  --batch_size 32
```

## 5. MoPoE
```bash
CUDA_VISIBLE_DEVICES=7 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_kirp \
  --modality mopoe \
  --bag_loss cox_surv \
  --eval_modalities all \
  --exp_group Table2_Baselines \
  --run_name tcga_kirp__mopoe \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_kirp \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/G/scFoundation_embedding_gene_raw \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/C/L4 \
  --k 1 --k_start 0 --k_end 1 \
  --max_epochs 1 \
  --batch_size 32
```

## 6. 主模型 B
```bash
CUDA_VISIBLE_DEVICES=5 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_kirp \
  --modality survtri_poe_vae \
  --poe_variant B \
  --bag_loss cox_surv \
  --eval_modalities all \
  --exp_group Table2_Baselines \
  --run_name tcga_kirp__survtri_poe_vae_B \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_kirp \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/G/scFoundation_embedding_gene_raw \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/C/L4 \
  --k 1 --k_start 0 --k_end 1 \
  --max_epochs 1 \
  --batch_size 32
```

上面这些命令和 Table2 批量 gen 一样：测试开 7 子集，训练 mask 关着。7 个测评共用一次训练。config 对应：

```bash
EVAL_MODALITIES=all
MISSING_MODE=model_gen
```

Table3 训练配比示例，测试 7 子集可以同时开：

```bash
CUDA_VISIBLE_DEVICES=3 /data/fangyuxuan/miniconda3/envs/SurvPGC/bin/python main.py \
  --study tcga_kirp \
  --modality survtri_poe_vae \
  --poe_variant B \
  --bag_loss cox_surv \
  --missing_mode unified_mask_csv \
  --missing_pattern 20,20,20 \
  --missing_seed 1 \
  --eval_modalities all \
  --exp_group Table3_MissingTrain \
  --run_name tcga_kirp__p20_20_20 \
  --split_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv/tcga_kirp \
  --data_root_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/P/uni_v1 \
  --gene_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/G/scFoundation_embedding_gene_raw \
  --clinic_dir /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace/tcga_kirp/C/L4 \
  --k 1 --k_start 0 --k_end 1 \
  --max_epochs 1 \
  --batch_size 32
```

config 对应：

```bash
MISSING_MODE=unified_mask_csv
MISSING_PATTERN=20,20,20
MISSING_SEED=1
EVAL_MODALITIES=all
```
