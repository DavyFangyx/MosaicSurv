# dataset_deployment

这里是多数据集的统一入口，负责“注册 + 构建 + 校验”。

**构建步骤**

1. 数据集注册  
`registry.yaml`：给人看，登记有哪些数据集、标准输出在哪里。  
`registry.py`：给代码读，训练脚本和构建脚本真正使用它。

2. 构建患者统计  
`data_tcgal_stats/<dataset>/`

单次构建全部患者统计：
```bash
python data_tcgal_stats/scripts/build_tcga_modal_stats.py
```

3. 构建训练输入  
`datasets_csv/metadata/`  
`datasets_csv/clinical_data/`  
`datasets_csv/raw_rna_data/combine/`  
`datasets_csv/feature_manifests/`

全数据集：
```bash
python dataset_deployment/scripts/generate_metadata_csv.py --all
python dataset_deployment/scripts/generate_clinical_csv.py --all
python dataset_deployment/scripts/generate_rna_clean_csv.py --all
python dataset_deployment/scripts/generate_feature_manifest.py --all
```

4. 构建划分  
`splits/5foldcv/<study>/`

全数据集：
```bash
python dataset_deployment/scripts/generate_5fold_splits.py --all
```

当前 `generate_5fold_splits.py` 的机制：

- 先读取 `data_tcgal_stats/<dataset>/*_patients.csv` 里的患者全集
- 再联合检查当前训练真实输入是否齐全：
  - `P`：`SurvPGC_Workspace/<study>/P/uni_v1`
  - `C`：`SurvPGC_Workspace/<study>/C/L0~L5`
  - `G foundation`：`SurvPGC_Workspace/<study>/G/*`
  - `G csv`：`datasets_csv/raw_rna_data/combine/<study>/rna_clean.csv`
- 只有 `P/C/G` 全部齐全的患者，才进入最终 split
- 生成：
  - `split_eligibility.csv`：逐患者记录是否合格以及缺失原因
  - `split_cohort_metadata.csv`：最终进入 split 的 metadata 子集
  - `splits_0.csv ~ splits_4.csv`：真正训练使用的五折划分
- 如果某个数据集当前没有任何合格样本：
  - 保留 `split_eligibility.csv`
  - 保留空的 `split_cohort_metadata.csv`
  - 不生成 `splits_0~4.csv`
  - 在 `splits/5foldcv/汇总.csv` 里记录状态
- 划分时按 `censorship` 做事件/删失分层，避免某一折验证集全删失

5. 对齐训练特征  
`SurvPGC_Workspace/<study>/P`  
`SurvPGC_Workspace/<study>/C`  
`SurvPGC_Workspace/<study>/G`

Workspace 生成：
```bash
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/SurvPGC_Workspace
bash build_workspace.sh
```

`tcga_kich` / `tcga_kirc` / `tcga_kirp` 虽然共享同一份 WSI 源目录 `TCGA-KICH-KIRC-KIRP`，但建链时会按各自 `data_tcgal_stats/<dataset>/*_patients.csv` 的 `submitter_id` 过滤，只保留本队列病人。

Workspace 校验：

```bash
python dataset_deployment/scripts/generate_dataset.py --all --validate-only
```

**常用命令**

单数据集：

```bash
python dataset_deployment/scripts/generate_dataset.py --study tcga_coad
```

全数据集：
```bash
python dataset_deployment/scripts/generate_dataset.py --all
```

只校验：

```bash
python dataset_deployment/scripts/generate_dataset.py --study tcga_coad --validate-only
```

`--all` 的意思不是“跑一个示例数据集”，而是：

对 `registry.py` / `registry.yaml` 里所有 `enabled: true` 的数据集，依次执行完整流程：

1. `clinical`
2. `metadata`
3. `rna_clean`
4. `feature_manifest`
5. `5-fold split`
6. `workspace` 校验

**目录分工**

- `registry.yaml`：数据集清单，给人看
- `registry.py`：数据集注册表，给代码读
- `scripts/`：实际生成与校验脚本
- `schemas/`：输出格式约定
- `datasets/`：每个数据集的原始结构说明

旧的 `generate_requirements/` 已移除，统一改用：

```bash
python dataset_deployment/scripts/...
```
