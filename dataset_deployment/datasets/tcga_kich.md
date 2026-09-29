# tcga_kich

- display_name: `TCGA-KICH`
- raw_root: `/data/lizhe/Medteam_projects/kindey_cancer_TCGA`
- stats_dir: `data_tcgal_stats/TCGA-KICH`
- workspace_root: `SurvPGC_Workspace/tcga_kich`
- clinic_jsons:
  - `clinical/clinical.cart.2026-03-17.json`
- gene_root: `Bulk_RNA/`
- wsi_root: `WSI/`
- case metadata dirs:
  - `clinical/`
- project_filter:
  - `TCGA-KICH`
- notes:
  - 与 `tcga_kirc`、`tcga_kirp` 共享同一原始根目录
  - 与 `tcga_kirc`、`tcga_kirp` 共享 WSI 源目录 `TCGA-KICH-KIRC-KIRP`，Workspace 按 `TCGA-KICH_patients.csv` 过滤
