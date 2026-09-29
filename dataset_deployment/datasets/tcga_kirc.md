# tcga_kirc

- display_name: `TCGA-KIRC`
- raw_root: `/data/lizhe/Medteam_projects/kindey_cancer_TCGA`
- stats_dir: `data_tcgal_stats/TCGA-KIRC`
- workspace_root: `SurvPGC_Workspace/tcga_kirc`
- clinic_jsons:
  - `clinical/clinical.cart.2026-03-17.json`
- gene_root: `Bulk_RNA/`
- wsi_root: `WSI/`
- case metadata dirs:
  - `clinical/`
- project_filter:
  - `TCGA-KIRC`
- notes:
  - 与 `tcga_kich`、`tcga_kirp` 共享 WSI 源目录 `TCGA-KICH-KIRC-KIRP`，Workspace 按 `TCGA-KIRC_patients.csv` 过滤
