# tcga_kirp

- display_name: `TCGA-KIRP`
- raw_root: `/data/lizhe/Medteam_projects/kindey_cancer_TCGA`
- stats_dir: `data_tcgal_stats/TCGA-KIRP`
- workspace_root: `SurvPGC_Workspace/tcga_kirp`
- clinic_jsons:
  - `clinical/clinical.cart.2026-03-17.json`
- gene_root: `Bulk_RNA/`
- wsi_root: `WSI/`
- case metadata dirs:
  - `clinical/`
- project_filter:
  - `TCGA-KIRP`
- notes:
  - 与 `tcga_kich`、`tcga_kirc` 共享 WSI 源目录 `TCGA-KICH-KIRC-KIRP`，Workspace 按 `TCGA-KIRP_patients.csv` 过滤
