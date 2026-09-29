from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_STUDY = "tcga_coad"
DEFAULT_WSI_EXPERIMENT = "uni_v1"
DEFAULT_CLINIC_EXPERIMENT = "L4"
DEFAULT_GENE_EXPERIMENT = "scFoundation_embedding_gene_raw"


@dataclass(frozen=True)
class RawDatasetConfig:
    raw_root: str
    gene_root: str
    wsi_root: str
    clinical_jsons: tuple[str, ...]
    case_dirs: tuple[str, ...]
    project_filter: tuple[str, ...] = ()
    rna_manifest: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetConfig:
    study: str
    display_name: str
    stats_dir: str
    metadata_csv: str
    clinical_csv: str
    rna_dir: str
    feature_manifest_csv: str
    split_dir: str
    workspace_root: str
    enabled: bool
    raw: RawDatasetConfig

    @property
    def study_subtype(self) -> str:
        return self.study.replace("tcga_", "", 1)

    @property
    def patient_table_csv(self) -> str:
        return f"{self.stats_dir}/{self.display_name}_patients.csv"


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "tcga_brca": DatasetConfig(
        study="tcga_brca",
        display_name="TCGA-BRCA",
        stats_dir="data_tcgal_stats/TCGA-BRCA",
        metadata_csv="datasets_csv/metadata/tcga_brca.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_brca_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/brca",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_brca.csv",
        split_dir="splits/5foldcv/tcga_brca",
        workspace_root="SurvPGC_Workspace/tcga_brca",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/TCGA-BRCA",
            gene_root="/data/lizhe/Medteam_projects/TCGA-BRCA/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/TCGA-BRCA/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/TCGA-BRCA/Bulk_RNA_case/clinical.cart.2026-06-29.json",
                "/data/lizhe/Medteam_projects/TCGA-BRCA/WSI_case/clinical.cart.2026-06-29.json",
                "/data/lizhe/Medteam_projects/TCGA-BRCA/Mutation_case/clinical.cart.2026-06-29.json",
                "/data/lizhe/Medteam_projects/TCGA-BRCA/Mythylation_case/clinical.cart.2026-06-29.json",
                "/data/lizhe/Medteam_projects/TCGA-BRCA/RPPA_case/clinical.cart.2026-06-29.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/TCGA-BRCA/Bulk_RNA_case",
                "/data/lizhe/Medteam_projects/TCGA-BRCA/WSI_case",
                "/data/lizhe/Medteam_projects/TCGA-BRCA/Mutation_case",
                "/data/lizhe/Medteam_projects/TCGA-BRCA/Mythylation_case",
                "/data/lizhe/Medteam_projects/TCGA-BRCA/RPPA_case",
            ),
            notes=(
                "BRCA uses *_case directories instead of *_cases.",
                "Mutation / Mythylation / RPPA case metadata also carry clinical JSON copies.",
            ),
        ),
    ),
    "tcga_coad": DatasetConfig(
        study="tcga_coad",
        display_name="TCGA-COAD",
        stats_dir="data_tcgal_stats/TCGA-COAD",
        metadata_csv="datasets_csv/metadata/tcga_coad.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_coad_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/coad",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_coad.csv",
        split_dir="splits/5foldcv/tcga_coad",
        workspace_root="SurvPGC_Workspace/tcga_coad",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/TCGA-COAD",
            gene_root="/data/lizhe/Medteam_projects/TCGA-COAD/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/TCGA-COAD/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/TCGA-COAD/Bulk_RNA_cases/clinical.cart.2026-06-18.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/TCGA-COAD/Bulk_RNA_cases",
                "/data/lizhe/Medteam_projects/TCGA-COAD/WSI_cases",
            ),
            notes=(
                "COAD stores the observed clinical JSON under Bulk_RNA_cases.",
            ),
        ),
    ),
    "tcga_kich": DatasetConfig(
        study="tcga_kich",
        display_name="TCGA-KICH",
        stats_dir="data_tcgal_stats/TCGA-KICH",
        metadata_csv="datasets_csv/metadata/tcga_kich.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_kich_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/kich",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_kich.csv",
        split_dir="splits/5foldcv/tcga_kich",
        workspace_root="SurvPGC_Workspace/tcga_kich",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA",
            gene_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical/clinical.cart.2026-03-17.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical",
            ),
            project_filter=("TCGA-KICH",),
            rna_manifest="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/gdc_manifest.2026-03-18.111422_Bulk.txt",
            notes=(
                "Kidney datasets share the same raw root and are split by project.project_id.",
                "Shared WSI source TCGA-KICH-KIRC-KIRP is filtered by TCGA-KICH_patients.csv.",
            ),
        ),
    ),
    "tcga_kirc": DatasetConfig(
        study="tcga_kirc",
        display_name="TCGA-KIRC",
        stats_dir="data_tcgal_stats/TCGA-KIRC",
        metadata_csv="datasets_csv/metadata/tcga_kirc.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_kirc_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/kirc",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_kirc.csv",
        split_dir="splits/5foldcv/tcga_kirc",
        workspace_root="SurvPGC_Workspace/tcga_kirc",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA",
            gene_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical/clinical.cart.2026-03-17.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical",
            ),
            project_filter=("TCGA-KIRC",),
            rna_manifest="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/gdc_manifest.2026-03-18.111422_Bulk.txt",
            notes=(
                "Kidney datasets share the same raw root and are split by project.project_id.",
                "Shared WSI source TCGA-KICH-KIRC-KIRP is filtered by TCGA-KIRC_patients.csv.",
            ),
        ),
    ),
    "tcga_kirp": DatasetConfig(
        study="tcga_kirp",
        display_name="TCGA-KIRP",
        stats_dir="data_tcgal_stats/TCGA-KIRP",
        metadata_csv="datasets_csv/metadata/tcga_kirp.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_kirp_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/kirp",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_kirp.csv",
        split_dir="splits/5foldcv/tcga_kirp",
        workspace_root="SurvPGC_Workspace/tcga_kirp",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA",
            gene_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical/clinical.cart.2026-03-17.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical",
            ),
            project_filter=("TCGA-KIRP",),
            rna_manifest="/data/lizhe/Medteam_projects/kindey_cancer_TCGA/gdc_manifest.2026-03-18.111422_Bulk.txt",
            notes=(
                "Kidney datasets share the same raw root and are split by project.project_id.",
                "Shared WSI source TCGA-KICH-KIRC-KIRP is filtered by TCGA-KIRP_patients.csv.",
            ),
        ),
    ),
    "tcga_lihc": DatasetConfig(
        study="tcga_lihc",
        display_name="TCGA_LIHC",
        stats_dir="data_tcgal_stats/TCGA_LIHC",
        metadata_csv="datasets_csv/metadata/tcga_lihc.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_lihc_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/lihc",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_lihc.csv",
        split_dir="splits/5foldcv/tcga_lihc",
        workspace_root="SurvPGC_Workspace/tcga_lihc",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/TCGA_LIHC",
            gene_root="/data/lizhe/Medteam_projects/TCGA_LIHC/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/TCGA_LIHC/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/TCGA_LIHC/clinical/clinical.cart.2026-06-01.json",
                "/data/lizhe/Medteam_projects/TCGA_LIHC/clinical.cart.2026-03-26.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/TCGA_LIHC/clinical",
                "/data/lizhe/Medteam_projects/TCGA_LIHC",
            ),
            rna_manifest="/data/lizhe/Medteam_projects/TCGA_LIHC/gdc_manifest.2026-06-10.160348_rna.txt",
            notes=(
                "LIHC keeps clinic files in both clinical/ and the dataset root.",
                "Local RNA file-to-case metadata is incomplete; GDC API lookup may be needed.",
            ),
        ),
    ),
    "tcga_prad": DatasetConfig(
        study="tcga_prad",
        display_name="TCGA-PRAD",
        stats_dir="data_tcgal_stats/TCGA-PRAD",
        metadata_csv="datasets_csv/metadata/tcga_prad.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_prad_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/prad",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_prad.csv",
        split_dir="splits/5foldcv/tcga_prad",
        workspace_root="SurvPGC_Workspace/tcga_prad",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/TCGA-PRAD",
            gene_root="/data/lizhe/Medteam_projects/TCGA-PRAD/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/TCGA-PRAD/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/TCGA-PRAD/Bulk_RNA_cases/clinical.cart.2026-06-18.json",
                "/data/lizhe/Medteam_projects/TCGA-PRAD/WSI_cases/clinical.cart.2026-06-18.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/TCGA-PRAD/Bulk_RNA_cases",
                "/data/lizhe/Medteam_projects/TCGA-PRAD/WSI_cases",
            ),
        ),
    ),
    "tcga_read": DatasetConfig(
        study="tcga_read",
        display_name="TCGA-READ",
        stats_dir="data_tcgal_stats/TCGA-READ",
        metadata_csv="datasets_csv/metadata/tcga_read.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_read_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/read",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_read.csv",
        split_dir="splits/5foldcv/tcga_read",
        workspace_root="SurvPGC_Workspace/tcga_read",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/TCGA-READ",
            gene_root="/data/lizhe/Medteam_projects/TCGA-READ/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/TCGA-READ/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/TCGA-READ/Bulk_RNA_cases/clinical.cart.2026-06-21.json",
                "/data/lizhe/Medteam_projects/TCGA-READ/WSI_cases/clinical.cart.2026-06-21.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/TCGA-READ/Bulk_RNA_cases",
                "/data/lizhe/Medteam_projects/TCGA-READ/WSI_cases",
            ),
        ),
    ),
    "tcga_stad": DatasetConfig(
        study="tcga_stad",
        display_name="TCGA-STAD",
        stats_dir="data_tcgal_stats/TCGA-STAD",
        metadata_csv="datasets_csv/metadata/tcga_stad.csv",
        clinical_csv="datasets_csv/clinical_data/tcga_stad_clinical.csv",
        rna_dir="datasets_csv/raw_rna_data/combine/stad",
        feature_manifest_csv="datasets_csv/feature_manifests/tcga_stad.csv",
        split_dir="splits/5foldcv/tcga_stad",
        workspace_root="SurvPGC_Workspace/tcga_stad",
        enabled=True,
        raw=RawDatasetConfig(
            raw_root="/data/lizhe/Medteam_projects/TCGA-STAD",
            gene_root="/data/lizhe/Medteam_projects/TCGA-STAD/Bulk_RNA",
            wsi_root="/data/lizhe/Medteam_projects/TCGA-STAD/WSI",
            clinical_jsons=(
                "/data/lizhe/Medteam_projects/TCGA-STAD/Bulk_RNA_cases/clinical.cart.2026-06-21.json",
                "/data/lizhe/Medteam_projects/TCGA-STAD/WSI_cases/clinical.cart.2026-06-21.json",
            ),
            case_dirs=(
                "/data/lizhe/Medteam_projects/TCGA-STAD/Bulk_RNA_cases",
                "/data/lizhe/Medteam_projects/TCGA-STAD/WSI_cases",
            ),
            notes=(
                "Current Bulk_RNA looks like a mirrored WSI tree; gene outputs may legitimately be empty.",
            ),
        ),
    ),
}


def list_enabled_studies() -> list[str]:
    return [study for study, config in DATASET_CONFIGS.items() if config.enabled]


def get_dataset_config(study: str) -> DatasetConfig:
    try:
        return DATASET_CONFIGS[study]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown study '{study}'. Expected one of: {valid}") from exc


def infer_standard_paths(
    study: str,
    repo_root: str | Path = ".",
    *,
    which_splits: str = "5foldcv",
    type_of_path: str = "combine",
    wsi_experiment: str = DEFAULT_WSI_EXPERIMENT,
    clinic_experiment: str = DEFAULT_CLINIC_EXPERIMENT,
    gene_experiment: str = DEFAULT_GENE_EXPERIMENT,
) -> dict[str, Path]:
    config = get_dataset_config(study)
    root = Path(repo_root).resolve()
    workspace_root = root / config.workspace_root
    return {
        "label_file": root / config.metadata_csv,
        "clinical_file": root / config.clinical_csv,
        "omics_dir": root / config.rna_dir.replace("/combine/", f"/{type_of_path}/"),
        "feature_manifest_csv": root / config.feature_manifest_csv,
        "split_dir": root / "splits" / which_splits / config.study,
        "workspace_root": workspace_root,
        "data_root_dir": workspace_root / "P" / wsi_experiment,
        "clinic_dir": workspace_root / "C" / clinic_experiment,
        "gene_dir": workspace_root / "G" / gene_experiment,
        "patient_table_csv": root / config.patient_table_csv,
    }
