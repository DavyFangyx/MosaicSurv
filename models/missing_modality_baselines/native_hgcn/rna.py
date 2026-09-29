from __future__ import annotations

import json
import joblib
import math
import re
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from dataset_deployment.registry import get_dataset_config, list_enabled_studies
from dataset_deployment.workspace_features import load_patient_allowlist

from .edges import HGCN_PAD_DIM, full_connect_edge_index, pad_vector
from .pack import load_metadata
from .paths import REPO_ROOT, resolve_study_name


STAT_ROW_PREFIXES = ("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous")

HGCN_RNA_FAMILY_NAMES = [
    "Tumor Suppressor Genes",
    "Oncogenes",
    "Protein Kinases",
    "Cell Differentiation Markers",
    "Translocated Cancer Genes",
    "Cytokines and Growth Factors",
    "Homeodomain Proteins",
    "Transcription Factors",
]
HGCN_RNA_FAMILY_COLUMNS = HGCN_RNA_FAMILY_NAMES
DEFAULT_SIGNATURE_CSV = REPO_ROOT / "datasets_csv" / "metadata" / "signatures.csv"
DEFAULT_MSIGDB_CACHE_DIR = REPO_ROOT / "models" / "missing_modality_baselines" / "native_hgcn" / "msigdb_cache"
DEFAULT_MSIGDB_FAMILY_GMT = DEFAULT_MSIGDB_CACHE_DIR / "msigdb_gene_families.gmt"
DEFAULT_MSIGDB_FAMILY_JSON = DEFAULT_MSIGDB_CACHE_DIR / "msigdb_gene_families.json"
HGCN_RNA_EXPERIMENT = "msigdb_gsea_families"
HGCN_RNA_PKL_NAME = "t_rna_fea.pkl"
DEFAULT_MSIGDB_USER_AGENT = "SurvPGC-HGCN/1.0"
DEFAULT_MSIGDB_FAMILY_PAGE_URLS = (
    "https://www.gsea-msigdb.org/gsea/msigdb/gene_families.jsp?ex=1",
    "https://www.gsea-msigdb.org/gsea/msigdb/human/gene_families.jsp",
)
FAMILY_NAME_ALIASES = {
    "tumor suppressor genes": "Tumor Suppressor Genes",
    "tumor suppressors": "Tumor Suppressor Genes",
    "oncogenes": "Oncogenes",
    "protein kinases": "Protein Kinases",
    "cell differentiation markers": "Cell Differentiation Markers",
    "translocated cancer genes": "Translocated Cancer Genes",
    "cytokines and growth factors": "Cytokines and Growth Factors",
    "homeodomain proteins": "Homeodomain Proteins",
    "transcription factors": "Transcription Factors",
    "c4 cgn": "Tumor Suppressor Genes",
    "gencdnf_tumor_suppressor_genes": "Tumor Suppressor Genes",
    "gencdnf_oncogenes": "Oncogenes",
    "gencdnf_protein_kinases": "Protein Kinases",
    "gencdnf_cell_differentiation_markers": "Cell Differentiation Markers",
    "gencdnf_translocated_cancer_genes": "Translocated Cancer Genes",
    "gencdnf_cytokines_and_growth_factors": "Cytokines and Growth Factors",
    "gencdnf_homeodomain_proteins": "Homeodomain Proteins",
    "gencdnf_transcription_factors": "Transcription Factors",
}
COLUMN_FAMILY_RE = re.compile(
    r'">\s*([^<]+?)\s*</a>\s*<form name="columnHeading\d+"[^>]*>\s*'
    r'<input type="hidden" name="geneList" value="([^"]+)"',
    re.I | re.S,
)


def _normalize_family_name(name: str) -> str:
    key = " ".join(str(name).replace("_", " ").replace("-", " ").split()).strip().lower()
    return FAMILY_NAME_ALIASES.get(key, str(name).strip())


def _unique_genes(values: Iterable[str]) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    for value in values:
        gene = str(value).strip()
        if not gene or gene in seen:
            continue
        seen.add(gene)
        genes.append(gene)
    return genes


def parse_gmt_families(text: str) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name = _normalize_family_name(parts[0])
        genes = _unique_genes(parts[2:])
        if name not in HGCN_RNA_FAMILY_NAMES or not genes:
            continue
        families[name] = genes
    return families


def parse_html_families(text: str) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for raw_name, raw_genes in COLUMN_FAMILY_RE.findall(text):
        name = _normalize_family_name(unescape(raw_name).strip())
        genes = _unique_genes(unescape(raw_genes).split(","))
        if name not in HGCN_RNA_FAMILY_NAMES or not genes:
            continue
        current = families.get(name, [])
        if len(genes) >= len(current):
            families[name] = genes
    return families


def families_to_gmt(families: dict[str, list[str]]) -> str:
    lines = []
    for name in HGCN_RNA_FAMILY_NAMES:
        genes = families[name]
        lines.append("\t".join([name.replace(" ", "_"), "MSigDB_gene_families", *genes]))
    return "\n".join(lines) + "\n"


def parse_csv_families(path: Path) -> dict[str, list[str]]:
    table = pd.read_csv(path)
    families: dict[str, list[str]] = {}
    for column in table.columns:
        name = _normalize_family_name(column)
        if name not in HGCN_RNA_FAMILY_NAMES:
            continue
        families[name] = _unique_genes(table[column].dropna().astype(str).tolist())
    return families


def ordered_family_lists(families: dict[str, list[str]]) -> list[list[str]]:
    missing = [name for name in HGCN_RNA_FAMILY_NAMES if name not in families or not families[name]]
    if missing:
        raise KeyError("Missing MSigDB gene-family sets: " + ", ".join(missing))
    return [list(families[name]) for name in HGCN_RNA_FAMILY_NAMES]


def _download_text(url: str, *, timeout: int = 60) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_MSIGDB_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    for encoding in ("utf-8", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace")


def ensure_msigdb_family_gmt(
    gmt_path: str | Path | None = None,
    *,
    urls: Iterable[str] = DEFAULT_MSIGDB_FAMILY_PAGE_URLS,
    force: bool = False,
) -> Path:
    path = Path(gmt_path) if gmt_path is not None else DEFAULT_MSIGDB_FAMILY_GMT
    if path.exists() and path.stat().st_size > 0 and not force:
        try:
            ordered_family_lists(parse_gmt_families(path.read_text(encoding="utf-8")))
            return path
        except KeyError:
            pass
    last_error: Exception | None = None
    for url in urls:
        try:
            html = _download_text(url)
            families = parse_html_families(html)
            ordered_family_lists(families)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(families_to_gmt(families), encoding="utf-8")
            DEFAULT_MSIGDB_FAMILY_JSON.write_text(
                json.dumps({name: families[name] for name in HGCN_RNA_FAMILY_NAMES}, indent=2),
                encoding="utf-8",
            )
            return path
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError, KeyError) as exc:
            last_error = exc
            continue
    raise FileNotFoundError(
        f"Could not download official MSigDB gene-family page into {path}. "
        "Place a GMT at that path and rerun. Last error: "
        f"{last_error}"
    )


def load_gene_families(
    signature_csv: str | Path | None = None,
    *,
    gmt_path: str | Path | None = None,
) -> list[list[str]]:
    if gmt_path is None and signature_csv is None:
        gmt_path = DEFAULT_MSIGDB_FAMILY_GMT
        if not Path(gmt_path).exists():
            gmt_path = ensure_msigdb_family_gmt(gmt_path)
    if gmt_path is not None:
        path = Path(gmt_path)
        families = parse_gmt_families(path.read_text(encoding="utf-8"))
        return ordered_family_lists(families)
    path = Path(signature_csv) if signature_csv is not None else DEFAULT_SIGNATURE_CSV
    return ordered_family_lists(parse_csv_families(path))


def read_tpm_by_gene(tsv_path: Path) -> dict[str, float]:
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    df = df[~df["gene_id"].isin(STAT_ROW_PREFIXES)]
    if "gene_type" in df.columns:
        df = df[df["gene_type"] == "protein_coding"]
    series = df.groupby("gene_name")["tpm_unstranded"].sum()
    return {str(name): float(value) for name, value in series.items()}


def ranked_gene_list(tpm: dict[str, float]) -> list[str]:
    items = [(gene, float(value)) for gene, value in tpm.items() if math.isfinite(float(value))]
    items.sort(key=lambda item: (-item[1], item[0]))
    return [gene for gene, _ in items]


def running_enrichment_score(ranked_genes: list[str], gene_set: set[str]) -> np.ndarray:
    n = len(ranked_genes)
    hits = np.fromiter((gene in gene_set for gene in ranked_genes), dtype=np.float64, count=n)
    n_hit = float(hits.sum())
    if n == 0 or n_hit == 0.0 or n_hit == n:
        return np.zeros((n,), dtype=np.float64)
    hit_step = hits / n_hit
    miss_step = (1.0 - hits) / (n - n_hit)
    return np.cumsum(hit_step - miss_step, dtype=np.float64)


def gsea_contribution_vector(
    ranked_genes: list[str],
    family_genes: list[str],
    *,
    dim: int = HGCN_PAD_DIM,
) -> np.ndarray:
    gene_set = set(family_genes)
    scores = running_enrichment_score(ranked_genes, gene_set)
    values: list[float] = []
    for gene, score in zip(ranked_genes, scores):
        if gene in gene_set:
            values.append(float(score))
    return pad_vector(values, dim=dim)


def family_vectors_from_tpm(
    tpm: dict[str, float],
    families: list[list[str]],
    *,
    dim: int = HGCN_PAD_DIM,
) -> np.ndarray:
    ranked = ranked_gene_list(tpm)
    rows = [gsea_contribution_vector(ranked, genes, dim=dim) for genes in families]
    return np.stack(rows, axis=0).astype(np.float32)


_GENE_LOOKUP_CACHE: dict[str, dict[str, tuple[str, Path]]] = {}


def _gene_lookup(study: str) -> dict[str, tuple[str, Path]]:
    study = resolve_study_name(study)
    cached = _GENE_LOOKUP_CACHE.get(study)
    if cached is not None:
        return cached
    from dataset_deployment.scripts.pipeline import build_gene_lookup

    config = get_dataset_config(study)
    lookup, _ = build_gene_lookup(config, allow_gdc_api=False)
    _GENE_LOOKUP_CACHE[study] = lookup
    return lookup


def resolve_rna_tsv(study: str, case_id: str, rna_file_name: str | None) -> Path | None:
    config = get_dataset_config(resolve_study_name(study))
    gene_root = Path(config.raw.gene_root)
    token = str(rna_file_name or "").strip()
    if token and token.lower() not in {"nan", "none"}:
        token_path = Path(token)
        candidates = []
        if token_path.is_absolute():
            candidates.append(token_path)
        else:
            candidates.append(gene_root / token)
            if token.lower().endswith(".tsv"):
                candidates.append(gene_root / token_path.parent.name / token_path.name)
            else:
                uuid_dir = gene_root / token
                if uuid_dir.is_dir():
                    candidates.extend(sorted(uuid_dir.glob("*.rna_seq.augmented_star_gene_counts.tsv")))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                hits = sorted(candidate.glob("*.rna_seq.augmented_star_gene_counts.tsv"))
                if hits:
                    return hits[0]
    mapped = _gene_lookup(study).get(str(case_id))
    if mapped is None:
        return None
    path = mapped[1]
    return path if path.is_file() else None


def build_case_rna(
    study: str,
    case_id: str,
    rna_file_name: str | None,
    families: list[list[str]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    tsv_path = resolve_rna_tsv(study, case_id, rna_file_name)
    if tsv_path is None:
        return None
    tpm = read_tpm_by_gene(tsv_path)
    x_rna = torch.tensor(family_vectors_from_tpm(tpm, families), dtype=torch.float32)
    if int(x_rna.shape[0]) == 0:
        return None
    return x_rna, full_connect_edge_index(int(x_rna.shape[0]))


def hgcn_rna_dir(study: str, repo_root: str | Path | None = None) -> Path:
    import hgcn_paths

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return hgcn_paths.hgcn_data_root(root) / resolve_study_name(study) / "G" / HGCN_RNA_EXPERIMENT


def generate_study_rna_graphs(
    study: str,
    *,
    repo_root: Path | None = None,
    families: list[list[str]] | None = None,
    gmt_path: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    max_cases: int | None = None,
) -> dict[str, int]:
    study = resolve_study_name(study)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    if families is None:
        families = load_gene_families(gmt_path=gmt_path)
    metadata = load_metadata(study, repo_root=root)
    allowlist = load_patient_allowlist(root / get_dataset_config(study).patient_table_csv)
    case_rows = []
    seen: set[str] = set()
    for _, row in metadata.iterrows():
        case_id = str(row["case_id"])
        if case_id in seen:
            continue
        if allowlist and case_id.upper() not in allowlist:
            continue
        seen.add(case_id)
        case_rows.append((case_id, str(row.get("rna_file_name", "") or "")))
        if max_cases is not None and len(case_rows) >= int(max_cases):
            break

    dest_dir = hgcn_rna_dir(study, root)
    dest_pkl = dest_dir / HGCN_RNA_PKL_NAME
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for leftover in dest_dir.glob("*.pt"):
            leftover.unlink()
    stats = {"wrote": 0, "skipped": 0, "failed": 0, "missing": 0}
    if dest_pkl.exists() and not force and not dry_run:
        stats["skipped"] = 1
        return stats

    t_rna_fea: dict[str, list[list[float]]] = {}
    for case_id, rna_file in case_rows:
        pack = build_case_rna(study, case_id, rna_file, families)
        if pack is None:
            stats["missing"] += 1
            continue
        x_rna, _edge_index = pack
        t_rna_fea[case_id] = x_rna.detach().cpu().tolist()
        stats["wrote"] += 1
    if dry_run:
        return stats
    if not dry_run:
        joblib.dump(t_rna_fea, dest_pkl)
    if not t_rna_fea:
        print(f"[HGCN-RNA] {study} wrote empty {dest_pkl}", flush=True)
    return stats


def generate_rna_graphs(
    studies: Iterable[str] | None = None,
    *,
    repo_root: Path | None = None,
    gmt_path: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    max_cases: int | None = None,
) -> dict[str, dict[str, int]]:
    selected = list(studies) if studies else list(list_enabled_studies())
    families = load_gene_families(gmt_path=gmt_path)
    results: dict[str, dict[str, int]] = {}
    for study in selected:
        results[resolve_study_name(study)] = generate_study_rna_graphs(
            study,
            repo_root=repo_root,
            families=families,
            gmt_path=gmt_path,
            force=force,
            dry_run=dry_run,
            max_cases=max_cases,
        )
    return results
