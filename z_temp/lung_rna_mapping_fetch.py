#!/usr/bin/env python3
"""Fetch lung RNA file->case mapping from GDC API with heavy retries and cache it.

Writes the same mapping to both per-study caches consumed by
dataset_deployment (pipeline.py) and data_tcgal_stats scripts:
    dataset_deployment/mappings/tcga_luad_rna_file_case_map.csv
    dataset_deployment/mappings/tcga_lusc_rna_file_case_map.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MANIFEST = Path("/data/lizhe/Medteam_projects/Lung/gdc_manifest.2026-06-10.161746_rna.txt")
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "dataset_deployment" / "mappings"
CHUNK = 50
MAX_ATTEMPTS = 8
BACKOFF = (2, 4, 8, 16, 32, 64, 120)
TIMEOUT = 45


def fetch_chunk(file_ids: list[str]) -> list[dict]:
    request = urllib.request.Request(
        "https://api.gdc.cancer.gov/files",
        data=json.dumps(
            {
                "filters": {"op": "in", "content": {"field": "files.file_id", "value": file_ids}},
                "fields": "file_id,file_name,cases.submitter_id,cases.case_id",
                "format": "JSON",
                "size": len(file_ids),
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("data", {}).get("hits", [])
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS - 1:
                wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
                print(f"  attempt {attempt + 1}/{MAX_ATTEMPTS} failed ({exc}); retry in {wait}s", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"chunk failed after {MAX_ATTEMPTS} attempts: {last_exc}")


def main() -> int:
    with MANIFEST.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    file_ids = [row["id"] for row in rows if row.get("id") and row.get("filename")]
    print(f"manifest files: {len(file_ids)}", flush=True)

    mapping: dict[str, tuple[str, str]] = {}
    for start in range(0, len(file_ids), CHUNK):
        chunk = file_ids[start : start + CHUNK]
        hits = fetch_chunk(chunk)
        for hit in hits:
            file_name = hit.get("file_name")
            cases = hit.get("cases") or []
            case = cases[0] if cases else {}
            patient_id = case.get("submitter_id")
            case_uuid = case.get("case_id") or ""
            if file_name and patient_id:
                mapping[file_name] = (patient_id, case_uuid)
        print(f"  {min(start + CHUNK, len(file_ids))}/{len(file_ids)} done, mapped={len(mapping)}", flush=True)

    print(f"total mapped: {len(mapping)} / {len(file_ids)}", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["file_name", "case_id", "submitter_id", "case_uuid"]
    for study in ("tcga_luad", "tcga_lusc"):
        cache_path = CACHE_DIR / f"{study}_rna_file_case_map.csv"
        with cache_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for file_name, (patient_id, case_uuid) in sorted(mapping.items()):
                writer.writerow([file_name, patient_id, patient_id, case_uuid])
        print(f"wrote {cache_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
