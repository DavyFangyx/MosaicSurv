#!/bin/bash
set -euo pipefail

# Generate Table2 missing-modality configs for rows enabled by the T2 column.
# Baselines stay in results/Table2_Baselines and are not regenerated here.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export HPARAM_CSV="${HPARAM_CSV:-$SCRIPT_DIR/cfilm_table1_hparams.csv}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/configs/queue}"
export RESULTS_BASE="${RESULTS_BASE:-./results/Cfilm_Hparam_Eval/Table2}"
export STUDIES="${STUDIES:-tcga_brca tcga_coad tcga_kirc tcga_kirp tcga_lihc}"
export CLINIC_EXPERIMENT="${CLINIC_EXPERIMENT:-L4}"
export GENE_EXPERIMENT="${GENE_EXPERIMENT:-scFoundation_embedding_gene_raw}"
export WSI_EXPERIMENT="${WSI_EXPERIMENT:-uni_v1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MAX_EPOCHS="${MAX_EPOCHS:-20}"
export MAX_EPOCHS_STAGE1="${MAX_EPOCHS_STAGE1:-10}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
export BATCH_SIZE_STAGE1="${BATCH_SIZE_STAGE1:-128}"
export EXPECTED_HPARAM_ROWS="${EXPECTED_HPARAM_ROWS:-22}"

if [[ ! -f "$HPARAM_CSV" ]]; then
    echo "Missing Cfilm hyperparam CSV: $HPARAM_CSV" >&2
    exit 1
fi

python3 - <<'PY'
import csv
import os
from pathlib import Path


def gene_tag(experiment: str) -> str:
    return {
        "scFoundation_embedding_cell_norm": "cell_norm",
        "scFoundation_embedding_cell_raw": "cell_raw",
        "scFoundation_embedding_gene_norm": "gene_norm",
        "scFoundation_embedding_gene_raw": "gene_raw",
    }.get(experiment, experiment)


hparam_csv = Path(os.environ["HPARAM_CSV"])
out_dir = Path(os.environ["OUT_DIR"])
results_base = os.environ["RESULTS_BASE"]
studies = os.environ["STUDIES"].split()
clinic_experiment = os.environ["CLINIC_EXPERIMENT"]
gene_experiment = os.environ["GENE_EXPERIMENT"]
wsi_experiment = os.environ["WSI_EXPERIMENT"]
wandb_mode = os.environ["WANDB_MODE"]
max_epochs = os.environ["MAX_EPOCHS"]
max_epochs_stage1 = os.environ["MAX_EPOCHS_STAGE1"]
warmup_epochs = os.environ["WARMUP_EPOCHS"]
batch_size_stage1 = os.environ["BATCH_SIZE_STAGE1"]
expected_rows = int(os.environ["EXPECTED_HPARAM_ROWS"])
preset = "survtri_poe_vae_C_film"

with hparam_csv.open(newline="", encoding="utf-8") as handle:
    all_rows = list(csv.DictReader(handle))

if len(all_rows) != expected_rows:
    raise SystemExit(
        "Expected %s Cfilm hyperparam rows, got %s from %s"
        % (expected_rows, len(all_rows), hparam_csv)
    )
if not studies:
    raise SystemExit("STUDIES must contain at least one study")

run_ids = [row["run_id"].strip() for row in all_rows]
if any(not run_id for run_id in run_ids) or len(set(run_ids)) != len(run_ids):
    raise SystemExit("Cfilm run_id values must be non-empty and unique")

def enabled(row, column):
    value = row.get(column, "").strip().lower()
    if not value:
        return False
    if value in {"1", "true", "yes", "on", column.lower()}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SystemExit("Invalid %s switch %r for run_id=%s" % (column, value, row.get("run_id", "")))

rows = [(index, row) for index, row in enumerate(all_rows, start=1) if enabled(row, "T2")]

out_dir.mkdir(parents=True, exist_ok=True)
created = 0
skipped = 0
total = 0
tag = gene_tag(gene_experiment)

for study_index, study in enumerate(studies):
    for row_index, row in rows:
        total += 1
        run_id = row["run_id"].strip()
        target = out_dir / (
            "table2_cfilm_hparam__%04d__%s__%s__%s.conf"
            % (study_index * len(all_rows) + row_index, study, preset, run_id)
        )
        if target.exists():
            skipped += 1
            continue

        lines = [
            "RESULTS_BASE=%s" % results_base,
            "EXP_GROUP=%s" % run_id,
            "RUN_NAME=%s__%s__%s__%s" % (study, clinic_experiment, tag, wsi_experiment),
            "PRESET=%s" % preset,
            "STUDY=%s" % study,
            "CLINIC_EXPERIMENT=%s" % clinic_experiment,
            "GENE_EXPERIMENT=%s" % gene_experiment,
            "WSI_EXPERIMENT=%s" % wsi_experiment,
            "WANDB_MODE=%s" % wandb_mode,
            "BAG_LOSS=cox_surv",
            "BATCH_SIZE=%s" % row["batch_size"],
            "BATCH_SIZE_STAGE1=%s" % batch_size_stage1,
            "MAX_EPOCHS=%s" % max_epochs,
            "MAX_EPOCHS_STAGE1=%s" % max_epochs_stage1,
            "WARMUP_EPOCHS=%s" % warmup_epochs,
            "EVAL_MODALITIES=all",
            "MISSING_MODE=model_gen",
            "LR=%s" % row["lr"],
            "REG=%s" % row["reg"],
            "POE_SURV_LAMBDA=%s" % row["poe_surv_lambda"],
            "POE_BETA_TARGET=%s" % row["poe_beta_target"],
            "BETAFIX=%s" % row["betafix"],
            "POE_MODALITY_DROPOUT=%s" % row["poe_modality_dropout"],
            "POE_MMHID=%s" % row["poe_mmhid"],
            "ALPHAFIX=%s" % row["alphafix"],
            "ALPHAPGC=%s" % row["alphapgc"].strip(),
            "",
        ]
        target.write_text("\n".join(lines), encoding="utf-8")
        created += 1

print("Generated %s new configs in %s" % (created, out_dir))
print("Total indexed configs this round: %s" % total)
print("Skipped existing configs: %s" % skipped)
print("Enabled T2 rows: %s; disabled rows: %s" % (len(rows), len(all_rows) - len(rows)))
print("Training jobs: %s rows x %s studies = %s" % (len(rows), len(studies), total))
print("Results base: %s/<run_id>/" % results_base)
PY
