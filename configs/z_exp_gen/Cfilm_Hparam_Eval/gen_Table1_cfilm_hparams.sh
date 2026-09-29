#!/bin/bash
set -euo pipefail

# Table1 Cfilm-only hyperparam rerun.
# Reads rows enabled by the T1 manifest column and writes
# only survtri_poe_vae_C_film configs. Does not touch gen_ours.sh.
# The checked-in CSV contains 8 existing fixed-alpha/beta rows plus 9
# learning-state rows (alpha, beta, and alpha+beta learning for 3 settings),
# plus 5 optuna-success rows from the C_alphafix_search study
# (trial 15/17/18/19/22: learnable alpha, beta warmup).
# Original Table1 generation remains:
#   bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_ours.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export HPARAM_CSV="${HPARAM_CSV:-$SCRIPT_DIR/cfilm_table1_hparams.csv}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/configs/queue}"
export CLINIC_EXPERIMENT="${CLINIC_EXPERIMENT:-L0}"
export GENE_EXPERIMENT="${GENE_EXPERIMENT:-scFoundation_embedding_cell_norm}"
export WSI_EXPERIMENT="${WSI_EXPERIMENT:-uni_v1}"
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

hparam_csv = Path(os.environ["HPARAM_CSV"])
out_dir = Path(os.environ["OUT_DIR"])
clinic_experiment = os.environ["CLINIC_EXPERIMENT"]
gene_experiment = os.environ["GENE_EXPERIMENT"]
wsi_experiment = os.environ["WSI_EXPERIMENT"]
max_epochs = os.environ["MAX_EPOCHS"]
max_epochs_stage1 = os.environ["MAX_EPOCHS_STAGE1"]
warmup_epochs = os.environ["WARMUP_EPOCHS"]
batch_size_stage1 = os.environ["BATCH_SIZE_STAGE1"]
expected_rows = int(os.environ["EXPECTED_HPARAM_ROWS"])
preset = "survtri_poe_vae_C_film"
studies = ["tcga_brca", "tcga_coad", "tcga_kirc", "tcga_kirp", "tcga_lihc"]

out_dir.mkdir(parents=True, exist_ok=True)

with hparam_csv.open(newline="") as handle:
    all_rows = list(csv.DictReader(handle))

if len(all_rows) != expected_rows:
    raise SystemExit(
        "Expected %s Cfilm hyperparam rows, got %s from %s"
        % (expected_rows, len(all_rows), hparam_csv)
    )

def enabled(row, column):
    value = row.get(column, "").strip().lower()
    if not value:
        return False
    if value in {"1", "true", "yes", "on", column.lower()}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SystemExit("Invalid %s switch %r for run_id=%s" % (column, value, row.get("run_id", "")))

rows = [(index, row) for index, row in enumerate(all_rows, start=1) if enabled(row, "T1")]

created = 0
skipped = 0
total = 0

for study in studies:
    study_tag = study.replace("tcga_", "")
    exp_group = "L0_%s_poe_model_val" % study_tag.upper()
    for seq, row in rows:
        total += 1
        run_id = row["run_id"].strip()
        alphapgc = row["alphapgc"].strip()
        run_name = "%s__%s__cell_norm__%s__%s__%s" % (
            study,
            clinic_experiment,
            wsi_experiment,
            preset,
            run_id,
        )
        fname = "L0_%s_poe_cfilm_hparam__%03d__PCG__%s__%s.conf" % (
            study_tag,
            seq,
            preset,
            run_id,
        )
        target = out_dir / fname
        if target.exists():
            skipped += 1
            continue
        lines = [
            "EXP_GROUP=%s" % exp_group,
            "RUN_NAME=%s" % run_name,
            "PRESET=%s" % preset,
            "STUDY=%s" % study,
            "CLINIC_EXPERIMENT=%s" % clinic_experiment,
            "GENE_EXPERIMENT=%s" % gene_experiment,
            "WSI_EXPERIMENT=%s" % wsi_experiment,
            "BAG_LOSS=cox_surv",
            "BATCH_SIZE=%s" % row["batch_size"],
            "BATCH_SIZE_STAGE1=%s" % batch_size_stage1,
            "MAX_EPOCHS=%s" % max_epochs,
            "MAX_EPOCHS_STAGE1=%s" % max_epochs_stage1,
            "WARMUP_EPOCHS=%s" % warmup_epochs,
            "LR=%s" % row["lr"],
            "REG=%s" % row["reg"],
            "POE_SURV_LAMBDA=%s" % row["poe_surv_lambda"],
            "POE_BETA_TARGET=%s" % row["poe_beta_target"],
            "BETAFIX=%s" % row["betafix"],
            "POE_MODALITY_DROPOUT=%s" % row["poe_modality_dropout"],
            "POE_MMHID=%s" % row["poe_mmhid"],
            "ALPHAFIX=%s" % row["alphafix"],
            "ALPHAPGC=%s" % alphapgc,
            "",
        ]
        target.write_text("\n".join(lines))
        created += 1

print("Generated %s new configs in %s" % (created, out_dir))
print("Total indexed configs this round: %s" % total)
print("Skipped existing configs: %s" % skipped)
print("Hyperparam CSV: %s" % hparam_csv)
print("Enabled T1 rows: %s; disabled rows: %s" % (len(rows), len(all_rows) - len(rows)))
print("Preset: %s" % preset)
print("Studies: %s" % " ".join(studies))
PY
