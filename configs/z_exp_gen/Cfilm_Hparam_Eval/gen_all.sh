#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/gen_Table1_cfilm_hparams.sh"
bash "$SCRIPT_DIR/gen_Table2_cfilm_hparams.sh"
bash "$SCRIPT_DIR/gen_Table3_cfilm_hparams.sh"
