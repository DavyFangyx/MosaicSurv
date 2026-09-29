#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "${STUDIES:-}" ]]; then
    read -r -a studies <<< "$STUDIES"
else
    studies=(tcga_brca tcga_coad tcga_kirc tcga_kirp tcga_lihc)
fi

for study in "${studies[@]}"; do
    case "$study" in
        brca|tcga_brca) script="gen_tcga_brca_ablation_test.sh" ;;
        coad|tcga_coad) script="gen_tcga_coad_ablation_test.sh" ;;
        kich|tcga_kich) script="gen_tcga_kich_ablation_test.sh" ;;
        kirc|tcga_kirc) script="gen_tcga_kirc_ablation_test.sh" ;;
        kirp|tcga_kirp) script="gen_tcga_kirp_ablation_test.sh" ;;
        lihc|tcga_lihc) script="gen_tcga_lihc_ablation_test.sh" ;;
        prad|tcga_prad) script="gen_tcga_prad_ablation_test.sh" ;;
        read|tcga_read) script="gen_tcga_read_ablation_test.sh" ;;
        *) echo "Unsupported Table4 study: $study" >&2; exit 2 ;;
    esac
    bash "$SCRIPT_DIR/$script"
done
