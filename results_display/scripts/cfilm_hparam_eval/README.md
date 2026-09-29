# Cfilm hyperparameter evaluation

Collect all tables:

```bash
conda activate SurvPGC
python results_display/scripts/cfilm_hparam_eval/collect_all.py
```

Collect one table only:

```bash
python results_display/scripts/cfilm_hparam_eval/table1.py
python results_display/scripts/cfilm_hparam_eval/table2.py
python results_display/scripts/cfilm_hparam_eval/table3.py
```

All commands accept `--results-root`, `--output-root`, and `--hparams`.
Missing experiments are recorded as `incomplete`, so the same command can be
run repeatedly while training results arrive.
Only rows enabled by the matching `T1`, `T2`, or `T3` manifest switch are
collected. A final-ranking row without all three switches is `not_selected`.

Default inputs:

- Table1 Cfilm: `results/L0_<STUDY>_poe_model_val/` (the canonical
  `results/Cfilm_Hparam_Eval/Table1/<run_id>/` layout is also supported).
- Table2 Cfilm: `results/Cfilm_Hparam_Eval/Table2/<run_id>/`.
- Table2 baselines: `results/Table2_Baselines/`.
- Table3 Cfilm: `results/Cfilm_Hparam_Eval/Table3/<run_id>/` (the legacy
  `results/Cfilm_Hparam_Eval/Table3_MissingRate/<run_id>/` layout is also
  supported).
- Table3 HCGN: `results/Table3_MissingRate/`.

Default output: `results_display/Cfilm_Hparam_Eval/`.

Table1 writes `Table1/cfilm_vs_classic_ranking.csv`. It reports each Cfilm
hyperparameter group's five-dataset C-index and its rank after adding that
group to the classic five-dataset Table1 comparison. The `grade` column marks
groups as A when all five ranks are 1, B when all five ranks are at most 3,
and C otherwise. Table1 also writes the same-format A/B checklist to
`Table1/cfilm_table1_hparams_rank_le_3.csv`. Table1 does not write
`parameter_ranking.csv` or `threshold_report.csv`. The classic table
defaults to:
`results_display/Table1_Cindex_Main/L0Test After fixed modal missing/summary_5datasets.csv`.
Use `--classic-summary` to provide a different comparison CSV.

Evaluation rules:

- Table1 applies the existing per-study Optuna C-index floors and requires all
  five studies to pass.
- Table2 evaluates the six missing-modality test subsets against zero, mean,
  HCGN, and the best of those three baselines.
- For each T2-enabled run, Table2 also writes
  `Table2/<run_id>/baseline_ranking.csv`. It ranks Cfilm plus the three
  baselines within each dataset and test subset, and records `vs_hgcn` as
  `↑` or `↓` for values above or below HCGN. T1 and T3 use the same manifest
  filtering rule, so only rows with the corresponding `T1`/`T2`/`T3` switch
  enabled are processed.
- Table3 counts wins against HCGN across 245 comparable cells. Stability is the
  mean C-index standard deviation across the seven training missing patterns
  for each study and test subset.
- For each T3-enabled run, Table3 writes the classic Table3 layout under
  `Table3/<run_id>/`: `train_missing_rate.png` (combined 3x3 grid) plus, for
  each test subset, `{subset}/{subset}.csv`,
  `{subset}/stability_summary.csv`, and `{subset}/train_missing_rate.png`.
- Final ranking includes only parameters that pass Table1 and have complete
  Table1, Table2, and Table3 results.
