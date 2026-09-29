# Cfilm hyperparameter config generation

The shared experiment manifest is:

```text
configs/z_exp_gen/Cfilm_Hparam_Eval/cfilm_table1_hparams.csv
```

Its `T1`, `T2`, `T3`, and `T4` columns control config generation independently.
An empty cell disables that table. Accepted enabled values are the column name
itself (`T1`, `T2`, or `T3`), `1`, `true`, `yes`, and `on`. Accepted explicit
disabled values are `0`, `false`, `no`, and `off`.

Examples:

```csv
...,T1,T2,T3,T4
...,,T2,
...,,,
```

The first row generates all four tables, the second generates Table2 only,
and the third generates none.

Generate all selected configs:

```bash
bash configs/z_exp_gen/Cfilm_Hparam_Eval/gen_all.sh
```

Generate one table only:

```bash
bash configs/z_exp_gen/Cfilm_Hparam_Eval/gen_Table1_cfilm_hparams.sh
bash configs/z_exp_gen/Cfilm_Hparam_Eval/gen_Table2_cfilm_hparams.sh
bash configs/z_exp_gen/Cfilm_Hparam_Eval/gen_Table3_cfilm_hparams.sh
```

Table4 不再读取本清单的 `T4` 列（单组超参，来自
`configs/z_exp_gen/cfilm_hparams.sh`）：

```bash
bash configs/z_exp_gen/gen_Table4_Abaltion_Test.sh
```
