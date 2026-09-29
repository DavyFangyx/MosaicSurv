# 主训练 vs Optuna：协议差异与风险清单

目标：判断 Optuna 最优超参能否直接指导 Table1 主训练结果。
代码依据：`main.py`、`main_tune_optuna.py`、`utils/optuna_utils.py`、`utils/core_utils.py`、`bg_tune.sh`、`configs/z_exp_gen/Table1_cindex_main/gen_ours/_common.bash`。

两条路径最终都调用 `_train_val_test() -> _step()`。Stage2 选点阈值已共享；不共享的是 trial 目标、剪枝、以及实际入口默认配置。

## 风险清单

| ID | 严重度 | 差异 | 主训练 | Optuna | 为何会误导主结果 |
|---|---|---|---|---|---|
| R1 | 已对齐 | Stage2 最早可保存 epoch | 普通模型默认 `epoch >= 10`；缺失模态 baseline 为 `0` | 不再覆盖 `save_best_from_epoch`，走 `_step()` 与主训练相同的默认阈值 | 调参选点窗口已与主训练一致。旧 study 若仍按 epoch 0 起存，不能和新协议混比。 |
| R2 | 高 | 选超参的目标指标 | Table1 汇总 `test_cindex` 的 5 折均值 | 选 trial 用 `mean(best val_cindex)`；每个 fold 虽会测 test，但 test 不进入 Optuna objective | 验证最优 trial 不等于测试最优配置。直接把 Optuna best_trial 当 Table1 配置，会把验证泄漏/过拟合配置搬进主结果。 |
| R3 | 高 | 调参只在单数据集完成 | Table1 主结果覆盖 BRCA/COAD/KIRC/KIRP/LIHC 等 | 当前调参入口和 `bg_tune.sh` 默认路径锁在 `tcga_lihc` | LIHC 上最优的 lr/beta/dropout/batch 不一定迁移到其他癌种。未做跨 study 复验就把同一组超参写进 `_common.bash`，会系统性偏置非 LIHC 结果。 |
| R4 | 中高 | epoch 预算经常不一致 | Table1 ours 默认 `max_epochs=20`、`max_epochs_stage1=10`、`warmup_epochs=3` | argparse 默认 `max_epochs=20`、`max_epochs_stage1=40`；`bg_tune.sh` 内部默认 stage1=40，注释示例却是 B=10/25、C=20 | 更长或更短的预算会改变最佳超参。把短预算搜到的 lr 放到更长主训练，或反过来，都不能当同一协议。 |
| R5 | 已对齐 | Model C 的 stage2 batch | Table1 Model C 固定 `BATCH_SIZE=16`；B 固定 128 | Model C 不再搜索 `batch_size`，直接继承命令行/`bg_tune.sh` 的 `--batch_size`；B 仍不搜索 stage2 batch，只搜索 `batch_size_stage1` | 新 study 里 C 的 lr/reg/lambda 不再和搜索到的 batch 耦合。旧 study 若搜过 `batch_size`，不能直接搬回固定 batch=16 的主训练。 |
| R6 | 中 | MedianPruner 提前终止 | 主训练跑满 `max_epochs`，再从允许窗口里选最佳 checkpoint | 默认 `optuna_pruner=median`，每个 epoch 用当前 `val_cindex` report；`n_warmup_steps=3` 后可剪枝。剪枝发生在保存判断之前 | 被剪枝 trial 没有完整训练轨迹，不能和主训练的“跑满再选点”对比。`mean_cv` 下 step 还是 fold 展开的全局 epoch，前几折的早期分数就可能杀掉整个 trial。 |
| R7 | 中 | 单折调参 vs 5 折主结果 | 主结果固定 5-fold test 均值 | `--optuna_fold_mode single` 只优化 `--optuna_fold`（默认 0） | 单折验证最优会过拟合该折划分，不能代表 5 折主表。当前默认是 `mean_cv`，只有显式改成 `single` 时触发。 |
| R8 | 中 | 默认 study 名带 `fold{optuna_fold}` | 主训练每个 study/run 目录独立 | 即使 `mean_cv`，默认 study 名仍是 `{study}_{modality}_{variant}[_betafix]_alphafix_search_fold{optuna_fold}`，且 `load_if_exists=True` | 不同 fold_mode / 旧 study 可能写进同一个 sqlite。看起来像“同一组调参结果”，实际混了不同协议的 trial。 |
| R9 | 中 | Optuna 最优 checkpoint 不能直接当主结果 | 主训练按主协议重新训练，再汇总 test | 每个 trial 已经写出 fold checkpoint 和 test 分数，但这些分数不是 Table1 口径 | 用 trial 目录里的 test_cindex 填主表，等于用“验证选 trial”的结果冒充主训练。必须按主协议重跑。 |
| R10 | 低/易混 | Stage1 选点规则不同，且 Optuna 不覆盖它 | A/B 的 Stage1 按验证 VAE loss 最小保存 | 同左。`save_best_from_epoch` 只作用于 Stage2 | 不要把 Stage2 C-index 选点规则套到 Stage1。Model B 的 `--poe_scan_stage1_ckpts` 也不是 Optuna 机制。 |
| R11 | 低/配置 | 入口默认超参与 Table1 固定值不同 | Table1 已把 B/C 最优值写死：B 为 lr=6e-4, lr_stage1=8e-5, reg=7e-5, beta=0.0164, dropout=0.25, mmhid=128, decoder=512；C 为 batch=16, lr=1.49e-4, reg=1.25e-5, beta=0.268, lambda=0.361, betafix=true | 搜索空间更宽，且 argparse 默认 beta=1.0、dropout=0.2、mmhid=256、lr=5e-4 | 这不是协议 bug，但是把“未调参默认值”和“Table1 固定超参”混比会误判调参收益。 |

## 1. Stage2 最优 epoch 机制

共同逻辑在 `utils/core_utils.py::_step()`：

```text
best_score = 0
for epoch = 0 .. max_epochs - 1:
    train(epoch)
    val_cindex = validate()
    maybe_optuna_report_and_prune(val_cindex)
    if epoch >= best_start_epoch and val_cindex >= best_score:
        best_score = val_cindex
        save s_<fold>_checkpoint.pt
reload checkpoint, then test
```

- epoch 是零基编号。日志 `Epoch: 10 is the best.` 表示第 11 次循环。
- 平局用 `>=`：同分时后面的 epoch 覆盖前面。
- 测试加载的是验证选出的 checkpoint，不是 last epoch。
- `val_result_fold<fold>.csv` 里的 `val_cindex` 是这个最佳分数。

主训练阈值：

```python
if args.modality in MISSING_MODALITY_BASELINES:  # modality_concat / mvae_poe / mopoe
    best_start_epoch = getattr(args, "save_best_from_epoch", 0)
else:
    best_start_epoch = getattr(args, "save_best_from_epoch", 10)
```

`build_trial_args()` 不再覆盖 `save_best_from_epoch`。因此 `survtri_poe_vae` 的 Optuna trial 与主训练一样：默认忽略 epoch 0-9，从 epoch 10 起按验证 C-index 选 checkpoint。缺失模态 baseline 仍默认从 epoch 0 起存。普通模型若 `max_epochs <= 10`，可能根本不存 checkpoint。

## 2. Optuna 怎么评价一个 trial

`main_tune_optuna.py`：

1. 采样超参，复制 args；Stage2 选点沿用主训练的 `save_best_from_epoch` 默认值。
2. `single`：只跑 `--optuna_fold`；`mean_cv`：跑 `_get_start_end(args)` 得到的 folds。
3. 每个 fold 完整调用 `_train_val_test()`，因此内部会测 test，但 objective 只读 `val_result_fold<fold>.csv`。
4. 返回 `mean(best_val_cindex_of_each_fold)`。
5. study direction 默认 maximize。

所以 Optuna 的“最好”是验证 C-index 最好的超参组合，不是测试最好，也不是主训练协议下重训后最好。

## 3. 剪枝时序

每个 epoch：

```text
validate -> trial.report(val_cindex, step=epoch_log_step) -> maybe prune -> save best
```

- `epoch_log_step = fold_index * max_epochs + epoch`
- 默认 MedianPruner：`n_startup_trials=5`，`n_warmup_steps=3`
- 剪枝异常抛在保存之前，该 epoch 不会成为 checkpoint
- 被剪枝 trial 没有完整 fold 结果，不能拿去和主训练比

`bg_tune.sh` 注释示例用 `--optuna_pruner none`，但脚本内部默认仍是 `median`。命令行后置参数可以覆盖。核对实际 study 时不要只看注释。

## 4. 当前实际入口差异

Table1 ours 生成器 `_common.bash`：

- 数据：`WSI=uni_v1`，`GENE=scFoundation_embedding_cell_norm`，`CLINIC=L0`
- 预算：`MAX_EPOCHS=20`，`MAX_EPOCHS_STAGE1=10`，`WARMUP_EPOCHS=3`
- B 的 stage2 batch=128；C 的 stage2 batch=16
- B/C 超参已写死为上面表格中的固定值
- 选点：走 `main.py`，`survtri_poe_vae` 默认 `epoch >= 10`
- 汇总：`Table1_Cindex_Main.py` 读 `test_cindex`

`bg_tune.sh`：

- 同样覆盖成 LIHC 的 L0 / cell_norm / uni_v1
- 但 `MAX_EPOCHS_STAGE1=40`，`MAX_EPOCHS=20`
- 只默认 LIHC；没有 Table1 的多癌种循环
- 默认 pruner=median

直接跑 `python main_tune_optuna.py` 还不覆盖数据路径，会落到 argparse 默认的 L4 + gene_raw。

## 5. AI 使用规则

1. 不能把 Optuna `best_value` 或 trial 目录中的 test_cindex 直接填进 Table1。
2. 要把 best_trial 搬回主结果，必须按主协议重跑：Table1 的数据源、epoch 预算、对应模型的固定 batch、目标癌种的 5 折。Stage2 选点已与主训练对齐，但仍不能用 trial 目录里的 test 分数填表。
3. 若 best_trial 的 `max_epochs` / 数据路径与 Table1 不同，或来自仍搜索 `batch_size` 的旧 C study，先当协议不一致，再谈超参好坏。
4. 单数据集调参结果默认只对 LIHC 有候选资格；其他癌种需要单独验证。
5. Stage1 仍按验证 VAE loss 最小选点，和 Stage2 的 C-index 选点无关。

