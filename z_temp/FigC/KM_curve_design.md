**Table1 KM Curves (FigC)**

## Summary

对 `results/Table1_Cindex_Main` 下指定实验组和数据集的全部模型绘制 Kaplan-Meier 曲线。不重跑 `.pt`，只读取已有 5-fold OOF 预测 `split_{k}_results.pkl`，按风险中位数分成 High/Low，输出单模型图和一张全部模型大拼图。

```bash
python results_display/scripts/FigC_KM_curves.py \
  --test-dir "L0Test After fixed modal missing" \
  --dataset BRCA

python results_display/scripts/FigC_KM_curves.py \
  --test-dir "L0Test After fixed modal missing" \
  --dataset all \
  --time-unit year
```

## Data And Split

- 结果根目录：`results/Table1_Cindex_Main/{test-dir}`
- 扫描 `{Lx}_{STUDY}_full_model_val` 与 `{Lx}_{STUDY}_poe_model_val` 下全部 `test_result.csv`，再读同目录 `split_{0-4}_results.pkl`
- 模型命名与 Table1 一致：baseline 用目录名，Ours 归一化为 `B` / `B_single` / `C_film` 等
- 合并 5 个互斥 test fold 的 OOF 预测；`event = 1 - censorship`；丢掉 NaN risk 和 `time <= 0`
- 按合并后 risk 中位数二分：High `risk >= median`（红 `#d62728`），Low `risk < median`（绿 `#2ca02c`）
- 中位数并列分到 High；若风险全相同或某一组为空，跳过该模型并 warning

## Plot Spec

- 单模型图：KM 阶梯曲线、删失刻度、左上角图例 `Low risk (n=...)` / `High risk (n=...)`、图内标注 log-rank `p-value`
- 大拼图：该实验组-数据集下全部成功模型，按 Table1 类型顺序 `P, C, G, P+C, P+G, C+G, P+C+G, Ours`
- 拼图共享左上角绿/红图例；每个子图保留模型标题和 p-value
- KM 与两样本 log-rank 用 numpy/scipy 实现，不依赖 `lifelines`
- KM 曲线叠加 Greenwood 95% CI 半透明带；单模型图和大拼图都画
- 时间轴：pkl 内存的是月；`--time-unit month` 原样显示，`year` 将横坐标除以 12

## Outputs

`results_display/FigC_KM_curves/{test-dir}/{dataset}/`

- `per_model/{model}.png`
- `all_models.png`
- `all_models.pdf`
- `km_stats.csv`：`model_type, model, n, n_low, n_high, events_low, events_high, median_risk, logrank_p`

## CLI

| Flag | Default | Meaning |
| --- | --- | --- |
| `--test-dir` | required | 实验组文件夹名，例如 `L0Test After fixed modal missing` |
| `--dataset` | required | `BRCA`、`tcga_brca`，或 `all` 扫描该实验组下实际存在的全部数据集 |
| `--time-unit` | `month` | `month` 或 `year`。pkl 内存的是月；`year` 把横坐标除以 12 |
| `--group-dir` | `Table1_Cindex_Main` | 结果大组 |
| `--results-root` | `results` | 结果根目录 |
| `--output-root` | `results_display/FigC_KM_curves` | 输出根目录 |

未知 `--test-dir` / `--dataset` 直接报错并列出可用项。缺 pkl 或退化分组只跳过该模型，不中断整次运行。
