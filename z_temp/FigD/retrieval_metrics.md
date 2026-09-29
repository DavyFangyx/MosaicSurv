## FigD Retrieval 指标说明

文件：`results_display/FigD_Generated_Heatmaps/<dataset>/fold<k>/<poe_model>/retrieval/retrieval_summary.csv`

对每个 query 病人 i：在指定缺失设定下生成缺失模态 `G'_i` 或 `C'_i`，再去全体真实 `G` / `C` gallery 里按 cosine 相似度检索，看真实的 `G_i` / `C_i` 排第几。rank=1 最好。

## 实验设定列

| 列 | 含义 |
|---|---|
| `eval_subset` | 输入哪些真实模态。`P`=只病理，补 `G'` 和 `C'`；`PC`=病理+clinic，补 `G'`；`PG`=病理+gene，补 `C'`；`C`=只 clinic，补 `G'`；`G`=只 gene，补 `C'` |
| `query_split` | query 来自哪一折划分，这里是 `test` |
| `gallery` | 检索库范围。`all`=该数据集全部真实病例，不只 test |
| `metric` | 相似度。`cosine`=L2 normalize 后点积，越大越近 |
| `query_modality` | 用来检索的补全向量，`G'` 或 `C'` |
| `gallery_modality` | 被检索的真实库，`G` 或 `C` |
| `n_query` | query 数。这里 72=LIHC fold0 test |
| `n_gallery` | 真实库大小 N。Gene=371，Clinic=365 |

## 检索指标

对每个 query，真实匹配的 rank ∈ `[1, N]`。下面都是对 `n_query` 条 query 取平均。

| 列 | 定义 | 怎么读 |
|---|---|---|
| `chance_top1` | `1 / N` | 随机猜中第一名的概率。N=371 时约 0.0027 |
| `top1` | `rank == 1` 的比例 | 补全向量是否把“这个病人”排第一。应远高于 chance |
| `top5` | `rank <= 5` 的比例 | 真实样本是否进前 5 |
| `mrr` | 平均 `1 / rank` | Mean Reciprocal Rank。全中第一名=1；随机大约 `ln(N)/N` |
| `mean_rank` | 平均 rank | 越小越好。随机约 `N/2`（这里约 180–190） |
| `median_rank` | rank 中位数 | 比 mean 更抗极端值 |
| `mean_percentile_rank` | 平均 `100 * (N - rank + 1) / N` | 越高越好。第 1 名≈100%；随机约 50% |

## 读数口径

- 好的补全：`top1` / `top5` / `mrr` 明显高于 chance，`mean_rank` 远小于 `N/2`，`mean_percentile_rank` 接近 100。
- 当前这张表接近随机：`top1≈0`，`mean_rank≈N/2`，`mean_percentile_rank≈50`。说明 `G'`/`C'` 没有保住病人级 identity，更像平均病人，而不是“这个病人”。
