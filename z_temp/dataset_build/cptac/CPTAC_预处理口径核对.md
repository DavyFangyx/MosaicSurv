# CPTAC 预处理与外部前向口径

初始调研：2026-10-14；现行代码与产出核对：2026-11-18。实验入口见 `z_temp/EVE/EVE_design.md`。

## 常规 P/G/C 特征

- TCGA 与 CPTAC 的常规 WSI 使用 UNI v1 预训练模型；原调研记录的提取参数为 20x / 256px / 0 重叠，维度 1024。TCGA 原始切片为 svs，CPTAC 为 IDC derived SM dcm。
- `features_uni_v1_ft` 等队友分类项目的微调特征不属于本实验输入。
- 实际特征路径由源模型 run.log 决定并映射至目标队列，不能按目录名推定所有模型使用同一特征。

## HGCN 原生特征

- P 使用 KimiaNet 原生切块编码，G 使用 GSEA 通路，C 使用原生临床图编码，不是 UNI/scFoundation 的替代路径。
- `models/missing_modality_baselines/native_hgcn/wsi.py` 已索引 dcm，`dcm_slide.py` 提供读取适配；旧“只支持 svs、无法生成”的状态已过期。
- CPTAC-COAD/BRCA 的图 pack 已用于 `results_display/scripts/EVE_HGCN_Forward.py`；原报告记录为 100/95 例。
- KimiaNet tiling、组织过滤与编码仍走原生管线；具体输入尺寸及资产以生成配置和 pack 为准。

## 生存与划分

- 终点为 OS。TCGA 从 GDC 的死亡时间或末次随访提取；CPTAC-COAD 用 OS_MONTHS/OS_STATUS，BRCA 用 OS_days/OS_event。
- metadata 的 survival_months 为月，天按 30.44 换算；COAD 原 OS_MONTHS 曾经天作中转换算。
- CPTAC registry 使用 `split_mode=external_test`：五份 splits 对应五个源模型，外部训练列为空，评估队列相同，不是五个独立外部抽样。

## 源 bins 与 scaler

`EVE_External_Forward.py::process_model_dir()` 从源 TCGA 工厂读取 bins，通过 `bins_override` 传给目标工厂；目标时间范围仅调整端点，内部边界沿用源队列。各折使用对应源训练集 MinMaxScaler。

`datasets/dataset_survival.py` 普通加载仍可自行 qcut，但 EVE 外部前向已经使用 override；旧“源 bins 冻结未实施、外部实验未接入”的描述不再适用。

## 可用范围

CPTAC-COAD 为 100 例/8 事件，BRCA 为 95 例/2 事件；低事件数限制估计精度。EVE1/2 已有产出，诊断和结果见 `z_temp/history/EVE/EVE_实验结果与诊断.md`。本次未改动外部影像、数据、分箱或实验代码。
