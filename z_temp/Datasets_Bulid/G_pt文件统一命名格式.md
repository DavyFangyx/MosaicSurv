# G pt文件统一命名格式

## 推荐最终格式

Gene foundation 患者级特征统一命名为：

```text
TCGA-XX-XXXX.pt
```

示例：

```text
TCGA-3L-AA1B.pt
TCGA-KL-8323.pt
TCGA-BC-A10Q.pt
```


## 命名原则

- 一个 `.pt` 对应一个患者 `case_id`
- 文件名必须能直接包含患者 ID
- 最终训练读取应按患者级而不是 sample UUID 读取

