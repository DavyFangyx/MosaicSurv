# C pt文件统一命名格式

## 推荐最终格式

Clinic 患者级特征统一命名为：

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
- 文件名必须能直接看出患者 ID
- 不要保留 slide 级后缀
- 不要保留 UUID 作为主文件名
