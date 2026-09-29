# Table4: Cfilm 消融实验清单

主模型：`survtri_poe_vae_c_film`（C 联合训练 + FiLM 条件化生存头）。

## A 组：读出层 × 训练方式

| 实验 | preset / modality | 实现 |
|---|---|---|
| c_single | `survtri_poe_vae_C_single` | C 联合训练，单一生存头 |
| c_single_enum | `survtri_poe_vae_C_single_enum` | C 联合训练，7-pattern PoE 枚举，共享单头 |
| c_film_noenum | `survtri_poe_vae_C_film_noenum` | C 联合训练，按当前 `avail` 单次 PoE，不做 pattern 枚举 |
| c_film | `survtri_poe_vae_C_film` | C 联合训练，FiLM 条件化头，主模型 |
| c_multi | `survtri_poe_vae_C_multi` | C 联合训练，7 个 pattern-specific fuse head |

## B 组：训练范式

| 实验 | preset / modality | 实现 |
|---|---|---|
| b_film | `survtri_poe_vae_B_film` | B 两阶段训练，冻结 Enc/PoE/Dec，仅训练 FiLM 头 |
| a_film | `survtri_poe_vae_A_film` | A 训练范式，冻结 backbone，仅训练 FiLM 读出头 |

## C 组：损失项

| 实验 | preset / modality | 实现 |
|---|---|---|
| c_film_kl | `survtri_poe_vae_C_film_kl` | Cfilm，Jeffreys 替换为标准 KL |
| c_film_beta0 | `survtri_poe_vae_C_film_beta0` | Cfilm，去除联合损失中的 J 项（`POE_BETA_TARGET=0`） |
| c_film_surv0 | `survtri_poe_vae_C_film_surv0` | Cfilm，去除联合损失中的 L_surv 项（FiLM 头保留 λL_surv；L_surv 不回传 Encoder/PoE alpha） |

c_film_surv0实现语义：surv_detach_poe=True → multi_pattern_surv_step 里 PoE 计算包在 torch.no_grad() 中。实测梯度：encoder_grad=None，head_grad=有。即 L_surv 仍保留在损失里、仍训练 FiLM 头，只是不回传 encoder/PoE alpha。

## 生成与汇总

默认生成 BRCA、COAD、KIRC、KIRP、LIHC 五个数据集：

```bash
bash configs/z_exp_gen/gen_Table4_Abaltion_Test.sh
```

Cfilm 使用单组公共超参（所有消融 preset 共享），来源为
`configs/z_exp_gen/cfilm_hparams.sh`。

结果目录为 `results/Table4_Abaltion_Test/`，汇总脚本为：

```bash
python3 results_display/scripts/Table4_Abaltion_Test.py
```
