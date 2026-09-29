"""MosaicSurv 家族的集中命名映射（display 层唯一来源）。

- ``FORMAL_NAMES``：注册键 / 结果目录名 → 论文表格里的正式名（Table1-4、FigC 共用）。
- ``LEGACY_ALIASES``：Cfilm 时代的旧拼写 → 新键。官方结果目录已随代码改名，
  但遗留区 ``results/Cfilm_Hparam_Eval/``（保持原样、未改名）以及历史缓存
  CSV 里仍是旧拼写，display 脚本读这些遗留数据时靠别名归一化。
- ``display_name()``：任意拼写 → 展示名；映射内的模型给论文正式名，
  映射外（如遗留 preset ``survtri_poe_vae_C`` / ``survtri_poe_vae_B_single``）
  保持原来的 CamelCase 规则，避免 display 输出出现未登记的键。
"""

from __future__ import annotations

MAIN_MODEL = "mosaic_surv"

# 新注册键 → 论文正式名（顺序与 configs/presets.sh 的 C 系 case 块一致）
FORMAL_NAMES: dict[str, str] = {
    "mosaic_surv": "Mosaic-Surv (Ours)",
    "mosaic_surv_noenum": "Mosaic-Surv w/o Pattern Enumeration",
    "mosaic_surv_single": "Mosaic-Surv w/ Single Head",
    "mosaic_surv_single_enum": "Mosaic-Surv w/ Single Head + Enumeration",
    "mosaic_surv_multi": "Mosaic-Surv w/ Multi-Head",
    "mosaic_surv_twostage": "Mosaic-Surv (Two-Stage)",
    "mosaic_surv_frozen": "Mosaic-Surv (Pretrained VAE, Frozen Backbone)",
    "mosaic_surv_kl": "Mosaic-Surv w/ KL",
    "mosaic_surv_nojeffreys": "Mosaic-Surv w/o Jeffreys",
    "mosaic_surv_detached": "Mosaic-Surv w/ Detached L_surv",
}

# 旧拼写（Cfilm 时代，全部小写做键）→ 新注册键
LEGACY_ALIASES: dict[str, str] = {
    "survtri_poe_vae_c_film": MAIN_MODEL,
    "c_film": MAIN_MODEL,
    "survtri_poe_vae_c_film_noenum": "mosaic_surv_noenum",
    "c_film_noenum": "mosaic_surv_noenum",
    "survtri_poe_vae_c_single": "mosaic_surv_single",
    "c_single": "mosaic_surv_single",
    "survtri_poe_vae_c_single_enum": "mosaic_surv_single_enum",
    "c_single_enum": "mosaic_surv_single_enum",
    "survtri_poe_vae_c_multi": "mosaic_surv_multi",
    "c_multi": "mosaic_surv_multi",
    "survtri_poe_vae_b_film": "mosaic_surv_twostage",
    "b_film": "mosaic_surv_twostage",
    "survtri_poe_vae_a_film": "mosaic_surv_frozen",
    "a_film": "mosaic_surv_frozen",
    "survtri_poe_vae_c_film_kl": "mosaic_surv_kl",
    "c_film_kl": "mosaic_surv_kl",
    "survtri_poe_vae_c_film_beta0": "mosaic_surv_nojeffreys",
    "c_film_beta0": "mosaic_surv_nojeffreys",
    "survtri_poe_vae_c_film_surv0": "mosaic_surv_detached",
    "c_film_surv0": "mosaic_surv_detached",
    # 早期批次的双下划线拼写（目录仍在 Table4 树里，未改名）
    "survtri_poe_vae_c_film__beta0": "mosaic_surv_nojeffreys",
    "survtri_poe_vae_c_film__surv0": "mosaic_surv_detached",
}


def _legacy_camel_case(token: str) -> str:
    """Cfilm 时代的展示规则：首段大写、其余小写（survtri_poe_vae_C → C）。"""
    parts = [part for part in str(token).split("_") if part]
    if not parts:
        return str(token)
    parts[0] = parts[0].upper()
    parts[1:] = [part.lower() for part in parts[1:]]
    return "_".join(parts)


DISPLAY_TO_KEY: dict[str, str] = {name.lower(): key for key, name in FORMAL_NAMES.items()}


def canonical_model_key(token: str) -> str:
    """任意拼写（旧/新、正式名、可带 run 名前缀）→ 注册键；映射外返回小写原样。"""
    text = str(token).strip().strip("/")
    if not text:
        return text
    low = text.lower()
    if low in FORMAL_NAMES:
        return low
    if low in DISPLAY_TO_KEY:
        return DISPLAY_TO_KEY[low]
    if low in LEGACY_ALIASES:
        return LEGACY_ALIASES[low]
    if "__" in low:
        # 整目录名（如 tcga_brca__L0__cell_norm__uni_v1__mosaic_surv）取最后一段
        head, _, tail = low.rpartition("__")
        if head and (tail in FORMAL_NAMES or tail in LEGACY_ALIASES):
            return canonical_model_key(tail)
    return low


def display_name(token: str) -> str:
    """展示名：映射内的模型 → 论文正式名；映射外保持旧的 CamelCase 规则。"""
    key = canonical_model_key(token)
    if key in FORMAL_NAMES:
        return FORMAL_NAMES[key]
    if key.startswith(MAIN_MODEL):
        # 未登记的 mosaic_surv* 变体：直接返回注册键，不猜测正式名
        return key
    return _legacy_camel_case(token)


def is_main_model(token: str) -> bool:
    return canonical_model_key(token) == MAIN_MODEL
