"""Cfilm，去除联合损失中的 L_surv 项（surv0）。

FiLM 头保留 lambda * L_surv，但 L_surv 不回传到 Encoder 与 PoE alpha：
这两个模块只收到 L_rec 与 beta * J 的梯度。
"""

from models.ablation_models.model_C_film import SurvTriPoEVAE_CFiLM


class SurvTriPoEVAE_CFiLMNoSurvGrad(SurvTriPoEVAE_CFiLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surv_detach_poe = True
