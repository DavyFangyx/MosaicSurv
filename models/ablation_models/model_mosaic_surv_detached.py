"""MosaicSurv w/ Detached L_surv（原 Cfilm surv0 消融），去除联合损失中的 L_surv 项。

FiLM 头保留 lambda * L_surv，但 L_surv 不回传到 Encoder 与 PoE alpha：
这两个模块只收到 L_rec 与 beta * J 的梯度。
"""

from models.ablation_models.model_mosaic_surv import MosaicSurv


class MosaicSurvDetached(MosaicSurv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surv_detach_poe = True
