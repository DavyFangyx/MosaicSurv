"""MosaicSurv w/ Single Head: C 联合训练 + mask-corrected 单一生存头。"""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE


class MosaicSurvSingle(SurvTriPoEVAE):
    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "C"
        super().__init__(*args, **kwargs)

