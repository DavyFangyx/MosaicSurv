"""Model C + single head: mask-corrected joint training."""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE


class SurvTriPoEVAE_CSingle(SurvTriPoEVAE):
    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "C"
        super().__init__(*args, **kwargs)

