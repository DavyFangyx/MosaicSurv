"""Model B ablation: use KL instead of Jeffreys in stage1 VAE regularization."""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE_KL


class SurvTriPoEVAE_BKL(SurvTriPoEVAE_KL):
    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "B"
        super().__init__(*args, **kwargs)
