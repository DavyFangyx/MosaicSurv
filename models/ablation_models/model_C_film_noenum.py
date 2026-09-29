"""C + FiLM head using the sample availability pattern once per forward."""

from models.ablation_models.model_C_film import SurvTriPoEVAE_CFiLM


class SurvTriPoEVAE_CFiLMNoEnum(SurvTriPoEVAE_CFiLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uses_multi_pattern_surv = False
