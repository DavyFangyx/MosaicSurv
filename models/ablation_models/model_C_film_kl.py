"""Model C + FiLM head using standard KL regularization."""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE_KL
from models.model_utils import FiLMHead


class SurvTriPoEVAE_CFiLMKL(SurvTriPoEVAE_KL):
    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "C"
        super().__init__(*args, **kwargs)
        self.pattern_head = FiLMHead(
            d_z=self.latent_dim,
            mmhid=self.mmhid,
            dropout=0.1,
            label_dim=self.label_dim,
        )
        self.uses_multi_pattern_surv = True
        self._disable_legacy_survival_head()
