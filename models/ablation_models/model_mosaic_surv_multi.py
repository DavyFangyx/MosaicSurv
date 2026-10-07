"""MosaicSurv w/ Multi-Head: C 联合训练 + 7 个 pattern-specific 头。"""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE
from models.model_utils import MultiPatternHead


class MosaicSurvMulti(SurvTriPoEVAE):
    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "C"
        super().__init__(*args, **kwargs)
        self.pattern_head = MultiPatternHead(
            d_z=self.latent_dim,
            mmhid=self.mmhid,
            dropout=0.1,
            label_dim=self.label_dim,
        )
        self.uses_multi_pattern_surv = True
        self._disable_legacy_survival_head()
