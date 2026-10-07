"""MosaicSurv (Pretrained VAE, Frozen Backbone): A 范式 + FiLM head 冻结 backbone probe。"""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE
from models.model_utils import FiLMHead


class MosaicSurvFrozen(SurvTriPoEVAE):
    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "A"
        super().__init__(*args, **kwargs)
        self.pattern_head = FiLMHead(
            d_z=self.latent_dim,
            mmhid=self.mmhid,
            dropout=0.1,
            label_dim=self.label_dim,
        )
        self.uses_multi_pattern_surv = True
        self._disable_legacy_survival_head()

    def freeze_backbone_for_probe(self):
        super().freeze_backbone_for_probe()
        for parameter in self.linear_probe.parameters():
            parameter.requires_grad = False
        for parameter in self.pattern_head.parameters():
            parameter.requires_grad = True
        self.head_trainable = True
