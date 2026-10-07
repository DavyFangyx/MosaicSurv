"""MosaicSurv w/ Single Head + Enumeration: 共享单头 + 7-pattern PoE 枚举。"""

import torch.nn as nn

from models.model_SurvTriPoEVAE import SurvTriPoEVAE


class _SharedSingleHead(nn.Module):
    """Adapter used by ``multi_pattern_surv_step``; pattern_id is ignored."""

    def __init__(self, latent_dim, mmhid, label_dim):
        super().__init__()
        self.fuse_fc = nn.Sequential(
            nn.Linear(latent_dim, mmhid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mmhid, mmhid),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(mmhid, label_dim)

    def forward(self, mu_joint, pattern_id):
        del pattern_id
        return self.classifier(self.fuse_fc(mu_joint))


class MosaicSurvSingleEnum(SurvTriPoEVAE):
    """C 范式 + 一个共享头 + 7-pattern PoE 枚举。"""

    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "C"
        super().__init__(*args, **kwargs)
        self.pattern_head = _SharedSingleHead(self.latent_dim, self.mmhid, self.label_dim)
        self.uses_multi_pattern_surv = True
        self._disable_legacy_survival_head()
