"""Model B ablation: configurable cross-study stage1 pretraining before stage2 finetuning."""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE


class SurvTriPoEVAE_BCrossStage1(SurvTriPoEVAE):
    def __init__(self, *args, **kwargs):
        self.poe_stage1_studies = kwargs.pop("poe_stage1_studies", "")
        kwargs["poe_variant"] = "B"
        super().__init__(*args, **kwargs)
