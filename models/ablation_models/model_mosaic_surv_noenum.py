"""MosaicSurv w/o Pattern Enumeration: FiLM head using the availability pattern once per forward."""

from models.ablation_models.model_mosaic_surv import MosaicSurv


class MosaicSurvNoEnum(MosaicSurv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uses_multi_pattern_surv = False
