from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

from .paths import DEFAULT_KIMIANET_WEIGHTS


class fully_connected(nn.Module):
    """Same head as HGCN cut_and_pretrain.py / KimiaNet PyTorch sample."""

    def __init__(self, model, num_ftrs, num_classes):
        super().__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        return out_1, out_3


def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if all(str(key).startswith('module.') for key in state):
        return {str(key)[len('module.'):]: value for key, value in state.items()}
    return state


def kimianet_transform():
    # HGCN cut_and_pretrain.py uses ToTensor only. Official KimiaNet sample adds ImageNet norm.
    return transforms.Compose([transforms.ToTensor()])


def load_kimianet(
    weights: str | Path | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    weights_path = Path(weights) if weights is not None else DEFAULT_KIMIANET_WEIGHTS
    if not weights_path.is_file():
        raise FileNotFoundError(f'KimiaNet weights not found: {weights_path}')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    try:
        densenet = models.densenet121(pretrained=True)
    except TypeError:
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for param in densenet.parameters():
        param.requires_grad = False
    densenet.eval()
    densenet.features = nn.Sequential(densenet.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    num_ftrs = densenet.classifier.in_features
    model_final = fully_connected(densenet.features, num_ftrs, 30)
    state = torch.load(str(weights_path), map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    model_final.load_state_dict(_strip_module_prefix(state))
    model_final.to(device)
    model_final.eval()
    for param in model_final.parameters():
        param.requires_grad = False
    return model_final
