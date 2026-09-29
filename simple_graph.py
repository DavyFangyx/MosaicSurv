from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch


@dataclass
class GraphData:
    x_img: torch.Tensor
    x_rna: torch.Tensor
    x_cli: torch.Tensor
    sur_type: torch.Tensor
    data_id: str
    data_type: list[str]
    edge_index_model: torch.Tensor
    edge_index_image: torch.Tensor
    edge_index_rna: torch.Tensor
    edge_index_cli: torch.Tensor
    surv_time: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to(self, device):
        # Copy tensors onto the target device and leave the cached CPU graph intact.
        # In-place .to(device) would pin every all_data[id] on GPU until the process exits.
        updates = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if torch.is_tensor(value)
        }
        return replace(self, **updates)
