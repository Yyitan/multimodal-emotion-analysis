"""Missing-modality augmentation that never confuses padding with missing spans."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def continuous_span_mask(
    features: Dict[str, torch.Tensor],
    observed: Dict[str, torch.Tensor],
    valid: torch.Tensor,
    probability: float = 0.5,
    min_fraction: float = 0.1,
    max_fraction: float = 0.4,
    modalities: Iterable[str] = ("text", "audio", "vision"),
    generator: torch.Generator | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Zero a continuous span inside each sample's valid region and update observed only."""
    augmented = {k: v.clone() for k, v in features.items()}
    masks = {k: v.clone() for k, v in observed.items()}
    for name in modalities:
        if name not in augmented:
            continue
        for i in range(valid.shape[0]):
            if torch.rand((), generator=generator).item() >= probability:
                continue
            positions = torch.where(valid[i])[0]
            if positions.numel() == 0:
                continue
            fraction = min_fraction + (max_fraction - min_fraction) * torch.rand((), generator=generator).item()
            span = max(1, min(int(round(positions.numel() * fraction)), positions.numel()))
            start_max = positions.numel() - span
            offset = int(torch.randint(start_max + 1, (), generator=generator).item()) if start_max else 0
            selected = positions[offset : offset + span]
            augmented[name][i, selected] = 0
            masks[name][i, selected] = False
    return augmented, masks
