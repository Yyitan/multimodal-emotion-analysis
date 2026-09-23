from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def supervised_multitask_loss(
    output: Dict[str, torch.Tensor],
    classes: torch.Tensor,
    regression: torch.Tensor,
    classification_weight: float = 1.0,
    regression_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    class_mask = classes.ge(0)
    reg_mask = torch.isfinite(regression)
    zero = output["logits"].sum() * 0.0
    class_loss = F.cross_entropy(output["logits"][class_mask], classes[class_mask]) if class_mask.any() else zero
    reg_loss = F.smooth_l1_loss(output["regression"][reg_mask], regression[reg_mask].clamp(-3, 3)) if reg_mask.any() else zero
    total = classification_weight * class_loss + regression_weight * reg_loss
    return {"loss": total, "classification_loss": class_loss, "regression_loss": reg_loss}


def consistency_loss(clean: Dict[str, torch.Tensor], augmented: Dict[str, torch.Tensor]) -> torch.Tensor:
    target_prob = torch.softmax(clean["logits"].detach(), dim=-1)
    class_consistency = F.kl_div(
        torch.log_softmax(augmented["logits"], dim=-1), target_prob, reduction="batchmean"
    )
    regression_consistency = F.mse_loss(augmented["regression"], clean["regression"].detach())
    return class_consistency + regression_consistency
