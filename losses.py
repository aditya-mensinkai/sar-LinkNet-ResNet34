"""Loss functions for binary oil-spill segmentation."""

import torch
from torch import nn
import segmentation_models_pytorch as smp


class DiceBCELoss(nn.Module):
    """Equal-weight Dice and BCE-with-logits loss."""

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        weight = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.dice(logits, targets) + 0.5 * self.bce(logits, targets)
