"""
Binary Cross-Entropy Loss for DDI prediction.
"""
import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with optional class weighting.
    """
    
    def __init__(self, pos_weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        return loss
