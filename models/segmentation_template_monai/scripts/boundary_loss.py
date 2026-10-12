"""Loss used for the SwinUNETR boundary-aware ablation.

The original experiment uses MONAI's DiceCELoss.  This module deliberately
keeps that term unchanged and adds a small distance-transform Hausdorff
surrogate:

    total = DiceCE + boundary_weight * log(1 + HausdorffDT)

The logarithm keeps the auxiliary term from overwhelming DiceCE when a patch
contains a distant false-positive component.  Background is excluded from the
Hausdorff term so each of the four SegTHOR organs contributes equally.
"""

from __future__ import annotations

import torch
from monai.losses import DiceCELoss, HausdorffDTLoss
from torch import nn


class DiceCEHausdorffLoss(nn.Module):
    """Unmodified DiceCE plus a low-weight boundary-aware auxiliary loss.

    Parameters mirror the original ``DiceCELoss`` configuration in
    ``configs/train.yaml``.  ``boundary_weight=0`` is therefore exactly the
    original loss and is useful as a regression check.

    ``HausdorffDTLoss`` is a differentiable distance-transform surrogate; it
    is not the evaluation HD95 itself.  HD95 still has to be calculated on the
    reconstructed patient volumes in physical millimetres.
    """

    def __init__(
        self,
        boundary_weight: float = 0.01,
        boundary_alpha: float = 2.0,
        include_background: bool = True,
        boundary_include_background: bool = False,
        to_onehot_y: bool = True,
        softmax: bool = True,
        squared_pred: bool = True,
        batch: bool = True,
    ) -> None:
        super().__init__()
        if boundary_weight < 0:
            raise ValueError("boundary_weight must be non-negative")
        if boundary_alpha <= 0:
            raise ValueError("boundary_alpha must be positive")

        self.boundary_weight = float(boundary_weight)
        self.region_loss = DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            squared_pred=squared_pred,
            batch=batch,
        )
        self.hausdorff_loss = HausdorffDTLoss(
            alpha=boundary_alpha,
            include_background=boundary_include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            reduction="mean",
            batch=False,
        )

    def boundary_term(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the stabilized boundary component before weighting."""

        return torch.log1p(self.hausdorff_loss(prediction, target))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        region = self.region_loss(prediction, target)
        if self.boundary_weight == 0.0:
            # Besides saving distance-transform work, this makes the regression
            # comparison bit-for-bit identical to the original DiceCE loss.
            return region
        return region + self.boundary_weight * self.boundary_term(prediction, target)
