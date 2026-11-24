from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """Multiclass focal and soft-Dice losses with explicit ignored-pixel masking."""

    def __init__(
        self,
        *,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        include_background_in_dice: bool = False,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        if focal_gamma < 0:
            raise ValueError("focal_gamma cannot be negative")
        if focal_weight < 0 or dice_weight < 0 or focal_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative and have a positive sum")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        if class_weights is not None and class_weights.ndim != 1:
            raise ValueError("class_weights must be one-dimensional")

        self.register_buffer(
            "class_weights",
            None if class_weights is None else class_weights.detach().float().clone(),
        )
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.include_background_in_dice = include_background_in_dice
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("expected logits [B,C,H,W] and target [B,H,W]")
        if logits.shape[0] != target.shape[0]:
            raise ValueError("logits and target batch sizes differ")
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits.float(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            logits = logits.float()

        num_classes = logits.shape[1]
        if self.class_weights is not None and self.class_weights.numel() != num_classes:
            raise ValueError(
                f"class_weights has {self.class_weights.numel()} entries for {num_classes} classes"
            )

        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0).long()
        valid_float = valid.to(dtype=logits.dtype)
        valid_count = valid_float.sum().clamp_min(1.0)

        probabilities = logits.softmax(dim=1)
        target_probability = probabilities.gather(
            1, safe_target.unsqueeze(1)
        ).squeeze(1)
        target_probability = target_probability.clamp_min(torch.finfo(logits.dtype).tiny)
        negative_log_probability = -target_probability.log()
        focal = (
            (1.0 - target_probability).pow(self.focal_gamma)
            * negative_log_probability
        )
        if self.class_weights is not None:
            focal = focal * self.class_weights[safe_target]
        focal_loss = (focal * valid_float).sum() / valid_count

        masked_probabilities = probabilities * valid_float.unsqueeze(1)
        predicted_counts = masked_probabilities.sum(dim=(0, 2, 3))
        flat_target = safe_target.reshape(-1)
        target_probability_flat = target_probability.reshape(-1) * valid_float.reshape(-1)
        intersections = torch.zeros(
            num_classes,
            dtype=logits.dtype,
            device=logits.device,
        ).scatter_add_(0, flat_target, target_probability_flat)
        target_counts = torch.bincount(
            target[valid].long(), minlength=num_classes
        ).to(dtype=logits.dtype)

        dice = (2.0 * intersections + self.smooth) / (
            predicted_counts + target_counts + self.smooth
        )
        active_classes = target_counts > 0
        if not self.include_background_in_dice and num_classes > 1:
            active_classes[0] = False
        active_float = active_classes.to(dtype=logits.dtype)
        dice_loss = (
            ((1.0 - dice) * active_float).sum()
            / active_float.sum().clamp_min(1.0)
        )

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
