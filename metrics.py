from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from foodseg_data import IGNORE_INDEX


def update_confusion_matrix(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    valid = labels != IGNORE_INDEX
    if not torch.any(valid):
        return
    num_labels = confusion.shape[0]
    indices = labels[valid].to(torch.int64) * num_labels + predictions[valid].to(torch.int64)
    counts = torch.bincount(indices, minlength=num_labels * num_labels)
    confusion += counts.reshape(num_labels, num_labels)


def _safe_mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None and math.isfinite(value)]
    return sum(present) / len(present) if present else None


def summarize_confusion_matrix(
    confusion: torch.Tensor,
    id2label: dict[int, str],
) -> tuple[dict, list[dict]]:
    matrix = confusion.detach().to(device="cpu", dtype=torch.float64)
    true_positive = matrix.diag()
    target_pixels = matrix.sum(dim=1)
    predicted_pixels = matrix.sum(dim=0)
    union = target_pixels + predicted_pixels - true_positive
    dice_denominator = target_pixels + predicted_pixels

    ious: list[float | None] = []
    dices: list[float | None] = []
    per_class: list[dict] = []
    for class_id in range(matrix.shape[0]):
        class_iou = (
            float(true_positive[class_id] / union[class_id])
            if union[class_id] > 0
            else None
        )
        class_dice = (
            float(2 * true_positive[class_id] / dice_denominator[class_id])
            if dice_denominator[class_id] > 0
            else None
        )
        ious.append(class_iou)
        dices.append(class_dice)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": id2label.get(class_id, f"class_{class_id}"),
                "ground_truth_pixels": int(target_pixels[class_id]),
                "predicted_pixels": int(predicted_pixels[class_id]),
                "iou": class_iou,
                "dice": class_dice,
            }
        )

    total_pixels = int(matrix.sum())
    correct_pixels = int(true_positive.sum())
    summary = {
        "pixel_accuracy": correct_pixels / total_pixels if total_pixels else None,
        "mean_iou_including_background": _safe_mean(ious),
        "mean_iou_foreground": _safe_mean(ious[1:]),
        "mean_dice_including_background": _safe_mean(dices),
        "mean_dice_foreground": _safe_mean(dices[1:]),
        "classes_with_ground_truth_or_predictions": sum(value is not None for value in ious),
        "foreground_classes_with_ground_truth_or_predictions": sum(
            value is not None for value in ious[1:]
        ),
        "evaluated_pixels": total_pixels,
    }
    return summary, per_class


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    num_labels: int,
    device: torch.device,
    *,
    use_amp: bool = True,
) -> torch.Tensor:
    model.eval()
    confusion = torch.zeros(
        (num_labels, num_labels), dtype=torch.int64, device=device
    )
    amp_enabled = use_amp and device.type == "cuda"
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(
            outputs.logits.float(),
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        predictions = logits.argmax(dim=1)
        update_confusion_matrix(confusion, predictions, labels)
    return confusion
