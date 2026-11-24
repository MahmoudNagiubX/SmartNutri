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
        class_precision = (
            float(true_positive[class_id] / predicted_pixels[class_id])
            if predicted_pixels[class_id] > 0
            else None
        )
        class_recall = (
            float(true_positive[class_id] / target_pixels[class_id])
            if target_pixels[class_id] > 0
            else None
        )
        ious.append(class_iou)
        dices.append(class_dice)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": id2label.get(class_id, f"class_{class_id}"),
                "true_positive_pixels": int(true_positive[class_id]),
                "ground_truth_pixels": int(target_pixels[class_id]),
                "predicted_pixels": int(predicted_pixels[class_id]),
                "iou": class_iou,
                "dice": class_dice,
                "pixel_precision": class_precision,
                "pixel_recall": class_recall,
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


def update_image_presence_confusion(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Count per-image foreground class presence as target/prediction 2x2 tables."""
    if predictions.shape != labels.shape or labels.ndim != 3:
        raise ValueError("predictions and labels must both have shape [B,H,W]")
    if confusion.ndim != 3 or confusion.shape[1:] != (2, 2):
        raise ValueError("presence confusion must have shape [classes,2,2]")

    num_classes = confusion.shape[0]
    for image_predictions, image_labels in zip(predictions, labels, strict=True):
        valid = image_labels != ignore_index
        target_present = torch.zeros(num_classes, dtype=torch.bool, device=labels.device)
        predicted_present = torch.zeros_like(target_present)
        if torch.any(valid):
            target_present[image_labels[valid].long().unique()] = True
            predicted_present[image_predictions[valid].long().unique()] = True
        class_ids = torch.arange(num_classes, device=confusion.device)
        confusion[class_ids, target_present.long(), predicted_present.long()] += 1


def summarize_image_presence_confusion(
    confusion: torch.Tensor,
    id2label: dict[int, str],
) -> list[dict]:
    """Return image-level presence precision/recall for each FoodSeg class."""
    matrix = confusion.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 3 or matrix.shape[1:] != (2, 2):
        raise ValueError("presence confusion must have shape [classes,2,2]")

    results: list[dict] = []
    for class_id in range(matrix.shape[0]):
        true_positive = matrix[class_id, 1, 1]
        false_positive = matrix[class_id, 0, 1]
        false_negative = matrix[class_id, 1, 0]
        true_negative = matrix[class_id, 0, 0]
        predicted_positive = true_positive + false_positive
        actual_positive = true_positive + false_negative
        precision = (
            float(true_positive / predicted_positive)
            if predicted_positive > 0
            else None
        )
        recall = float(true_positive / actual_positive) if actual_positive > 0 else None
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if precision is not None and recall is not None and precision + recall > 0
            else (0.0 if precision == 0.0 or recall == 0.0 else None)
        )
        results.append(
            {
                "class_id": class_id,
                "class_name": id2label.get(class_id, f"class_{class_id}"),
                "true_positive_images": int(true_positive),
                "false_positive_images": int(false_positive),
                "false_negative_images": int(false_negative),
                "true_negative_images": int(true_negative),
                "presence_precision": precision,
                "presence_recall": recall,
                "presence_f1": f1,
            }
        )
    return results


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    num_labels: int,
    device: torch.device,
    *,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
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
            dtype=amp_dtype,
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
