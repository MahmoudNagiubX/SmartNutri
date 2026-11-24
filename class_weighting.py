from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from foodseg_data import ParquetSplit, _image_bytes, load_label_map
from image_processing import IGNORE_INDEX, NUM_CLASSES, PreprocessingConfig


@dataclass(frozen=True)
class ClassWeightConfig:
    experiment_name: str
    output_dir: str
    num_classes: int
    ignore_index: int
    frequency_split: str
    frequency_space: str
    tempering_exponent: float
    max_weight: float
    normalization: str
    absent_class_policy: str


@dataclass(frozen=True)
class PixelFrequencySummary:
    class_pixel_counts: np.ndarray
    image_count: int
    ignored_padding_pixels: int

    @property
    def valid_pixel_count(self) -> int:
        return int(self.class_pixel_counts.sum())


def load_class_weight_config(path: str | Path) -> ClassWeightConfig:
    config_path = Path(path)
    raw: Any = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Class-weight configuration must be a JSON object: {config_path}")

    expected_fields = set(ClassWeightConfig.__dataclass_fields__)
    if set(raw) != expected_fields:
        missing = sorted(expected_fields - set(raw))
        extra = sorted(set(raw) - expected_fields)
        raise ValueError(f"Invalid class-weight config fields; missing={missing}, extra={extra}")

    config = ClassWeightConfig(**raw)
    if not config.experiment_name.strip() or not config.output_dir.strip():
        raise ValueError("experiment_name and output_dir must be non-empty")
    if config.num_classes != NUM_CLASSES:
        raise ValueError(f"FoodSeg103 requires exactly {NUM_CLASSES} classes")
    if config.ignore_index != IGNORE_INDEX:
        raise ValueError(f"FoodSeg103 padding must use ignore label {IGNORE_INDEX}")
    if config.frequency_split != "train":
        raise ValueError("class frequencies must be computed from the FoodSeg103 train split")
    if config.frequency_space != "preprocessed_baseline_pixels":
        raise ValueError("class frequencies must match the baseline-preprocessed training masks")
    if config.normalization != "arithmetic_mean_one_with_cap":
        raise ValueError("normalization must be arithmetic_mean_one_with_cap")
    if config.absent_class_policy != "median_frequency_imputation":
        raise ValueError("absent classes must use median-frequency imputation")
    if not math.isfinite(config.tempering_exponent) or not 0.0 <= config.tempering_exponent <= 1.0:
        raise ValueError("tempering_exponent must be finite and in [0, 1]")
    if not math.isfinite(config.max_weight) or config.max_weight < 1.0:
        raise ValueError("max_weight must be finite and at least 1 for mean-one normalization")
    return config


def count_training_pixel_frequencies(
    foodseg_root: str | Path,
    processing_config: PreprocessingConfig,
    *,
    include_ids: Iterable[int | str] | None = None,
) -> PixelFrequencySummary:
    """Count only selected training-split labels; never reads validation rows."""
    root = Path(foodseg_root).expanduser().resolve()
    load_label_map(root)
    source = ParquetSplit(root, split="train")
    selected_ids = None if include_ids is None else frozenset(str(value) for value in include_ids)
    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_count = 0
    ignored_padding_pixels = 0

    for row in source.iter_rows(
        shuffle=False,
        rng=random.Random(0),
        include_ids=selected_ids,
    ):
        with Image.open(BytesIO(_image_bytes(row, "image"))) as source_image:
            image_size = source_image.size
            if source_image.getexif().get(274) in {5, 6, 7, 8}:
                image_size = (image_size[1], image_size[0])
        with Image.open(BytesIO(_image_bytes(row, "label"))) as source_mask:
            source_mask.load()
            raw_mask = np.asarray(source_mask)
            if raw_mask.ndim != 2:
                raise ValueError(
                    f"Expected a single-channel indexed mask in train row {row.get('id')}"
                )
            valid_values = ((raw_mask >= 0) & (raw_mask < NUM_CLASSES)) | (raw_mask == IGNORE_INDEX)
            if not np.all(valid_values):
                invalid = np.unique(raw_mask[~valid_values])[:8].tolist()
                raise ValueError(
                    f"Train row {row.get('id')} has labels outside 0..{NUM_CLASSES - 1} "
                    f"and ignore_index {IGNORE_INDEX}: {invalid}"
                )
            if source_mask.size != image_size:
                raise ValueError(
                    f"EXIF-corrected image size {image_size} does not match "
                    f"mask size {source_mask.size} in train row {row.get('id')}"
                )
            mask = Image.fromarray(raw_mask.astype(np.uint8, copy=False), mode="L")

        width, height = image_size
        scale = min(processing_config.image_size / width, processing_config.image_size / height)
        resized_size = (
            min(processing_config.image_size, max(1, round(width * scale))),
            min(processing_config.image_size, max(1, round(height * scale))),
        )
        resized_mask = mask.resize(resized_size, Image.Resampling.NEAREST)
        canvas = Image.new(
            "L",
            (processing_config.image_size, processing_config.image_size),
            IGNORE_INDEX,
        )
        left = (processing_config.image_size - resized_size[0]) // 2
        top = (processing_config.image_size - resized_size[1]) // 2
        canvas.paste(resized_mask, (left, top))
        values = np.asarray(canvas, dtype=np.uint8).reshape(-1)
        valid = values != IGNORE_INDEX
        if np.any(valid):
            class_pixel_counts += np.bincount(
                values[valid].astype(np.int64, copy=False), minlength=NUM_CLASSES
            )
        ignored_padding_pixels += int(np.count_nonzero(~valid))
        image_count += 1

    expected_images = source.num_rows if selected_ids is None else len(selected_ids)
    if image_count != expected_images:
        raise RuntimeError(
            f"Read {image_count} selected training rows but expected {expected_images}"
        )
    if image_count == 0 or class_pixel_counts.sum() == 0:
        raise ValueError("FoodSeg103 train split contains no valid labeled pixels")
    return PixelFrequencySummary(
        class_pixel_counts=class_pixel_counts,
        image_count=image_count,
        ignored_padding_pixels=ignored_padding_pixels,
    )


def compute_class_weights(
    class_pixel_counts: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    *,
    tempering_exponent: float = 0.5,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Return 104 finite weights with arithmetic mean 1 and a strict upper cap.

    Present classes use ``(median_positive_count / count) ** exponent``. An absent
    class is imputed at the median frequency (raw weight 1), avoiding division by
    zero while retaining its output channel. A scalar normalization is found by
    bisection so ``mean(min(scale * raw_weight, max_weight)) == 1``.
    """
    counts = torch.as_tensor(class_pixel_counts, dtype=torch.float64, device="cpu")
    if counts.ndim != 1 or counts.numel() != NUM_CLASSES:
        raise ValueError(f"class_pixel_counts must contain exactly {NUM_CLASSES} values")
    if not torch.isfinite(counts).all() or torch.any(counts < 0):
        raise ValueError("class_pixel_counts must be finite and non-negative")
    if not torch.equal(counts, counts.round()):
        raise ValueError("class_pixel_counts must contain whole pixel counts")
    if float(counts.sum()) <= 0:
        raise ValueError("at least one class must have training pixels")
    if not math.isfinite(tempering_exponent) or not 0.0 <= tempering_exponent <= 1.0:
        raise ValueError("tempering_exponent must be finite and in [0, 1]")
    if not math.isfinite(max_weight) or max_weight < 1.0:
        raise ValueError("max_weight must be finite and at least 1")

    present = counts > 0
    reference_count = float(np.median(counts[present].numpy()))
    raw_weights = torch.ones(NUM_CLASSES, dtype=torch.float64)
    raw_weights[present] = (reference_count / counts[present]).pow(tempering_exponent)

    if max_weight == 1.0:
        return torch.ones(NUM_CLASSES, dtype=torch.float32)

    low = 0.0
    high = max_weight / float(raw_weights.min())
    for _ in range(128):
        scale = (low + high) / 2.0
        mean_weight = torch.clamp(raw_weights * scale, max=max_weight).mean().item()
        if mean_weight < 1.0:
            low = scale
        else:
            high = scale

    weights = torch.clamp(raw_weights * ((low + high) / 2.0), max=max_weight)
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise RuntimeError("computed class weights are not finite and positive")
    if float(weights.max()) > max_weight + 1e-10:
        raise RuntimeError("computed class weights exceeded the configured cap")
    if abs(float(weights.mean()) - 1.0) > 1e-8:
        raise RuntimeError("computed class weights failed mean-one normalization")
    return weights.to(torch.float32)


def build_class_weight_metadata(
    config: ClassWeightConfig,
    frequency_summary: PixelFrequencySummary,
    class_weights: torch.Tensor,
    id2label: dict[int, str],
) -> dict[str, Any]:
    if len(id2label) != NUM_CLASSES or set(id2label) != set(range(NUM_CLASSES)):
        raise ValueError(f"id2label must map every FoodSeg103 class 0..{NUM_CLASSES - 1}")
    weights = class_weights.detach().to(device="cpu", dtype=torch.float64)
    if weights.shape != (NUM_CLASSES,):
        raise ValueError(f"class_weights must contain exactly {NUM_CLASSES} values")
    valid_pixel_count = frequency_summary.valid_pixel_count
    return {
        "experiment_name": config.experiment_name,
        "frequency_split": config.frequency_split,
        "frequency_space": config.frequency_space,
        "training_images": frequency_summary.image_count,
        "valid_training_pixels": valid_pixel_count,
        "ignored_padding_pixels": frequency_summary.ignored_padding_pixels,
        "num_classes": config.num_classes,
        "ignore_index": config.ignore_index,
        "tempering_exponent": config.tempering_exponent,
        "max_weight": config.max_weight,
        "normalization": config.normalization,
        "absent_class_policy": config.absent_class_policy,
        "weight_min": float(weights.min().item()),
        "weight_max": float(weights.max().item()),
        "weight_mean": float(weights.mean().item()),
        "classes": [
            {
                "class_id": class_id,
                "class_name": id2label[class_id],
                "pixel_count": int(frequency_summary.class_pixel_counts[class_id]),
                "frequency": float(
                    frequency_summary.class_pixel_counts[class_id] / valid_pixel_count
                ),
                "weight": float(weights[class_id].item()),
            }
            for class_id in range(NUM_CLASSES)
        ],
    }


def weighted_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute weighted CE after aligning SegFormer logits to target resolution."""
    if logits.ndim != 4:
        raise ValueError("logits must have shape [batch, classes, height, width]")
    if labels.ndim != 3:
        raise ValueError("labels must have shape [batch, height, width]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logit and label batch sizes must match")
    if class_weights.ndim != 1 or class_weights.numel() != logits.shape[1]:
        raise ValueError("class_weights must contain one value per logit class")
    if not torch.isfinite(class_weights).all() or torch.any(class_weights <= 0):
        raise ValueError("class_weights must be finite and positive")

    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits.float(),
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    else:
        logits = logits.float()
    labels = labels.to(device=logits.device, dtype=torch.long)
    weights = class_weights.to(device=logits.device, dtype=torch.float32)
    valid = labels != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0

    per_pixel_loss = F.cross_entropy(
        logits,
        labels,
        weight=weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    normalizer = weights[labels[valid]].sum()
    return per_pixel_loss[valid].sum() / normalizer
