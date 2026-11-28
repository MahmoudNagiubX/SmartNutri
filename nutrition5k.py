from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
from torchvision.transforms import ColorJitter

TARGETS = (
    ("calories", "total_calories", "kcal/g", "kcal"),
    ("protein", "protein_g", "g/g", "g"),
    ("carbohydrates", "carb_g", "g/g", "g"),
    ("fat", "fat_g", "g/g", "g"),
)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
IMAGE_FILL = tuple(round(value * 255) for value in IMAGE_MEAN)
_DISH_ID = re.compile(r"^dish_(\d{10})$")


@dataclass(frozen=True)
class DishRecord:
    dish_id: str
    image_path: Path
    group_id: str
    mass_g: float
    totals: tuple[float, float, float, float]
    densities: tuple[float, float, float, float]


@dataclass(frozen=True)
class DataPartitions:
    training: list[DishRecord]
    validation: list[DishRecord]
    test: list[DishRecord]
    excluded_ids: list[str]
    audit: dict[str, int]
    seed: int
    validation_fraction: float
    group_method: str = "dish_id_utc_day"

    def manifest(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "validation_fraction_target": self.validation_fraction,
            "group_method": self.group_method,
            "training_ids": [record.dish_id for record in self.training],
            "validation_ids": [record.dish_id for record in self.validation],
            "official_test_ids": [record.dish_id for record in self.test],
            "excluded_unmatched_ids": self.excluded_ids,
            "training_groups": sorted({record.group_id for record in self.training}),
            "validation_groups": sorted({record.group_id for record in self.validation}),
            "counts": {
                "training": len(self.training),
                "validation": len(self.validation),
                "official_test": len(self.test),
                "excluded_unmatched": len(self.excluded_ids),
            },
            "audit": self.audit,
        }


def _read_id_file(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Official Nutrition5k split file is missing: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Official split file contains duplicate IDs: {path}")
    return set(values)


def _dish_group(dish_id: str) -> str:
    match = _DISH_ID.fullmatch(dish_id)
    if match is None:
        raise ValueError(f"Expected Nutrition5k dish_[10-digit timestamp] ID, got {dish_id!r}")
    timestamp = int(match.group(1))
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def _row_to_record(row: dict[str, str], image_dir: Path) -> DishRecord:
    dish_id = row["dish_id"].strip()
    group_id = _dish_group(dish_id)
    try:
        mass_g = float(row["total_mass_g"])
        totals = tuple(float(row[field]) for _, field, _, _ in TARGETS)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric nutrition metadata for {dish_id}") from error
    numeric = (mass_g, *totals)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"Non-finite nutrition metadata for {dish_id}")
    if mass_g <= 0 or any(value < 0 for value in totals):
        raise ValueError(f"Mass must be positive and nutrient totals nonnegative for {dish_id}")
    image_path = image_dir / f"{dish_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"RGB image for {dish_id} is missing: {image_path}")
    densities = tuple(value / mass_g for value in totals)
    return DishRecord(
        dish_id=dish_id,
        image_path=image_path,
        group_id=group_id,
        mass_g=mass_g,
        totals=totals,
        densities=densities,
    )


def _split_by_day_groups(
    records: list[DishRecord], *, validation_fraction: float, seed: int
) -> tuple[list[DishRecord], list[DishRecord]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    grouped: dict[str, list[DishRecord]] = {}
    for record in records:
        grouped.setdefault(record.group_id, []).append(record)
    if len(grouped) < 2:
        raise ValueError("At least two UTC acquisition-day groups are required")

    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    target_count = max(1, round(len(records) * validation_fraction))
    validation_groups: set[str] = set()
    validation_count = 0
    for group in groups:
        if validation_count >= target_count:
            break
        if len(validation_groups) + 1 >= len(groups):
            break
        validation_groups.add(group)
        validation_count += len(grouped[group])
    if not validation_groups or len(validation_groups) == len(groups):
        raise ValueError("Could not construct nonempty day-disjoint training and validation sets")
    validation = [record for record in records if record.group_id in validation_groups]
    training = [record for record in records if record.group_id not in validation_groups]
    return training, validation


def load_partitions(
    data_root: str | Path,
    *,
    split_dir: str | Path | None = None,
    seed: int = 42,
    validation_fraction: float = 0.10,
) -> DataPartitions:
    data_root = Path(data_root).expanduser().resolve()
    split_dir = (
        Path(split_dir).expanduser().resolve()
        if split_dir is not None
        else data_root / "dish_ids" / "splits"
    )
    metadata_path = data_root / "metadata.csv"
    image_dir = data_root / "images"
    required_columns = {"dish_id", "total_mass_g", *(field for _, field, _, _ in TARGETS)}
    with metadata_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        missing_columns = required_columns - columns
        if missing_columns:
            raise ValueError(f"Nutrition5k metadata is missing columns: {sorted(missing_columns)}")
        rows = list(reader)

    train_ids = _read_id_file(split_dir / "rgb_train_ids.txt")
    test_ids = _read_id_file(split_dir / "rgb_test_ids.txt")
    if train_ids & test_ids:
        raise ValueError("Official Nutrition5k RGB train/test ID lists overlap")

    seen: set[str] = set()
    all_records: dict[str, DishRecord] = {}
    for row in rows:
        dish_id = row["dish_id"].strip()
        if dish_id in seen:
            raise ValueError(f"Duplicate Nutrition5k metadata ID: {dish_id}")
        seen.add(dish_id)
        all_records[dish_id] = _row_to_record(row, image_dir)

    official_training = sorted(
        (all_records[dish_id] for dish_id in train_ids & all_records.keys()),
        key=lambda record: record.dish_id,
    )
    official_test = sorted(
        (all_records[dish_id] for dish_id in test_ids & all_records.keys()),
        key=lambda record: record.dish_id,
    )
    excluded_ids = sorted(all_records.keys() - train_ids - test_ids)
    training, validation = _split_by_day_groups(
        official_training, validation_fraction=validation_fraction, seed=seed
    )
    training_groups = {record.group_id for record in training}
    validation_groups = {record.group_id for record in validation}
    if training_groups & validation_groups:
        raise RuntimeError("UTC-day group leakage detected between train and validation")
    if {record.dish_id for record in training + validation} & {
        record.dish_id for record in official_test
    }:
        raise RuntimeError("Official test leakage detected")

    return DataPartitions(
        training=training,
        validation=validation,
        test=official_test,
        excluded_ids=excluded_ids,
        audit={
            "metadata_rows": len(rows),
            "unique_metadata_ids": len(seen),
            "official_train_ids": len(train_ids),
            "official_test_ids": len(test_ids),
            "matched_official_train_images": len(official_training),
            "matched_official_test_images": len(official_test),
            "unmatched_local_ids_excluded": len(excluded_ids),
        },
        seed=seed,
        validation_fraction=validation_fraction,
    )


def split_fingerprint(manifest: dict[str, Any]) -> str:
    payload = {
        key: manifest[key]
        for key in ("training_ids", "validation_ids", "official_test_ids", "excluded_unmatched_ids")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_target_scales(records: Iterable[DishRecord]) -> np.ndarray:
    densities = np.asarray([record.densities for record in records], dtype=np.float64)
    if densities.ndim != 2 or densities.shape[1] != len(TARGETS) or densities.shape[0] == 0:
        raise ValueError("Target scales require a nonempty N x 4 training-density array")
    scales = np.median(densities, axis=0)
    fallback = np.quantile(densities, 0.75, axis=0)
    scales = np.where(scales > 1e-8, scales, fallback)
    scales = np.where(scales > 1e-8, scales, 1.0)
    if not np.isfinite(scales).all():
        raise ValueError("Training-only target scales must be finite")
    return scales.astype(np.float32)


def mean_training_density(records: Iterable[DishRecord]) -> np.ndarray:
    densities = np.asarray([record.densities for record in records], dtype=np.float64)
    if densities.ndim != 2 or densities.shape[1] != len(TARGETS) or densities.shape[0] == 0:
        raise ValueError("Mean-density baseline requires a nonempty N x 4 training array")
    return densities.mean(axis=0).astype(np.float32)


class NutritionImageTransform:
    def __init__(
        self,
        image_size: int = 288,
        *,
        training: bool = False,
        augmentation: dict[str, float] | None = None,
    ) -> None:
        if image_size < 32:
            raise ValueError("EfficientNet input size must be at least 32 pixels")
        self.image_size = image_size
        self.training = training
        augmentation = augmentation or {}
        self.horizontal_flip_probability = float(
            augmentation.get("horizontal_flip_probability", 0.5)
        )
        self.color_jitter_probability = float(
            augmentation.get("color_jitter_probability", 0.7)
        )
        self.rotation_probability = float(augmentation.get("rotation_probability", 0.25))
        self.rotation_degrees = float(augmentation.get("rotation_degrees", 5.0))
        self.color_jitter = ColorJitter(
            brightness=0.12, contrast=0.12, saturation=0.10, hue=0.015
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.training:
            if random.random() < self.horizontal_flip_probability:
                image = ImageOps.mirror(image)
            if random.random() < self.color_jitter_probability:
                image = self.color_jitter(image)
            if random.random() < self.rotation_probability:
                image = image.rotate(
                    random.uniform(-self.rotation_degrees, self.rotation_degrees),
                    resample=Image.Resampling.BICUBIC,
                    fillcolor=IMAGE_FILL,
                )

        width, height = image.size
        scale = min(self.image_size / width, self.image_size / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.image_size, self.image_size), IMAGE_FILL)
        canvas.paste(
            resized,
            ((self.image_size - resized.width) // 2, (self.image_size - resized.height) // 2),
        )
        array = np.array(canvas, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGE_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGE_STD, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std


class Nutrition5kDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[DishRecord], transform: NutritionImageTransform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "density": torch.tensor(record.densities, dtype=torch.float32),
            "totals": torch.tensor(record.totals, dtype=torch.float32),
            "mass_g": torch.tensor(record.mass_g, dtype=torch.float32),
            "dish_id": record.dish_id,
        }


class PositiveDensityModel(nn.Module):
    """Four positive, training-scale-normalized density predictions."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.backbone(pixel_values))


def build_model(*, pretrained: bool) -> PositiveDensityModel:
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = efficientnet_b2(weights=weights)
    final_layer = backbone.classifier[-1]
    backbone.classifier[-1] = nn.Linear(final_layer.in_features, len(TARGETS))
    return PositiveDensityModel(backbone)


def resolve_amp_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    if requested == "none" or device.type != "cuda":
        return None
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 AMP was requested but this CUDA device does not support it")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {requested}")


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_scales: np.ndarray | torch.Tensor,
    amp_dtype: torch.dtype | None,
) -> tuple[list[str], np.ndarray]:
    model.eval()
    scales = torch.as_tensor(target_scales, dtype=torch.float32, device=device)
    ids: list[str] = []
    outputs: list[np.ndarray] = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            normalized_density = model(pixels).float()
        density = normalized_density * scales
        outputs.append(density.cpu().numpy())
        ids.extend(batch["dish_id"])
    if not outputs:
        raise ValueError("Cannot evaluate an empty Nutrition5k partition")
    return ids, np.concatenate(outputs, axis=0)


def _metric_set(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.mean(np.abs(truth)))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "normalized_mae": (mae / denominator if denominator > 1e-12 else None),
        "signed_bias": float(np.mean(error)),
    }


def evaluation_report(
    records: list[DishRecord],
    predicted_density: np.ndarray,
    baseline_density: np.ndarray,
) -> dict[str, Any]:
    if len(records) != len(predicted_density):
        raise ValueError("Predictions must contain exactly one row per dish record")
    true_density = np.asarray([record.densities for record in records], dtype=np.float64)
    true_totals = np.asarray([record.totals for record in records], dtype=np.float64)
    masses = np.asarray([record.mass_g for record in records], dtype=np.float64)[:, None]
    predicted_density = np.asarray(predicted_density, dtype=np.float64)
    baseline_density = np.asarray(baseline_density, dtype=np.float64)
    predicted_totals = predicted_density * masses
    baseline_totals = baseline_density[None, :] * masses
    metrics: dict[str, Any] = {}
    for index, (name, _, density_unit, total_unit) in enumerate(TARGETS):
        metrics[name] = {
            "total_unit": total_unit,
            "density_unit": density_unit,
            "model_total": _metric_set(predicted_totals[:, index], true_totals[:, index]),
            "mean_density_baseline_total": _metric_set(
                baseline_totals[:, index], true_totals[:, index]
            ),
            "model_density": _metric_set(predicted_density[:, index], true_density[:, index]),
            "mean_density_baseline_density": _metric_set(
                np.full(len(records), baseline_density[index]), true_density[:, index]
            ),
        }
    return {
        "sample_count": len(records),
        "normalized_mae_definition": "MAE divided by mean absolute ground-truth value",
        "signed_bias_definition": "mean(prediction minus ground truth)",
        "nutrients": metrics,
    }


def validation_score(
    records: list[DishRecord],
    predicted_density: np.ndarray,
    target_scales: np.ndarray,
) -> float:
    truth = np.asarray([record.densities for record in records], dtype=np.float64)
    scales = np.asarray(target_scales, dtype=np.float64)
    if truth.shape != predicted_density.shape or scales.shape != (len(TARGETS),):
        raise ValueError("Validation targets, predictions, and scales have incompatible shapes")
    return float(np.mean(np.abs(predicted_density - truth) / scales[None, :]))


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[PositiveDensityModel, dict[str, Any]]:
    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    model = build_model(pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval(), state


def load_partition_by_name(partitions: DataPartitions, name: str) -> list[DishRecord]:
    if name == "train":
        return partitions.training
    if name == "validation":
        return partitions.validation
    if name == "test":
        return partitions.test
    raise ValueError(f"Unsupported Nutrition5k partition: {name}")
