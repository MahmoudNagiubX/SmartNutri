from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

NUM_CLASSES = 104
IGNORE_INDEX = 255
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.0
    scale_probability: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    rotation_probability: float = 0.0
    rotation_degrees: float = 0.0
    photometric_probability: float = 1.0
    brightness_delta: float = 0.0
    contrast_delta: float = 0.0
    saturation_delta: float = 0.0
    gaussian_blur_probability: float = 0.0
    gaussian_blur_radius: float = 0.0
    gaussian_noise_probability: float = 0.0
    gaussian_noise_std: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "horizontal_flip_probability",
            "scale_probability",
            "rotation_probability",
            "photometric_probability",
            "gaussian_blur_probability",
            "gaussian_noise_probability",
        ):
            _validate_probability(name, getattr(self, name))
        for name in (
            "scale_min",
            "scale_max",
            "rotation_degrees",
            "brightness_delta",
            "contrast_delta",
            "saturation_delta",
            "gaussian_blur_radius",
            "gaussian_noise_std",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.scale_min <= 0 or self.scale_max < self.scale_min:
            raise ValueError("scale bounds must satisfy 0 < scale_min <= scale_max")


@dataclass(frozen=True)
class EnhancementConfig:
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    gamma: float = 1.0
    denoising: bool = False
    denoising_kernel_size: int = 3

    def __post_init__(self) -> None:
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if self.clahe_grid_size < 2:
            raise ValueError("clahe_grid_size must be at least 2")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.denoising_kernel_size < 3 or self.denoising_kernel_size % 2 == 0:
            raise ValueError("denoising_kernel_size must be an odd integer of at least 3")


@dataclass(frozen=True)
class PreprocessingConfig:
    name: str = "baseline"
    image_size: int = 512
    mean: tuple[float, float, float] = IMAGE_MEAN
    std: tuple[float, float, float] = IMAGE_STD
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    enhancements: EnhancementConfig = field(default_factory=EnhancementConfig)

    def __post_init__(self) -> None:
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("SegFormer normalization mean and std must have 3 channels")
        if any(not 0.0 <= value <= 1.0 for value in self.mean):
            raise ValueError("normalization mean values must be between 0 and 1")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization std values must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PreprocessingConfig:
        augmentation = AugmentationConfig(**values.get("augmentation", {}))
        enhancements = EnhancementConfig(**values.get("enhancements", {}))
        return cls(
            name=str(values.get("name", "custom")),
            image_size=int(values.get("image_size", 512)),
            mean=tuple(float(value) for value in values.get("mean", IMAGE_MEAN)),
            std=tuple(float(value) for value in values.get("std", IMAGE_STD)),
            augmentation=augmentation,
            enhancements=enhancements,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_preprocessing_config(path: str | Path) -> PreprocessingConfig:
    config_path = Path(path)
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Preprocessing config must be a JSON object: {config_path}")
    return PreprocessingConfig.from_dict(values)


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_mask(mask: Image.Image) -> Image.Image:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"Expected a single-channel indexed mask; found shape {values.shape}")
    valid = ((values >= 0) & (values < NUM_CLASSES)) | (values == IGNORE_INDEX)
    if not np.all(valid):
        invalid = np.unique(values[~valid])[:8].tolist()
        raise ValueError(
            f"Mask contains values outside FoodSeg103 labels 0..103 and ignore_index "
            f"{IGNORE_INDEX}: {invalid}"
        )
    return Image.fromarray(values.astype(np.uint8, copy=True), mode="L")


def _clahe_channel(channel: np.ndarray, *, clip_limit: float, grid_size: int) -> np.ndarray:
    height, width = channel.shape
    row_tiles = np.array_split(np.arange(height), min(grid_size, height))
    column_tiles = np.array_split(np.arange(width), min(grid_size, width))
    maps = np.empty((len(row_tiles), len(column_tiles), 256), dtype=np.float32)
    row_centers = np.asarray([(tile[0] + tile[-1]) / 2 for tile in row_tiles])
    column_centers = np.asarray([(tile[0] + tile[-1]) / 2 for tile in column_tiles])

    for row_index, rows in enumerate(row_tiles):
        for column_index, columns in enumerate(column_tiles):
            tile = channel[np.ix_(rows, columns)]
            histogram = np.bincount(tile.reshape(-1), minlength=256)
            limit = max(1, int(clip_limit * tile.size / 256))
            excess = int(np.maximum(histogram - limit, 0).sum())
            histogram = np.minimum(histogram, limit)
            if excess:
                histogram += excess // 256
                histogram[: excess % 256] += 1

            cumulative = np.cumsum(histogram)
            first_nonzero = int(np.flatnonzero(histogram)[0])
            denominator = tile.size - cumulative[first_nonzero]
            if denominator <= 0:
                lookup = np.arange(256, dtype=np.float32)
            else:
                lookup = np.maximum(cumulative - cumulative[first_nonzero], 0)
                lookup = lookup.astype(np.float32) * (255.0 / denominator)
            maps[row_index, column_index] = np.clip(lookup, 0, 255)

    def weights(coordinates: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        upper = np.searchsorted(centers, coordinates, side="right")
        lower = np.clip(upper - 1, 0, len(centers) - 1)
        upper = np.clip(upper, 0, len(centers) - 1)
        distance = centers[upper] - centers[lower]
        fraction = np.zeros(coordinates.shape, dtype=np.float32)
        separated = distance > 0
        fraction[separated] = (
            (coordinates[separated] - centers[lower[separated]]) / distance[separated]
        )
        return lower, upper, np.clip(fraction, 0.0, 1.0)

    y0, y1, wy = weights(np.arange(height), row_centers)
    x0, x1, wx = weights(np.arange(width), column_centers)
    values = channel.astype(np.int64)
    top_left = maps[y0[:, None], x0[None, :], values]
    top_right = maps[y0[:, None], x1[None, :], values]
    bottom_left = maps[y1[:, None], x0[None, :], values]
    bottom_right = maps[y1[:, None], x1[None, :], values]
    top = top_left * (1.0 - wx[None, :]) + top_right * wx[None, :]
    bottom = bottom_left * (1.0 - wx[None, :]) + bottom_right * wx[None, :]
    result = top * (1.0 - wy[:, None]) + bottom * wy[:, None]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _apply_enhancements(image: Image.Image, config: EnhancementConfig) -> Image.Image:
    if config.clahe:
        ycbcr = image.convert("YCbCr")
        luminance, chroma_blue, chroma_red = ycbcr.split()
        enhanced = _clahe_channel(
            np.asarray(luminance, dtype=np.uint8),
            clip_limit=config.clahe_clip_limit,
            grid_size=config.clahe_grid_size,
        )
        image = Image.merge(
            "YCbCr", (Image.fromarray(enhanced, mode="L"), chroma_blue, chroma_red)
        ).convert("RGB")

    if config.gamma != 1.0:
        lookup = np.clip(
            np.rint(np.power(np.arange(256, dtype=np.float32) / 255.0, config.gamma) * 255),
            0,
            255,
        ).astype(np.uint8)
        image = image.point(np.tile(lookup, 3).tolist())

    if config.denoising:
        image = image.filter(ImageFilter.MedianFilter(size=config.denoising_kernel_size))
    return image


def _apply_photometric_augmentations(
    image: Image.Image,
    config: AugmentationConfig,
    rng: random.Random,
) -> Image.Image:
    has_color_adjustment = any(
        value > 0
        for value in (config.brightness_delta, config.contrast_delta, config.saturation_delta)
    )
    if has_color_adjustment and rng.random() < config.photometric_probability:
        if config.brightness_delta:
            factor = rng.uniform(1.0 - config.brightness_delta, 1.0 + config.brightness_delta)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if config.contrast_delta:
            factor = rng.uniform(1.0 - config.contrast_delta, 1.0 + config.contrast_delta)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if config.saturation_delta:
            factor = rng.uniform(1.0 - config.saturation_delta, 1.0 + config.saturation_delta)
            image = ImageEnhance.Color(image).enhance(factor)

    if config.gaussian_blur_probability and rng.random() < config.gaussian_blur_probability:
        radius = rng.uniform(0.0, config.gaussian_blur_radius)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    if config.gaussian_noise_probability and rng.random() < config.gaussian_noise_probability:
        pixels = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(rng.getrandbits(64)).normal(
            0.0, config.gaussian_noise_std, pixels.shape
        )
        image = Image.fromarray(np.clip(np.rint(pixels + noise), 0, 255).astype(np.uint8))
    return image


def preprocess_pair(
    image: Image.Image,
    mask: Image.Image,
    config: PreprocessingConfig,
    *,
    training: bool = False,
    rng: random.Random | None = None,
    force_horizontal_flip: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """Preprocess one FoodSeg103 pair; masks never receive photometric transforms."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    mask = _validate_mask(mask)
    if image.size != mask.size:
        raise ValueError(
            f"EXIF-corrected image size {image.size} does not match mask size {mask.size}"
        )

    rng = rng or random.Random()
    image = _apply_enhancements(image, config.enhancements)
    width, height = image.size
    scale = min(config.image_size / width, config.image_size / height)
    resized_size = (
        min(config.image_size, max(1, round(width * scale))),
        min(config.image_size, max(1, round(height * scale))),
    )
    image = image.resize(resized_size, Image.Resampling.BILINEAR)
    mask = mask.resize(resized_size, Image.Resampling.NEAREST)

    if training:
        image = _apply_photometric_augmentations(image, config.augmentation, rng)

    left = (config.image_size - resized_size[0]) // 2
    top = (config.image_size - resized_size[1]) // 2
    padding_color = tuple(int(round(channel * 255)) for channel in config.mean)
    image_canvas = Image.new("RGB", (config.image_size, config.image_size), padding_color)
    mask_canvas = Image.new("L", (config.image_size, config.image_size), IGNORE_INDEX)
    image_canvas.paste(image, (left, top))
    mask_canvas.paste(mask, (left, top))

    if force_horizontal_flip is not None:
        should_flip = force_horizontal_flip
    else:
        should_flip = training and rng.random() < config.augmentation.horizontal_flip_probability
    if should_flip:
        image_canvas = image_canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_canvas = mask_canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if training and config.augmentation.scale_probability and (
        rng.random() < config.augmentation.scale_probability
    ):
        scale_factor = rng.uniform(
            config.augmentation.scale_min,
            config.augmentation.scale_max,
        )
        if scale_factor != 1.0:
            center_x = (config.image_size - 1) / 2.0
            center_y = (config.image_size - 1) / 2.0
            inverse_scale = 1.0 / scale_factor
            transform = (
                inverse_scale,
                0.0,
                center_x - inverse_scale * center_x,
                0.0,
                inverse_scale,
                center_y - inverse_scale * center_y,
            )
            image_canvas = image_canvas.transform(
                image_canvas.size,
                Image.Transform.AFFINE,
                transform,
                resample=Image.Resampling.BILINEAR,
                fillcolor=padding_color,
            )
            mask_canvas = mask_canvas.transform(
                mask_canvas.size,
                Image.Transform.AFFINE,
                transform,
                resample=Image.Resampling.NEAREST,
                fillcolor=IGNORE_INDEX,
            )

    if training and config.augmentation.rotation_probability and (
        rng.random() < config.augmentation.rotation_probability
    ):
        angle = rng.uniform(
            -config.augmentation.rotation_degrees,
            config.augmentation.rotation_degrees,
        )
        image_canvas = image_canvas.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=padding_color,
        )
        mask_canvas = mask_canvas.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=False,
            fillcolor=IGNORE_INDEX,
        )

    image_values = np.asarray(image_canvas, dtype=np.float32) / 255.0
    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)
    image_values = (image_values - mean) / std
    pixel_values = torch.from_numpy(
        np.ascontiguousarray(image_values.transpose(2, 0, 1))
    )
    label_values = torch.from_numpy(np.asarray(mask_canvas, dtype=np.uint8).astype(np.int64))
    return pixel_values, label_values, image_canvas
