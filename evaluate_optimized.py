from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import SegformerForSemanticSegmentation

from foodseg_data import ParquetSplit, _image_bytes, load_label_map, prepare_example
from image_processing import IGNORE_INDEX, _validate_mask, load_preprocessing_config
from metrics import (
    summarize_confusion_matrix,
    summarize_image_presence_confusion,
    update_confusion_matrix,
    update_image_presence_confusion,
)
from train_optimized import DEFAULT_FOODSEG_ROOT, PROJECT_DIR, resolve_amp_dtype

DEFAULT_PREPROCESSING = PROJECT_DIR / "configs" / "image_processing" / "baseline.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SmartNutri segmentation checkpoints with one shared evaluation protocol."
    )
    parser.add_argument("--model-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG_ROOT)
    parser.add_argument("--split", choices=("validation", "external"), default="validation")
    parser.add_argument("--external-root", type=Path, default=None)
    parser.add_argument("--preprocessing-config", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "none"), default="auto")
    parser.add_argument("--original-resolution", action="store_true")
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def create_evaluation_output_dir(root: Path, *, timestamp: str | None = None) -> Path:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = f"segmentation-comparison-{timestamp}"
    suffix = 0
    while True:
        name = base if suffix == 0 else f"{base}-{suffix:02d}"
        candidate = root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1


def iter_external_rows(root: Path) -> Iterator[dict[str, Any]]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"External test set must contain images/ and masks/ directories: {root}"
        )
    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    masks = {path.stem: path for path in mask_dir.glob("*.png") if path.is_file()}
    image_stems = {path.stem for path in images}
    missing_masks = sorted(image_stems - set(masks))
    extra_masks = sorted(set(masks) - image_stems)
    if missing_masks or extra_masks:
        raise ValueError(
            f"External image/mask names must match; missing masks={missing_masks[:5]}, "
            f"unpaired masks={extra_masks[:5]}"
        )
    if not images:
        raise ValueError(f"No supported test images found under {image_dir}")
    for image_path in images:
        yield {
            "id": image_path.stem,
            "image": {"bytes": image_path.read_bytes(), "path": str(image_path)},
            "label": {"bytes": masks[image_path.stem].read_bytes(), "path": str(masks[image_path.stem])},
        }


def iter_evaluation_rows(
    *,
    split: str,
    foodseg_root: Path,
    external_root: Path | None,
) -> Iterator[dict[str, Any]]:
    if split == "external":
        if external_root is None:
            raise ValueError("--external-root is required when --split external is selected")
        yield from iter_external_rows(external_root)
        return
    if external_root is not None:
        raise ValueError("--external-root is only used with --split external")
    source = ParquetSplit(foodseg_root, split="validation")
    yield from source.iter_rows(shuffle=False, rng=random.Random(0))


def resize_logits_argmax(
    logits: torch.Tensor,
    output_size: tuple[int, int],
    *,
    tile_rows: int = 128,
    max_full_logits_values: int = 50_000_000,
) -> torch.Tensor:
    """Upsample logits and argmax, tiling large original-resolution masks."""
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1,C,H,W]")
    height, width = output_size
    if height < 1 or width < 1:
        raise ValueError("output_size dimensions must be positive")
    logits = logits.float()
    if logits.shape[1] * height * width <= max_full_logits_values:
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[0].argmax(dim=0)

    x = torch.arange(width, dtype=logits.dtype, device=logits.device)
    x_grid = (2.0 * (x + 0.5) / width - 1.0).view(1, width).expand(tile_rows, width)
    output = torch.empty((height, width), dtype=torch.long, device=logits.device)
    for row_start in range(0, height, tile_rows):
        rows = min(tile_rows, height - row_start)
        y = torch.arange(row_start, row_start + rows, dtype=logits.dtype, device=logits.device)
        y_grid = (2.0 * (y + 0.5) / height - 1.0).view(rows, 1).expand(rows, width)
        grid = torch.stack((x_grid[:rows], y_grid), dim=-1).unsqueeze(0)
        resized_tile = F.grid_sample(
            logits,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0]
        output[row_start : row_start + rows] = resized_tile.argmax(dim=0)
    return output


def unletterbox_logits_argmax(
    logits: torch.Tensor,
    original_size: tuple[int, int],
    *,
    image_size: int,
) -> torch.Tensor:
    """Remove deterministic square letterboxing before source-size evaluation."""
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1,C,H,W]")
    original_height, original_width = original_size
    if original_height < 1 or original_width < 1 or image_size < 1:
        raise ValueError("image dimensions and image_size must be positive")

    scale = min(image_size / original_width, image_size / original_height)
    resized_width = min(image_size, max(1, round(original_width * scale)))
    resized_height = min(image_size, max(1, round(original_height * scale)))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2

    square_logits = F.interpolate(
        logits.float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    image_logits = square_logits[
        :, :, top : top + resized_height, left : left + resized_width
    ]
    return resize_logits_argmax(image_logits, original_size)


def load_evaluation_model(
    model_dir: Path,
    id2label: dict[int, str],
) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir, use_safetensors=True)
    if model.config.num_labels != len(id2label) or model.decode_head.classifier.out_channels != len(id2label):
        raise ValueError(f"Checkpoint must provide exactly {len(id2label)} FoodSeg103 logits: {model_dir}")
    model_map = {int(key): str(value).strip() for key, value in model.config.id2label.items()}
    if model_map != id2label:
        raise ValueError(f"Checkpoint class mapping differs from FoodSeg103: {model_dir}")
    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    return model


def _forward_logits(
    model: SegformerForSemanticSegmentation,
    pixel_values: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    hflip_tta: bool,
) -> torch.Tensor:
    amp_enabled = amp_dtype is not None
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=amp_enabled,
    ):
        logits = model(pixel_values=pixel_values).logits
        if hflip_tta:
            flipped_logits = model(pixel_values=pixel_values.flip(-1)).logits.flip(-1)
            logits = (logits.float() + flipped_logits.float()) * 0.5
    return logits


def _collect_latency_start(device: torch.device) -> Any:
    if device.type == "cuda":
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return time.perf_counter()


def _collect_latency_end(device: torch.device, start: Any) -> Any:
    if device.type == "cuda":
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        return start, end
    return (time.perf_counter() - start) * 1000.0


def _new_bucket(num_classes: int) -> dict[str, Any]:
    return {
        "confusion_512": torch.zeros((num_classes, num_classes), dtype=torch.int64),
        "confusion_original": torch.zeros((num_classes, num_classes), dtype=torch.int64),
        "presence_confusion": torch.zeros((num_classes, 2, 2), dtype=torch.int64),
        "latencies": [],
    }


def _macro_presence(per_class: list[dict[str, Any]]) -> dict[str, float | None]:
    foreground = per_class[1:]
    precision = [row["presence_precision"] for row in foreground if row["presence_precision"] is not None]
    recall = [row["presence_recall"] for row in foreground if row["presence_recall"] is not None]
    f1 = [row["presence_f1"] for row in foreground if row["presence_f1"] is not None]
    return {
        "mean_foreground_presence_precision": statistics.fmean(precision) if precision else None,
        "mean_foreground_presence_recall": statistics.fmean(recall) if recall else None,
        "mean_foreground_presence_f1": statistics.fmean(f1) if f1 else None,
    }


def _write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_csv(path: Path, confusion: torch.Tensor, id2label: dict[int, str]) -> None:
    matrix = confusion.detach().cpu().numpy()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_class_id", "true_class_name", *[f"predicted_{i}" for i in range(len(id2label))]])
        for class_id, row in enumerate(matrix):
            writer.writerow([class_id, id2label[class_id], *[int(value) for value in row]])


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "model"


def _evaluate_one_model(
    model_dir: Path,
    *,
    rows_factory,
    id2label: dict[int, str],
    processing_config,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    include_original_resolution: bool,
    include_hflip_tta: bool,
    output_dir: Path,
) -> dict[str, Any]:
    model = load_evaluation_model(model_dir, id2label).to(device).eval()
    if device.type == "cuda":
        model_loaded_allocated = torch.cuda.memory_allocated(device)
        model_loaded_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        model_loaded_allocated = None
        model_loaded_reserved = None
    variants = ["single"] + (["hflip_tta"] if include_hflip_tta else [])
    buckets = {variant: _new_bucket(len(id2label)) for variant in variants}
    sample_count = 0
    timing_starts: dict[str, list[Any]] = {variant: [] for variant in variants}
    timing_ends: dict[str, list[Any]] = {variant: [] for variant in variants}

    warmup_input = torch.zeros(
        (1, 3, processing_config.image_size, processing_config.image_size),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for variant in variants:
            for _ in range(5):
                _forward_logits(
                    model,
                    warmup_input,
                    device=device,
                    amp_dtype=amp_dtype,
                    hflip_tta=variant == "hflip_tta",
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del warmup_input

    with torch.inference_mode():
        for row in rows_factory():
            pixel_values, labels_512, _ = prepare_example(
                row,
                processing_config.image_size,
                processing_config=processing_config,
                training=False,
            )
            with Image.open(BytesIO(_image_bytes(row, "label"))) as source_mask:
                source_mask.load()
                original_mask = _validate_mask(source_mask)
                original_array = np.array(original_mask, dtype=np.uint8, copy=True)
            with Image.open(BytesIO(_image_bytes(row, "image"))) as source_image:
                oriented_image = ImageOps.exif_transpose(source_image)
                oriented_size = (oriented_image.height, oriented_image.width)
            if oriented_size != original_mask.size[::-1]:
                raise ValueError(
                    f"Image/mask source dimensions differ after EXIF orientation for "
                    f"{row.get('id')}: image={oriented_size[::-1]}, mask={original_mask.size}"
                )
            labels_original = torch.from_numpy(original_array.astype(np.int64, copy=False))
            model_input = pixel_values.unsqueeze(0).to(device, non_blocking=True)
            true_512 = labels_512.unsqueeze(0)
            true_original = labels_original.unsqueeze(0)
            sample_count += 1

            for variant in variants:
                is_tta = variant == "hflip_tta"
                start_event = _collect_latency_start(device)
                logits = _forward_logits(
                    model,
                    model_input,
                    device=device,
                    amp_dtype=amp_dtype,
                    hflip_tta=is_tta,
                )
                end_event = _collect_latency_end(device, start_event)
                if device.type == "cuda":
                    timing_starts[variant].append(end_event[0])
                    timing_ends[variant].append(end_event[1])
                else:
                    buckets[variant]["latencies"].append(end_event)

                logits_for_scoring = logits.detach().float()
                prediction_512 = resize_logits_argmax(
                    logits_for_scoring,
                    tuple(labels_512.shape[-2:]),
                ).cpu()
                update_confusion_matrix(
                    buckets[variant]["confusion_512"],
                    prediction_512.unsqueeze(0),
                    true_512,
                )
                update_image_presence_confusion(
                    buckets[variant]["presence_confusion"],
                    prediction_512.unsqueeze(0),
                    true_512,
                    ignore_index=IGNORE_INDEX,
                )
                if include_original_resolution:
                    prediction_original = unletterbox_logits_argmax(
                        logits_for_scoring,
                        tuple(labels_original.shape[-2:]),
                        image_size=processing_config.image_size,
                    ).cpu()
                    update_confusion_matrix(
                        buckets[variant]["confusion_original"],
                        prediction_original.unsqueeze(0),
                        true_original,
                    )

    if sample_count == 0:
        raise ValueError("Evaluation split contains no examples")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        for variant in variants:
            buckets[variant]["latencies"] = [
                start.elapsed_time(end)
                for start, end in zip(timing_starts[variant], timing_ends[variant], strict=True)
            ]
        gpu_memory = {
            "device_name": torch.cuda.get_device_name(device),
            "model_loaded_allocated_bytes": model_loaded_allocated,
            "model_loaded_reserved_bytes": model_loaded_reserved,
            "peak_allocated_bytes_during_warmup_and_inference": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes_during_warmup_and_inference": torch.cuda.max_memory_reserved(device),
            "inference_batch_size": 1,
        }
    else:
        gpu_memory = None

    model_name = model_dir.parent.name if model_dir.name == "best_model" else model_dir.name
    model_slug = _safe_slug(model_name)
    output: dict[str, Any] = {
        "checkpoint": str(model_dir),
        "examples": sample_count,
        "gpu_memory": gpu_memory,
        "variants": {},
    }
    for variant, bucket in buckets.items():
        summary_512, per_class_512 = summarize_confusion_matrix(
            bucket["confusion_512"], id2label
        )
        presence_per_class = summarize_image_presence_confusion(
            bucket["presence_confusion"], id2label
        )
        presence_macro = _macro_presence(presence_per_class)
        class_coverage = {
            "foreground_classes_present_in_ground_truth": sum(
                row["ground_truth_pixels"] > 0 for row in per_class_512[1:]
            ),
            "foreground_classes_predicted": sum(
                row["predicted_pixels"] > 0 for row in per_class_512[1:]
            ),
            "foreground_classes_with_true_positive_pixels": sum(
                row["true_positive_pixels"] > 0 for row in per_class_512[1:]
            ),
        }
        positive_support = [row for row in per_class_512[1:] if row["ground_truth_pixels"] > 0]
        rare_count = max(1, math.ceil(len(positive_support) * 0.25)) if positive_support else 0
        rare_ids = {
            row["class_id"]
            for row in sorted(
                positive_support,
                key=lambda value: (value["ground_truth_pixels"], value["class_id"]),
            )[:rare_count]
        }
        for row in per_class_512:
            row["is_rare_by_evaluation_support"] = row["class_id"] in rare_ids
        rare_summary = {
            "definition": "bottom quartile of foreground classes by ground-truth pixel support on this split",
            "class_count": len(rare_ids),
            "classes_with_predictions": sum(
                row["class_id"] in rare_ids and row["predicted_pixels"] > 0
                for row in per_class_512
            ),
            "classes_with_true_positive_pixels": sum(
                row["class_id"] in rare_ids and row["true_positive_pixels"] > 0
                for row in per_class_512
            ),
            "class_ids": sorted(rare_ids),
        }
        latency = bucket["latencies"]
        if latency:
            latency_summary = {
                "measurement": "model_forward_only_batch_1; preprocessing and transfer excluded",
                "tta_forward_passes": 2 if variant == "hflip_tta" else 1,
                "images": sample_count,
                "mean_ms_per_image": statistics.fmean(latency),
                "p50_ms_per_image": statistics.median(latency),
                "p95_ms_per_image": float(np.percentile(latency, 95)),
            }
        else:
            latency_summary = {
                "measurement": "model_forward_only_batch_1; preprocessing and transfer excluded",
                "tta_forward_passes": 2 if variant == "hflip_tta" else 1,
                "images": sample_count,
                "mean_ms_per_image": None,
                "p50_ms_per_image": None,
                "p95_ms_per_image": None,
            }
        variant_summary = {
            "input_resolution_metrics": summary_512,
            "class_coverage": class_coverage,
            "ingredient_presence_metrics": presence_macro,
            "ingredient_presence_resolution": "512x512 preprocessed input",
            "rare_class_coverage": rare_summary,
            "inference_latency": latency_summary,
        }
        if include_original_resolution:
            summary_original, per_class_original = summarize_confusion_matrix(
                bucket["confusion_original"], id2label
            )
            for row in per_class_original:
                row["is_rare_by_evaluation_support"] = row["class_id"] in rare_ids
            variant_summary["original_resolution_metrics"] = summary_original
            _write_per_class_csv(
                output_dir / f"{model_slug}__{variant}__original_per_class.csv",
                per_class_original,
            )
            _write_confusion_csv(
                output_dir / f"{model_slug}__{variant}__original_confusion.csv",
                bucket["confusion_original"],
                id2label,
            )
        _write_per_class_csv(
            output_dir / f"{model_slug}__{variant}__512_per_class.csv",
            per_class_512,
        )
        _write_per_class_csv(
            output_dir / f"{model_slug}__{variant}__presence_per_class.csv",
            presence_per_class,
        )
        _write_confusion_csv(
            output_dir / f"{model_slug}__{variant}__512_confusion.csv",
            bucket["confusion_512"],
            id2label,
        )
        output["variants"][variant] = variant_summary

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def main() -> None:
    args = parse_args()
    model_dirs = [path.expanduser().resolve() for path in args.model_dirs]
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")
    if args.image_size < 32:
        raise ValueError("image-size must be at least 32")
    if args.split == "external" and args.external_root is None:
        raise ValueError("--external-root is required for an independent real-world test set")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    id2label = load_label_map(args.foodseg_root.expanduser().resolve())
    processing_config = load_preprocessing_config(args.preprocessing_config)
    if args.image_size != processing_config.image_size:
        processing_config = type(processing_config).from_dict(
            {**processing_config.to_dict(), "image_size": args.image_size}
        )

    output_dir = (
        create_evaluation_output_dir(PROJECT_DIR / "runs")
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    if args.output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)

    def rows_factory():
        return iter_evaluation_rows(
            split=args.split,
            foodseg_root=args.foodseg_root.expanduser().resolve(),
            external_root=(args.external_root.expanduser().resolve() if args.external_root else None),
        )

    results = []
    for model_dir in model_dirs:
        print(f"Evaluating {model_dir}", flush=True)
        result = _evaluate_one_model(
            model_dir,
            rows_factory=rows_factory,
            id2label=id2label,
            processing_config=processing_config,
            device=device,
            amp_dtype=amp_dtype,
            include_original_resolution=args.original_resolution,
            include_hflip_tta=args.hflip_tta,
            output_dir=output_dir,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    report = {
        "dataset": "FoodSeg103" if args.split == "validation" else "SmartNutri real-world overhead-camera test set",
        "split": args.split,
        "examples": results[0]["examples"] if results else 0,
        "input_size": processing_config.image_size,
        "preprocessing_config": processing_config.to_dict(),
        "device": str(device),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "evaluation_protocol": {
            "all_checkpoints_share_the_same_rows_and_preprocessing": True,
            "original_resolution_enabled": args.original_resolution,
            "horizontal_flip_tta_enabled": args.hflip_tta,
            "tta_reported_as_a_separate_variant": True,
            "latency_batch_size": 1,
            "latency_includes_preprocessing": False,
            "gpu_memory_includes_model_load": False,
            "gpu_memory_includes_warmup_and_inference": True,
        },
        "models": results,
        "output_dir": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation report: {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
