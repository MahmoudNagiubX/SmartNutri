from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import SegformerForSemanticSegmentation

from foodseg_data import ParquetSplit, _image_bytes, load_label_map, prepare_example
from image_processing import load_preprocessing_config
from train_optimized import DEFAULT_FOODSEG_ROOT, PROJECT_DIR, resolve_amp_dtype

DEFAULT_B0 = PROJECT_DIR / "runs" / "foodseg103-segformer-b0" / "best_model"
DEFAULT_B2 = (
    PROJECT_DIR / "runs" / "foodseg103-segformer-b2-optimized-20261124_075454" / "best_model"
)
MEAT_NAMES = {
    "steak",
    "pork",
    "chicken duck",
    "sausage",
    "fried meat",
    "lamb",
    "crab",
    "fish",
    "shellfish",
    "shrimp",
    "hamburg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate actual B0/B2 FoodSeg103 prediction comparisons and training curves."
    )
    parser.add_argument("--b0-model", type=Path, default=DEFAULT_B0)
    parser.add_argument("--b2-model", type=Path, default=DEFAULT_B2)
    parser.add_argument("--b0-history", type=Path, default=DEFAULT_B0.parent / "training_history.json")
    parser.add_argument("--b2-history", type=Path, default=DEFAULT_B2.parent / "training_history.json")
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG_ROOT)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=PROJECT_DIR / "configs" / "image_processing" / "baseline.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "none"), default="fp16")
    parser.add_argument("--num-cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(model_dir: Path, id2label: dict[int, str], device: torch.device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        use_safetensors=True,
        local_files_only=True,
    )
    mapped = {int(key): str(value).strip() for key, value in model.config.id2label.items()}
    if model.config.num_labels != 104 or model.decode_head.classifier.out_channels != 104:
        raise ValueError(f"Expected a 104-class checkpoint: {model_dir}")
    if mapped != id2label:
        raise ValueError(f"FoodSeg103 class map differs in {model_dir}")
    return model.to(device).eval()


@torch.inference_mode()
def predict_mask(model, pixel_values: torch.Tensor, *, device: torch.device, amp_dtype) -> np.ndarray:
    values = pixel_values.unsqueeze(0).to(device, non_blocking=True)
    enabled = amp_dtype is not None
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=enabled,
    ):
        logits = model(pixel_values=values).logits
    prediction = F.interpolate(
        logits.float(),
        size=pixel_values.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)[0]
    return prediction.cpu().numpy().astype(np.uint8, copy=False)


def score_one_image(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    id2label: dict[int, str],
) -> dict[str, Any]:
    valid = labels != 255
    targets = labels[valid].astype(np.int64, copy=False)
    guesses = predictions[valid].astype(np.int64, copy=False)
    matrix = np.bincount(targets * 104 + guesses, minlength=104 * 104).reshape(104, 104)
    true_positive = np.diag(matrix)
    target_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    union = target_count + predicted_count - true_positive
    selected = np.flatnonzero(union[1:] > 0) + 1
    class_ious = {
        int(class_id): float(true_positive[class_id] / union[class_id])
        for class_id in selected
    }
    meat_ids = {class_id for class_id, name in id2label.items() if name.strip().casefold() in MEAT_NAMES}
    meat_pixels = (targets != guesses) & np.isin(targets, list(meat_ids)) & (guesses > 0)
    meat_pairs = Counter(
        (int(target), int(guess))
        for target, guess in zip(targets[meat_pixels], guesses[meat_pixels], strict=True)
    )
    return {
        "mean_iou": float(statistics.fmean(class_ious.values())) if class_ious else 0.0,
        "class_ious": class_ious,
        "ground_truth_classes": [int(x) for x in np.flatnonzero(target_count[1:] > 0) + 1],
        "predicted_classes": [int(x) for x in np.flatnonzero(predicted_count[1:] > 0) + 1],
        "meat_confusion_pixels": int(meat_pixels.sum()),
        "meat_confusion_pairs": dict(meat_pairs),
    }


def _candidate_order(records: list[dict[str, Any]], predicate, score) -> list[dict[str, Any]]:
    selected = [record for record in records if predicate(record)]
    return sorted(selected, key=score)


def choose_cases(
    records: list[dict[str, Any]],
    pixel_support: np.ndarray,
    id2label: dict[int, str],
    *,
    case_count: int,
) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}

    def add(pool: list[dict[str, Any]], tag: str, count: int) -> int:
        added = 0
        for record in pool:
            index = record["row_index"]
            if index in selected:
                continue
            selected[index] = {**record, "selection_tags": [tag]}
            added += 1
            if added == count or len(selected) == case_count:
                break
        return added

    add(
        _candidate_order(
            records,
            lambda r: len(r["b2"]["ground_truth_classes"]) >= 2,
            lambda r: (-r["b2"]["mean_iou"], r["row_index"]),
        ),
        "successful_prediction",
        6,
    )
    add(
        _candidate_order(
            records,
            lambda r: len(r["b2"]["ground_truth_classes"]) >= 2,
            lambda r: (r["b2"]["mean_iou"], r["row_index"]),
        ),
        "difficult_case",
        6,
    )
    add(
        _candidate_order(
            records,
            lambda r: r["b2"]["meat_confusion_pixels"] > 0,
            lambda r: (-r["b2"]["meat_confusion_pixels"], r["b2"]["mean_iou"], r["row_index"]),
        ),
        "meat_confusion",
        6,
    )

    present_foreground = [class_id for class_id in range(1, 104) if pixel_support[class_id] > 0]
    rare_count = max(1, math.ceil(len(present_foreground) * 0.25)) if present_foreground else 0
    rare_ids = sorted(present_foreground, key=lambda class_id: (pixel_support[class_id], class_id))[:rare_count]
    rare_added = 0
    for class_id in rare_ids:
        candidates = _candidate_order(
            records,
            lambda r, c=class_id: c in r["b2"]["ground_truth_classes"],
            lambda r, c=class_id: (
                r["b2"]["class_ious"].get(c, 0.0),
                r["row_index"],
            ),
        )
        new_candidates = [candidate for candidate in candidates if candidate["row_index"] not in selected]
        if not new_candidates:
            continue
        tag = f"rare_class_{class_id}_{_slug(id2label[class_id])}"
        selected[new_candidates[0]["row_index"]] = {
            **new_candidates[0],
            "selection_tags": [tag],
        }
        rare_added += 1
        if rare_added >= 6 or len(selected) >= case_count:
            break

    add(
        _candidate_order(
            records,
            lambda r: len(r["b2"]["ground_truth_classes"]) >= 4,
            lambda r: (
                -len(r["b2"]["ground_truth_classes"]),
                r["b2"]["mean_iou"],
                r["row_index"],
            ),
        ),
        "mixed_food_plate",
        6,
    )

    if len(selected) < case_count:
        ordered = sorted(records, key=lambda r: (r["b2"]["mean_iou"], r["row_index"]))
        cursor = 0
        while len(selected) < case_count and cursor < len(ordered):
            record = ordered[cursor]
            cursor += 1
            if record["row_index"] not in selected:
                selected[record["row_index"]] = {**record, "selection_tags": ["additional_representative"]}
    if len(selected) < case_count:
        raise RuntimeError(f"Only selected {len(selected)} cases; expected {case_count}")
    return list(selected.values())[:case_count]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "class"


def _palette() -> np.ndarray:
    colors = np.zeros((256, 3), dtype=np.uint8)
    for class_id in range(1, 104):
        hue = (class_id * 0.61803398875) % 1.0
        red, green, blue = matplotlib.colors.hsv_to_rgb((hue, 0.78, 0.98))
        colors[class_id] = (round(red * 255), round(green * 255), round(blue * 255))
    colors[255] = (150, 150, 150)
    return colors


def _overlay(image: np.ndarray, mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    colors = palette[mask]
    alpha = np.where(mask == 255, 0.35, np.where((mask > 0) & (mask < 104), 0.48, 0.0))
    blended = image.astype(np.float32) * (1.0 - alpha[..., None]) + colors.astype(np.float32) * alpha[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _top_classes(mask: np.ndarray, id2label: dict[int, str], *, limit: int = 7) -> str:
    values, counts = np.unique(mask[(mask > 0) & (mask < 104)], return_counts=True)
    order = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{id2label[class_id]} ({class_id})" for class_id, _ in order) or "none"


def _save_case(
    path: Path,
    *,
    image: np.ndarray,
    ground_truth: np.ndarray,
    b0_prediction: np.ndarray,
    b2_prediction: np.ndarray,
    b0_score: float,
    b2_score: float,
    sample_id: str,
    tags: list[str],
    id2label: dict[int, str],
    palette: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.4), constrained_layout=True)
    panels = [
        (image, "Input (512×512)"),
        (_overlay(image, ground_truth, palette), "Ground truth"),
        (_overlay(image, b0_prediction, palette), f"B0  |  mIoU {b0_score:.3f}"),
        (_overlay(image, b2_prediction, palette), f"B2  |  mIoU {b2_score:.3f}"),
    ]
    for axis, (pixels, title) in zip(axes, panels, strict=True):
        axis.imshow(pixels)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    gt_names = _top_classes(ground_truth, id2label)
    b0_names = _top_classes(b0_prediction, id2label)
    b2_names = _top_classes(b2_prediction, id2label)
    fig.suptitle(
        f"FoodSeg103 image {sample_id}  |  {', '.join(tags)}\n"
        f"GT: {gt_names}\nB0: {b0_names}\nB2: {b2_names}",
        fontsize=10,
    )
    fig.savefig(path, dpi=115, facecolor="white")
    plt.close(fig)


def _plot_histories(b0_history_path: Path, b2_history_path: Path, output_path: Path) -> dict[str, Any]:
    histories = {
        "B0": json.loads(b0_history_path.read_text(encoding="utf-8")),
        "B2": json.loads(b2_history_path.read_text(encoding="utf-8")),
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    curves = [
        ("train_loss", "Training loss", axes[0, 0]),
        ("mean_iou_foreground", "Internal validation foreground mIoU", axes[0, 1]),
        ("mean_dice_foreground", "Internal validation foreground Dice", axes[1, 0]),
        ("pixel_accuracy", "Internal validation pixel accuracy", axes[1, 1]),
    ]
    best_epochs: dict[str, int] = {}
    for name, history in histories.items():
        epochs = [int(record["epoch"]) for record in history]
        best_epochs[name] = int(max(history, key=lambda record: record["validation"]["mean_iou_foreground"])["epoch"])
        for metric, title, axis in curves:
            values = [
                record["train_loss"] if metric == "train_loss" else record["validation"][metric]
                for record in history
            ]
            axis.plot(epochs, values, marker=".", markersize=3, label=name)
            if name == "B2" and metric != "train_loss":
                best_value = history[best_epochs[name] - 1]["validation"][metric]
                axis.scatter([best_epochs[name]], [best_value], marker="*", s=90, zorder=5)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(True, alpha=0.25)
    for axis in axes.flat:
        axis.legend()
    figure.suptitle("SmartNutri FoodSeg103 training histories (saved run records)")
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)
    return {
        "b0_epochs": len(histories["B0"]),
        "b2_epochs": len(histories["B2"]),
        "b0_best_epoch": best_epochs["B0"],
        "b2_best_epoch": best_epochs["B2"],
    }


def _write_contact_sheet(case_paths: list[Path], output_path: Path, *, columns: int = 5) -> None:
    thumbnails = []
    for path in case_paths:
        with Image.open(path) as opened:
            thumb = opened.convert("RGB")
            thumb.thumbnail((420, 132), Image.Resampling.LANCZOS)
            thumbnails.append(thumb.copy())
    cell_width = 430
    cell_height = 145
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, thumb) in enumerate(zip(case_paths, thumbnails, strict=True)):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(thumb, (x + 4, y + 3))
        draw.text((x + 4, y + 136), path.stem, fill=(20, 20, 20))
    sheet.save(output_path, quality=88, optimize=True)


def main() -> None:
    args = parse_args()
    if args.num_cases != 30:
        raise ValueError("This report workflow generates exactly 30 representative cases")
    b0_dir = args.b0_model.expanduser().resolve()
    b2_dir = args.b2_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    case_dir = output_dir / "visualizations"
    case_dir.mkdir()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    processing_config = load_preprocessing_config(args.preprocessing_config)
    id2label = load_label_map(args.foodseg_root)
    b0_model = load_model(b0_dir, id2label, device)
    b2_model = load_model(b2_dir, id2label, device)
    split = ParquetSplit(args.foodseg_root, "validation")
    records: list[dict[str, Any]] = []
    pixel_support = np.zeros(104, dtype=np.int64)

    print("Scanning held-out images for representative-case selection…", flush=True)
    for row_index, row in enumerate(split.iter_rows(shuffle=False, rng=random.Random(args.seed))):
        pixel_values, labels, _ = prepare_example(
            row,
            processing_config.image_size,
            processing_config=processing_config,
            training=False,
        )
        target = labels.numpy().astype(np.uint8, copy=False)
        valid_target = target[target != 255]
        pixel_support += np.bincount(valid_target.astype(np.int64), minlength=104)
        b0_prediction = predict_mask(b0_model, pixel_values, device=device, amp_dtype=amp_dtype)
        b2_prediction = predict_mask(b2_model, pixel_values, device=device, amp_dtype=amp_dtype)
        records.append(
            {
                "row_index": row_index,
                "sample_id": str(row.get("id", row_index)),
                "b0": score_one_image(target, b0_prediction, id2label=id2label),
                "b2": score_one_image(target, b2_prediction, id2label=id2label),
            }
        )
        if (row_index + 1) % 250 == 0:
            print(f"Scanned {row_index + 1}/{split.num_rows} images", flush=True)

    if len(records) != 2135:
        raise RuntimeError(f"Expected 2,135 validation records, found {len(records)}")
    cases = choose_cases(records, pixel_support, id2label, case_count=args.num_cases)
    print(f"Selected {len(cases)} cases; rendering comparisons…", flush=True)
    palette = _palette()
    case_paths: list[Path] = []
    csv_rows: list[dict[str, Any]] = []
    for case_number, case in enumerate(cases, start=1):
        row = split.row_at(case["row_index"])
        pixel_values, labels, image_canvas = prepare_example(
            row,
            processing_config.image_size,
            processing_config=processing_config,
            training=False,
        )
        b0_prediction = predict_mask(b0_model, pixel_values, device=device, amp_dtype=amp_dtype)
        b2_prediction = predict_mask(b2_model, pixel_values, device=device, amp_dtype=amp_dtype)
        sample_id = _slug(case["sample_id"])
        path = case_dir / f"case_{case_number:02d}_id-{sample_id}.png"
        _save_case(
            path,
            image=np.asarray(image_canvas, dtype=np.uint8),
            ground_truth=labels.numpy().astype(np.uint8, copy=False),
            b0_prediction=b0_prediction,
            b2_prediction=b2_prediction,
            b0_score=case["b0"]["mean_iou"],
            b2_score=case["b2"]["mean_iou"],
            sample_id=case["sample_id"],
            tags=case["selection_tags"],
            id2label=id2label,
            palette=palette,
        )
        case_paths.append(path)
        csv_rows.append(
            {
                "case": case_number,
                "sample_id": case["sample_id"],
                "selection_tags": ";".join(case["selection_tags"]),
                "b0_per_image_foreground_miou": case["b0"]["mean_iou"],
                "b2_per_image_foreground_miou": case["b2"]["mean_iou"],
                "ground_truth_classes": ";".join(
                    f"{c}:{id2label[c]}" for c in case["b2"]["ground_truth_classes"]
                ),
                "b0_predicted_classes": ";".join(
                    f"{c}:{id2label[c]}" for c in case["b0"]["predicted_classes"]
                ),
                "b2_predicted_classes": ";".join(
                    f"{c}:{id2label[c]}" for c in case["b2"]["predicted_classes"]
                ),
                "b2_meat_confusion_pixels": case["b2"]["meat_confusion_pixels"],
            }
        )
        print(f"Rendered {case_number}/{len(cases)}: {case['sample_id']}", flush=True)

    _write_contact_sheet(case_paths, output_dir / "visualization_contact_sheet.jpg")
    with (output_dir / "visual_case_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    history_summary = _plot_histories(
        args.b0_history.expanduser().resolve(),
        args.b2_history.expanduser().resolve(),
        output_dir / "training_curves.png",
    )
    summary = {
        "validation_examples_scanned": len(records),
        "visualizations_generated": len(case_paths),
        "categories": sorted({tag for case in cases for tag in case["selection_tags"]}),
        "class_support": [
            {"class_id": class_id, "class_name": id2label[class_id], "ground_truth_pixels": int(pixel_support[class_id])}
            for class_id in range(104)
        ],
        "training_history": history_summary,
        "visualization_dir": str(case_dir),
        "contact_sheet": str(output_dir / "visualization_contact_sheet.jpg"),
        "training_curves": str(output_dir / "training_curves.png"),
        "case_index": str(output_dir / "visual_case_index.csv"),
    }
    (output_dir / "visual_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "class_support"}, indent=2))


if __name__ == "__main__":
    main()
