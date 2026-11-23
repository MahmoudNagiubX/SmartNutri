from __future__ import annotations

import argparse
import colorsys
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import SegformerForSemanticSegmentation

from foodseg_data import IGNORE_INDEX, ParquetSplit, load_label_map, prepare_example
from image_processing import PreprocessingConfig, load_preprocessing_config
from train import DEFAULT_FOODSEG_ROOT, DEFAULT_PREPROCESSING_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save validation images with ground-truth and predicted masks."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG_ROOT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=None,
        help="Optional preprocessing profile override; otherwise use checkpoint metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Defaults to CUDA when available.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def class_color(class_id: int) -> tuple[int, int, int]:
    if class_id == 0:
        return (0, 0, 0)
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.96)
    return round(red * 255), round(green * 255), round(blue * 255)


def overlay_mask(image: Image.Image, labels: np.ndarray) -> Image.Image:
    original = np.asarray(image, dtype=np.float32)
    colored = np.zeros_like(original)
    alpha = np.zeros(labels.shape, dtype=np.float32)
    for raw_id in np.unique(labels):
        class_id = int(raw_id)
        if class_id <= 0 or class_id == IGNORE_INDEX:
            continue
        selected = labels == class_id
        colored[selected] = class_color(class_id)
        alpha[selected] = 0.48
    blended = original * (1.0 - alpha[..., None]) + colored * alpha[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def save_comparison(
    original: Image.Image,
    target: np.ndarray,
    prediction: np.ndarray,
    id2label: dict[int, str],
    output_path: Path,
) -> list[int]:
    image_size = original.width
    panels = [
        original,
        overlay_mask(original, target),
        overlay_mask(original, prediction),
    ]
    canvas = Image.new("RGB", (image_size * 3, image_size + 78), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    titles = ("Image", "Ground truth", "SegFormer prediction")
    for column, (panel, title) in enumerate(zip(panels, titles)):
        x = column * image_size
        draw.text((x + 10, 8), title, fill=(20, 20, 20))
        canvas.paste(panel, (x, 30))

    present = Counter(
        int(class_id)
        for class_id in np.concatenate((target.reshape(-1), prediction.reshape(-1)))
        if int(class_id) not in {0, IGNORE_INDEX}
    )
    top_classes = [class_id for class_id, _ in present.most_common(8)]
    legend = "Classes: " + " | ".join(
        f"{class_id} {id2label.get(class_id, str(class_id))}"
        for class_id in top_classes
    )
    if not top_classes:
        legend = "No foreground classes in this sample"
    draw.text((10, image_size + 43), legend[:240], fill=(30, 30, 30))
    canvas.save(output_path)
    return top_classes


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.start_index < 0:
        raise ValueError("count must be positive and start_index cannot be negative")

    model_dir = args.model_dir.expanduser().resolve()
    foodseg_root = args.foodseg_root.expanduser().resolve()
    metadata_path = model_dir / "training_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    if args.preprocessing_config is not None:
        processing_config = load_preprocessing_config(args.preprocessing_config)
    elif isinstance(metadata.get("preprocessing_config"), dict):
        processing_config = PreprocessingConfig.from_dict(metadata["preprocessing_config"])
    else:
        processing_config = load_preprocessing_config(DEFAULT_PREPROCESSING_CONFIG)
        if "image_size" in metadata:
            processing_config = replace(
                processing_config, image_size=int(metadata["image_size"])
            )
    if args.image_size is not None:
        processing_config = replace(processing_config, image_size=args.image_size)
    image_size = processing_config.image_size
    if image_size < 32:
        raise ValueError("image_size must be at least 32 pixels")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    id2label = load_label_map(foodseg_root)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
    if model.config.num_labels != len(id2label):
        raise ValueError(
            f"Checkpoint has {model.config.num_labels} classes but FoodSeg103 has "
            f"{len(id2label)}"
        )
    model.eval()

    split = ParquetSplit(foodseg_root, "validation")
    if args.start_index + args.count > split.num_rows:
        raise ValueError(
            f"Requested validation examples through index {args.start_index + args.count - 1}, "
            f"but the split contains {split.num_rows} rows"
        )

    output_dir = args.output_dir or (model_dir / "visualizations")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_enabled = device.type == "cuda" and not args.no_amp
    report = []

    for index in range(args.start_index, args.start_index + args.count):
        row = split.row_at(index)
        pixel_values, labels, original = prepare_example(
            row, image_size, processing_config=processing_config
        )
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(pixel_values=pixel_values.unsqueeze(0).to(device))
            logits = F.interpolate(
                outputs.logits.float(),
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        target = labels.numpy().astype(np.uint8)
        prediction[target == IGNORE_INDEX] = IGNORE_INDEX
        output_path = output_dir / f"validation_{index:05d}.png"
        present_classes = save_comparison(
            original, target, prediction, id2label, output_path
        )
        report.append(
            {
                "validation_index": index,
                "split_local_id": int(row["id"]),
                "source_image_name": row["image"].get("path"),
                "classes_shown": present_classes,
                "output": str(output_path),
            }
        )
        print(f"Saved {output_path}", flush=True)

    (output_dir / "visualizations.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
