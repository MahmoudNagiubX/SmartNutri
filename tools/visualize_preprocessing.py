from __future__ import annotations

import argparse
import colorsys
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from foodseg_data import ParquetSplit, load_label_map, prepare_example

DEFAULT_FOODSEG_ROOT = PROJECT_DIR.parent / "Datasets" / "FoodSeg103"
TILE_SIZE = 320


def class_color(class_id: int) -> tuple[int, int, int]:
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.96)
    return round(red * 255), round(green * 255), round(blue * 255)


def fit_to_tile(image: Image.Image) -> Image.Image:
    image = ImageOps.contain(image.convert("RGB"), (TILE_SIZE, TILE_SIZE))
    tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (235, 235, 235))
    tile.paste(image, ((TILE_SIZE - image.width) // 2, (TILE_SIZE - image.height) // 2))
    return tile


def ground_truth_overlay(image: Image.Image, labels: np.ndarray) -> Image.Image:
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    colors = np.zeros_like(source)
    selected = (labels > 0) & (labels < 104)
    for raw_id in np.unique(labels[selected]):
        class_id = int(raw_id)
        colors[labels == class_id] = class_color(class_id)
    blended = source.copy()
    blended[selected] = source[selected] * 0.5 + colors[selected] * 0.5
    blended[labels == 255] = source[labels == 255] * 0.72 + 74 * 0.28
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create read-only before/after panels from local FoodSeg103 samples."
    )
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG_ROOT)
    parser.add_argument("--indices", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "docs" / "assets" / "preprocessing_before_after.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.indices or any(index < 0 for index in args.indices):
        raise ValueError("indices must contain non-negative validation row indices")

    foodseg_root = args.foodseg_root.expanduser().resolve()
    labels_by_id = load_label_map(foodseg_root)
    split = ParquetSplit(foodseg_root, "validation")
    if any(index >= split.num_rows for index in args.indices):
        raise ValueError(f"Validation split contains {split.num_rows} rows")

    title_height = 56
    row_height = TILE_SIZE + 80
    margin = 22
    canvas = Image.new(
        "RGB",
        (margin * 2 + TILE_SIZE * 3 + 32 * 2, margin * 2 + title_height + row_height * len(args.indices)),
        (248, 247, 243),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (margin, margin),
        "FoodSeg103 preprocessing: local validation examples",
        fill=(24, 31, 38),
        font=font,
    )
    headers = (
        "Before: decoded source RGB",
        "After: oriented, resized, padded",
        "After: aligned ground-truth overlay",
    )
    column_x = [margin, margin + TILE_SIZE + 16, margin + (TILE_SIZE + 16) * 2]

    for row_number, index in enumerate(args.indices):
        row = split.row_at(index)
        with Image.open(io.BytesIO(row["image"]["bytes"])) as source_image:
            source_image.load()
            before = source_image.convert("RGB")
        with Image.open(io.BytesIO(row["label"]["bytes"])) as source_mask:
            source_mask.load()
            raw_mask = source_mask.copy()

        _, labels, processed = prepare_example(row, 512)
        row_y = margin + title_height + row_number * row_height
        source_name = row["image"].get("path") or f"split-local id {row['id']}"
        draw.text(
            (margin, row_y),
            f"Validation row {index} | id {row['id']} | {source_name}",
            fill=(48, 58, 67),
            font=font,
        )
        for column, header in enumerate(headers):
            x = column_x[column]
            draw.text((x, row_y + 18), header, fill=(37, 55, 68), font=font)

        target = labels.numpy().astype(np.uint8)
        overlay = ground_truth_overlay(processed, target)
        panels = (before, processed, overlay)
        for column, panel in enumerate(panels):
            tile = fit_to_tile(panel)
            canvas.paste(tile, (column_x[column], row_y + 36))

        class_ids = [int(value) for value in np.unique(np.asarray(raw_mask)) if 0 < value < 104]
        class_names = ", ".join(labels_by_id.get(class_id, str(class_id)) for class_id in class_ids[:5])
        if len(class_ids) > 5:
            class_names += f", +{len(class_ids) - 5} more"
        draw.text(
            (margin, row_y + 36 + TILE_SIZE + 7),
            f"Source-mask classes: {class_names or 'background only'}",
            fill=(72, 78, 82),
            font=font,
        )

    foot_y = canvas.height - margin - 16
    draw.text(
        (margin, foot_y),
        "Real local validation rows; source Parquet data is read-only. Display panels precede tensor normalization.",
        fill=(80, 84, 87),
        font=font,
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
