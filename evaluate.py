from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from foodseg_data import FoodSegParquetDataset, load_label_map
from image_processing import PreprocessingConfig, load_preprocessing_config
from metrics import evaluate_model, summarize_confusion_matrix
from train import DEFAULT_FOODSEG_ROOT, DEFAULT_PREPROCESSING_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SmartNutri SegFormer checkpoint on FoodSeg103."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG_ROOT)
    parser.add_argument(
        "--split", choices=("validation", "train"), default="validation"
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=None,
        help="Optional preprocessing profile override; otherwise use checkpoint metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Defaults to CUDA when available.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    foodseg_root = args.foodseg_root.expanduser().resolve()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

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

    id2label = load_label_map(foodseg_root)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
    if model.config.num_labels != len(id2label):
        raise ValueError(
            f"Checkpoint has {model.config.num_labels} classes but FoodSeg103 has "
            f"{len(id2label)}"
        )
    model.to(device)

    dataset = FoodSegParquetDataset(
        foodseg_root,
        args.split,
        image_size,
        training=False,
        processing_config=processing_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    confusion = evaluate_model(
        model,
        loader,
        len(id2label),
        device,
        use_amp=not args.no_amp,
    )
    summary, per_class = summarize_confusion_matrix(confusion, id2label)
    report = {
        "dataset": "FoodSeg103",
        "split": args.split,
        "examples": len(dataset),
        "image_size": image_size,
        "checkpoint": str(model_dir),
        "metrics": summary,
    }

    output_dir = args.output_dir or (model_dir / f"evaluation-{args.split}")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)

    print(json.dumps(report, indent=2), flush=True)
    print(f"Per-class metrics: {output_dir / 'per_class_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
