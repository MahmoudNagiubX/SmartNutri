from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from foodseg_data import FoodSegParquetDataset, load_label_map
from image_processing import load_preprocessing_config
from metrics import evaluate_model, summarize_confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FOODSEG_ROOT = PROJECT_DIR.parent / "Datasets" / "FoodSeg103"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "foodseg103-segformer-b0"
DEFAULT_PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_PREPROCESSING_CONFIG = (
    PROJECT_DIR / "configs" / "image_processing" / "baseline.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SegFormer-B0 on the local FoodSeg103 Parquet files."
    )
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained-model", default=DEFAULT_PRETRAINED_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=DEFAULT_PREPROCESSING_CONFIG,
        help="JSON preprocessing profile; choose safe_augmentation.json to enable train-only augmentation.",
    )
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.early_stopping_patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, early-stopping patience, and batch_size must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1")

    processing_config = load_preprocessing_config(args.preprocessing_config)
    if args.image_size is not None:
        processing_config = replace(processing_config, image_size=args.image_size)
    image_size = processing_config.image_size
    if image_size < 32:
        raise ValueError("image_size must be at least 32 pixels")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device.index} is not available")

    foodseg_root = args.foodseg_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (foodseg_root / "id2label.json").is_file():
        raise FileNotFoundError(f"FoodSeg103 label map not found under {foodseg_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new --output-dir."
        )

    set_seed(args.seed)
    id2label = load_label_map(foodseg_root)
    label2id = {name: class_id for class_id, name in id2label.items()}

    train_dataset = FoodSegParquetDataset(
        foodseg_root,
        "train",
        image_size,
        training=True,
        seed=args.seed,
        processing_config=processing_config,
    )
    validation_dataset = FoodSegParquetDataset(
        foodseg_root,
        "validation",
        image_size,
        training=False,
        seed=args.seed,
        processing_config=processing_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained_model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.semantic_loss_ignore_index = 255
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_updates - warmup_steps)
        return max(0.0, (total_updates - step) / decay_steps)

    scheduler = LambdaLR(optimizer, learning_rate_scale)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": "FoodSeg103",
        "foodseg_root": str(foodseg_root),
        "official_splits": {
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
        },
        "pretrained_model": args.pretrained_model,
        "model": "SegFormer-B0",
        "num_labels": len(id2label),
        "image_size": image_size,
        "preprocessing_config": processing_config.to_dict(),
        "preprocessing_config_path": str(args.preprocessing_config.expanduser().resolve()),
        "padding_label": 255,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision_fp16": amp_enabled,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    history: list[dict] = []
    best_foreground_miou = -1.0
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        total_loss = 0.0
        seen_examples = 0
        num_batches = len(train_loader)

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            batch_examples = pixel_values.shape[0]

            group_start = (step // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps, num_batches - group_start
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            total_loss += float(loss.detach()) * batch_examples
            seen_examples += batch_examples
            scaler.scale(loss / group_size).backward()

            is_update_step = (
                (step + 1) % args.gradient_accumulation_steps == 0
                or step + 1 == num_batches
            )
            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not amp_enabled or scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = total_loss / max(1, seen_examples)
        confusion = evaluate_model(
            model,
            validation_loader,
            len(id2label),
            device,
            use_amp=not args.no_amp,
        )
        validation_summary, _ = summarize_confusion_matrix(confusion, id2label)
        score = validation_summary["mean_iou_foreground"]
        if score is None:
            raise RuntimeError("Validation produced no foreground class pixels")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation": validation_summary,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        if score > best_foreground_miou:
            best_foreground_miou = score
            epochs_without_improvement = 0
            best_dir = output_dir / "best_model"
            model.save_pretrained(best_dir, safe_serialization=True)
            (best_dir / "training_metadata.json").write_text(
                json.dumps(
                    {
                        "best_epoch": epoch + 1,
                        "validation_mean_iou_foreground": score,
                        "image_size": image_size,
                        "seed": args.seed,
                        "preprocessing_config": processing_config.to_dict(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"validation foreground mIoU {score:.4f} | "
            f"foreground Dice {validation_summary['mean_dice_foreground']:.4f} | "
            f"seconds {epoch_record['epoch_seconds']:.1f}",
            flush=True,
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping after {args.early_stopping_patience} epochs without "
                f"validation foreground mIoU improvement.",
                flush=True,
            )
            break

    print(f"Best model: {output_dir / 'best_model'}", flush=True)
    print(f"Training history: {output_dir / 'training_history.json'}", flush=True)


if __name__ == "__main__":
    main()
