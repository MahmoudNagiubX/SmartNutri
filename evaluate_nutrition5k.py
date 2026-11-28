from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nutrition5k import (
    Nutrition5kDataset,
    NutritionImageTransform,
    TARGETS,
    atomic_write_json,
    collect_predictions,
    evaluation_report,
    load_model_checkpoint,
    load_partition_by_name,
    load_partitions,
    resolve_amp_dtype,
    split_fingerprint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an EfficientNet-B2 Nutrition5k checkpoint without test-set tuning."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "none"), default="auto")
    return parser.parse_args()


def _write_predictions(
    path: Path,
    records,
    predictions: np.ndarray,
    baseline_density: np.ndarray,
) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(
            [
                "dish_id",
                "mass_g",
                *[f"true_{name}_total" for name, *_ in TARGETS],
                *[f"predicted_{name}_density" for name, *_ in TARGETS],
                *[f"predicted_{name}_total" for name, *_ in TARGETS],
                *[f"mean_density_baseline_{name}_total" for name, *_ in TARGETS],
                *[f"model_{name}_signed_error" for name, *_ in TARGETS],
            ]
        )
        for index, record in enumerate(records):
            predicted_total = predictions[index] * record.mass_g
            baseline_total = baseline_density * record.mass_g
            writer.writerow(
                [
                    record.dish_id,
                    record.mass_g,
                    *record.totals,
                    *predictions[index].tolist(),
                    *predicted_total.tolist(),
                    *baseline_total.tolist(),
                    *(predicted_total - np.asarray(record.totals)).tolist(),
                ]
            )
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    model, checkpoint = load_model_checkpoint(args.checkpoint, device=device)
    seed = int(checkpoint["seed"])
    validation_fraction = float(checkpoint["validation_fraction"])
    partitions = load_partitions(
        args.data_root,
        split_dir=args.split_dir,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    manifest = partitions.manifest()
    fingerprint = split_fingerprint(manifest)
    if fingerprint != checkpoint["split_fingerprint"]:
        raise ValueError("Current official split files/data do not match the checkpoint split fingerprint")
    records = load_partition_by_name(partitions, args.partition)
    if not records:
        raise ValueError(f"Nutrition5k {args.partition} partition is empty")

    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    dataset = Nutrition5kDataset(
        records,
        NutritionImageTransform(
            image_size=int(checkpoint["input_size"]),
            training=False,
            augmentation=checkpoint["training_config"]["augmentation"],
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(checkpoint["training_config"]["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    prediction_ids, predictions = collect_predictions(
        model, loader, device, checkpoint["target_scales"], amp_dtype
    )
    by_id = {dish_id: predictions[index] for index, dish_id in enumerate(prediction_ids)}
    if len(by_id) != len(prediction_ids) or set(by_id) != {record.dish_id for record in records}:
        raise RuntimeError("Evaluation loader IDs do not exactly match the chosen partition")
    predictions = np.stack([by_id[record.dish_id] for record in records])
    report = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "partition": args.partition,
        "official_test_used_for_selection": False,
        "split_fingerprint": fingerprint,
        "mean_density_baseline_training_count": len(partitions.training),
        "mean_density_baseline_training_only": checkpoint["baseline_mean_density"],
        "metrics": evaluation_report(
            records,
            predictions,
            np.asarray(checkpoint["baseline_mean_density"], dtype=np.float32),
        ),
    }
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else args.checkpoint.expanduser().resolve().parent / f"evaluation_{args.partition}.json"
    )
    output_csv = output_json.with_suffix(".csv")
    if output_json.exists() or output_csv.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_json} or {output_csv}")
    atomic_write_json(report, output_json)
    _write_predictions(
        output_csv,
        records,
        predictions,
        np.asarray(checkpoint["baseline_mean_density"], dtype=np.float32),
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Per-dish predictions: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
