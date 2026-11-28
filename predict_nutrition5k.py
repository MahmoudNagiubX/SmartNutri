from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from nutrition5k import (
    NutritionImageTransform,
    TARGETS,
    load_model_checkpoint,
    resolve_amp_dtype,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict nutrient densities and meal totals for one RGB food image."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mass-g", type=float, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "none"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mass_g <= 0:
        raise ValueError("--mass-g must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    model, checkpoint = load_model_checkpoint(args.checkpoint, device=device)
    transform = NutritionImageTransform(
        image_size=int(checkpoint["input_size"]),
        training=False,
        augmentation=checkpoint["training_config"]["augmentation"],
    )
    with Image.open(args.image) as source:
        pixels = transform(source.convert("RGB")).unsqueeze(0).to(device)
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=amp_dtype is not None and device.type == "cuda",
    ):
        normalized_density = model(pixels).float().cpu().squeeze(0)
    target_scales = torch.tensor(checkpoint["target_scales"], dtype=torch.float32)
    densities = normalized_density * target_scales
    totals = densities * float(args.mass_g)
    result = {
        "image": str(args.image.expanduser().resolve()),
        "measured_mass_g": args.mass_g,
        "predictions": {
            name: {
                "density": float(densities[index]),
                "density_unit": density_unit,
                "estimated_total": float(totals[index]),
                "total_unit": total_unit,
            }
            for index, (name, _, density_unit, total_unit) in enumerate(TARGETS)
        },
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
