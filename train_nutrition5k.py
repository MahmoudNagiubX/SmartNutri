from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nutrition5k import (
    IMAGE_MEAN,
    IMAGE_STD,
    TARGETS,
    Nutrition5kDataset,
    NutritionImageTransform,
    atomic_torch_save,
    atomic_write_json,
    build_model,
    collect_predictions,
    evaluation_report,
    fit_target_scales,
    load_partitions,
    mean_training_density,
    resolve_amp_dtype,
    split_fingerprint,
    validation_score,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "training" / "nutrition5k_efficientnet_b2.json"
DEFAULT_DATA_ROOT = PROJECT_DIR.parent / "Datasets" / "Nutrition5k_RGB"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ImageNet-pretrained EfficientNet-B2 nutrient-density model."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _configuration(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if config.get("architecture") != "torchvision-efficientnet-b2":
        raise ValueError("Nutrition5k training is configured for torchvision EfficientNet-B2 only")
    if config.get("target_order") != [name for name, *_ in TARGETS]:
        raise ValueError("Nutrition5k target order must be calories, protein, carbohydrates, fat")
    training = config["training"]
    if training["validation_group"] != "dish_id_utc_day":
        raise ValueError("Nutrition5k validation must keep UTC acquisition-day groups intact")
    if config.get("augmentation") != {
        "horizontal_flip_probability": 0.5,
        "color_jitter_probability": 0.7,
        "rotation_probability": 0.25,
        "rotation_degrees": 5.0,
    }:
        raise ValueError("Nutrition5k augmentation must remain moderate and reproducible")
    if training["batch_size"] * training["gradient_accumulation_steps"] != training["effective_batch_size"]:
        raise ValueError("Physical batch times accumulation must equal effective_batch_size")
    if training["effective_batch_size"] != 8:
        raise ValueError("The approved Nutrition5k effective batch size is 8")
    if training["max_epochs"] < 1 or training["early_stopping_patience"] < 1:
        raise ValueError("max_epochs and early_stopping_patience must be positive")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return config, hashlib.sha256(encoded).hexdigest()


def _loader(
    records,
    *,
    image_size: int,
    batch_size: int,
    training: bool,
    augmentation: dict[str, float],
    num_workers: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = Nutrition5kDataset(
        records,
        NutritionImageTransform(
            image_size=image_size, training=training, augmentation=augmentation
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _match_prediction_order(records, prediction_ids, predictions: np.ndarray) -> np.ndarray:
    if len(prediction_ids) != len(predictions) or len(set(prediction_ids)) != len(prediction_ids):
        raise RuntimeError("Prediction loader returned duplicate or misaligned dish IDs")
    by_id = {dish_id: predictions[index] for index, dish_id in enumerate(prediction_ids)}
    expected = {record.dish_id for record in records}
    if set(by_id) != expected:
        raise RuntimeError("Prediction IDs do not match the requested dataset partition")
    return np.stack([by_id[record.dish_id] for record in records])


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    target_scales: torch.Tensor,
    training_config: dict[str, Any],
) -> dict[str, float | int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    batch_size = int(training_config["batch_size"])
    accumulation_steps = int(training_config["gradient_accumulation_steps"])
    loss_total = 0.0
    examples_seen = 0
    optimizer_updates = 0
    started = time.perf_counter()

    for step, batch in enumerate(loader):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        densities = batch["density"].to(device, non_blocking=True)
        batch_examples = pixels.shape[0]
        group_start = (step // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, num_batches - group_start)
        group_examples = min(
            batch_size * group_size,
            len(loader.dataset) - group_start * batch_size,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            normalized_prediction = model(pixels)
            normalized_target = densities / target_scales
            loss = F.smooth_l1_loss(
                normalized_prediction,
                normalized_target,
                beta=float(training_config["smooth_l1_beta"]),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite normalized SmoothL1 loss at batch {step}")
        loss_total += float(loss.detach()) * batch_examples
        examples_seen += batch_examples
        scaler.scale(loss * batch_examples / group_examples).backward()

        should_update = (step + 1) % accumulation_steps == 0 or step + 1 == num_batches
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1

    return {
        "mean_normalized_smooth_l1": loss_total / max(1, examples_seen),
        "examples_seen": examples_seen,
        "optimizer_updates": optimizer_updates,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = _parse_args()
    config, config_fingerprint = _configuration(args.config)
    training_config = config["training"]
    data_root = args.data_root.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    seed = int(training_config["seed"])
    _set_seed(seed)

    partitions = load_partitions(
        data_root,
        split_dir=args.split_dir,
        seed=seed,
        validation_fraction=float(training_config["validation_fraction"]),
    )
    manifest = partitions.manifest()
    fingerprint = split_fingerprint(manifest)
    train_records = partitions.training
    validation_records = partitions.validation
    target_scales = fit_target_scales(train_records)
    baseline_density = mean_training_density(train_records)

    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        output_dir = resume_path.parent
        if args.output_dir is not None and args.output_dir.expanduser().resolve() != output_dir:
            raise ValueError("--output-dir must match the parent directory of --resume")
        saved_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
        if saved_config.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Resume configuration differs from the saved run")
        if Path(saved_config.get("data_root", "")).resolve() != data_root:
            raise ValueError("Resume dataset root differs from the saved run")
        if saved_config.get("split_fingerprint") != fingerprint:
            raise ValueError("Resume data split differs from the saved run")
    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required for a new Nutrition5k training run")
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        run_config = {
            "dataset": "Nutrition5k RGB",
            "data_root": str(data_root),
            "split_dir": str(
                Path(args.split_dir).expanduser().resolve()
                if args.split_dir is not None
                else data_root / "dish_ids" / "splits"
            ),
            "architecture": config["architecture"],
            "weights": config["weights"],
            "target_order": config["target_order"],
            "config": config,
            "config_fingerprint": config_fingerprint,
            "split_fingerprint": fingerprint,
            "split_counts": manifest["counts"],
            "target_scales_fitted_from_training_partition": target_scales.tolist(),
            "mean_density_baseline_from_training_partition": baseline_density.tolist(),
            "image_normalization": {"mean": IMAGE_MEAN, "std": IMAGE_STD},
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
        }
        atomic_write_json(run_config, output_dir / "run_config.json")
        atomic_write_json(manifest, output_dir / "split_manifest.json")

    log_path = output_dir / "training.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as sink:
            sink.write(message.rstrip() + "\n")
        print(message, flush=True)

    model = build_model(pretrained=args.resume is None).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config["scheduler_factor"]),
        patience=int(training_config["scheduler_patience"]),
        min_lr=float(training_config["minimum_learning_rate"]),
    )
    amp_dtype_name = training_config["amp_dtype"]
    amp_dtype = resolve_amp_dtype(device, amp_dtype_name)
    amp_enabled = amp_dtype is not None
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    scales_device = torch.as_tensor(target_scales, dtype=torch.float32, device=device)

    history: list[dict[str, Any]] = []
    best_score = math.inf
    epochs_without_improvement = 0
    start_epoch = 0
    if args.resume is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Resume checkpoint config fingerprint does not match")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["completed_epoch"])
        best_score = float(state["best_validation_score"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = list(state["history"])
        _restore_rng_state(state["rng_state"])
        log(f"Resuming {output_dir} at epoch {start_epoch + 1}")
    else:
        validation_loader = _loader(
            validation_records,
            image_size=config["image_size"],
            batch_size=training_config["batch_size"],
            training=False,
            augmentation=config["augmentation"],
            num_workers=training_config["num_workers"],
            device=device,
        )
        prediction_ids, initial_predictions = collect_predictions(
            model, validation_loader, device, target_scales, amp_dtype
        )
        initial_predictions = _match_prediction_order(
            validation_records, prediction_ids, initial_predictions
        )
        initial_report = evaluation_report(
            validation_records, initial_predictions, baseline_density
        )
        initial_report["model_selection_score_normalized_density_mae"] = validation_score(
            validation_records, initial_predictions, target_scales
        )
        atomic_write_json(initial_report, output_dir / "initial_validation.json")
        atomic_write_json(
            {
                "target_scales": target_scales.tolist(),
                "training_sample_count": len(train_records),
                "mean_density_baseline": baseline_density.tolist(),
                "target_order": config["target_order"],
            },
            output_dir / "training_target_scaling.json",
        )
        log(
            f"Warm-start validation: n={len(validation_records)}, "
            f"normalized_density_mae={initial_report['model_selection_score_normalized_density_mae']:.6f}, "
            f"amp={amp_dtype}, device={device}"
        )

    train_dataset = Nutrition5kDataset(
        train_records,
        NutritionImageTransform(
            image_size=config["image_size"],
            training=True,
            augmentation=config["augmentation"],
        ),
    )
    validation_loader = _loader(
        validation_records,
        image_size=config["image_size"],
        batch_size=training_config["batch_size"],
        training=False,
        augmentation=config["augmentation"],
        num_workers=training_config["num_workers"],
        device=device,
    )
    total_epochs = int(training_config["max_epochs"])
    batch_size = int(training_config["batch_size"])
    accumulation_steps = int(training_config["gradient_accumulation_steps"])
    log(
        f"Training EfficientNet-B2 on {len(train_records)} dishes; "
        f"validation={len(validation_records)}; official test reserved; "
        f"physical_batch={batch_size}, accumulation={accumulation_steps}, "
        f"effective_batch={training_config['effective_batch_size']}, "
        f"amp={amp_dtype}, output={output_dir}"
    )

    for epoch in range(start_epoch, total_epochs):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=training_config["num_workers"],
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        epoch_started = time.perf_counter()
        train_result = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            amp_dtype=amp_dtype,
            target_scales=scales_device,
            training_config=training_config,
        )
        train_loss = float(train_result["mean_normalized_smooth_l1"])
        prediction_ids, validation_predictions = collect_predictions(
            model, validation_loader, device, target_scales, amp_dtype
        )
        validation_predictions = _match_prediction_order(
            validation_records, prediction_ids, validation_predictions
        )
        validation_report = evaluation_report(
            validation_records, validation_predictions, baseline_density
        )
        score = validation_score(validation_records, validation_predictions, target_scales)
        scheduler.step(score)
        improved = score < best_score
        if improved:
            best_score = score
            epochs_without_improvement = 0
            best_checkpoint = {
                "model_state_dict": _cpu_state_dict(model),
                "target_scales": target_scales.tolist(),
                "baseline_mean_density": baseline_density.tolist(),
                "target_order": config["target_order"],
                "input_size": config["image_size"],
                "image_mean": list(IMAGE_MEAN),
                "image_std": list(IMAGE_STD),
                "architecture": config["architecture"],
                "weights": config["weights"],
                "training_config": config,
                "config_fingerprint": config_fingerprint,
                "split_fingerprint": fingerprint,
                "seed": seed,
                "validation_fraction": training_config["validation_fraction"],
                "best_epoch": epoch + 1,
                "best_validation_score_normalized_density_mae": best_score,
            }
            atomic_torch_save(best_checkpoint, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch + 1,
            "train_normalized_smooth_l1": train_loss,
            "optimizer_updates": train_result["optimizer_updates"],
            "validation_model_selection_score_normalized_density_mae": score,
            "validation": validation_report,
            "best_validation_score": best_score,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        atomic_write_json(history, output_dir / "training_history.json")
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_epoch": epoch + 1,
            "best_validation_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "rng_state": _rng_state(),
            "config_fingerprint": config_fingerprint,
            "split_fingerprint": fingerprint,
            "data_root": str(data_root),
        }
        atomic_torch_save(state, output_dir / "last_training_state.pt")
        log(json.dumps(record, separators=(",", ":")))
        if epochs_without_improvement >= training_config["early_stopping_patience"]:
            log(
                f"Early stopping after {epoch + 1} epochs; "
                f"best normalized density MAE={best_score:.6f}"
            )
            break

    log(f"Best inference checkpoint: {output_dir / 'best_model.pt'}")
    log(f"Resume checkpoint: {output_dir / 'last_training_state.pt'}")


if __name__ == "__main__":
    main()
