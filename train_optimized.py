from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from class_weighting import compute_class_weights, count_training_pixel_frequencies
from foodseg_data import FoodSegParquetDataset, ParquetSplit, load_label_map
from image_processing import IGNORE_INDEX, PreprocessingConfig, load_preprocessing_config
from metrics import evaluate_model, summarize_confusion_matrix
from training_losses import FocalDiceLoss

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "training" / "segformer_b2_foodseg103.json"
DEFAULT_FOODSEG_ROOT = PROJECT_DIR.parent / "Datasets" / "FoodSeg103"
DEFAULT_RUNS_DIR = PROJECT_DIR / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the single optimized SegFormer-B2 FoodSeg103 configuration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a run's last_training_state.pt checkpoint.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Start a new fine-tuning run from a local FoodSeg103 best_model directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def deterministic_train_validation_split(
    sample_ids: Iterable[int | str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int | str], list[int | str]]:
    """Carve a stable internal holdout from training IDs only."""
    ids = list(sample_ids)
    normalized_ids = [str(value) for value in ids]
    if len(set(normalized_ids)) != len(ids):
        raise ValueError("training sample IDs must be unique")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if len(ids) < 2:
        raise ValueError("at least two training sample IDs are required")

    ids.sort(key=lambda value: (type(value).__name__, str(value)))
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
    validation_set = {str(value) for value in shuffled[:validation_count]}
    train_ids = [value for value in ids if str(value) not in validation_set]
    validation_ids = [value for value in ids if str(value) in validation_set]
    return train_ids, validation_ids


def create_timestamped_run_dir(
    root: Path,
    run_name: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Create a never-reused run directory, adding a suffix on timestamp collision."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_name = f"{run_name}-{timestamp}"
    suffix = 0
    while True:
        candidate_name = base_name if suffix == 0 else f"{base_name}-{suffix:02d}"
        candidate = root / candidate_name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1


def load_training_config(path: Path) -> tuple[dict[str, Any], PreprocessingConfig, str]:
    config_path = path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Training configuration must be a JSON object: {config_path}")
    preprocessing_path = (config_path.parent / config["preprocessing_profile"]).resolve()
    processing_config = load_preprocessing_config(preprocessing_path)
    config["preprocessing_profile"] = str(preprocessing_path)
    config["preprocessing_config"] = processing_config.to_dict()
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    validate_training_config(config, processing_config)
    return config, processing_config, fingerprint


def validate_training_config(
    config: dict[str, Any],
    processing_config: PreprocessingConfig,
) -> None:
    expected_classes = config["dataset"]["num_classes"]
    if expected_classes != 104:
        raise ValueError("FoodSeg103 training requires exactly 104 output classes")
    if config["dataset"]["ignore_index"] != IGNORE_INDEX:
        raise ValueError(f"FoodSeg103 padding must use ignore_index={IGNORE_INDEX}")
    if config["architecture"]["name"] != "SegFormer-B2":
        raise ValueError("This optimized training configuration is fixed to SegFormer-B2")
    pretrained = config["pretrained"]
    expected_artifact = (
        "model.safetensors" if pretrained["use_safetensors"] else "pytorch_model.bin"
    )
    if pretrained["artifact_file"] != expected_artifact:
        raise ValueError(
            "artifact_file and use_safetensors must select the same checkpoint format"
        )
    if config["training"]["input_size"] != processing_config.image_size:
        raise ValueError("input_size must match the preprocessing profile")
    if config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"] != 8:
        raise ValueError("effective batch size must remain 8")
    if config["dataset"]["internal_validation_fraction"] != 0.10:
        raise ValueError("internal model-selection split must remain 10% of train")
    if config["dataset"]["internal_validation_seed"] != config["training"]["seed"]:
        raise ValueError("internal-validation and training seeds must match")
    if config["training"]["max_epochs"] < 1 or config["training"]["early_stopping_patience"] < 1:
        raise ValueError("max_epochs and early_stopping_patience must be positive")
    if config["loss"]["class_weight_cap"] > 2.0:
        raise ValueError("class_weight_cap must remain moderate (at most 2.0)")
    training = config["training"]
    has_encoder_lr = "encoder_learning_rate" in training
    has_decoder_lr = "decoder_learning_rate" in training
    if has_encoder_lr != has_decoder_lr:
        raise ValueError("encoder_learning_rate and decoder_learning_rate must be set together")
    for key in ("learning_rate", "encoder_learning_rate", "decoder_learning_rate"):
        if key in training and (not math.isfinite(training[key]) or training[key] <= 0):
            raise ValueError(f"{key} must be a finite positive value")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_amp_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    if requested == "none" or device.type != "cuda":
        return None
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 AMP was requested but the CUDA device does not support it")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    raise ValueError(f"Unknown amp_dtype: {requested}")


def _normalise_model_label_map(raw: dict[Any, Any]) -> dict[int, str]:
    return {int(class_id): str(name).strip() for class_id, name in raw.items()}


def _validate_foodseg_b2_model(
    model: SegformerForSemanticSegmentation,
    id2label: dict[int, str],
) -> SegformerForSemanticSegmentation:
    label2id = {name: class_id for class_id, name in id2label.items()}
    model_id2label = _normalise_model_label_map(model.config.id2label or {})
    model_label2id = {
        str(name).strip(): int(class_id)
        for name, class_id in (model.config.label2id or {}).items()
    }
    if model.config.num_labels != 104 or model.decode_head.classifier.out_channels != 104:
        raise RuntimeError("Loaded SegFormer-B2 does not expose exactly 104 output channels")
    if model_id2label != id2label or model_label2id != label2id:
        raise RuntimeError("Loaded model label IDs do not match the FoodSeg103 label map")
    depths = list(model.config.depths)
    hidden_sizes = list(model.config.hidden_sizes)
    decoder_hidden_size = int(model.config.decoder_hidden_size)
    if (
        depths != [3, 4, 6, 3]
        or hidden_sizes != [64, 128, 320, 512]
        or decoder_hidden_size != 768
    ):
        raise RuntimeError(
            "Expected MiT-B2 depths/sizes and decoder width 768, found "
            f"depths={depths}, hidden_sizes={hidden_sizes}, "
            f"decoder_hidden_size={decoder_hidden_size}"
        )
    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    return model


def load_foodseg_b2_model(
    pretrained: dict[str, Any],
    id2label: dict[int, str],
) -> SegformerForSemanticSegmentation:
    label2id = {name: class_id for class_id, name in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained["model_id"],
        revision=pretrained["revision"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        use_safetensors=bool(pretrained["use_safetensors"]),
    )
    return _validate_foodseg_b2_model(model, id2label)


def load_foodseg_b2_checkpoint(
    checkpoint_dir: str | Path,
    id2label: dict[int, str],
) -> SegformerForSemanticSegmentation:
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Local initialization checkpoint not found: {checkpoint_path}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(checkpoint_path),
        local_files_only=True,
    )
    return _validate_foodseg_b2_model(model, id2label)


def checkpoint_provenance(checkpoint_dir: str | Path) -> dict[str, str]:
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Local initialization checkpoint not found: {checkpoint_path}")
    if not (checkpoint_path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config.json not found: {checkpoint_path}")
    weights_path = next(
        (
            candidate
            for candidate in (
                checkpoint_path / "model.safetensors",
                checkpoint_path / "pytorch_model.bin",
            )
            if candidate.is_file()
        ),
        None,
    )
    if weights_path is None:
        raise FileNotFoundError(f"No supported local model weights found: {checkpoint_path}")
    digest = hashlib.sha256()
    with weights_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(checkpoint_path),
        "weights_file": weights_path.name,
        "weights_sha256": digest.hexdigest(),
    }


def optimizer_parameter_groups(
    model: torch.nn.Module,
    training_config: dict[str, Any],
) -> Any:
    encoder_lr = training_config.get("encoder_learning_rate")
    decoder_lr = training_config.get("decoder_learning_rate")
    if encoder_lr is None and decoder_lr is None:
        return model.parameters()
    if encoder_lr is None or decoder_lr is None:
        raise ValueError("encoder_learning_rate and decoder_learning_rate must be set together")
    if not hasattr(model, "segformer") or not hasattr(model, "decode_head"):
        raise ValueError("Separate learning rates require SegFormer encoder and decoder modules")

    encoder_parameters = [p for p in model.segformer.parameters() if p.requires_grad]
    decoder_parameters = [p for p in model.decode_head.parameters() if p.requires_grad]
    all_parameters = [p for p in model.parameters() if p.requires_grad]
    grouped_ids = [id(p) for p in encoder_parameters + decoder_parameters]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {
        id(p) for p in all_parameters
    }:
        raise ValueError("SegFormer encoder/decoder parameter groups must cover trainable weights once")
    return [
        {"params": encoder_parameters, "lr": encoder_lr, "group_name": "encoder"},
        {"params": decoder_parameters, "lr": decoder_lr, "group_name": "decoder"},
    ]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_best_huggingface_model(
    model: SegformerForSemanticSegmentation,
    output_dir: Path,
    *,
    training_metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(output_dir)
    temporary_model = output_dir / "model.tmp.safetensors"
    final_model = output_dir / "model.safetensors"
    state_dict = {
        key: tensor.detach().to(device="cpu").contiguous()
        for key, tensor in model.state_dict().items()
    }
    save_file(state_dict, str(temporary_model))
    temporary_model.replace(final_model)
    _write_json(output_dir / "training_metadata.json", training_metadata)


def _save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    completed_epoch: int,
    best_score: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    config_fingerprint: str,
) -> None:
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "completed_epoch": completed_epoch,
        "best_score": best_score,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "config_fingerprint": config_fingerprint,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    warmup_ratio: float,
    power: float,
) -> LambdaLR:
    warmup_updates = int(total_updates * warmup_ratio)

    def scale(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max(1, step + 1) / warmup_updates
        decay_updates = max(1, total_updates - warmup_updates)
        progress = min(1.0, max(0.0, (step - warmup_updates) / decay_updates))
        return (1.0 - progress) ** power

    return LambdaLR(optimizer, scale)


def _run_config(
    config: dict[str, Any],
    *,
    foodseg_root: Path,
    output_dir: Path,
    id2label: dict[int, str],
    train_ids: list[int | str],
    validation_ids: list[int | str],
    processing_config: PreprocessingConfig,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    class_weights: torch.Tensor,
    config_fingerprint: str,
    initial_checkpoint: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "dataset": "FoodSeg103",
        "foodseg_root": str(foodseg_root),
        "output_dir": str(output_dir),
        "dataset_split_usage": {
            "official_train_rows": len(train_ids) + len(validation_ids),
            "gradient_train_rows": len(train_ids),
            "internal_validation_rows": len(validation_ids),
            "external_evaluation_rows": config["dataset"]["expected_external_evaluation_rows"],
            "external_evaluation_split_used_for_training": False,
        },
        "architecture": config["architecture"],
        "pretrained": config["pretrained"],
        "initial_checkpoint": initial_checkpoint,
        "num_labels": len(id2label),
        "id2label": {str(key): value for key, value in id2label.items()},
        "training": config["training"],
        "loss": {
            **config["loss"],
            "class_weight_min": float(class_weights.min()),
            "class_weight_max": float(class_weights.max()),
            "class_weight_mean": float(class_weights.mean()),
            "frequency_source": "inner_training_subset_only",
        },
        "preprocessing_profile": config["preprocessing_profile"],
        "preprocessing_config": processing_config.to_dict(),
        "device": str(device),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "seed": config["training"]["seed"],
        "config_fingerprint": config_fingerprint,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    config, processing_config, config_fingerprint = load_training_config(args.config)
    initial_checkpoint = (
        checkpoint_provenance(args.init_checkpoint)
        if args.init_checkpoint is not None
        else None
    )
    foodseg_root = args.foodseg_root.expanduser().resolve()
    id2label = load_label_map(foodseg_root)
    train_source = ParquetSplit(foodseg_root, split="train")
    external_evaluation_source = ParquetSplit(foodseg_root, split="validation")
    expected_train_rows = config["dataset"]["expected_train_rows"]
    expected_external_rows = config["dataset"]["expected_external_evaluation_rows"]
    if train_source.num_rows != expected_train_rows:
        raise ValueError(
            f"Expected {expected_train_rows} official train rows; found {train_source.num_rows}"
        )
    if external_evaluation_source.num_rows != expected_external_rows:
        raise ValueError(
            f"Expected {expected_external_rows} external evaluation rows; "
            f"found {external_evaluation_source.num_rows}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    amp_dtype = resolve_amp_dtype(device, config["training"]["amp_dtype"])
    seed = config["training"]["seed"]
    set_seed(seed)

    train_ids, internal_validation_ids = deterministic_train_validation_split(
        train_source.list_ids(),
        validation_fraction=config["dataset"]["internal_validation_fraction"],
        seed=seed,
    )
    if len(train_ids) != 4485 or len(internal_validation_ids) != 498:
        raise RuntimeError(
            f"Expected internal split 4485/498, got {len(train_ids)}/{len(internal_validation_ids)}"
        )

    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        output_dir = resume_path.parent
        if args.output_dir is not None and args.output_dir.expanduser().resolve() != output_dir:
            raise ValueError("--output-dir must match the parent directory of --resume")
        saved_run_config = output_dir / "run_config.json"
        if not saved_run_config.is_file():
            raise FileNotFoundError(f"Run metadata not found for resume: {saved_run_config}")
        run_metadata = json.loads(saved_run_config.read_text(encoding="utf-8"))
        if run_metadata.get("config_fingerprint") != config_fingerprint:
            raise ValueError("resume configuration differs from the saved training configuration")
        if Path(run_metadata.get("foodseg_root", "")).resolve() != foodseg_root:
            raise ValueError("resume dataset root differs from the saved training run")
        if run_metadata.get("id2label") != {str(key): value for key, value in id2label.items()}:
            raise ValueError("resume FoodSeg103 label mapping differs from the saved training run")
        requested_amp = str(amp_dtype).replace("torch.", "") if amp_dtype else None
        if run_metadata.get("amp_dtype") != requested_amp:
            raise ValueError("resume mixed-precision mode differs from the saved training run")
    else:
        if args.output_dir is None:
            output_dir = create_timestamped_run_dir(
                DEFAULT_RUNS_DIR,
                config["run_name"],
            )
        else:
            output_dir = args.output_dir.expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=False)
        resume_path = None

    frequency_summary = count_training_pixel_frequencies(
        foodseg_root,
        processing_config,
        include_ids=train_ids,
    )
    if frequency_summary.image_count != len(train_ids):
        raise RuntimeError("class-frequency scan did not match the gradient-training subset")
    class_weights = compute_class_weights(
        frequency_summary.class_pixel_counts,
        tempering_exponent=config["loss"]["class_weight_tempering_exponent"],
        max_weight=config["loss"]["class_weight_cap"],
    )

    train_dataset = FoodSegParquetDataset(
        foodseg_root,
        "train",
        processing_config.image_size,
        training=True,
        seed=seed,
        processing_config=processing_config,
        include_ids=train_ids,
    )
    validation_dataset = FoodSegParquetDataset(
        foodseg_root,
        "train",
        processing_config.image_size,
        training=False,
        seed=seed,
        processing_config=processing_config,
        include_ids=internal_validation_ids,
    )
    batch_size = config["training"]["batch_size"]
    loader_options = {
        "batch_size": batch_size,
        "num_workers": config["training"]["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, **loader_options)
    validation_loader = DataLoader(validation_dataset, **loader_options)
    updates_per_epoch = math.ceil(
        len(train_loader) / config["training"]["gradient_accumulation_steps"]
    )
    total_updates = updates_per_epoch * config["training"]["max_epochs"]

    if args.init_checkpoint is None:
        model = load_foodseg_b2_model(config["pretrained"], id2label)
    else:
        model = load_foodseg_b2_checkpoint(args.init_checkpoint, id2label)
    model = model.to(device)
    criterion = FocalDiceLoss(
        class_weights=class_weights,
        ignore_index=IGNORE_INDEX,
        focal_gamma=config["loss"]["focal_gamma"],
        focal_weight=config["loss"]["focal_weight"],
        dice_weight=config["loss"]["dice_weight"],
        include_background_in_dice=config["loss"]["include_background_in_dice"],
    ).to(device)
    class_weights = class_weights.to(device)
    optimizer = AdamW(
        optimizer_parameter_groups(model, config["training"]),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = _make_scheduler(
        optimizer,
        total_updates=total_updates,
        warmup_ratio=config["training"]["warmup_ratio"],
        power=config["training"]["polynomial_power"],
    )
    amp_enabled = amp_dtype is not None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    run_config = _run_config(
        config,
        foodseg_root=foodseg_root,
        output_dir=output_dir,
        id2label=id2label,
        train_ids=train_ids,
        validation_ids=internal_validation_ids,
        processing_config=processing_config,
        device=device,
        amp_dtype=amp_dtype,
        class_weights=class_weights,
        config_fingerprint=config_fingerprint,
        initial_checkpoint=initial_checkpoint,
    )
    if resume_path is None:
        _write_json(output_dir / "run_config.json", run_config)
        _write_json(
            output_dir / "split_manifest.json",
            {
                "seed": seed,
                "official_training_ids": [*train_ids, *internal_validation_ids],
                "gradient_training_ids": train_ids,
                "internal_validation_ids": internal_validation_ids,
                "external_evaluation_split": "validation",
                "external_evaluation_ids_loaded_for_training": False,
            },
        )
        _write_json(
            output_dir / "class_weight_summary.json",
            {
                "image_count": frequency_summary.image_count,
                "valid_pixel_count": frequency_summary.valid_pixel_count,
                "ignored_padding_pixels": frequency_summary.ignored_padding_pixels,
                "weight_min": float(class_weights.min()),
                "weight_max": float(class_weights.max()),
                "weight_mean": float(class_weights.mean()),
                "classes": [
                    {
                        "class_id": class_id,
                        "class_name": id2label[class_id],
                        "training_pixels": int(frequency_summary.class_pixel_counts[class_id]),
                        "weight": float(class_weights[class_id]),
                    }
                    for class_id in range(len(id2label))
                ],
            },
        )

    history: list[dict[str, Any]] = []
    best_score = -1.0
    epochs_without_improvement = 0
    start_epoch = 0
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("config_fingerprint") != config_fingerprint:
            raise ValueError("resume checkpoint fingerprint does not match the requested config")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["completed_epoch"])
        best_score = float(state["best_score"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = list(state["history"])
        _restore_rng_state(state)
        print(f"Resuming {output_dir} at epoch {start_epoch + 1}", flush=True)

    print(
        f"Training SegFormer-B2 on {len(train_ids)} images; internal validation "
        f"{len(internal_validation_ids)}; external 2,135-image split reserved; "
        f"device={device}, amp={amp_dtype}, effective_batch=8, output={output_dir}",
        flush=True,
    )
    training_config = config["training"]
    accumulation_steps = training_config["gradient_accumulation_steps"]
    output_state_path = output_dir / "last_training_state.pt"
    for epoch in range(start_epoch, training_config["max_epochs"]):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        total_loss = torch.zeros((), device=device, dtype=torch.float64)
        seen_examples = 0
        num_batches = len(train_loader)

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            batch_examples = pixel_values.shape[0]
            group_start = (step // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, num_batches - group_start)
            group_example_count = min(
                batch_size * group_size,
                len(train_dataset) - group_start * batch_size,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_enabled,
            ):
                logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)
            total_loss += loss.detach().to(torch.float64) * batch_examples
            seen_examples += batch_examples
            scaler.scale(loss * batch_examples / max(1, group_example_count)).backward()

            should_update = (
                (step + 1) % accumulation_steps == 0 or step + 1 == num_batches
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=training_config["max_grad_norm"],
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not scaler.is_enabled() or scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = float((total_loss / max(1, seen_examples)).item())
        confusion = evaluate_model(
            model,
            validation_loader,
            len(id2label),
            device,
            use_amp=amp_enabled,
            amp_dtype=amp_dtype or torch.float16,
        )
        validation_summary, _ = summarize_confusion_matrix(confusion, id2label)
        score = validation_summary["mean_iou_foreground"]
        if score is None:
            raise RuntimeError("internal validation produced no foreground class pixels")

        improved = score > best_score
        if improved:
            best_score = float(score)
            epochs_without_improvement = 0
            _save_best_huggingface_model(
                model,
                output_dir / "best_model",
                training_metadata={
                    "dataset": "FoodSeg103",
                    "model": config["architecture"]["name"],
                    "pretrained_model": config["pretrained"]["model_id"],
                    "initial_checkpoint": initial_checkpoint,
                    "num_labels": len(id2label),
                    "padding_label": IGNORE_INDEX,
                    "preprocessing_config": processing_config.to_dict(),
                    "best_internal_validation_foreground_miou": best_score,
                    "training_config_fingerprint": config_fingerprint,
                },
            )
        else:
            epochs_without_improvement += 1
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation": validation_summary,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rates": {
                group.get("group_name", f"group_{index}"): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "best_foreground_miou": best_score,
            "improved": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_record)
        _write_json(output_dir / "training_history.json", history)
        _save_training_state(
            output_state_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=epoch + 1,
            best_score=best_score,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            config_fingerprint=config_fingerprint,
        )
        print(json.dumps(epoch_record, separators=(",", ":")), flush=True)
        if epochs_without_improvement >= training_config["early_stopping_patience"]:
            print(
                f"Early stopping after {epoch + 1} epochs; best internal foreground mIoU={best_score:.5f}",
                flush=True,
            )
            break

    print(f"Best inference checkpoint: {output_dir / 'best_model'}", flush=True)
    print(f"Resume checkpoint: {output_state_path}", flush=True)


if __name__ == "__main__":
    main()
