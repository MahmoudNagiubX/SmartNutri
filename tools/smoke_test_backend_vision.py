from __future__ import annotations

import argparse
import io
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from foodseg_data import ParquetSplit, _image_bytes, load_label_map
from smartnutri_backend.agent import AgentContext, NutritionDecisionAgent
from smartnutri_backend.camera import FileCameraService, ImageQualityError
from smartnutri_backend.schemas import AgentAction
from smartnutri_backend.vision import SegFormerInferenceService
from train_optimized import DEFAULT_FOODSEG_ROOT


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "runs" / "foodseg103-segformer-b2-optimized-20261124_075454" / "best_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test a local FoodSeg103 checkpoint through SmartNutri's vision adapter."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG_ROOT)
    parser.add_argument("--model-variant", choices=("b0", "b2"), default="b2")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--min-region-area-ratio", type=float, default=0.001)
    parser.add_argument("--candidate-limit", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    foodseg_root = args.foodseg_root.expanduser().resolve()
    labels = load_label_map(foodseg_root)
    source = ParquetSplit(foodseg_root, "validation")

    with tempfile.TemporaryDirectory(prefix="smartnutri-vision-smoke-") as temporary:
        temp_root = Path(temporary)
        accepted_capture = None
        accepted_id = None
        for index, row in enumerate(
            source.iter_rows(shuffle=False, rng=random.Random(0))
        ):
            if index >= args.candidate_limit:
                break
            sample_id = str(row.get("id", index))
            with Image.open(io.BytesIO(_image_bytes(row, "image"))) as opened:
                image = opened.convert("RGB")
                image_path = temp_root / f"sample-{index:04d}.jpg"
                image.save(image_path, format="JPEG", quality=95)
            camera = FileCameraService(image_path, capture_retries=0)
            try:
                accepted_capture = camera.capture_best()
                accepted_id = sample_id
                break
            except ImageQualityError:
                continue
        if accepted_capture is None:
            raise RuntimeError(
                f"No acceptable local validation image found among {args.candidate_limit} samples"
            )

        service = SegFormerInferenceService(
            checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
            foodseg_root=foodseg_root,
            device=args.device,
            model_variant=args.model_variant,
            hflip_tta=args.hflip_tta,
            min_region_area_ratio=args.min_region_area_ratio,
        )
        started = time.perf_counter()
        result = service.infer(accepted_capture, scan_id="local-vision-smoke")
        elapsed_ms = (time.perf_counter() - started) * 1000.0

    with Image.open(io.BytesIO(result.semantic_mask_png)) as mask_image:
        mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
    mask_ids = sorted(int(value) for value in np.unique(mask))
    if mask.shape != (result.height, result.width):
        raise RuntimeError(
            f"Mask shape {mask.shape} differs from response size {(result.height, result.width)}"
        )
    if any(class_id not in labels for class_id in mask_ids):
        raise RuntimeError(f"Output mask contains unknown FoodSeg103 ids: {mask_ids}")
    if any(item.class_id not in range(1, 104) or item.label != labels[item.class_id] for item in result.predictions):
        raise RuntimeError("Backend predictions violate the 104-class FoodSeg103 contract")

    decision = NutritionDecisionAgent().decide(
        AgentContext(predictions=result.predictions)
    )
    if decision.action != AgentAction.RETRY_WEIGHT:
        raise RuntimeError("Nutrition workflow must wait for a real scale observation")

    print(
        json.dumps(
            {
                "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
                "model_variant": args.model_variant,
                "hflip_tta": args.hflip_tta,
                "sample_id": accepted_id,
                "adapter_model_loaded": service.model_loaded,
                "adapter_source": result.source,
                "prediction_count": len(result.predictions),
                "mask_shape_hw": list(mask.shape),
                "mask_class_ids": mask_ids,
                "input_capture_is_test_fixture": accepted_capture.simulated,
                "agent_action_without_scale": decision.action.value,
                "portion_or_nutrition_created": False,
                "end_to_end_inference_ms": elapsed_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
