"""CPU-only, local-checkpoint SegFormer smoke test; performs no training or writes."""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartnutri_backend.camera import CapturedImage, MockCameraService
from smartnutri_backend.schemas import ImageQuality
from smartnutri_backend.vision import (
    FoodSegLabelMap,
    SegFormerInferenceService,
    validate_foodseg103_label_anchors,
)


def _capture_from_file(path: Path) -> CapturedImage:
    image_bytes = path.read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image_format = (opened.format or "").upper()
        if image_format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("smoke-test image format must be JPEG, PNG, or WEBP")
        oriented = ImageOps.exif_transpose(opened).convert("RGB")
        brightness = ImageStat.Stat(oriented.convert("L")).mean[0]
        quality = ImageQuality(
            acceptable=True,
            reasons=[],
            width=oriented.width,
            height=oriented.height,
            format=image_format,
            byte_count=len(image_bytes),
            brightness_mean=float(brightness),
            clipped_pixel_fraction=0.0,
            edge_variance=0.0,
        )
    return CapturedImage(
        image_bytes=image_bytes,
        source="smoke-test-file",
        simulated=False,
        captured_at=datetime.now(timezone.utc).isoformat(),
        quality=quality,
        quality_score=1.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "foodseg103-segformer-b0" / "best_model",
        help="Existing local checkpoint directory; never downloaded.",
    )
    parser.add_argument(
        "--foodseg-root",
        type=Path,
        default=PROJECT_ROOT.parent / "Datasets" / "FoodSeg103",
        help="Existing local FoodSeg103 directory containing id2label.json.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local image. If omitted, the mock camera supplies a synthetic test image.",
    )
    args = parser.parse_args()

    label_map = FoodSegLabelMap.from_json(args.foodseg_root / "id2label.json")
    validate_foodseg103_label_anchors(label_map)
    capture = (
        _capture_from_file(args.image.expanduser())
        if args.image is not None
        else MockCameraService(capture_retries=0).capture_best()
    )
    service = SegFormerInferenceService(
        checkpoint_dir=args.checkpoint_dir,
        foodseg_root=args.foodseg_root,
        device="cpu",
    )
    result = service.infer(capture, scan_id="local-segformer-smoke")
    print(json.dumps({
        "status": "passed",
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "device": "cpu",
        "model_loaded": service.model_loaded,
        "num_labels": 104,
        "real_foodseg_label_anchors": True,
        "input_source": capture.source,
        "input_simulated": capture.simulated,
        "model_output_simulated": result.simulated,
        "foreground_class_ids": [item.class_id for item in result.predictions],
        "foreground_labels": [item.label for item in result.predictions],
        "note": "Smoke-test output only; this is not an accuracy evaluation.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
