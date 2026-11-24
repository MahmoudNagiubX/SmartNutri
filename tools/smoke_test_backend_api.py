"""Exercise the local B2 checkpoint through the full FastAPI mock-scale path."""

from __future__ import annotations

import io
import json
import random
import tempfile
import sys
import time
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foodseg_data import ParquetSplit, _image_bytes, load_label_map
from smartnutri_backend.api import create_app
from smartnutri_backend.camera import FileCameraService, ImageQualityError
from smartnutri_backend.settings import Settings


DEFAULT_FOODSEG = ROOT.parent / "Datasets" / "FoodSeg103"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--foodseg-root", type=Path, default=DEFAULT_FOODSEG)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--candidate-limit", type=int, default=40)
    args = parser.parse_args()

    checkpoint_setting = Settings(cv_model="b2", checkpoint_dir=args.checkpoint_dir)
    checkpoint = checkpoint_setting.selected_checkpoint_dir.expanduser().resolve()
    foodseg_root = args.foodseg_root.expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"local B2 checkpoint is unavailable: {checkpoint}")
    labels = load_label_map(foodseg_root)

    with tempfile.TemporaryDirectory(prefix="smartnutri-backend-b2-smoke-") as temporary:
        temp_root = Path(temporary)
        image_path: Path | None = None
        sample_id: str | None = None
        validation = ParquetSplit(foodseg_root, "validation")
        for index, row in enumerate(
            validation.iter_rows(shuffle=False, rng=random.Random(42))
        ):
            if index >= args.candidate_limit:
                break
            candidate_path = temp_root / f"validation-{index:04d}.jpg"
            with Image.open(io.BytesIO(_image_bytes(row, "image"))) as source:
                source.convert("RGB").save(candidate_path, format="JPEG", quality=95)
            try:
                FileCameraService(candidate_path, capture_retries=0).capture_best()
            except ImageQualityError:
                continue
            image_path = candidate_path
            sample_id = str(row.get("id", index))
            break
        if image_path is None:
            raise RuntimeError(f"no acceptable local FoodSeg103 validation image in first {args.candidate_limit} rows")

        settings = Settings(
            data_dir=temp_root / "runtime",
            scale_mode="mock",
            camera_mode="file",
            camera_file_path=image_path,
            segmentation_mode="segformer",
            cv_model="b2",
            checkpoint_dir=args.checkpoint_dir,
            foodseg_root=foodseg_root,
            cv_device=args.device,
            auto_scan_enabled=False,
            capture_retries=0,
        )
        with TestClient(create_app(settings)) as client:
            plate = client.post(
                "/api/v1/plates",
                json={"name": "B2 smoke fixture plate", "empty_plate_weight_g": 300, "make_default": True},
            )
            if plate.status_code != 201:
                raise RuntimeError(f"mock plate setup failed: {plate.status_code} {plate.text}")
            inference_started = time.perf_counter()
            response = client.post("/api/v1/scans/mock-placement", json={"gross_weight_g": 750})
            workflow_ms = (time.perf_counter() - inference_started) * 1000.0
            if response.status_code != 200:
                raise RuntimeError(f"backend B2 scan failed: {response.status_code} {response.text}")
            scan = response.json()["scan"]
            scan_id = scan["id"]
            foods = client.get(f"/api/v1/scans/{scan_id}/foods")
            if foods.status_code != 200:
                raise RuntimeError(f"food prediction route failed: {foods.status_code} {foods.text}")
            predictions = foods.json()["predictions"]
            if any(item["source"] != "segformer" or item["simulated"] for item in predictions):
                raise RuntimeError("backend returned mock predictions instead of local B2 output")
            if any(item["label"] != labels[item["class_id"]] for item in predictions):
                raise RuntimeError("backend FoodSeg103 class IDs and labels do not match the local mapping")
            nutrition = client.get(f"/api/v1/scans/{scan_id}/nutrition").json()
            if nutrition["totals"] is not None:
                raise RuntimeError("simulated scale input unexpectedly produced nutrition totals")
            mask = client.get(f"/api/v1/scans/{scan_id}/mask.png")
            if mask.status_code != 200 or mask.headers.get("content-type", "").split(";")[0] != "image/png":
                raise RuntimeError("backend did not expose the semantic mask as PNG")

            print(json.dumps({
                "status": "passed",
                "checkpoint": str(checkpoint),
                "architecture": "SegFormer-B2",
                "device_requested": args.device,
                "device_used": str(client.app.state.runtime.inference._device),
                "sample_id": sample_id,
                "input_camera": "local FoodSeg103 validation image (fixture source)",
                "scan_state": scan["state"],
                "net_weight_g": scan["measured_net_food_weight_g"],
                "prediction_count": len(predictions),
                "food_class_ids": [item["class_id"] for item in predictions],
                "food_labels": [item["label"] for item in predictions],
                "nutrition_totals": nutrition["totals"],
                "nutrition_blocked_for_simulated_scale": True,
                "mask_route": "passed",
                "full_mock_placement_workflow_ms": round(workflow_ms, 2),
            }, indent=2))


if __name__ == "__main__":
    main()
