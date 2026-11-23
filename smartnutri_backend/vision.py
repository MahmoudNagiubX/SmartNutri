from __future__ import annotations

import colorsys
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from smartnutri_backend.camera import CapturedImage
from smartnutri_backend.schemas import FoodPrediction


class FoodSegLabelMap:
    """Validated FoodSeg103 mapping. Runtime mappings come from local id2label.json."""

    def __init__(self, labels: dict[int | str, str], *, simulated: bool = False) -> None:
        normalized = {int(class_id): str(label).strip() for class_id, label in labels.items()}
        if set(normalized) != set(range(104)):
            raise ValueError("FoodSeg103 mapping must contain exactly class IDs 0..103")
        if normalized[0].casefold() != "background":
            raise ValueError("FoodSeg103 class 0 must be background")
        if any(not label for label in normalized.values()):
            raise ValueError("FoodSeg103 labels cannot be empty")
        self.labels = normalized
        self.simulated = simulated

    @classmethod
    def from_json(cls, path: str | Path) -> "FoodSegLabelMap":
        mapping_path = Path(path).expanduser()
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("FoodSeg103 id2label.json must contain a JSON object")
        return cls(raw)

    @classmethod
    def mock_fixture(cls) -> "FoodSegLabelMap":
        labels = {class_id: f"mock_food_class_{class_id}" for class_id in range(104)}
        labels.update({0: "background", 1: "rice", 2: "chicken", 3: "fries"})
        return cls(labels, simulated=True)

    def label_for(self, class_id: int) -> str:
        return self.labels[class_id]


@dataclass(frozen=True)
class SegmentationResult:
    predictions: list[FoodPrediction]
    semantic_mask_png: bytes
    annotated_image_bytes: bytes
    width: int
    height: int
    source: str
    simulated: bool


def _component_counts(mask: Image.Image) -> dict[int, int]:
    """Count 4-connected runs by class without turning semantic classes into instances."""
    width, height = mask.size
    pixels = mask.load()
    parent: dict[int, int] = {}
    parent_classes: dict[int, int] = {}
    next_component = 1
    previous: list[tuple[int, int, int, int]] = []
    for y in range(height):
        current: list[tuple[int, int, int, int]] = []
        x = 0
        while x < width:
            class_id = pixels[x, y]
            if class_id == 0:
                x += 1
                continue
            start = x
            x += 1
            while x < width and pixels[x, y] == class_id:
                x += 1
            end = x
            while previous and previous[0][1] <= start:
                previous.pop(0)
            overlaps = [
                run for run in previous
                if run[0] < end and run[1] > start and run[2] == class_id
            ]
            if overlaps:
                component = overlaps[0][3]
                for run in overlaps[1:]:
                    first, second = _find(parent, component), _find(parent, run[3])
                    if first != second:
                        parent[max(first, second)] = min(first, second)
                current.append((start, end, class_id, component))
            else:
                component = next_component
                next_component += 1
                parent[component] = component
                parent_classes[component] = class_id
                current.append((start, end, class_id, component))
        previous = current
    canonical: dict[int, set[int]] = {}
    for component, class_id in parent_classes.items():
        canonical.setdefault(class_id, set()).add(_find(parent, component))
    return {class_id: len(components) for class_id, components in canonical.items()}


def _find(parent: dict[int, int], component: int) -> int:
    root = component
    while parent.get(root, root) != root:
        root = parent[root]
    while component != root:
        next_component = parent[component]
        parent[component] = root
        component = next_component
    return root


def _annotation(original: Image.Image, semantic_mask: Image.Image, labels: dict[int, str]) -> bytes:
    palette: list[int] = []
    for class_id in range(256):
        if class_id == 0:
            palette.extend((0, 0, 0))
        else:
            red, green, blue = colorsys.hsv_to_rgb((class_id * 0.61803398875) % 1.0, 0.82, 1.0)
            palette.extend((int(red * 255), int(green * 255), int(blue * 255)))
    indexed = semantic_mask.convert("P")
    indexed.putpalette(palette)
    rgba = indexed.convert("RGBA")
    alpha = semantic_mask.point(lambda class_id: 0 if class_id == 0 else 105)
    rgba.putalpha(alpha)
    annotated = Image.alpha_composite(original.convert("RGBA"), rgba)
    draw = ImageDraw.Draw(annotated)
    visible_ids = [class_id for class_id, count in enumerate(semantic_mask.histogram()) if class_id and count]
    if visible_ids:
        summary = "  ".join(f"{class_id}: {labels[class_id]}" for class_id in visible_ids[:6])
        draw.rectangle((6, 6, min(annotated.width - 1, 12 + len(summary) * 7), 28), fill=(0, 0, 0, 190))
        draw.text((10, 10), summary, fill=(255, 255, 255, 255))
    output = io.BytesIO()
    annotated.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()


class MockSegmentationService:
    source = "mock"
    simulated = True

    def __init__(self, label_map: FoodSegLabelMap | None = None) -> None:
        self.label_map = label_map or FoodSegLabelMap.mock_fixture()
        if not self.label_map.simulated:
            # A mock predictor must not make mock output look like model inference.
            self.label_map = FoodSegLabelMap(self.label_map.labels, simulated=True)

    def infer(self, image: CapturedImage, *, scan_id: str) -> SegmentationResult:
        with Image.open(io.BytesIO(image.image_bytes)) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = original.size
        semantic = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(semantic)
        draw.rounded_rectangle((int(width * 0.22), int(height * 0.26), int(width * 0.42), int(height * 0.66)), radius=18, fill=1)
        draw.rounded_rectangle((int(width * 0.25), int(height * 0.70), int(width * 0.40), int(height * 0.82)), radius=12, fill=1)
        draw.ellipse((int(width * 0.50), int(height * 0.28), int(width * 0.75), int(height * 0.62)), fill=2)
        draw.polygon(
            [(int(width * 0.43), int(height * 0.69)), (int(width * 0.72), int(height * 0.67)),
             (int(width * 0.78), int(height * 0.82)), (int(width * 0.45), int(height * 0.84))],
            fill=3,
        )
        histogram = semantic.histogram()
        regions = _component_counts(semantic)
        predictions: list[FoodPrediction] = []
        for class_id, confidence in ((1, 0.96), (2, 0.93), (3, 0.91)):
            area = histogram[class_id]
            if not area:
                continue
            region_count = max(1, regions.get(class_id, 1))
            quality_status = "small_region" if area < max(32, width * height * 0.001) else "unverified"
            if region_count > 16:
                quality_status = "fragmented"
            predictions.append(
                FoodPrediction(
                    class_id=class_id,
                    label=self.label_map.label_for(class_id),
                    classification_confidence=confidence,
                    pixel_area=area,
                    region_count=region_count,
                    mask_ref=f"/api/v1/scans/{scan_id}/mask/{class_id}",
                    segmentation_quality=quality_status,
                    source="mock",
                    simulated=True,
                )
            )
        mask_output = io.BytesIO()
        semantic.save(mask_output, format="PNG", optimize=True)
        return SegmentationResult(
            predictions=predictions,
            semantic_mask_png=mask_output.getvalue(),
            annotated_image_bytes=_annotation(original, semantic, self.label_map.labels),
            width=width,
            height=height,
            source=self.source,
            simulated=self.simulated or image.simulated,
        )


class SegFormerInferenceService:
    """Lazy, local-only SegFormer inference adapter; no model is loaded on API startup."""

    source = "segformer"
    simulated = False

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path | None,
        foodseg_root: str | Path | None,
        preprocessing_config_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser() if checkpoint_dir else None
        self.foodseg_root = Path(foodseg_root).expanduser() if foodseg_root else None
        self.preprocessing_config_path = (
            Path(preprocessing_config_path).expanduser() if preprocessing_config_path else None
        )
        self.device_name = device
        self._model: Any = None
        self._torch: Any = None
        self._preprocessing: Any = None
        self._labels: FoodSegLabelMap | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.model_loaded:
            return
        if self.checkpoint_dir is None or not self.checkpoint_dir.is_dir():
            raise FileNotFoundError("configured local SegFormer checkpoint directory is unavailable")
        if self.foodseg_root is None:
            raise FileNotFoundError("SMARTNUTRI_FOODSEG_ROOT must point to the local FoodSeg103 label map")
        label_path = self.foodseg_root / "id2label.json"
        metadata_path = self.checkpoint_dir / "training_metadata.json"
        for required in (label_path, metadata_path):
            if not required.is_file():
                raise FileNotFoundError(f"required local inference metadata is missing: {required.name}")
        labels = FoodSegLabelMap.from_json(label_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset") != "FoodSeg103" or metadata.get("num_labels") != 104:
            raise ValueError("checkpoint metadata does not identify the 104-class FoodSeg103 task")
        if metadata.get("padding_label") != 255:
            raise ValueError("checkpoint metadata does not preserve ignore_index=255")
        preprocessing_values = metadata.get("preprocessing_config")
        if not isinstance(preprocessing_values, dict):
            raise ValueError("checkpoint metadata is missing its preprocessing configuration")

        import torch
        from transformers import SegformerForSemanticSegmentation
        from image_processing import PreprocessingConfig

        if self.preprocessing_config_path is not None:
            raise ValueError(
                "inference preprocessing is loaded from checkpoint metadata; explicit overrides are disabled"
            )
        model = SegformerForSemanticSegmentation.from_pretrained(
            str(self.checkpoint_dir), local_files_only=True, trust_remote_code=False
        )
        validate_model_label_contract(model.config, model.decode_head.classifier.out_channels, labels)
        model.eval()
        model.to(self.device_name)
        self._torch = torch
        self._preprocessing = PreprocessingConfig.from_dict(preprocessing_values)
        self._labels = labels
        self._model = model

    def infer(self, image: CapturedImage, *, scan_id: str) -> SegmentationResult:
        self._load()
        assert self._model is not None and self._labels is not None
        assert self._torch is not None and self._preprocessing is not None
        torch = self._torch
        import torch.nn.functional as functional
        with Image.open(io.BytesIO(image.image_bytes)) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        dummy_mask = Image.new("L", original.size, 0)
        from image_processing import preprocess_pair

        pixel_values, _, _ = preprocess_pair(
            original,
            dummy_mask,
            self._preprocessing,
            training=False,
        )
        size = self._preprocessing.image_size
        with torch.inference_mode():
            output = self._model(pixel_values=pixel_values.unsqueeze(0).to(self.device_name))
            logits = output.logits.float()
            low_res_class_ids = logits.argmax(dim=1)[0]
            low_res_probabilities = torch.softmax(logits[0], dim=0)
            canvas_class_ids = functional.interpolate(
                low_res_class_ids[None, None].float(),
                size=(size, size),
                mode="nearest",
            )[0, 0].to(torch.uint8).cpu().numpy()
            confidence_map = low_res_probabilities.cpu()
            low_ids = low_res_class_ids.cpu()

        original_width, original_height = original.size
        scale = min(size / original_width, size / original_height)
        resized_width = min(size, max(1, round(original_width * scale)))
        resized_height = min(size, max(1, round(original_height * scale)))
        left = (size - resized_width) // 2
        top = (size - resized_height) // 2
        canvas = Image.fromarray(canvas_class_ids, mode="L")
        content = canvas.crop((left, top, left + resized_width, top + resized_height))
        semantic = content.resize((original_width, original_height), Image.Resampling.NEAREST)
        histogram = semantic.histogram()
        regions = _component_counts(content)
        predictions: list[FoodPrediction] = []
        for class_id in range(1, 104):
            area = histogram[class_id]
            if not area:
                continue
            selected = low_ids == class_id
            confidence = float(confidence_map[class_id][selected].mean().item()) if selected.any() else 0.0
            region_count = max(1, regions.get(class_id, 1))
            quality_status = "small_region" if area < max(32, original_width * original_height * 0.001) else "unverified"
            if region_count > 16:
                quality_status = "fragmented"
            predictions.append(
                FoodPrediction(
                    class_id=class_id,
                    label=self._labels.label_for(class_id),
                    classification_confidence=confidence,
                    pixel_area=area,
                    region_count=region_count,
                    mask_ref=f"/api/v1/scans/{scan_id}/mask/{class_id}",
                    segmentation_quality=quality_status,
                    source="segformer",
                    simulated=False,
                )
            )
        predictions.sort(key=lambda prediction: prediction.pixel_area, reverse=True)
        mask_output = io.BytesIO()
        semantic.save(mask_output, format="PNG", optimize=True)
        return SegmentationResult(
            predictions=predictions,
            semantic_mask_png=mask_output.getvalue(),
            annotated_image_bytes=_annotation(original, semantic, self._labels.labels),
            width=original_width,
            height=original_height,
            source=self.source,
            simulated=False,
        )


def validate_model_label_contract(config: Any, head_channels: int, labels: FoodSegLabelMap) -> None:
    if getattr(config, "num_labels", None) != 104 or head_channels != 104:
        raise ValueError("SegFormer model must configure and output exactly 104 classes")
    id2label = {
        int(class_id): str(label).strip()
        for class_id, label in (getattr(config, "id2label", None) or {}).items()
    }
    label2id = {
        str(label).strip(): int(class_id)
        for label, class_id in (getattr(config, "label2id", None) or {}).items()
    }
    expected_label2id = {label: class_id for class_id, label in labels.labels.items()}
    if id2label != labels.labels or label2id != expected_label2id:
        raise ValueError("SegFormer id2label/label2id must exactly match local FoodSeg103 id2label.json")
