from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from smartnutri_backend.agent import AgentContext, NutritionDecisionAgent
from smartnutri_backend.camera import CameraService, CameraUnavailable, ImageQualityError
from smartnutri_backend.nutrition import NutritionCalculation, NutritionService
from smartnutri_backend.portion import PortionEstimator
from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.schemas import (
    AgentAction,
    AgentDecision,
    ScanState,
    ScaleObservation,
)
from smartnutri_backend.vision import SegmentationResult


class ScanWorkflow:
    """Coordinates one stable placement into an auditable, persisted scan."""

    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        camera: CameraService,
        inference: Any,
        portion_estimator: PortionEstimator,
        agent: NutritionDecisionAgent,
        nutrition: NutritionService,
        image_dir: str | Path,
    ) -> None:
        self.repository = repository
        self.camera = camera
        self.inference = inference
        self.portion_estimator = portion_estimator
        self.agent = agent
        self.nutrition = nutrition
        self.image_dir = Path(image_dir)
        self._cancelled: dict[str, asyncio.Event] = {}
        self._busy: set[str] = set()

    def cancel(self, scan_id: str) -> None:
        self._cancelled.setdefault(scan_id, asyncio.Event()).set()

    def finish(self, scan_id: str) -> None:
        self._cancelled.pop(scan_id, None)

    async def process(
        self,
        *,
        scan_id: str,
        observation: ScaleObservation,
    ) -> dict[str, Any]:
        if scan_id in self._busy:
            raise RuntimeError("this scan is already being processed")
        self._busy.add(scan_id)
        try:
            return await self._process_once(scan_id=scan_id, observation=observation)
        finally:
            self._busy.discard(scan_id)

    async def _process_once(
        self,
        *,
        scan_id: str,
        observation: ScaleObservation,
    ) -> dict[str, Any]:
        cancelled = self._cancelled.setdefault(scan_id, asyncio.Event())
        scan = self.repository.get_scan(scan_id)
        if scan is None:
            raise KeyError(scan_id)
        if cancelled.is_set():
            return self._mark_removed(scan_id)
        self.repository.update_scan(scan_id, state=ScanState.CAPTURING.value)
        tare_config = self.repository.get_tare_configuration()
        plate_profile = self.repository.get_default_plate()
        tare_mode = tare_config.get("mode", "saved_plate")
        plate_profile_id: str | None = None
        plate_tare_weight_g: float | None = None

        if tare_mode == "live_plate":
            if not tare_config.get("live_tare_active"):
                return self._needs_plate_tare(scan_id, "Live plate tare has not been confirmed.")
            plate_tare_weight_g = tare_config.get("empty_plate_weight_g")
            # Firmware already subtracts the live plate tare; do not subtract it again.
            net_weight_g = observation.weight_g
        else:
            if plate_profile is None:
                return self._needs_plate_tare(scan_id, "Register a default empty plate profile first.")
            plate_profile_id = plate_profile["id"]
            plate_tare_weight_g = float(plate_profile["empty_plate_weight_g"])
            gross_weight_g = observation.weight_g
            if gross_weight_g is None:
                return self._needs_plate_tare(scan_id, "Stable gross weight is unavailable.")
            if gross_weight_g <= plate_tare_weight_g:
                return self._needs_plate_tare(
                    scan_id,
                    "Measured gross weight does not exceed the selected empty plate weight. "
                    "Verify the plate profile and tare mode before retrying.",
                )
            net_weight_g = gross_weight_g - plate_tare_weight_g

        if net_weight_g is None or net_weight_g <= 0:
            return self._needs_plate_tare(scan_id, "Net food weight must be positive after the configured tare.")
        self.repository.update_scan(
            scan_id,
            tare_mode=tare_mode,
            plate_profile_id=plate_profile_id,
            measured_gross_weight_g=observation.weight_g if tare_mode == "saved_plate" else None,
            plate_tare_weight_g=plate_tare_weight_g,
            measured_net_food_weight_g=net_weight_g,
            state=ScanState.CAPTURING.value,
        )

        try:
            capture = await asyncio.to_thread(self.camera.capture_best)
        except (CameraUnavailable, ImageQualityError) as exc:
            quality = exc.quality.model_dump(mode="json") if isinstance(exc, ImageQualityError) and exc.quality else None
            decision = self.agent.decide(
                AgentContext(
                    scale=observation,
                    plate_tare_known=True,
                    net_weight_g=net_weight_g,
                    camera_quality=exc.quality if isinstance(exc, ImageQualityError) else None,
                    capture_attempts_remaining=0,
                )
            )
            return self._mark_review(
                scan_id,
                decision,
                result_message=str(exc),
                error_code="camera_unavailable" if isinstance(exc, CameraUnavailable) else "image_quality_rejected",
                image_quality=quality,
            )

        if cancelled.is_set():
            return self._mark_removed(scan_id)
        capture_path = self._store(scan_id, "capture.jpg", capture.image_bytes)
        self.repository.update_scan(
            scan_id,
            capture_source=capture.source,
            capture_simulated=capture.simulated,
            capture_path=str(capture_path),
            image_quality=capture.quality.model_dump(mode="json"),
            state=ScanState.ANALYZING.value,
        )
        if cancelled.is_set():
            return self._mark_removed(scan_id)

        try:
            segmentation: SegmentationResult = await asyncio.to_thread(
                self.inference.infer, capture, scan_id=scan_id
            )
        except Exception as exc:
            return self._mark_review(
                scan_id,
                self.agent.decide(
                    AgentContext(
                        scale=observation,
                        plate_tare_known=True,
                        net_weight_g=net_weight_g,
                        camera_quality=capture.quality,
                        capture_attempts_remaining=0,
                    )
                ),
                result_message=f"Segmentation service unavailable ({type(exc).__name__}).",
                error_code="inference_unavailable",
                image_quality=capture.quality.model_dump(mode="json"),
            )
        if cancelled.is_set():
            return self._mark_removed(scan_id)

        self._store(scan_id, "mask.png", segmentation.semantic_mask_png)
        annotated_path = self._store(scan_id, "annotated.jpg", segmentation.annotated_image_bytes)
        predictions = [item.model_dump(mode="json") for item in segmentation.predictions]
        with Image.open(io.BytesIO(segmentation.semantic_mask_png)) as mask_image:
            low_confidence_predictions = [
                item
                for item in segmentation.predictions
                if item.classification_confidence < self.agent.min_classification_confidence
            ]
            usable_predictions = [
                item
                for item in segmentation.predictions
                if item.classification_confidence >= self.agent.min_classification_confidence
            ]
            portions = self.portion_estimator.estimate(
                net_weight_g,
                usable_predictions,
                semantic_mask=mask_image.convert("L"),
                unidentified_area_px=sum(item.pixel_area for item in low_confidence_predictions),
                simulated=observation.simulated or capture.simulated or segmentation.simulated,
            )
        nutrition_calculation: NutritionCalculation = self.nutrition.calculate(
            portions.estimates,
            simulated=observation.simulated or capture.simulated or segmentation.simulated,
        )
        nutrition_simulated = observation.simulated or capture.simulated or segmentation.simulated
        verified_classes = {
            prediction.class_id
            for prediction in segmentation.predictions
            if self.nutrition.lookup_for_class(prediction.class_id, simulated=nutrition_simulated) is not None
        }
        context = AgentContext(
            scale=observation,
            plate_tare_known=True,
            net_weight_g=net_weight_g,
            camera_quality=capture.quality,
            predictions=segmentation.predictions,
            portions=portions,
            verified_nutrition_classes=verified_classes,
            capture_attempts_remaining=0,
        )
        decision = self.agent.decide(context)
        if decision.action == AgentAction.ACCEPT_ANALYSIS:
            decision = self.agent.decide(context.model_copy(update={"analysis_accepted": True}))

        nutrition_items = {item.class_id: item for item in nutrition_calculation.items}
        portion_items = {item.class_id: item for item in portions.estimates}
        prediction_items = {item.class_id: item for item in segmentation.predictions}
        meal_items: list[dict[str, Any]] = []
        for class_id, portion in portion_items.items():
            prediction = prediction_items[class_id]
            nutrient = nutrition_items.get(class_id)
            meal_items.append(
                {
                    "class_id": class_id,
                    "food_label": prediction.label,
                    "classification_confidence": prediction.classification_confidence,
                    "pixel_area": prediction.pixel_area,
                    "region_count": prediction.region_count,
                    "estimated_weight_g": portion.estimated_weight_g,
                    "portion_method": portion.method,
                    "portion_confidence": portion.confidence,
                    "portion_uncertainty_g": portion.uncertainty_g,
                    "nutrition_record_id": nutrient.nutrition_record_id if nutrient else None,
                    "nutrients": nutrient.nutrients.model_dump(mode="json") if nutrient and nutrient.nutrients else None,
                    "needs_review": decision.requires_user or portion.review_required or nutrient is None or nutrient.nutrients is None,
                }
            )

        state = ScanState.COMPLETED.value if decision.action == AgentAction.SAVE_MEAL else ScanState.NEEDS_REVIEW.value
        nutrition_complete = nutrition_calculation.complete and portions.allocation_complete
        nutrition_totals = (
            nutrition_calculation.totals.model_dump(mode="json")
            if nutrition_complete and nutrition_calculation.totals else None
        )
        self.repository.update_scan(
            scan_id,
            state=state,
            decision=decision.action.value,
            capture_path=str(capture_path),
            annotated_path=str(annotated_path),
            image_quality=capture.quality.model_dump(mode="json"),
            predictions=predictions,
            portions=portions.model_dump(mode="json"),
            nutrition_totals=nutrition_totals,
            result_message=decision.rationale,
            error_code=None,
        )
        meal = self.repository.save_meal(
            scan_id=scan_id,
            measured_net_food_weight_g=net_weight_g,
            items=meal_items,
            totals=nutrition_totals,
            state="saved" if state == ScanState.COMPLETED.value else "needs_review",
        )
        result = self.repository.get_scan(scan_id)
        assert result is not None
        result["meal"] = meal
        result["agent_decision"] = decision.model_dump(mode="json")
        result["nutrition_complete"] = nutrition_complete
        return result

    def _needs_plate_tare(self, scan_id: str, message: str) -> dict[str, Any]:
        return self._mark_review(
            scan_id,
            AgentDecision(
                action=AgentAction.REQUEST_PLATE_TARE,
                rationale=message,
                requires_user=True,
            ),
            result_message=message,
            error_code="plate_tare_required",
        )

    def _mark_review(
        self,
        scan_id: str,
        decision: Any,
        *,
        result_message: str,
        error_code: str,
        image_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.repository.update_scan(
            scan_id,
            state=ScanState.NEEDS_REVIEW.value,
            decision=decision.action.value,
            result_message=result_message[:500],
            error_code=error_code,
            image_quality=image_quality,
        )
        result["agent_decision"] = decision.model_dump(mode="json")
        return result

    def _mark_removed(self, scan_id: str) -> dict[str, Any]:
        return self._mark_review(
            scan_id,
            AgentDecision(
                action=AgentAction.RETRY_WEIGHT,
                rationale="Plate was removed before scan processing completed.",
                requires_user=True,
                retryable=True,
            ),
            result_message="Plate was removed before analysis completed.",
            error_code="plate_removed_during_scan",
        )

    def _store(self, scan_id: str, filename: str, content: bytes) -> Path:
        target_dir = self.image_dir / "scans"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{scan_id}-{filename}"
        temp = target.with_suffix(target.suffix + ".tmp")
        with temp.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(target)
        return target
