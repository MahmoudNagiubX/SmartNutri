from __future__ import annotations

import math

from pydantic import Field

from smartnutri_backend.schemas import (
    AgentAction,
    AgentDecision,
    FoodPrediction,
    ImageQuality,
    PortionEstimationResult,
    ScaleObservation,
    StrictModel,
)


class AgentContext(StrictModel):
    scale: ScaleObservation | None = None
    plate_tare_known: bool = False
    net_weight_g: float | None = None
    camera_quality: ImageQuality | None = None
    predictions: list[FoodPrediction] = Field(default_factory=list)
    portions: PortionEstimationResult | None = None
    verified_nutrition_classes: set[int] = Field(default_factory=set)
    capture_attempts_remaining: int = 0
    current_meal_state: str = "analyzing"
    analysis_accepted: bool = False


class NutritionDecisionAgent:
    """Transparent backend policy; this is not a separately trained AI model."""

    def __init__(self, *, min_classification_confidence: float = 0.55) -> None:
        if not math.isfinite(min_classification_confidence) or not 0 <= min_classification_confidence <= 1:
            raise ValueError("classification confidence threshold must be in [0, 1]")
        self.min_classification_confidence = min_classification_confidence

    def decide(self, context: AgentContext) -> AgentDecision:
        reading = context.scale
        if reading is None or not reading.usable_stable:
            status = reading.status if reading is not None else "missing"
            severe = status in {"sensor_disconnected", "sensor_error", "overload"}
            return AgentDecision(
                action=AgentAction.RETRY_WEIGHT,
                rationale=f"Scale reading is not usable and stable ({status}).",
                requires_user=severe,
                retryable=not severe,
            )

        if (
            not context.plate_tare_known
            or context.net_weight_g is None
            or not math.isfinite(context.net_weight_g)
            or context.net_weight_g <= 0
        ):
            return AgentDecision(
                action=AgentAction.REQUEST_PLATE_TARE,
                rationale="A valid plate tare is required to determine positive net food mass.",
                requires_user=True,
            )

        if context.camera_quality is None or not context.camera_quality.acceptable:
            has_retries = context.capture_attempts_remaining > 0
            return AgentDecision(
                action=AgentAction.RECAPTURE_IMAGE,
                rationale="The captured image failed acquisition quality checks.",
                requires_user=not has_retries,
                retryable=has_retries,
            )

        if not context.predictions:
            return AgentDecision(
                action=AgentAction.REQUEST_FOOD_PREPARATION,
                rationale="No non-background food regions were identified; arrange the plate and retry.",
                requires_user=True,
            )

        low_confidence = [
            item for item in context.predictions
            if item.classification_confidence < self.min_classification_confidence
        ]
        if low_confidence:
            labels = ", ".join(item.label for item in low_confidence[:4])
            return AgentDecision(
                action=AgentAction.REVIEW_FOOD_PREDICTIONS,
                rationale=f"Food class confidence is below threshold for: {labels}.",
                requires_user=True,
            )

        missing_nutrition = {
            item.class_id for item in context.predictions
            if item.class_id not in context.verified_nutrition_classes
        }
        if missing_nutrition:
            return AgentDecision(
                action=AgentAction.REQUEST_FOOD_PREPARATION,
                rationale="Verified per-100 g nutrition mapping or preparation state is missing.",
                requires_user=True,
            )

        portions = context.portions
        if (
            portions is None
            or not portions.allocation_complete
            or portions.unallocated_weight_g not in (None, 0)
            or portions.review_required
            or any(item.review_required for item in portions.estimates)
            or abs(sum(item.estimated_weight_g for item in portions.estimates) - context.net_weight_g) > 0.05
        ):
            return AgentDecision(
                action=AgentAction.REVIEW_PORTION_ESTIMATES,
                rationale="Per-food mass is experimental, incomplete, or does not conserve the measured total.",
                requires_user=True,
            )

        if context.analysis_accepted:
            return AgentDecision(
                action=AgentAction.SAVE_MEAL,
                rationale="All inputs passed policy checks and the analysis was accepted.",
                requires_user=False,
            )
        return AgentDecision(
            action=AgentAction.ACCEPT_ANALYSIS,
            rationale="Scale, image, food classes, portions, and verified nutrition are ready.",
            requires_user=False,
        )
