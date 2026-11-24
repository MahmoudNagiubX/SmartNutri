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
    prediction_mask_consistent: bool = True
    portions: PortionEstimationResult | None = None
    verified_nutrition_classes: set[int] = Field(default_factory=set)
    capture_attempts_remaining: int = 0
    current_meal_state: str = "analyzing"
    analysis_accepted: bool = False


class NutritionDecisionAgent:
    """Transparent backend policy; this is not a separately trained AI model."""

    ambiguous_foodseg_classes = {
        9: "cheese or butter",
        48: "chicken or duck",
        89: "cilantro or mint",
        94: "pepper variety",
        103: "other ingredients",
    }

    def __init__(
        self,
        *,
        min_classification_confidence: float = 0.55,
        max_portion_uncertainty_fraction: float = 0.20,
    ) -> None:
        if not math.isfinite(min_classification_confidence) or not 0 <= min_classification_confidence <= 1:
            raise ValueError("classification confidence threshold must be in [0, 1]")
        if not math.isfinite(max_portion_uncertainty_fraction) or not 0 <= max_portion_uncertainty_fraction <= 1:
            raise ValueError("portion uncertainty threshold must be in [0, 1]")
        self.min_classification_confidence = min_classification_confidence
        self.max_portion_uncertainty_fraction = max_portion_uncertainty_fraction

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

        if not context.prediction_mask_consistent:
            return AgentDecision(
                action=AgentAction.REVIEW_FOOD_PREDICTIONS,
                rationale="Food class predictions conflict with the semantic mask; no nutrition calculation is supported.",
                requires_user=True,
            )

        uncertain_predictions = [
            item for item in context.predictions
            if item.classification_confidence < self.min_classification_confidence
            or item.segmentation_quality != "unverified"
        ]
        ambiguous_predictions = [
            item for item in context.predictions
            if item.class_id in self.ambiguous_foodseg_classes
        ]
        if uncertain_predictions or ambiguous_predictions:
            labels = ", ".join(
                dict.fromkeys(
                    item.label
                    for item in (uncertain_predictions + ambiguous_predictions)
                )
            )[:200]
            return AgentDecision(
                action=AgentAction.REVIEW_FOOD_PREDICTIONS,
                rationale=f"Food identity is uncertain, low confidence, or has a small/fragmented region: {labels}.",
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
        portion_uncertainty_too_high = bool(
            portions
            and context.net_weight_g
            and any(
                item.uncertainty_g / context.net_weight_g > self.max_portion_uncertainty_fraction
                for item in portions.estimates
            )
        )
        if (
            portions is None
            or not portions.allocation_complete
            or portions.unallocated_weight_g not in (None, 0)
            or portions.review_required
            or any(item.review_required for item in portions.estimates)
            or portion_uncertainty_too_high
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
