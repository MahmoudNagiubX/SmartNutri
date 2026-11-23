import unittest

from smartnutri_backend.agent import AgentContext, NutritionDecisionAgent
from smartnutri_backend.schemas import (
    AgentAction,
    FoodPrediction,
    ImageQuality,
    NutrientEstimate,
    PortionEstimate,
    PortionEstimationResult,
    ScaleObservation,
)


def scale(status="stable", weight=450, *, stable=True, simulated=False):
    source = "mock" if simulated else "hx711"
    error = status in {"sensor_disconnected", "sensor_error", "overload"}
    return ScaleObservation(
        ok=not error,
        source=source,
        simulated=simulated,
        sensor_connected=not error,
        tared=True,
        calibrated=True,
        stable=stable,
        status=status,
        unit="g",
        weight_g=None if error else weight,
    )


def food(confidence=0.95):
    return FoodPrediction(
        class_id=1,
        label="rice",
        classification_confidence=confidence,
        pixel_area=500,
        region_count=1,
        source="mock",
        simulated=True,
    )


def good_portions(review=False):
    item = PortionEstimate(
        class_id=1,
        label="rice",
        estimated_weight_g=450,
        method="user_corrected" if not review else "experimental_visual_area_proportion_v1",
        confidence=1.0 if not review else 0.25,
        uncertainty_g=0 if not review else 120,
        review_required=review,
        simulated=True,
    )
    return PortionEstimationResult(
        measured_net_weight_g=450,
        method=item.method,
        estimates=[item],
        allocation_complete=True,
        review_required=review,
        simulated=True,
    )


def camera(acceptable=True):
    return ImageQuality(
        acceptable=acceptable,
        reasons=[] if acceptable else ["image_blurry_or_textureless"],
        width=640,
        height=480,
        format="JPEG",
        byte_count=30000,
        brightness_mean=120,
        clipped_pixel_fraction=0.01,
        edge_variance=100 if acceptable else 0,
    )


class NutritionDecisionAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = NutritionDecisionAgent(min_classification_confidence=0.55)
        self.base = AgentContext(
            scale=scale(simulated=True),
            plate_tare_known=True,
            net_weight_g=450,
            camera_quality=camera(),
            predictions=[food()],
            portions=good_portions(),
            verified_nutrition_classes={1},
        )

    def test_weight_errors_and_unstable_readings_retry(self):
        for state, stable in (("unstable", False), ("sensor_disconnected", False), ("overload", False)):
            with self.subTest(state=state):
                context = self.base.model_copy(update={"scale": scale(state, stable=stable)})
                self.assertEqual(self.agent.decide(context).action, AgentAction.RETRY_WEIGHT)

    def test_unknown_tare_requires_plate_registration_or_live_tare(self):
        context = self.base.model_copy(update={"plate_tare_known": False, "net_weight_g": None})
        self.assertEqual(self.agent.decide(context).action, AgentAction.REQUEST_PLATE_TARE)

    def test_bad_camera_requests_recapture_then_user_if_retries_exhausted(self):
        context = self.base.model_copy(update={"camera_quality": camera(False), "capture_attempts_remaining": 1})
        decision = self.agent.decide(context)
        self.assertEqual(decision.action, AgentAction.RECAPTURE_IMAGE)
        self.assertTrue(decision.retryable)
        exhausted = self.agent.decide(context.model_copy(update={"capture_attempts_remaining": 0}))
        self.assertTrue(exhausted.requires_user)

    def test_low_confidence_requests_food_review(self):
        context = self.base.model_copy(update={"predictions": [food(0.4)]})
        self.assertEqual(self.agent.decide(context).action, AgentAction.REVIEW_FOOD_PREDICTIONS)

    def test_missing_nutrition_mapping_requests_preparation_or_mapping(self):
        context = self.base.model_copy(update={"verified_nutrition_classes": set()})
        self.assertEqual(self.agent.decide(context).action, AgentAction.REQUEST_FOOD_PREPARATION)

    def test_uncertain_experimental_portions_are_reviewed(self):
        context = self.base.model_copy(update={"portions": good_portions(review=True)})
        self.assertEqual(self.agent.decide(context).action, AgentAction.REVIEW_PORTION_ESTIMATES)

    def test_portion_uncertainty_fraction_is_configurable_policy_input(self):
        item = PortionEstimate.model_validate(
            {**good_portions().estimates[0].model_dump(), "uncertainty_g": 100}
        )
        portions = PortionEstimationResult(
            measured_net_weight_g=450,
            method="test",
            estimates=[item],
            allocation_complete=True,
            review_required=False,
            simulated=True,
        )
        context = self.base.model_copy(update={"portions": portions})
        self.assertEqual(self.agent.decide(context).action, AgentAction.REVIEW_PORTION_ESTIMATES)

    def test_good_analysis_is_accepted_then_saved_automatically(self):
        accepted = self.agent.decide(self.base)
        self.assertEqual(accepted.action, AgentAction.ACCEPT_ANALYSIS)
        saved = self.agent.decide(self.base.model_copy(update={"analysis_accepted": True}))
        self.assertEqual(saved.action, AgentAction.SAVE_MEAL)
        self.assertFalse(saved.requires_user)


if __name__ == "__main__":
    unittest.main()
