import unittest

from pydantic import ValidationError

from smartnutri_backend.schemas import (
    FoodPrediction,
    NutritionRecord,
    PortionEstimationResult,
    ScaleObservation,
)


def stable_reading(**overrides):
    values = {
        "ok": True,
        "source": "hx711",
        "simulated": False,
        "sensor_connected": True,
        "tared": True,
        "calibrated": True,
        "stable": True,
        "status": "stable",
        "unit": "g",
        "weight_g": 750.0,
    }
    values.update(overrides)
    return values


class BackendContractTests(unittest.TestCase):
    def test_real_stable_scale_reading_requires_ready_hardware_flags(self):
        observation = ScaleObservation.model_validate(stable_reading())
        self.assertEqual(observation.source, "hx711")
        self.assertFalse(observation.simulated)
        self.assertEqual(observation.weight_g, 750.0)

        with self.assertRaises(ValidationError):
            ScaleObservation.model_validate(stable_reading(simulated=True))
        with self.assertRaises(ValidationError):
            ScaleObservation.model_validate(stable_reading(sensor_connected=False))

    def test_simulated_measurement_cannot_claim_real_hardware_source(self):
        with self.assertRaises(ValidationError):
            ScaleObservation.model_validate(stable_reading(source="mock", simulated=False))
        simulated = stable_reading(source="mock", simulated=True)
        observation = ScaleObservation.model_validate(simulated)
        self.assertTrue(observation.simulated)

    def test_food_prediction_excludes_background_and_preserves_foodseg_ids(self):
        prediction = FoodPrediction(
            class_id=103,
            label="example food",
            classification_confidence=0.9,
            pixel_area=1200,
            region_count=2,
        )
        self.assertEqual(prediction.class_id, 103)
        self.assertEqual(prediction.region_count, 2)
        for invalid_id in (0, 104):
            with self.subTest(class_id=invalid_id), self.assertRaises(ValidationError):
                FoodPrediction(
                    class_id=invalid_id,
                    label="background",
                    classification_confidence=0.9,
                    pixel_area=1200,
                    region_count=1,
                )

    def test_unavailable_portion_estimate_keeps_measured_total_without_fabrication(self):
        result = PortionEstimationResult(
            measured_net_weight_g=450.0,
            method="unavailable",
            estimates=[],
            allocation_complete=False,
            review_required=True,
        )
        self.assertEqual(result.measured_net_weight_g, 450.0)
        self.assertFalse(result.allocation_complete)

    def test_complete_portions_must_sum_to_measured_net_weight(self):
        with self.assertRaises(ValidationError):
            PortionEstimationResult(
                measured_net_weight_g=450.0,
                method="experimental_visual_area_proportion_v1",
                estimates=[],
                allocation_complete=True,
                review_required=True,
            )

    def test_nutrition_requires_source_and_version(self):
        record = NutritionRecord(
            food_identity="rice",
            preparation_state="cooked",
            kcal_per_100g=130,
            protein_g_per_100g=2.7,
            carbohydrates_g_per_100g=28.2,
            fat_g_per_100g=0.3,
            source="synthetic-test-fixture",
            version="test-v1",
        )
        self.assertEqual(record.source, "synthetic-test-fixture")
        with self.assertRaises(ValidationError):
            NutritionRecord.model_validate({**record.model_dump(), "source": ""})


if __name__ == "__main__":
    unittest.main()
