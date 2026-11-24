import unittest

from smartnutri_backend.portion import ExperimentalVisualProportionEstimator
from smartnutri_backend.schemas import FoodPrediction


def prediction(class_id, label, area, *, simulated=False, confidence=0.95):
    return FoodPrediction(
        class_id=class_id,
        label=label,
        classification_confidence=confidence,
        pixel_area=area,
        region_count=1,
        source="mock" if simulated else "segformer",
        simulated=simulated,
    )


class PortionEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.estimator = ExperimentalVisualProportionEstimator()
        self.foods = [
            prediction(1, "rice", 20),
            prediction(2, "chicken", 50),
            prediction(3, "fries", 30),
        ]

    def test_experimental_area_baseline_conserves_mass_and_requires_review(self):
        result = self.estimator.estimate(450, self.foods)
        self.assertEqual(result.method, "experimental_visual_area_proportion_v1")
        self.assertTrue(result.allocation_complete)
        self.assertTrue(result.review_required)
        self.assertAlmostEqual(sum(item.estimated_weight_g for item in result.estimates), 450, places=2)
        estimates = {item.label: item.estimated_weight_g for item in result.estimates}
        self.assertEqual(estimates, {"rice": 90.0, "chicken": 225.0, "fries": 135.0})
        self.assertTrue(all(item.uncertainty_g > 0 for item in result.estimates))

    def test_synthetic_known_multi_food_portions_report_absolute_error(self):
        # Synthetic fixture only: this quantifies arithmetic behavior, not model accuracy.
        synthetic_foods = [
            prediction(1, "rice", 20, simulated=True),
            prediction(2, "chicken", 50, simulated=True),
            prediction(3, "fries", 30, simulated=True),
        ]
        synthetic_truth_g = {1: 110.0, 2: 220.0, 3: 120.0}
        result = self.estimator.estimate(450, synthetic_foods)

        absolute_errors = [
            abs(item.estimated_weight_g - synthetic_truth_g[item.class_id])
            for item in result.estimates
        ]
        mean_absolute_error_g = sum(absolute_errors) / len(absolute_errors)

        self.assertTrue(result.simulated)
        self.assertAlmostEqual(mean_absolute_error_g, 40 / 3, places=2)

    def test_unknown_visible_area_remains_unallocated(self):
        result = self.estimator.estimate(450, self.foods, unidentified_area_px=20)
        self.assertFalse(result.allocation_complete)
        self.assertTrue(result.review_required)
        self.assertEqual(result.unallocated_weight_g, 75.0)
        self.assertAlmostEqual(
            sum(item.estimated_weight_g for item in result.estimates) + result.unallocated_weight_g,
            450,
            places=2,
        )

    def test_missing_food_regions_do_not_create_fake_portions(self):
        result = self.estimator.estimate(450, [])
        self.assertEqual(result.method, "unavailable")
        self.assertFalse(result.allocation_complete)
        self.assertTrue(result.review_required)
        self.assertEqual(result.estimates, [])

    def test_mock_estimates_are_never_labelled_real(self):
        simulated_foods = [prediction(1, "rice", 20, simulated=True)]
        result = self.estimator.estimate(100, simulated_foods)
        self.assertTrue(result.simulated)
        self.assertTrue(result.estimates[0].simulated)

    def test_prior_or_model_estimates_are_not_silently_used(self):
        with self.assertRaises(NotImplementedError):
            self.estimator.estimate(450, self.foods, prior_weights_g={1: 200})

    def test_invalid_total_and_duplicate_classes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.estimator.estimate(0, self.foods)
        with self.assertRaises(ValueError):
            self.estimator.estimate(450, [self.foods[0], self.foods[0]])


if __name__ == "__main__":
    unittest.main()
