import tempfile
import unittest
from pathlib import Path

from smartnutri_backend.nutrition import NutritionService
from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.schemas import NutritionRecord, PortionEstimate


class NutritionServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = SQLiteRepository(Path(self.temp_dir.name) / "nutrition.sqlite3")
        self.service = NutritionService(self.repository)

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def record(self, **overrides):
        values = {
            "class_id": 1,
            "food_identity": "rice",
            "preparation_state": "cooked",
            "kcal_per_100g": 130,
            "protein_g_per_100g": 2.7,
            "carbohydrates_g_per_100g": 28.2,
            "fat_g_per_100g": 0.3,
            "source": "synthetic-test-fixture",
            "version": "fixture-v1",
            "verified": True,
        }
        values.update(overrides)
        return NutritionRecord(**values)

    def estimate(self, weight=150, class_id=1, label="rice"):
        return PortionEstimate(
            class_id=class_id,
            label=label,
            estimated_weight_g=weight,
            method="synthetic-test-fixture",
            confidence=1,
            uncertainty_g=0,
            review_required=False,
            simulated=True,
        )

    def test_calculates_macros_from_verified_per_100g_source(self):
        self.service.register(self.record())
        result = self.service.calculate([self.estimate()])
        item = result.items[0]
        self.assertEqual(item.nutrients.kcal, 195.0)
        self.assertEqual(item.nutrients.protein_g, 4.05)
        self.assertEqual(item.nutrients.carbohydrates_g, 42.3)
        self.assertEqual(item.nutrients.fat_g, 0.45)
        self.assertEqual(item.nutrients.source, "synthetic-test-fixture")
        self.assertTrue(item.nutrients.estimated)
        self.assertTrue(result.simulated)

    def test_unverified_or_missing_records_do_not_create_nutrient_values(self):
        self.service.register(self.record(verified=False))
        result = self.service.calculate([self.estimate()])
        self.assertFalse(result.complete)
        self.assertIsNone(result.items[0].nutrients)
        self.assertIsNone(result.totals)

    def test_ambiguous_preparation_mapping_requires_user_selection(self):
        self.service.register(self.record())
        self.service.register(
            self.record(
                preparation_state="fried",
                source="second-source",
                version="v2",
                kcal_per_100g=180,
            )
        )
        result = self.service.calculate([self.estimate()])
        self.assertFalse(result.complete)
        self.assertIsNone(result.items[0].nutrients)

    def test_meal_totals_are_only_returned_when_every_food_is_mapped(self):
        self.service.register(self.record())
        self.service.register(
            self.record(
                class_id=2,
                food_identity="chicken",
                preparation_state="roasted",
                source="fixture-chicken",
                kcal_per_100g=200,
                protein_g_per_100g=25,
                carbohydrates_g_per_100g=0,
                fat_g_per_100g=10,
            )
        )
        result = self.service.calculate([self.estimate(150), self.estimate(200, 2, "chicken")])
        self.assertTrue(result.complete)
        self.assertIsNotNone(result.totals)
        self.assertEqual(result.totals.kcal, 595.0)

    def test_invalid_nutrient_values_are_rejected(self):
        with self.assertRaises(Exception):
            self.record(kcal_per_100g=-1)


if __name__ == "__main__":
    unittest.main()
