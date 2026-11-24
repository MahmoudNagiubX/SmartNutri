import tempfile
import unittest
from pathlib import Path

from smartnutri_backend.nutrition import NutritionService
from smartnutri_backend.nutrition_catalog import CURATED_NUTRITION_RECORDS
from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.schemas import NutritionRecord, PortionEstimate


class NutritionCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = SQLiteRepository(Path(self.temp_dir.name) / "catalog.sqlite3")
        self.service = NutritionService(self.repository)

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def test_catalog_has_eight_food_identities_and_traceable_usda_records(self):
        self.assertEqual(len(CURATED_NUTRITION_RECORDS), 11)
        self.assertEqual(len({item["food_identity"] for item in CURATED_NUTRITION_RECORDS}), 8)
        self.assertEqual(
            {item["source_identifier"] for item in CURATED_NUTRITION_RECORDS},
            {f"FDC:{fdc_id}" for fdc_id in (168878, 168877, 171477, 171077, 172411, 170698, 168202, 173944, 169967, 170379, 170457)},
        )
        for item in CURATED_NUTRITION_RECORDS:
            with self.subTest(source_identifier=item["source_identifier"]):
                record = NutritionRecord.model_validate(item)
                self.assertEqual(record.source, "USDA FoodData Central")
                self.assertEqual(record.version, "SR Legacy final release, April 2018")
                self.assertTrue(record.verified)

    def test_catalog_seeding_is_idempotent_and_preserves_exact_source_ids(self):
        first_count = self.service.seed_catalog(CURATED_NUTRITION_RECORDS)
        second_count = self.service.seed_catalog(CURATED_NUTRITION_RECORDS)
        records = self.service.list_records()

        self.assertEqual(first_count, 11)
        self.assertEqual(second_count, 0)
        self.assertEqual(len(records), 11)
        self.assertEqual({item["source_identifier"] for item in records}, {
            item["source_identifier"] for item in CURATED_NUTRITION_RECORDS
        })

    def test_catalog_macro_values_match_the_curated_source_snapshot(self):
        expected = {
            "FDC:168878": (130.0, 2.69, 28.17, 0.28),
            "FDC:168877": (365.0, 7.13, 79.95, 0.66),
            "FDC:171477": (165.0, 31.02, 0.0, 3.57),
            "FDC:171077": (120.0, 22.5, 0.0, 2.62),
            "FDC:172411": (201.0, 23.48, 0.0, 11.2),
            "FDC:170698": (312.0, 3.43, 41.44, 14.73),
            "FDC:168202": (57.0, 0.28, 13.6, 0.15),
            "FDC:173944": (89.0, 1.09, 22.84, 0.33),
            "FDC:169967": (35.0, 2.38, 7.18, 0.41),
            "FDC:170379": (34.0, 2.82, 6.64, 0.37),
            "FDC:170457": (18.0, 0.88, 3.89, 0.2),
        }
        actual = {
            item["source_identifier"]: (
                item["kcal_per_100g"],
                item["protein_g_per_100g"],
                item["carbohydrates_g_per_100g"],
                item["fat_g_per_100g"],
            )
            for item in CURATED_NUTRITION_RECORDS
        }
        self.assertEqual(actual, expected)

    def test_verified_multi_food_totals_match_independent_per_100g_calculation(self):
        self.service.seed_catalog(CURATED_NUTRITION_RECORDS)
        records = {
            item["source_identifier"]: item
            for item in self.service.list_records()
        }
        portions = [
            PortionEstimate(
                class_id=class_id,
                label=label,
                estimated_weight_g=weight_g,
                method="user_correction",
                confidence=1.0,
                uncertainty_g=0.0,
                review_required=False,
            )
            for class_id, label, weight_g in (
                (66, "rice", 150.0),
                (48, "chicken", 200.0),
                (3, "french fries", 100.0),
            )
        ]
        calculation = self.service.calculate(
            portions,
            nutrition_record_ids={
                66: records["FDC:168878"]["id"],
                48: records["FDC:171477"]["id"],
                3: records["FDC:170698"]["id"],
            },
        )

        self.assertTrue(calculation.complete)
        self.assertEqual(
            calculation.totals.model_dump(exclude={"estimated", "source", "source_identifier", "version"}),
            {
                "kcal": 837.0,
                "protein_g": 69.51,
                "carbohydrates_g": 83.7,
                "fat_g": 22.29,
            },
        )
        self.assertEqual(
            {item.source_identifier for item in (portion.nutrients for portion in calculation.items)},
            {"FDC:168878", "FDC:171477", "FDC:170698"},
        )

    def test_ambiguous_food_and_preparation_classes_require_explicit_selection(self):
        self.service.seed_catalog(CURATED_NUTRITION_RECORDS)

        for class_id in (25, 29, 48, 66, 73, 87):
            with self.subTest(class_id=class_id):
                self.assertIsNone(self.service.lookup_for_class(class_id))

        fries = self.service.lookup_for_class(3)
        self.assertIsNotNone(fries)
        self.assertEqual(fries[1].source_identifier, "FDC:170698")

        rice = next(item for item in self.service.list_records(class_id=66) if item["source_identifier"] == "FDC:168878")
        selected = self.service.lookup_for_class(66, nutrition_record_id=rice["id"])
        self.assertIsNotNone(selected)
        self.assertEqual(selected[1].preparation_state, "cooked; long-grain, regular, enriched")

    def test_chicken_duck_class_requires_explicit_species_and_source_record(self):
        self.service.seed_catalog(CURATED_NUTRITION_RECORDS)
        self.assertIsNone(self.service.lookup_for_class(48))

        duck = next(
            item for item in self.service.list_records(class_id=48)
            if item["source_identifier"] == "FDC:172411"
        )
        selected = self.service.lookup_for_class(48, nutrition_record_id=duck["id"])
        self.assertIsNotNone(selected)
        self.assertEqual(selected[1].food_identity, "duck, domesticated, meat only")
        self.assertEqual(selected[1].protein_g_per_100g, 23.48)
        self.assertEqual(selected[1].source_identifier, "FDC:172411")

    def test_strawberry_cherry_and_blueberry_without_records_keep_nutrients_unavailable(self):
        self.service.seed_catalog(CURATED_NUTRITION_RECORDS)
        portions = [
            PortionEstimate(
                class_id=class_id,
                label=label,
                estimated_weight_g=100,
                method="audit-fixture",
                confidence=1.0,
                uncertainty_g=0,
                review_required=True,
                simulated=False,
            )
            for class_id, label in ((30, "strawberry"), (31, "cherry"), (32, "blueberry"))
        ]

        result = self.service.calculate(portions, simulated=False)

        self.assertFalse(result.complete)
        self.assertIsNone(result.totals)
        self.assertEqual(
            [(item.class_id, item.nutrients, item.mapping_status) for item in result.items],
            [
                (30, None, "missing_or_ambiguous_verified_record"),
                (31, None, "missing_or_ambiguous_verified_record"),
                (32, None, "missing_or_ambiguous_verified_record"),
            ],
        )

    def test_mock_fixture_ids_never_resolve_real_catalog_records(self):
        self.service.seed_catalog(CURATED_NUTRITION_RECORDS)
        mock_portion = PortionEstimate(
            class_id=3,
            label="fries",
            estimated_weight_g=100,
            method="mock-fixture",
            confidence=1,
            uncertainty_g=0,
            review_required=False,
            simulated=True,
        )
        without_fixture = self.service.calculate([mock_portion], simulated=True)
        self.assertFalse(without_fixture.complete)
        self.assertIsNone(without_fixture.items[0].nutrients)

        fixture = NutritionRecord(
            class_id=3,
            food_identity="mock fries",
            preparation_state="simulated",
            kcal_per_100g=310,
            protein_g_per_100g=3.4,
            carbohydrates_g_per_100g=41,
            fat_g_per_100g=15,
            source="synthetic-test-fixture",
            source_identifier="test-fixture:mock-fries",
            version="fixture-v1",
            verified=True,
        )
        self.service.register(fixture)
        simulated_result = self.service.calculate([mock_portion], simulated=True)
        real_result = self.service.calculate([mock_portion.model_copy(update={"simulated": False})], simulated=False)

        self.assertEqual(simulated_result.items[0].source_identifier, "test-fixture:mock-fries")
        self.assertEqual(real_result.items[0].source_identifier, "FDC:170698")


if __name__ == "__main__":
    unittest.main()
