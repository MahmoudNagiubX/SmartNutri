import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from smartnutri_backend.api import create_app
from smartnutri_backend.camera import MockCameraService
from smartnutri_backend.schemas import NutritionRecord
from smartnutri_backend.settings import Settings
from smartnutri_backend.vision import SegmentationResult


def fixture_nutrients(runtime, classes=(1, 2, 3)):
    rows = [
        (1, "rice", "cooked", 130, 2.7, 28.2, 0.3),
        (2, "chicken", "roasted", 200, 25, 0, 10),
        (3, "fries", "fried", 310, 3.4, 41, 15),
    ]
    for class_id, identity, preparation, kcal, protein, carbs, fat in rows:
        if class_id not in classes:
            continue
        runtime.nutrition.register(
            NutritionRecord(
                class_id=class_id,
                food_identity=identity,
                preparation_state=preparation,
                kcal_per_100g=kcal,
                protein_g_per_100g=protein,
                carbohydrates_g_per_100g=carbs,
                fat_g_per_100g=fat,
                source="synthetic-test-fixture",
                source_identifier=f"test-fixture:{class_id}:{identity}:{preparation}",
                version="fixture-v1",
                verified=True,
            )
        )


class AutomaticScanApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        settings = Settings(
            data_dir=Path(self.temp_dir.name),
            scale_mode="mock",
            auto_scan_enabled=False,
            post_stability_delay_s=0.01,
            stable_confirmations=2,
            capture_retries=0,
        )
        self.context = TestClient(create_app(settings))
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def test_case_a_auto_whole_plate_analysis_persists_simulated_meal(self):
        self.client.post(
            "/api/v1/plates",
            json={"name": "test plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        runtime = self.client.app.state.runtime
        fixture_nutrients(runtime)

        response = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        scan = body["scan"]
        self.assertTrue(body["simulated"])
        self.assertIn("synthetic", body["simulation_notice"])
        self.assertEqual(scan["measured_gross_weight_g"], 750)
        self.assertEqual(scan["plate_tare_weight_g"], 300)
        self.assertEqual(scan["measured_net_food_weight_g"], 450)
        self.assertEqual({item["label"] for item in scan["predictions"]}, {"rice", "chicken", "fries"})
        self.assertTrue(scan["nutrition_totals"]["estimated"])
        self.assertEqual(scan["state"], "NEEDS_REVIEW")

        meal = scan["meal"]
        self.assertTrue(meal["simulated"])
        self.assertEqual(meal["measured_net_food_weight_g"], 450)
        self.assertEqual(len(meal["items"]), 3)
        self.assertEqual(meal["state"], "needs_review")
        self.assertEqual(len(self.client.get("/api/v1/meals").json()), 1)
        self.assertEqual(self.client.get(scan["annotated_url"]).status_code, 200)
        self.assertEqual(self.client.get(scan["mask_url"]).status_code, 200)

        duplicate = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        )
        self.assertEqual(duplicate.status_code, 409)

        other_plate = self.client.post(
            "/api/v1/plates",
            json={"name": "second plate", "empty_plate_weight_g": 450, "make_default": False},
        )
        changed_default = self.client.put(
            "/api/v1/plates/default",
            json={"plate_profile_id": other_plate.json()["id"]},
        )
        self.assertEqual(changed_default.status_code, 409)

    def test_case_b_live_plate_tare_is_applied_once(self):
        runtime = self.client.app.state.runtime
        runtime.scale_client.set_weight(300)
        tare = self.client.post(
            "/api/v1/scale/tare", json={"mode": "empty_plate", "confirm": "TARE"}
        )
        self.assertEqual(tare.status_code, 200, tare.text)
        self.assertEqual(tare.json()["mode"], "live_plate")
        self.assertTrue(tare.json()["live_tare_active"])
        self.assertEqual(tare.json()["empty_plate_weight_g"], 300)

        response = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 450}
        )
        self.assertEqual(response.status_code, 200, response.text)
        scan = response.json()["scan"]
        self.assertIsNone(scan["measured_gross_weight_g"])
        self.assertEqual(scan["plate_tare_weight_g"], 300)
        self.assertEqual(scan["measured_net_food_weight_g"], 450)
        self.assertEqual(scan["tare_mode"], "live_plate")

    def test_live_tare_setup_prevents_empty_plate_autoscan_then_measures_added_food_once(self):
        runtime = self.client.app.state.runtime
        prepared = self.client.post(
            "/api/v1/scale/tare",
            json={"mode": "prepare_empty_plate", "confirm": "PREPARE"},
        )
        self.assertEqual(prepared.status_code, 200, prepared.text)
        self.assertTrue(prepared.json()["live_tare_pending"])

        empty_plate = runtime.scale_client.set_weight(300)
        self.assertIsNone(runtime.monitor.monitor.observe(empty_plate, now=1.0))
        self.assertEqual(runtime.repository.list_scans(), [])

        tared = self.client.post(
            "/api/v1/scale/tare",
            json={"mode": "empty_plate", "confirm": "TARE"},
        )
        self.assertEqual(tared.status_code, 200, tared.text)
        self.assertTrue(tared.json()["live_tare_active"])
        self.assertFalse(tared.json()["live_tare_pending"])

        loaded = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 450}
        )
        self.assertEqual(loaded.status_code, 200, loaded.text)
        scan = loaded.json()["scan"]
        self.assertEqual(scan["tare_mode"], "live_plate")
        self.assertIsNone(scan["measured_gross_weight_g"])
        self.assertEqual(scan["plate_tare_weight_g"], 300)
        self.assertEqual(scan["measured_net_food_weight_g"], 450)

    def test_saved_plate_empty_first_waits_for_food_then_scans_without_replacement(self):
        profile = self.client.post(
            "/api/v1/plates",
            json={"name": "test plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        self.assertEqual(profile.status_code, 201, profile.text)
        runtime = self.client.app.state.runtime

        empty_placement = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 300}
        )
        self.assertEqual(empty_placement.status_code, 409)
        self.assertEqual(runtime.repository.list_scans(), [])

        loaded = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        )
        self.assertEqual(loaded.status_code, 200, loaded.text)
        scans = runtime.repository.list_scans()
        self.assertEqual(len(scans), 1)
        scan = runtime.repository.get_scan(scans[0]["id"])
        self.assertIsNotNone(scan)
        self.assertEqual(scan["measured_gross_weight_g"], 750)
        self.assertEqual(scan["plate_tare_weight_g"], 300)
        self.assertEqual(scan["measured_net_food_weight_g"], 450)
        self.assertEqual(runtime.monitor.state.value, "WAITING_FOR_REMOVAL")
        self.assertEqual(
            self.client.post("/api/v1/scans/mock-placement", json={"gross_weight_g": 750}).status_code,
            409,
        )

    def test_no_plate_profile_requests_registration_without_inventing_net_weight(self):
        response = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        )
        self.assertEqual(response.status_code, 200, response.text)
        scan = response.json()["scan"]
        self.assertEqual(scan["decision"], "REQUEST_PLATE_TARE")
        self.assertIsNone(scan["measured_net_food_weight_g"])
        self.assertIsNone(scan["meal"])
        self.assertIsNone(scan["capture_url"])

    def test_wrong_default_plate_weight_is_rejected_without_negative_mass(self):
        self.client.post(
            "/api/v1/plates",
            json={"name": "incorrect plate profile", "empty_plate_weight_g": 800, "make_default": True},
        )
        response = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.app.state.runtime.repository.list_scans(), [])

    def test_missing_verified_nutrition_mapping_never_fabricates_totals(self):
        self.client.post(
            "/api/v1/plates",
            json={"name": "test plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        runtime = self.client.app.state.runtime
        fixture_nutrients(runtime, classes=(1, 2))

        response = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        )
        scan = response.json()["scan"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(scan["decision"], "REQUEST_FOOD_PREPARATION")
        self.assertIsNone(scan["nutrition_totals"])
        self.assertFalse(scan["meal"]["state"] == "saved")

    def test_foods_contract_preserves_simulated_capture_provenance_with_real_model_output(self):
        self.client.post(
            "/api/v1/plates",
            json={"name": "test plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        runtime = self.client.app.state.runtime
        mock_inference = runtime.workflow.inference

        class RealModelOutputFromSyntheticInput:
            def infer(self, image, *, scan_id):
                result = mock_inference.infer(image, scan_id=scan_id)
                predictions = [
                    item.model_copy(update={"source": "segformer", "simulated": False})
                    for item in result.predictions
                ]
                return SegmentationResult(
                    predictions=predictions,
                    semantic_mask_png=result.semantic_mask_png,
                    annotated_image_bytes=result.annotated_image_bytes,
                    width=result.width,
                    height=result.height,
                    source="segformer",
                    simulated=False,
                )

        runtime.workflow.inference = RealModelOutputFromSyntheticInput()
        scan_response = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        )
        self.assertEqual(scan_response.status_code, 200, scan_response.text)
        scan = scan_response.json()["scan"]
        self.assertTrue(scan["capture_simulated"])
        self.assertTrue(scan["simulated"])

        foods = self.client.get(f"/api/v1/scans/{scan['id']}/foods")
        self.assertEqual(foods.status_code, 200, foods.text)
        self.assertEqual(foods.json()["source"], "segformer")
        self.assertTrue(foods.json()["simulated"])
        self.assertTrue(all(not item["simulated"] for item in foods.json()["predictions"]))

    def test_unidentified_food_mass_suppresses_partial_meal_nutrition_totals(self):
        self.client.post(
            "/api/v1/plates",
            json={"name": "test plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        runtime = self.client.app.state.runtime
        fixture_nutrients(runtime)
        base_inference = runtime.inference

        class LowConfidenceInference:
            def infer(self, image, *, scan_id):
                result = base_inference.infer(image, scan_id=scan_id)
                predictions = [
                    item.model_copy(update={"classification_confidence": 0.4})
                    if item.class_id == 3 else item
                    for item in result.predictions
                ]
                return SegmentationResult(
                    predictions=predictions,
                    semantic_mask_png=result.semantic_mask_png,
                    annotated_image_bytes=result.annotated_image_bytes,
                    width=result.width,
                    height=result.height,
                    source=result.source,
                    simulated=result.simulated,
                )

        runtime.workflow.inference = LowConfidenceInference()
        response = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        )

        self.assertEqual(response.status_code, 200, response.text)
        scan = response.json()["scan"]
        self.assertEqual(scan["decision"], "REVIEW_FOOD_PREDICTIONS")
        self.assertFalse(scan["portions"]["allocation_complete"])
        self.assertIsNone(scan["nutrition_totals"])
        self.assertIsNone(scan["meal"]["kcal_total"])
        self.assertTrue(all(item["needs_review"] for item in scan["meal"]["items"]))

    def test_mock_weight_payload_is_validated_and_real_controls_require_token(self):
        invalid = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": -5}
        )
        self.assertEqual(invalid.status_code, 422)

        protected = TestClient(
            create_app(
                Settings(
                    data_dir=Path(self.temp_dir.name) / "protected",
                    scale_mode="real",
                    control_token="test-control-token",
                    auto_scan_enabled=False,
                )
            )
        )
        with protected:
            denied = protected.post(
                "/api/v1/scale/tare",
                json={"mode": "platform_empty", "confirm": "TARE"},
            )
            self.assertEqual(denied.status_code, 403)

    def test_camera_unavailable_requests_one_guarded_recapture(self):
        self.context.__exit__(None, None, None)
        settings = Settings(
            data_dir=Path(self.temp_dir.name) / "camera-retry",
            scale_mode="mock",
            camera_mode="file",
            camera_file_path=Path(self.temp_dir.name) / "missing-image.jpg",
            auto_scan_enabled=False,
            post_stability_delay_s=0.01,
            stable_confirmations=2,
            capture_retries=0,
        )
        self.context = TestClient(create_app(settings))
        self.client = self.context.__enter__()
        self.client.post(
            "/api/v1/plates",
            json={"name": "test plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        first = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        ).json()["scan"]
        self.assertEqual(first["error_code"], "camera_unavailable")
        self.assertEqual(first["decision"], "RECAPTURE_IMAGE")
        self.assertIsNone(first["meal"])

        runtime = self.client.app.state.runtime
        runtime.camera = MockCameraService(capture_retries=0)
        runtime.workflow.camera = runtime.camera
        recaptured = self.client.post(f"/api/v1/scans/{first['id']}/recapture")
        self.assertEqual(recaptured.status_code, 200, recaptured.text)
        self.assertEqual(recaptured.json()["state"], "NEEDS_REVIEW")
        self.assertIsNotNone(recaptured.json()["capture_url"])
        self.assertIsNotNone(recaptured.json()["meal"])

    def test_user_portion_correction_recalculates_verified_fixture_nutrition(self):
        self.client.post(
            "/api/v1/plates",
            json={"name": "test plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        runtime = self.client.app.state.runtime
        fixture_nutrients(runtime)
        scan = self.client.post(
            "/api/v1/scans/mock-placement", json={"gross_weight_g": 750}
        ).json()["scan"]
        meal_id = scan["meal"]["id"]

        correction = self.client.post(
            f"/api/v1/meals/{meal_id}/corrections",
            json={
                "items": [
                    {"class_id": 1, "label": "rice", "weight_g": 100},
                    {"class_id": 2, "label": "chicken", "weight_g": 200},
                    {"class_id": 3, "label": "fries", "weight_g": 150},
                ]
            },
        )
        self.assertEqual(correction.status_code, 200, correction.text)
        updated = correction.json()
        self.assertEqual(updated["correction_revision"], 1)
        self.assertEqual(updated["state"], "saved")
        self.assertTrue(all(item["user_corrected"] for item in updated["items"]))
        self.assertIsNotNone(updated["kcal_total"])
        self.assertTrue(updated["simulated"])


if __name__ == "__main__":
    unittest.main()
