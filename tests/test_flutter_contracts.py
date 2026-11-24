import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute

from smartnutri_backend.api import create_app
from smartnutri_backend.settings import Settings


class FlutterApiContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        settings = Settings(
            data_dir=Path(self.temp_dir.name),
            scale_mode="mock",
            auto_scan_enabled=False,
            cors_origins=(),
        )
        self.client_context = TestClient(create_app(settings))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def test_mobile_json_routes_have_named_openapi_response_models(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        expected = {
            ("/api/v1/health", "get"): ("200", "HealthResponse"),
            ("/api/v1/scale/calibrate", "post"): ("200", "ScaleCalibrationResponse"),
            ("/api/v1/camera/status", "get"): ("200", "CameraStatusResponse"),
            ("/api/v1/scans/{scan_id}/foods", "get"): ("200", "FoodPredictionsResponse"),
            ("/api/v1/scans/{scan_id}/portions", "get"): ("200", "PortionResponse"),
            ("/api/v1/scans/{scan_id}/nutrition", "get"): ("200", "ScanNutritionResponse"),
            ("/api/v1/nutrition/records", "get"): ("200", "NutritionRecordResponse"),
            ("/api/v1/nutrition/records", "post"): ("201", "NutritionRecordResponse"),
            ("/api/v1/agents/decisions/{scan_id}", "get"): ("200", "AgentDecisionResponse"),
        }

        for (path, method), (status_code, model_name) in expected.items():
            with self.subTest(path=path, method=method):
                response = paths[path][method]["responses"][status_code]
                schema = response["content"]["application/json"]["schema"]
                if path == "/api/v1/nutrition/records" and method == "get":
                    schema = schema["items"]
                self.assertEqual(schema["$ref"], f"#/components/schemas/{model_name}")

    def test_openapi_v1_matches_the_checked_in_flutter_contract(self):
        contract_path = Path(__file__).resolve().parents[1] / "docs" / "contracts" / "openapi-v1.json"
        expected = json.loads(contract_path.read_text(encoding="utf-8"))
        actual = self.client.get("/openapi.json").json()
        self.assertEqual(actual, expected)

    def test_frozen_contract_contains_every_required_backend_operation(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        expected = {
            ("/api/v1/health", "get"),
            ("/api/v1/scale/status", "get"),
            ("/api/v1/scale/tare-config", "get"),
            ("/api/v1/scale/tare", "post"),
            ("/api/v1/scale/calibrate", "post"),
            ("/api/v1/plates", "get"), ("/api/v1/plates", "post"),
            ("/api/v1/plates/default", "get"), ("/api/v1/plates/default", "put"),
            ("/api/v1/camera/status", "get"),
            ("/api/v1/scans", "get"), ("/api/v1/scans/active", "get"),
            ("/api/v1/scans/{scan_id}", "get"),
            ("/api/v1/scans/{scan_id}/foods", "get"),
            ("/api/v1/scans/{scan_id}/portions", "get"),
            ("/api/v1/scans/{scan_id}/nutrition", "get"),
            ("/api/v1/scans/{scan_id}/mask.png", "get"),
            ("/api/v1/scans/{scan_id}/mask/{class_id}", "get"),
            ("/api/v1/scans/{scan_id}/capture.jpg", "get"),
            ("/api/v1/scans/{scan_id}/annotated.jpg", "get"),
            ("/api/v1/camera/capture/{scan_id}", "post"),
            ("/api/v1/scans/{scan_id}/recapture", "post"),
            ("/api/v1/scans/mock-placement", "post"),
            ("/api/v1/nutrition/records", "get"), ("/api/v1/nutrition/records", "post"),
            ("/api/v1/meals", "get"), ("/api/v1/meals/{meal_id}", "get"),
            ("/api/v1/meals/{meal_id}/corrections", "post"),
            ("/api/v1/agents/decisions/{scan_id}", "get"),
        }
        actual = {
            (path, method)
            for path, operations in paths.items()
            for method in operations
            if method in {"get", "post", "put", "patch", "delete"}
        }
        self.assertSetEqual(actual, expected)

    def test_active_scan_contract_is_nullable_and_scan_polling_is_http_only(self):
        schema = self.client.get("/openapi.json").json()
        active_schema = schema["paths"]["/api/v1/scans/active"]["get"]["responses"]["200"]
        self.assertIn("anyOf", active_schema["content"]["application/json"]["schema"])
        self.assertFalse(any(isinstance(route, WebSocketRoute) for route in self.client.app.routes))

        active = self.client.get("/api/v1/scans/active")
        self.assertEqual(active.status_code, 200)
        self.assertIsNone(active.json())

    def test_home_and_camera_status_payloads_match_the_mobile_schemas(self):
        health = self.client.get("/api/v1/health")
        camera = self.client.get("/api/v1/camera/status")

        self.assertEqual(health.status_code, 200, health.text)
        self.assertEqual(health.json()["service"], "smartnutri-backend")
        self.assertEqual(camera.status_code, 200, camera.text)
        self.assertEqual(camera.json()["source"], "mock")
        self.assertTrue(camera.json()["simulated"])

    def test_image_resources_document_their_binary_media_types(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        expected_media = {
            "/api/v1/scans/{scan_id}/mask.png": "image/png",
            "/api/v1/scans/{scan_id}/mask/{class_id}": "image/png",
            "/api/v1/scans/{scan_id}/capture.jpg": "image/jpeg",
            "/api/v1/scans/{scan_id}/annotated.jpg": "image/jpeg",
        }
        for path, media_type in expected_media.items():
            with self.subTest(path=path):
                response = paths[path]["get"]["responses"]["200"]
                self.assertIn(media_type, response["content"])
                self.assertEqual(
                    response["content"][media_type]["schema"],
                    {"type": "string", "format": "binary"},
                )

    def test_nutrition_records_round_trip_as_typed_mobile_json(self):
        payload = {
            "class_id": 66,
            "food_identity": "cooked rice test fixture",
            "preparation_state": "boiled",
            "kcal_per_100g": 130,
            "protein_g_per_100g": 2.7,
            "carbohydrates_g_per_100g": 28.2,
            "fat_g_per_100g": 0.3,
            "source": "unit-test fixture only",
            "source_identifier": "test-fixture:rice-boiled",
            "version": "test-v1",
            "verified": True,
        }
        created = self.client.post("/api/v1/nutrition/records", json=payload)
        self.assertEqual(created.status_code, 201, created.text)
        record = created.json()
        self.assertTrue(record["verified"])
        self.assertEqual(record["class_id"], 66)
        self.assertEqual(record["source_identifier"], "test-fixture:rice-boiled")
        self.assertTrue(record["id"])
        self.assertTrue(record["created_at"])

        listed = self.client.get("/api/v1/nutrition/records", params={"class_id": 66})
        self.assertEqual(listed.status_code, 200)
        self.assertIn(record, listed.json())
        self.assertTrue(any(item["source_identifier"] == "FDC:168878" for item in listed.json()))

    def test_full_mock_scan_contract_exercises_flutter_read_paths_without_fake_nutrition(self):
        for path in (
            "/api/v1/health",
            "/api/v1/scale/status",
            "/api/v1/scale/tare-config",
            "/api/v1/camera/status",
            "/api/v1/plates",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

        plate = self.client.post(
            "/api/v1/plates",
            json={"name": "Contract fixture plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        self.assertEqual(plate.status_code, 201, plate.text)
        self.assertEqual(self.client.get("/api/v1/plates/default").json()["id"], plate.json()["id"])

        placement = self.client.post("/api/v1/scans/mock-placement", json={"gross_weight_g": 750})
        self.assertEqual(placement.status_code, 200, placement.text)
        payload = placement.json()
        self.assertTrue(payload["simulated"])
        self.assertIn("synthetic", payload["simulation_notice"])
        scan = payload["scan"]
        scan_id = scan["id"]
        self.assertEqual(scan["measured_net_food_weight_g"], 450)
        self.assertIsNone(scan["nutrition_totals"])
        active = self.client.get("/api/v1/scans/active")
        self.assertEqual(active.status_code, 200, active.text)
        self.assertEqual(active.json()["id"], scan_id)

        resources = {
            f"/api/v1/scans/{scan_id}": 200,
            f"/api/v1/scans/{scan_id}/foods": 200,
            f"/api/v1/scans/{scan_id}/portions": 200,
            f"/api/v1/scans/{scan_id}/nutrition": 200,
            f"/api/v1/agents/decisions/{scan_id}": 200,
            f"/api/v1/scans/{scan_id}/mask.png": 200,
            f"/api/v1/scans/{scan_id}/mask/1": 200,
            f"/api/v1/scans/{scan_id}/capture.jpg": 200,
            f"/api/v1/scans/{scan_id}/annotated.jpg": 200,
        }
        for path, expected_status in resources.items():
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, expected_status, response.text[:300])

        foods = self.client.get(f"/api/v1/scans/{scan_id}/foods").json()
        self.assertTrue(foods["predictions"])
        self.assertTrue(foods["simulated"])
        portion_response = self.client.get(f"/api/v1/scans/{scan_id}/portions").json()
        portions = portion_response["portions"]
        self.assertEqual(portions["method"], "experimental_visual_area_proportion_v1")
        self.assertTrue(portions["review_required"])
        nutrition = self.client.get(f"/api/v1/scans/{scan_id}/nutrition").json()
        self.assertIsNone(nutrition["totals"])
        self.assertTrue(nutrition["simulated"])

        meal = scan["meal"]
        self.assertIsNotNone(meal)
        self.assertEqual(self.client.get("/api/v1/scans").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/meals").status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/meals/{meal['id']}").status_code, 200)
        records = self.client.get("/api/v1/nutrition/records").json()
        self.assertTrue(records)
        self.assertTrue(all(item["source_identifier"] for item in records))

        correction = self.client.post(
            f"/api/v1/meals/{meal['id']}/corrections",
            json={"items": [{"class_id": 1, "label": "mock-confirmed rice", "weight_g": 450}]},
        )
        self.assertEqual(correction.status_code, 200, correction.text)
        self.assertEqual(correction.json()["items"][0]["estimated_weight_g"], 450)
        self.assertIsNone(correction.json()["kcal_total"])
        corrected_nutrition = self.client.get(f"/api/v1/scans/{scan_id}/nutrition").json()
        self.assertIsNone(corrected_nutrition["totals"])

    def test_native_mobile_defaults_do_not_enable_browser_cors(self):
        self.assertEqual(Settings().cors_origins, ())
        cors = self.client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotIn("access-control-allow-origin", cors.headers)


if __name__ == "__main__":
    unittest.main()
