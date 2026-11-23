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
            "version": "test-v1",
            "verified": True,
        }
        created = self.client.post("/api/v1/nutrition/records", json=payload)
        self.assertEqual(created.status_code, 201, created.text)
        record = created.json()
        self.assertTrue(record["verified"])
        self.assertEqual(record["class_id"], 66)
        self.assertTrue(record["id"])
        self.assertTrue(record["created_at"])

        listed = self.client.get("/api/v1/nutrition/records", params={"class_id": 66})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json(), [record])

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
