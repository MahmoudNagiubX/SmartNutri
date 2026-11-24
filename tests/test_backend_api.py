import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from smartnutri_backend.api import create_app
from smartnutri_backend.settings import Settings


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self.temp_dir.name),
            auto_scan_enabled=False,
            cors_origins=("http://localhost:5173",),
        )
        self.client_context = TestClient(create_app(self.settings))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def test_health_and_scale_status_do_not_fake_hardware(self):
        health = self.client.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        scale = self.client.get("/api/v1/scale/status")
        self.assertEqual(scale.status_code, 200)
        self.assertEqual(scale.json()["status"], "not_configured")
        self.assertIsNone(scale.json()["weight_g"])
        self.assertFalse(scale.json()["simulated"])

    def test_health_response_and_settings_repr_do_not_expose_device_secrets(self):
        protected = create_app(
            Settings(
                data_dir=Path(self.temp_dir.name) / "protected-health",
                scale_mode="real",
                scale_base_url="http://192.168.1.30",
                control_token="test-control-secret",
                scale_device_token="test-device-secret",
                auto_scan_enabled=False,
            )
        )
        with TestClient(protected) as client:
            health = client.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        self.assertNotIn("test-control-secret", health.text)
        self.assertNotIn("test-device-secret", health.text)
        self.assertNotIn("test-control-secret", repr(protected.state.runtime.settings))
        self.assertNotIn("test-device-secret", repr(protected.state.runtime.settings))

    def test_plate_profile_and_default_are_persisted(self):
        created = self.client.post(
            "/api/v1/plates",
            json={"name": "Ceramic dinner plate", "empty_plate_weight_g": 300.0, "make_default": True},
        )
        self.assertEqual(created.status_code, 201)
        plate = created.json()
        self.assertEqual(plate["empty_plate_weight_g"], 300.0)
        self.assertTrue(plate["is_default"])

        listed = self.client.get("/api/v1/plates")
        self.assertEqual(len(listed.json()), 1)
        self.assertEqual(self.client.get("/api/v1/plates/default").json()["id"], plate["id"])

    def test_default_plate_updates_automatic_placement_threshold(self):
        first = self.client.post(
            "/api/v1/plates",
            json={"name": "small plate", "empty_plate_weight_g": 300, "make_default": True},
        )
        second = self.client.post(
            "/api/v1/plates",
            json={"name": "large plate", "empty_plate_weight_g": 400, "make_default": False},
        )
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        runtime = self.client.app.state.runtime
        self.assertEqual(runtime.monitor.monitor.placement_threshold_g, 320)

        changed = self.client.put(
            "/api/v1/plates/default", json={"plate_profile_id": second.json()["id"]}
        )
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(runtime.monitor.monitor.placement_threshold_g, 420)
        self.assertEqual(changed.json()["id"], second.json()["id"])

    def test_plate_weight_and_name_are_validated(self):
        response = self.client.post(
            "/api/v1/plates",
            json={"name": "  ", "empty_plate_weight_g": -1},
        )
        self.assertEqual(response.status_code, 422)

    def test_cors_allows_only_configured_local_origin(self):
        allowed = self.client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "http://localhost:5173")

        denied = self.client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotIn("access-control-allow-origin", denied.headers)


if __name__ == "__main__":
    unittest.main()
