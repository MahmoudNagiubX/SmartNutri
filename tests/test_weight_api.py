from __future__ import annotations

import json
import io
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock, patch

from smartnutri_weight.client import WeightClient, WeightReading
from smartnutri_weight.logic import StabilityWindow, weight_delta_g


def mock_weight_payload(weight_g: float = 125.4, **overrides: object) -> dict[str, object]:
    """Synthetic test reading; source and simulated fields always say mock."""

    payload: dict[str, object] = {
        "ok": True,
        "source": "mock",
        "simulated": True,
        "sensor_connected": False,
        "tared": False,
        "calibrated": False,
        "stable": False,
        "status": "mock",
        "unit": "g",
        "weight_g": weight_g,
        "weight_delta_g": None,
        "stability_range_g": None,
        "stability_threshold_g": 2.0,
        "sample_count": 0,
        "window_size": 10,
        "timestamp_ms": 0,
        "device_session_id": "fixture-session",
        "measurement_id": "fixture-reading",
    }
    payload.update(overrides)
    return payload


def hardware_error_payload(
    status: str = "sensor_error", *, sensor_connected: bool = True, **overrides: object
) -> dict[str, object]:
    """Protocol fixture for an error state; it contains no measured weight."""

    payload: dict[str, object] = {
        "ok": False,
        "source": "hx711",
        "simulated": False,
        "sensor_connected": sensor_connected,
        "tared": True,
        "calibrated": True,
        "stable": False,
        "status": status,
        "unit": "g",
        "weight_g": None,
        "weight_delta_g": None,
        "stability_range_g": None,
        "stability_threshold_g": 2.0,
        "sample_count": 0,
        "window_size": 10,
        "timestamp_ms": 0,
        "device_session_id": "fixture-session",
        "measurement_id": "fixture-reading",
    }
    payload.update(overrides)
    return payload


def http_response(payload: dict[str, object]) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
    return response


class WeightApiTests(unittest.TestCase):
    def test_mock_api_payload_has_a_valid_format_and_explicit_simulation_marker(self) -> None:
        reading = WeightReading.from_payload(mock_weight_payload())

        self.assertEqual(reading.weight_g, 125.4)
        self.assertEqual(reading.source, "mock")
        self.assertTrue(reading.simulated)
        self.assertEqual(reading.status, "mock")
        self.assertEqual(reading.unit, "g")

    def test_hardware_error_response_has_no_measured_value(self) -> None:
        reading = WeightReading.from_payload(
            hardware_error_payload("sensor_disconnected", sensor_connected=False)
        )

        self.assertFalse(reading.ok)
        self.assertFalse(reading.sensor_connected)
        self.assertEqual(reading.source, "hx711")
        self.assertFalse(reading.simulated)
        self.assertIsNone(reading.weight_g)
        self.assertEqual(reading.status, "sensor_disconnected")

    def test_mock_source_cannot_be_mislabeled_as_hardware(self) -> None:
        with self.assertRaisesRegex(ValueError, "source and simulated"):
            WeightReading.from_payload(mock_weight_payload(source="hx711", simulated=True))

    def test_hardware_source_cannot_be_marked_simulated(self) -> None:
        with self.assertRaisesRegex(ValueError, "source and simulated"):
            WeightReading.from_payload(hardware_error_payload(simulated=True))

    def test_nonfinite_sensor_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            WeightReading.from_payload(mock_weight_payload(float("nan")))

    def test_negative_synthetic_reading_is_kept_explicitly_mock(self) -> None:
        reading = WeightReading.from_payload(
            mock_weight_payload(-1.25, weight_delta_g=-0.3)
        )

        self.assertEqual(reading.weight_g, -1.25)
        self.assertEqual(reading.weight_delta_g, -0.3)
        self.assertTrue(reading.simulated)

    def test_http_get_calls_weight_endpoint_and_preserves_mock_marker(self) -> None:
        with patch(
            "smartnutri_weight.client.urlopen", return_value=http_response(mock_weight_payload())
        ) as urlopen:
            reading = WeightClient("http://scale.local/").get_weight()

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://scale.local/weight")
        self.assertEqual(request.get_method(), "GET")
        self.assertTrue(reading.simulated)

    def test_real_client_never_treats_mock_tare_response_as_hardware(self) -> None:
        payload = mock_weight_payload() | {"operation": "tare", "operation_ok": True}
        with patch(
            "smartnutri_weight.client.urlopen", return_value=http_response(payload)
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                WeightClient("http://scale.local", device_token="test-device-token").tare(confirmation="EMPTY")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://scale.local/tare")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("X-smartnutri-device-token"), "test-device-token")
        self.assertNotIn("token=", request.full_url)

    def test_async_tare_poll_does_not_hide_a_simulated_result(self) -> None:
        pending = mock_weight_payload(
            operation="tare",
            operation_status="pending",
            operation_ok=None,
            operation_message="collecting_eight_samples",
        )
        completed = mock_weight_payload(
            operation="tare",
            operation_status="succeeded",
            operation_ok=True,
            operation_message="tare_saved",
        )
        client = WeightClient("http://scale.local", retry_delay_s=0, device_token="test-device-token")
        with patch.object(client, "_request", side_effect=[pending, completed]) as request:
            with patch("smartnutri_weight.client.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    client.tare(confirmation="EMPTY")

        self.assertEqual(request.call_args_list[-1].args, ("GET", "/weight"))

    def test_async_tare_propagates_sensor_failure(self) -> None:
        pending = mock_weight_payload(
            operation="tare",
            operation_status="pending",
            operation_ok=None,
        )
        failed = mock_weight_payload(
            operation="tare",
            operation_status="failed",
            operation_ok=False,
            operation_message="sensor_disconnected",
        )
        client = WeightClient("http://scale.local", retry_delay_s=0, device_token="test-device-token")
        with patch.object(client, "_request", side_effect=[pending, failed]):
            with patch("smartnutri_weight.client.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "sensor_disconnected"):
                    client.tare(confirmation="EMPTY")

    def test_confirmed_calibration_posts_expected_form_but_rejects_mock_result(self) -> None:
        pending = mock_weight_payload(
            operation="calibrate",
            operation_status="pending",
            operation_ok=None,
            operation_message="collecting_eight_samples",
        )
        completed = mock_weight_payload(
            operation="calibrate",
            operation_status="succeeded",
            operation_ok=True,
            operation_message="calibration_saved",
        )
        client = WeightClient("http://scale.local", retry_delay_s=0, device_token="test-device-token")
        with patch.object(client, "_request", side_effect=[pending, completed]) as request:
            with patch("smartnutri_weight.client.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    client.calibrate(500.0, confirmation="CALIBRATE")

        self.assertEqual(
            request.call_args_list[0].kwargs["form"],
            {"known_weight_g": "500", "confirm": "CALIBRATE"},
        )

    def test_get_retries_transient_disconnect_but_post_is_not_retried(self) -> None:
        client = WeightClient("http://scale.local", retry_delay_s=0, max_get_retries=1, device_token="test-device-token")
        with patch(
            "smartnutri_weight.client.urlopen",
            side_effect=[URLError("temporary disconnect"), http_response(mock_weight_payload())],
        ) as urlopen:
            reading = client.get_weight()
        self.assertTrue(reading.simulated)
        self.assertEqual(urlopen.call_count, 2)

        with patch("smartnutri_weight.client.urlopen", side_effect=URLError("offline")) as urlopen:
            with self.assertRaises(URLError):
                client.tare(confirmation="EMPTY")
        self.assertEqual(urlopen.call_count, 1)

    def test_http_error_body_is_reported_after_bounded_get_retry(self) -> None:
        errors = [
            HTTPError("http://scale.local/weight", 503, "busy", None, io.BytesIO(b"busy")),
            HTTPError("http://scale.local/weight", 503, "busy", None, io.BytesIO(b"busy")),
        ]
        client = WeightClient("http://scale.local", retry_delay_s=0, max_get_retries=1)
        with patch("smartnutri_weight.client.urlopen", side_effect=errors) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "HTTP 503: busy"):
                client.get_weight()
        self.assertEqual(urlopen.call_count, 2)

    def test_malformed_json_response_is_rejected(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"not-json"
        client = WeightClient("http://scale.local", max_get_retries=0)
        with patch("smartnutri_weight.client.urlopen", return_value=response):
            with self.assertRaises(json.JSONDecodeError):
                client.get_weight()

    def test_stable_read_gate_rejects_mock_and_hardware_error_states(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            WeightClient(mock_weight_g=75.0).get_stable_weight(timeout_s=0.1)

        disconnected = WeightReading.from_payload(
            hardware_error_payload("sensor_disconnected", sensor_connected=False)
        )
        disconnected_client = WeightClient("http://scale.local")
        with patch.object(disconnected_client, "get_weight", return_value=disconnected):
            with self.assertRaisesRegex(RuntimeError, "sensor_disconnected"):
                disconnected_client.get_stable_weight(timeout_s=0.1)

        sensor_error = WeightReading.from_payload(hardware_error_payload("sensor_error"))
        error_client = WeightClient("http://scale.local")
        with patch.object(error_client, "get_weight", return_value=sensor_error):
            with self.assertRaisesRegex(RuntimeError, "sensor_error"):
                error_client.get_stable_weight(timeout_s=0.1)

        overload = WeightReading.from_payload(hardware_error_payload("overload"))
        overload_client = WeightClient("http://scale.local")
        with patch.object(overload_client, "get_weight", return_value=overload):
            with self.assertRaisesRegex(RuntimeError, "overload"):
                overload_client.get_stable_weight(timeout_s=0.1)

    def test_operation_fields_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "operation_ok"):
            WeightReading.from_payload(
                mock_weight_payload(
                    operation="tare",
                    operation_status="pending",
                    operation_ok=True,
                )
            )

    def test_mock_tare_stays_explicitly_simulated(self) -> None:
        reading = WeightClient(mock_weight_g=250.0).tare(confirmation="EMPTY")

        self.assertEqual(reading.weight_g, 0.0)
        self.assertEqual(reading.source, "mock")
        self.assertTrue(reading.simulated)

    def test_calibration_requires_an_explicit_confirmation_token(self) -> None:
        client = WeightClient(base_url="http://127.0.0.1")

        with self.assertRaises(PermissionError):
            client.calibrate(500.0)

    def test_device_operations_fail_closed_without_device_token(self) -> None:
        client = WeightClient("http://scale.local")
        with patch("smartnutri_weight.client.urlopen") as urlopen:
            with self.assertRaisesRegex(PermissionError, "device token"):
                client.tare(confirmation="EMPTY")
            with self.assertRaisesRegex(PermissionError, "device token"):
                client.calibrate(500.0, confirmation="CALIBRATE")
        urlopen.assert_not_called()

    def test_tare_requires_specific_empty_confirmation(self) -> None:
        client = WeightClient("http://scale.local", device_token="test-device-token")
        with self.assertRaisesRegex(PermissionError, "EMPTY"):
            client.tare()

    def test_calibration_rejects_out_of_range_known_weights(self) -> None:
        client = WeightClient(base_url="http://127.0.0.1")

        with self.assertRaises(ValueError):
            client.calibrate(0.0, confirmation="CALIBRATE")
        with self.assertRaises(ValueError):
            client.calibrate(5000.1, confirmation="CALIBRATE")

    def test_mock_mode_cannot_calibrate_hardware(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "mock mode"):
            WeightClient(mock_weight_g=100.0).calibrate(500.0, confirmation="CALIBRATE")

    def test_stability_window_uses_only_explicitly_simulated_readings(self) -> None:
        window = StabilityWindow(window_size=5, threshold_g=0.8)
        simulated_values = (100.0, 100.3, 99.9, 100.2, 100.1)

        for value in simulated_values:
            reading = WeightClient(mock_weight_g=value).get_weight()
            self.assertEqual(reading.source, "mock")
            self.assertTrue(reading.simulated)
            snapshot = window.add(reading.weight_g)

        self.assertTrue(snapshot.stable)
        self.assertAlmostEqual(snapshot.stability_range_g, 0.4, places=6)

    def test_stability_window_rejects_a_simulated_spike(self) -> None:
        window = StabilityWindow(window_size=5, threshold_g=0.8)
        simulated_values = (100.0, 100.1, 100.2, 103.0, 100.1)

        for value in simulated_values:
            reading = WeightClient(mock_weight_g=value).get_weight()
            self.assertTrue(reading.simulated)
            snapshot = window.add(reading.weight_g)

        self.assertFalse(snapshot.stable)

    def test_weight_delta_is_signed_and_uses_simulated_readings(self) -> None:
        previous = WeightClient(mock_weight_g=125.0).get_weight()
        current = WeightClient(mock_weight_g=131.5).get_weight()
        self.assertTrue(previous.simulated and current.simulated)

        self.assertAlmostEqual(weight_delta_g(current.weight_g, previous.weight_g), 6.5)
        self.assertAlmostEqual(weight_delta_g(previous.weight_g, current.weight_g), -6.5)


if __name__ == "__main__":
    unittest.main()
