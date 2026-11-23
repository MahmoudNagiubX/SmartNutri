"""Small standard-library client for the independent ESP32 weight API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


_REQUIRED_FIELDS = (
    "ok",
    "source",
    "simulated",
    "sensor_connected",
    "tared",
    "calibrated",
    "stable",
    "status",
    "unit",
    "weight_g",
    "weight_delta_g",
    "stability_range_g",
    "stability_threshold_g",
    "sample_count",
    "window_size",
    "timestamp_ms",
)
_REAL_STATUSES = {
    "sensor_disconnected",
    "sensor_error",
    "overload",
    "not_tared",
    "not_calibrated",
    "taring",
    "calibrating",
    "warming_up",
    "unstable",
    "stable",
}


def _boolean(payload: dict[str, Any], key: str) -> bool:
    value = payload[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _number(payload: dict[str, Any], key: str, *, optional: bool = False) -> float | None:
    value = payload[key]
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{key} must be a number" + (" or null" if optional else ""))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class WeightReading:
    """Validated API reading. ``simulated`` must be checked before use."""

    ok: bool
    source: str
    simulated: bool
    sensor_connected: bool
    tared: bool
    calibrated: bool
    stable: bool
    status: str
    unit: str
    weight_g: float | None
    weight_delta_g: float | None
    stability_range_g: float | None
    stability_threshold_g: float
    sample_count: int
    window_size: int
    timestamp_ms: int
    operation: str = "none"
    operation_status: str = "idle"
    operation_ok: bool | None = None
    operation_message: str = "idle"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WeightReading":
        if not isinstance(payload, dict):
            raise ValueError("weight API response must be a JSON object")
        missing = [field for field in _REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"weight API response is missing fields: {', '.join(missing)}")

        source = payload["source"]
        if not isinstance(source, str) or source not in {"hx711", "mock"}:
            raise ValueError("source must be 'hx711' or 'mock'")
        simulated = _boolean(payload, "simulated")
        if (source == "mock") != simulated:
            raise ValueError("source and simulated flag disagree; mock readings must be marked simulated")

        status = payload["status"]
        if not isinstance(status, str):
            raise ValueError("status must be a string")
        if source == "mock":
            if status != "mock":
                raise ValueError("simulated readings must use status 'mock'")
        elif status not in _REAL_STATUSES:
            raise ValueError("unknown HX711 status")

        unit = payload["unit"]
        if unit != "g":
            raise ValueError("unit must be 'g'")
        ok = _boolean(payload, "ok")
        sensor_connected = _boolean(payload, "sensor_connected")
        tared = _boolean(payload, "tared")
        calibrated = _boolean(payload, "calibrated")
        stable = _boolean(payload, "stable")
        weight_g = _number(payload, "weight_g", optional=True)
        weight_delta_g = _number(payload, "weight_delta_g", optional=True)
        stability_range_g = _number(payload, "stability_range_g", optional=True)
        threshold = _number(payload, "stability_threshold_g")
        assert threshold is not None
        if threshold < 0:
            raise ValueError("stability_threshold_g must be non-negative")

        sample_count = _integer(payload, "sample_count")
        window_size = _integer(payload, "window_size")
        if window_size < 1 or sample_count > window_size:
            raise ValueError("sample_count/window_size are inconsistent")

        operation = payload.get("operation", "none")
        operation_ok = payload.get("operation_ok")
        operation_status = payload.get("operation_status")
        if operation_status is None:
            if operation == "none":
                operation_status = "idle"
            elif operation_ok is True:
                operation_status = "succeeded"
            elif operation_ok is False:
                operation_status = "failed"
            else:
                operation_status = "pending"
        operation_message = payload.get("operation_message", payload.get("message", "idle"))
        if not isinstance(operation, str) or operation not in {"none", "tare", "calibrate"}:
            raise ValueError("operation must be none, tare, or calibrate")
        if not isinstance(operation_status, str) or operation_status not in {
            "idle",
            "pending",
            "succeeded",
            "failed",
        }:
            raise ValueError("unknown operation_status")
        if operation_ok is not None and type(operation_ok) is not bool:
            raise ValueError("operation_ok must be a boolean or null")
        if not isinstance(operation_message, str):
            raise ValueError("operation_message must be a string")
        if operation_status == "idle":
            if operation != "none" or operation_ok is not None:
                raise ValueError("idle operation fields are inconsistent")
        elif operation == "none":
            raise ValueError("an active or completed operation must have a name")
        elif operation_status in {"pending", "succeeded"} and operation_ok is not (
            True if operation_status == "succeeded" else None
        ):
            raise ValueError("operation_ok conflicts with operation_status")
        elif operation_status == "failed" and operation_ok is not False:
            raise ValueError("failed operation must set operation_ok=false")

        if source == "hx711":
            if stable != (status == "stable"):
                raise ValueError("stable flag and status disagree")
            if not sensor_connected:
                if status != "sensor_disconnected" or ok or weight_g is not None:
                    raise ValueError("disconnected sensor payload contains an invalid state or weight")
            elif status == "sensor_disconnected":
                raise ValueError("connected sensor cannot report sensor_disconnected")

            no_weight_statuses = {
                "sensor_disconnected",
                "sensor_error",
                "overload",
                "not_tared",
                "not_calibrated",
                "taring",
                "calibrating",
            }
            if status in no_weight_statuses and any(
                value is not None for value in (weight_g, weight_delta_g, stability_range_g)
            ):
                raise ValueError(f"{status} payload must not expose weight or stability values")
            if status in {
                "sensor_disconnected",
                "sensor_error",
                "overload",
                "not_tared",
                "not_calibrated",
                "taring",
                "calibrating",
            } and ok:
                raise ValueError(f"{status} payload cannot be marked ok")
            if status in {"warming_up", "unstable", "stable"} and not ok:
                raise ValueError(f"{status} payload requires a ready scale")
            if status == "not_tared" and tared:
                raise ValueError("not_tared status conflicts with tared flag")
            if status == "not_calibrated" and calibrated:
                raise ValueError("not_calibrated status conflicts with calibrated flag")
            if status == "taring" and (operation_status != "pending" or operation != "tare"):
                raise ValueError("taring status requires a pending tare operation")
            if status == "calibrating" and (operation_status != "pending" or operation != "calibrate"):
                raise ValueError("tare/calibration status requires a pending operation")
            if status == "stable":
                if not (ok and sensor_connected and tared and calibrated and stable):
                    raise ValueError("stable status requires a ready, connected scale")
                if sample_count != window_size or weight_g is None or stability_range_g is None:
                    raise ValueError("stable status requires a full window and weight range")
                if stability_range_g > threshold:
                    raise ValueError("stable status exceeds the configured threshold")
            if status == "unstable" and (sample_count != window_size or weight_g is None):
                raise ValueError("unstable status requires a full window and a weight")

        return cls(
            ok=ok,
            source=source,
            simulated=simulated,
            sensor_connected=sensor_connected,
            tared=tared,
            calibrated=calibrated,
            stable=stable,
            status=status,
            unit=unit,
            weight_g=weight_g,
            weight_delta_g=weight_delta_g,
            stability_range_g=stability_range_g,
            stability_threshold_g=threshold,
            sample_count=sample_count,
            window_size=window_size,
            timestamp_ms=_integer(payload, "timestamp_ms"),
            operation=operation,
            operation_status=operation_status,
            operation_ok=operation_ok,
            operation_message=operation_message,
        )


class WeightClient:
    """HTTP client with an explicit local mock mode.

    Passing ``mock_weight_g`` avoids all networking. Mock responses always use
    ``source='mock'`` and ``simulated=true`` so they cannot be mistaken for an
    HX711 measurement.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_s: float = 3.0,
        mock_weight_g: float | None = None,
        operation_timeout_s: float = 10.0,
        max_get_retries: int = 1,
        retry_delay_s: float = 0.1,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not math.isfinite(operation_timeout_s) or operation_timeout_s <= 0:
            raise ValueError("operation_timeout_s must be finite and positive")
        if isinstance(max_get_retries, bool) or not isinstance(max_get_retries, int) or max_get_retries < 0:
            raise ValueError("max_get_retries must be a non-negative integer")
        if not math.isfinite(retry_delay_s) or retry_delay_s < 0:
            raise ValueError("retry_delay_s must be finite and non-negative")
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_s = float(timeout_s)
        self.operation_timeout_s = float(operation_timeout_s)
        self.max_get_retries = max_get_retries
        self.retry_delay_s = float(retry_delay_s)
        self.mock_weight_g = self._finite_weight(mock_weight_g) if mock_weight_g is not None else None
        self._mock_last_weight_g: float | None = None
        if self.mock_weight_g is None and not self.base_url:
            raise ValueError("base_url is required unless mock_weight_g is provided")

    @staticmethod
    def _finite_weight(weight_g: float | None) -> float:
        if weight_g is None or isinstance(weight_g, bool):
            raise ValueError("weight_g must be a finite number")
        value = float(weight_g)
        if not math.isfinite(value):
            raise ValueError("weight_g must be a finite number")
        return value

    def set_mock_weight(self, weight_g: float) -> WeightReading:
        if self.mock_weight_g is None:
            raise RuntimeError("set_mock_weight is available only in mock mode")
        self.mock_weight_g = self._finite_weight(weight_g)
        return self.get_weight()

    def get_weight(self) -> WeightReading:
        if self.mock_weight_g is not None:
            return self._mock_reading()
        return WeightReading.from_payload(self._request("GET", "/weight"))

    def get_stable_weight(
        self,
        *,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.2,
    ) -> WeightReading:
        """Wait for a stable, calibrated real reading or fail clearly.

        Mock readings are deliberately rejected because they cannot authorize
        recording a real food mass.
        """

        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be finite and positive")
        deadline = time.monotonic() + timeout_s
        while True:
            reading = self.get_weight()
            if reading.simulated or reading.source != "hx711":
                raise RuntimeError("a simulated reading cannot be accepted as real food weight")
            if reading.status in {
                "sensor_disconnected",
                "sensor_error",
                "overload",
                "not_tared",
                "not_calibrated",
            }:
                raise RuntimeError(f"scale is not ready for recording: {reading.status}")
            if reading.status == "stable" and reading.stable and reading.weight_g is not None:
                return reading
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("scale did not produce a stable reading before timeout")
            time.sleep(min(poll_interval_s, remaining))

    def tare(self) -> WeightReading:
        if self.mock_weight_g is not None:
            self.mock_weight_g = 0.0
            self._mock_last_weight_g = None
            return self._mock_reading()
        payload = self._request("POST", "/tare", form={})
        return self._await_operation(payload, "tare")

    def calibrate(self, known_weight_g: float, *, confirmation: str | None = None) -> WeightReading:
        known_weight = self._finite_weight(known_weight_g)
        if not 0 < known_weight <= 5000:
            raise ValueError("known_weight_g must be greater than 0 and at most 5000")
        if confirmation != "CALIBRATE":
            raise PermissionError("calibration requires confirmation='CALIBRATE'")
        if self.mock_weight_g is not None:
            raise RuntimeError("calibration is unavailable in mock mode")
        payload = self._request(
            "POST",
            "/calibrate",
            form={"known_weight_g": f"{known_weight:g}", "confirm": confirmation},
        )
        return self._await_operation(payload, "calibrate")

    def _await_operation(self, payload: dict[str, Any], operation: str) -> WeightReading:
        # Older firmware returned a completed operation directly; keep that
        # response form readable while the current firmware completes samples
        # asynchronously and is polled through GET /weight.
        if payload.get("operation_status") is None:
            if payload.get("operation_ok") is True:
                return self._hardware_operation_result(payload, operation)
            raise RuntimeError(f"scale {operation} failed: {payload.get('message', 'unknown error')}")

        deadline = time.monotonic() + self.operation_timeout_s
        current = payload
        while True:
            state = current.get("operation_status")
            if current.get("operation") != operation:
                raise RuntimeError(f"scale is reporting a different operation: {current.get('operation')}")
            if state == "succeeded" and current.get("operation_ok") is True:
                return self._hardware_operation_result(current, operation)
            if state == "failed" or current.get("operation_ok") is False:
                message = current.get("operation_message", current.get("message", "unknown error"))
                raise RuntimeError(f"scale {operation} failed: {message}")
            if state != "pending":
                raise ValueError("scale operation response has an invalid status")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"scale {operation} operation did not finish before timeout")
            time.sleep(min(self.retry_delay_s or 0.05, remaining))
            current = self._request("GET", "/weight")

    @staticmethod
    def _hardware_operation_result(payload: dict[str, Any], operation: str) -> WeightReading:
        reading = WeightReading.from_payload(payload)
        if reading.source != "hx711" or reading.simulated:
            raise RuntimeError(
                f"scale {operation} response is simulated; it cannot confirm a hardware operation"
            )
        return reading

    def _mock_reading(self) -> WeightReading:
        assert self.mock_weight_g is not None
        delta = (
            None
            if self._mock_last_weight_g is None
            else self.mock_weight_g - self._mock_last_weight_g
        )
        payload = {
            "ok": True,
            "source": "mock",
            "simulated": True,
            "sensor_connected": False,
            "tared": False,
            "calibrated": False,
            "stable": False,
            "status": "mock",
            "unit": "g",
            "weight_g": self.mock_weight_g,
            "weight_delta_g": delta,
            "stability_range_g": None,
            "stability_threshold_g": 2.0,
            "sample_count": 0,
            "window_size": 10,
            "timestamp_ms": 0,
            "operation": "none",
            "operation_status": "idle",
            "operation_ok": None,
            "operation_message": "idle",
        }
        self._mock_last_weight_g = self.mock_weight_g
        return WeightReading.from_payload(payload)

    def _request(self, method: str, path: str, *, form: dict[str, str] | None = None) -> dict[str, Any]:
        data = urlencode(form).encode("ascii") if form is not None else None
        headers = {"Accept": "application/json"}
        if form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        attempts = self.max_get_retries + 1 if method.upper() == "GET" else 1
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if method.upper() == "GET" and 500 <= exc.code < 600 and attempt + 1 < attempts:
                    time.sleep(self.retry_delay_s * (attempt + 1))
                    continue
                raise RuntimeError(f"weight API returned HTTP {exc.code}: {details}") from exc
            except (TimeoutError, URLError, OSError):
                if attempt + 1 >= attempts:
                    raise
                time.sleep(self.retry_delay_s * (attempt + 1))
        if not isinstance(payload, dict):
            raise ValueError("weight API response must be a JSON object")
        return payload
