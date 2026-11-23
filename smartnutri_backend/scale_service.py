from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.scale_monitor import MonitorEvent, ScaleMonitor
from smartnutri_backend.schemas import ScaleObservation, ScanState


class MockScaleClient:
    """Explicitly simulated, stable scale source for backend demos and tests."""

    source = "mock"
    simulated = True

    def __init__(self, weight_g: float = 0.0) -> None:
        self.weight_g = float(weight_g)
        self.tared = True
        self.calibrated = True

    def get_weight(self) -> ScaleObservation:
        return ScaleObservation(
            ok=True,
            source="mock",
            simulated=True,
            sensor_connected=False,
            tared=self.tared,
            calibrated=self.calibrated,
            stable=True,
            status="stable",
            unit="g",
            weight_g=self.weight_g,
            weight_delta_g=None,
            stability_range_g=0.0,
            timestamp_ms=int(time.monotonic() * 1000),
        )

    def set_weight(self, weight_g: float) -> ScaleObservation:
        value = float(weight_g)
        if not math.isfinite(value) or abs(value) > 5000:
            raise ValueError("simulated weight must be finite and within the 5 kg load-cell range")
        self.weight_g = value
        return self.get_weight()

    def tare(self) -> ScaleObservation:
        self.weight_g = 0.0
        self.tared = True
        return self.get_weight()


class ScaleMonitoringService:
    """Polls the existing weight client and dispatches exactly one scan per placement."""

    def __init__(
        self,
        *,
        client: Any,
        repository: SQLiteRepository,
        workflow: Any,
        settings: Any,
        allow_simulated: bool = False,
    ) -> None:
        self.client = client
        self.repository = repository
        self.workflow = workflow
        self.settings = settings
        self.allow_simulated = allow_simulated
        tare = repository.get_tare_configuration()
        default_plate = repository.get_default_plate()
        restored_latch = repository.get_monitor_latch()
        self.monitor = ScaleMonitor(
            placement_threshold_g=settings.placement_threshold_g,
            removal_threshold_g=settings.removal_threshold_g,
            stable_confirmations=settings.stable_confirmations,
            removal_confirmations=settings.removal_confirmations,
            post_stability_delay_s=settings.post_stability_delay_s,
            tare_mode=tare.get("mode", "saved_plate"),
            live_plate_weight_g=(
                tare.get("empty_plate_weight_g")
                if tare.get("mode") == "live_plate" else None
            ),
            saved_plate_weight_g=(
                default_plate.get("empty_plate_weight_g") if default_plate else None
            ),
            live_tare_pending=tare.get("live_tare_pending", False),
            allow_simulated=allow_simulated,
            restored_latch=restored_latch,
        )
        if restored_latch.get("latched") and restored_latch.get("placement_key"):
            interrupted = repository.get_scan_by_placement(restored_latch["placement_key"])
            if interrupted and interrupted.get("state") in {
                ScanState.WEIGHT_DETECTED.value,
                ScanState.WAITING_FOR_STABILITY.value,
                ScanState.CAPTURING.value,
                ScanState.ANALYZING.value,
            }:
                repository.update_scan(
                    interrupted["id"],
                    state=ScanState.NEEDS_REVIEW.value,
                    result_message="Backend restarted during scan; remove the plate before starting another scan.",
                    error_code="backend_restarted_during_scan",
                )
        self.last_observation: ScaleObservation | None = None
        self.last_error: str | None = None
        self._task: asyncio.Task | None = None
        self._scan_tasks: set[asyncio.Task] = set()
        self._stop = asyncio.Event()

    def refresh_saved_plate_profile(self) -> None:
        profile = self.repository.get_default_plate()
        self.monitor.set_saved_plate_weight(
            profile.get("empty_plate_weight_g") if profile is not None else None
        )

    @property
    def state(self) -> ScanState:
        return self.monitor.state

    async def start(self) -> None:
        if self.client is not None and self.settings.auto_scan_enabled and self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._poll_loop(), name="smartnutri-scale-monitor")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._scan_tasks:
            await asyncio.gather(*self._scan_tasks, return_exceptions=True)

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = await asyncio.to_thread(self.client.get_weight)
                observation = self.to_observation(raw)
                self.last_error = None
                await self.observe(observation)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = type(exc).__name__
                await self.observe(None)
            await asyncio.sleep(self.settings.scale_poll_interval_s)

    @staticmethod
    def to_observation(reading: Any) -> ScaleObservation:
        if isinstance(reading, ScaleObservation):
            return reading
        payload = asdict(reading) if is_dataclass(reading) else reading
        fields = {
            "ok", "source", "simulated", "sensor_connected", "tared", "calibrated",
            "stable", "status", "unit", "weight_g", "weight_delta_g",
            "stability_range_g", "timestamp_ms",
        }
        return ScaleObservation.model_validate({key: payload[key] for key in fields if key in payload})

    async def observe(self, observation: ScaleObservation | None, *, now: float | None = None) -> MonitorEvent | None:
        if observation is not None:
            self.last_observation = observation
            self.last_error = (
                observation.status
                if observation.status in {"sensor_disconnected", "sensor_error", "overload"}
                else None
            )
        event = self.monitor.observe(observation, now=now)
        if event is None:
            return None
        if event.kind == "capture":
            scan = self.repository.begin_scan_for_placement(
                placement_key=event.placement_key,
                source=event.observation.source,
                simulated=event.observation.simulated,
                state=ScanState.WEIGHT_DETECTED.value,
            )
            task = asyncio.create_task(
                self._run_scan(scan["id"], event.observation),
                name=f"smartnutri-scan-{scan['id']}",
            )
            self._scan_tasks.add(task)
            task.add_done_callback(self._scan_tasks.discard)
        elif event.kind == "removed":
            self.repository.set_monitor_latched(False, None)
            scan = self.repository.get_scan_by_placement(event.placement_key)
            if event.during_scan and scan is not None:
                self.workflow.cancel(scan["id"])
        return event

    async def _run_scan(self, scan_id: str, observation: ScaleObservation) -> None:
        try:
            await self.workflow.process(scan_id=scan_id, observation=observation)
        except Exception as exc:
            try:
                self.repository.update_scan(
                    scan_id,
                    state=ScanState.NEEDS_REVIEW.value,
                    result_message=f"Scan pipeline failed ({type(exc).__name__}).",
                    error_code="scan_pipeline_error",
                )
            except Exception:
                pass
            self.last_error = type(exc).__name__
        finally:
            self.workflow.finish(scan_id)
            if self.monitor.latched:
                self.monitor.state = ScanState.WAITING_FOR_REMOVAL

    async def inject_mock_placement(self, gross_weight_g: float) -> dict[str, Any]:
        if not self.allow_simulated or not isinstance(self.client, MockScaleClient):
            raise RuntimeError("mock placement is available only when the backend scale mode is mock")
        observation = self.client.set_weight(gross_weight_g)
        start = time.monotonic()
        await self.observe(observation, now=start)
        await self.observe(observation, now=start + 0.05)
        event = await self.observe(
            observation,
            now=start + max(0.1, self.settings.post_stability_delay_s + 0.1),
        )
        if event is None:
            raise RuntimeError("simulated placement did not produce a stable scan trigger")
        if self._scan_tasks:
            await asyncio.gather(*tuple(self._scan_tasks), return_exceptions=True)
        scan = self.repository.get_scan_by_placement(event.placement_key)
        if scan is None:
            raise RuntimeError("simulated scan record was not persisted")
        meal = self.repository.get_meal_by_scan(scan["id"])
        return {"scan": scan, "meal": meal, "simulated": True}

    async def inject_mock_removal(self) -> None:
        if not self.allow_simulated or not isinstance(self.client, MockScaleClient):
            raise RuntimeError("mock removal is available only when the backend scale mode is mock")
        removal_weight_g = (
            -self.monitor.live_plate_weight_g
            if self.monitor.tare_mode == "live_plate" and self.monitor.live_plate_weight_g is not None
            else 0.0
        )
        observation = self.client.set_weight(removal_weight_g)
        start = time.monotonic()
        for index in range(self.settings.removal_confirmations):
            await self.observe(observation, now=start + index * 0.1)

    def configure_tare(self, mode: str) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("scale is not configured")
        if self.monitor.latched:
            raise RuntimeError("remove the current plate before changing tare mode")
        current = self.to_observation(self.client.get_weight())
        tare_config = self.repository.get_tare_configuration()
        if mode == "prepare_empty_plate":
            if not current.usable_stable or current.weight_g is None:
                raise RuntimeError("prepare live tare on a stable, empty platform before placing the plate")
            if abs(current.weight_g) > self.settings.removal_threshold_g:
                raise RuntimeError("remove all weight, then prepare live tare before placing the empty plate")
            self.repository.set_tare_configuration(
                mode="saved_plate",
                live_tare_active=False,
                live_tare_pending=True,
                empty_plate_weight_g=None,
            )
            self.monitor.prepare_live_tare()
            return self.repository.get_tare_configuration()
        if mode == "empty_plate":
            if not current.usable_stable or current.weight_g is None:
                raise RuntimeError("live plate tare requires a stable calibrated scale reading")
            if current.weight_g <= 0 or current.weight_g > 5000:
                raise RuntimeError("live plate tare requires a stable positive empty plate weight")
            empty_plate_weight_g = current.weight_g
            tare_mode = "live_plate"
        elif mode == "platform_empty":
            if current.status == "not_tared":
                if not current.sensor_connected:
                    raise RuntimeError("platform tare requires a connected HX711 sensor")
            elif not current.usable_stable or current.weight_g is None:
                raise RuntimeError("platform tare requires a stable reading or an untared connected sensor")
            elif tare_config.get("mode") == "live_plate" and tare_config.get("live_tare_active"):
                empty_plate_weight = tare_config.get("empty_plate_weight_g")
                if empty_plate_weight is None or abs(current.weight_g + float(empty_plate_weight)) > max(
                    5.0, self.settings.removal_threshold_g
                ):
                    raise RuntimeError("remove the live-tared empty plate before taring the platform")
            elif abs(current.weight_g) > self.settings.removal_threshold_g:
                raise RuntimeError("platform tare requires the empty platform")
            empty_plate_weight_g = None
            tare_mode = "saved_plate"
        else:
            raise ValueError("tare mode must be platform_empty or empty_plate")
        result = self.client.tare()
        if isinstance(result, ScaleObservation):
            if result.simulated != current.simulated:
                raise RuntimeError("tare result source does not match the current scale source")
        self.repository.set_tare_configuration(
            mode=tare_mode,
            live_tare_active=tare_mode == "live_plate",
            live_tare_pending=False,
            empty_plate_weight_g=empty_plate_weight_g,
            live_tare_at=(datetime.now(timezone.utc).isoformat() if tare_mode == "live_plate" else None),
        )
        if tare_mode == "live_plate":
            assert empty_plate_weight_g is not None
            self.monitor.set_live_tare(empty_plate_weight_g)
        else:
            self.monitor.set_platform_tare()
        return self.repository.get_tare_configuration()
