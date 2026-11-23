from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass

from smartnutri_backend.schemas import ScanState, ScaleObservation


@dataclass(frozen=True)
class MonitorEvent:
    kind: str
    placement_key: str
    observation: ScaleObservation | None = None
    during_scan: bool = False


class ScaleMonitor:
    """Deterministic scale-placement state machine; no networking or camera I/O."""

    def __init__(
        self,
        *,
        placement_threshold_g: float,
        removal_threshold_g: float,
        stable_confirmations: int,
        removal_confirmations: int,
        post_stability_delay_s: float,
        tare_mode: str = "saved_plate",
        live_plate_weight_g: float | None = None,
        live_tare_pending: bool = False,
        allow_simulated: bool = False,
        restored_latch: dict | None = None,
    ) -> None:
        values = (placement_threshold_g, removal_threshold_g, post_stability_delay_s)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("scale monitor thresholds and delay must be finite and non-negative")
        if placement_threshold_g <= removal_threshold_g:
            raise ValueError("placement threshold must exceed removal threshold")
        if stable_confirmations < 1 or removal_confirmations < 1:
            raise ValueError("confirmation counts must be positive")
        if tare_mode not in {"saved_plate", "live_plate"}:
            raise ValueError("tare_mode must be saved_plate or live_plate")
        if tare_mode == "live_plate" and (
            live_plate_weight_g is None
            or not math.isfinite(live_plate_weight_g)
            or live_plate_weight_g <= 0
        ):
            raise ValueError("live plate mode requires its saved empty-plate tare weight")

        self.placement_threshold_g = float(placement_threshold_g)
        self.removal_threshold_g = float(removal_threshold_g)
        self.stable_confirmations = stable_confirmations
        self.removal_confirmations = removal_confirmations
        self.post_stability_delay_s = float(post_stability_delay_s)
        self.tare_mode = tare_mode
        self.live_plate_weight_g = float(live_plate_weight_g) if live_plate_weight_g is not None else None
        self.live_tare_pending = bool(live_tare_pending)
        self.allow_simulated = allow_simulated
        latch = restored_latch or {}
        self.active_placement_key = latch.get("placement_key") if latch.get("latched") else None
        self.state = (
            ScanState.WAITING_FOR_REMOVAL
            if self.active_placement_key
            else ScanState.IDLE
        )
        self.last_error: str | None = None
        self._stable_count = 0
        self._removal_count = 0
        self._stable_since: float | None = None

    @property
    def latched(self) -> bool:
        return self.active_placement_key is not None

    def set_state(self, state: ScanState | str) -> None:
        next_state = ScanState(state)
        if next_state in {ScanState.IDLE, ScanState.WEIGHT_DETECTED, ScanState.WAITING_FOR_STABILITY}:
            raise ValueError("candidate and idle states are owned by the monitor")
        if not self.latched:
            raise RuntimeError("a scan state cannot be set without an active placement")
        self.state = next_state

    def prepare_live_tare(self) -> None:
        if self.latched:
            raise RuntimeError("remove the current plate before preparing live tare")
        self.live_tare_pending = True
        self._reset_candidate(preserve_candidate=False)

    def set_live_tare(self, plate_weight_g: float) -> None:
        if not math.isfinite(plate_weight_g) or plate_weight_g <= 0:
            raise ValueError("live plate tare weight must be finite and positive")
        if self.latched:
            raise RuntimeError("remove the current plate before changing tare mode")
        self.tare_mode = "live_plate"
        self.live_plate_weight_g = float(plate_weight_g)
        self.live_tare_pending = False
        self._reset_candidate(preserve_candidate=False)

    def set_platform_tare(self) -> None:
        if self.latched:
            raise RuntimeError("remove the current plate before changing tare mode")
        self.tare_mode = "saved_plate"
        self.live_plate_weight_g = None
        self.live_tare_pending = False
        self._reset_candidate(preserve_candidate=False)

    def observe(
        self,
        observation: ScaleObservation | None,
        *,
        now: float | None = None,
    ) -> MonitorEvent | None:
        current_time = time.monotonic() if now is None else float(now)
        if self.latched:
            return self._observe_latched(observation)
        if self.live_tare_pending:
            if observation is not None and observation.status in {
                "sensor_disconnected", "sensor_error", "overload"
            }:
                self.last_error = observation.status
            elif observation is not None:
                self.last_error = None
            return None
        if observation is None:
            self._reset_candidate(preserve_candidate=True)
            return None
        if observation.status in {"sensor_disconnected", "sensor_error", "overload"}:
            self.last_error = observation.status
            self._reset_candidate(preserve_candidate=False)
            return None
        if observation.source == "mock" and not self.allow_simulated:
            return None
        if observation.source != "hx711" and observation.source != "mock":
            return None
        self.last_error = None

        weight = observation.weight_g
        if weight is None:
            self._reset_candidate(preserve_candidate=True)
            return None
        if self.state == ScanState.IDLE:
            if weight >= self.placement_threshold_g:
                self.state = ScanState.WEIGHT_DETECTED
                self._stable_count = 0
                self._stable_since = None
            return None

        if weight < self.placement_threshold_g:
            self._reset_candidate(preserve_candidate=False)
            return None
        if not observation.usable_stable:
            self._stable_count = 0
            self._stable_since = None
            self.state = ScanState.WEIGHT_DETECTED
            return None

        if self._stable_count == 0:
            self._stable_since = current_time
        self._stable_count += 1
        self.state = ScanState.WAITING_FOR_STABILITY
        if (
            self._stable_count >= self.stable_confirmations
            and self._stable_since is not None
            and current_time - self._stable_since >= self.post_stability_delay_s
        ):
            placement_key = str(uuid.uuid4())
            self.active_placement_key = placement_key
            self._removal_count = 0
            self.state = ScanState.CAPTURING
            return MonitorEvent("capture", placement_key, observation)
        return None

    def _observe_latched(self, observation: ScaleObservation | None) -> MonitorEvent | None:
        if observation is None or not observation.usable_stable:
            self._removal_count = 0
            if observation is not None and observation.status in {
                "sensor_disconnected", "sensor_error", "overload"
            }:
                self.last_error = observation.status
            return None
        self.last_error = None
        if observation.source == "mock" and not self.allow_simulated:
            self._removal_count = 0
            return None
        weight = observation.weight_g
        if weight is None:
            self._removal_count = 0
            return None
        if self.tare_mode == "live_plate":
            assert self.live_plate_weight_g is not None
            tolerance_g = min(self.removal_threshold_g, self.live_plate_weight_g * 0.05)
            removed_sample = weight <= -(self.live_plate_weight_g - tolerance_g)
        else:
            removed_sample = abs(weight) <= self.removal_threshold_g
        if not removed_sample:
            self._removal_count = 0
            return None

        self._removal_count += 1
        if self._removal_count < self.removal_confirmations:
            return None
        key = self.active_placement_key
        during_scan = self.state in {ScanState.CAPTURING, ScanState.ANALYZING}
        self.active_placement_key = None
        self.state = ScanState.IDLE
        self._removal_count = 0
        self._stable_count = 0
        self._stable_since = None
        self.last_error = None
        return MonitorEvent("removed", key or "", observation, during_scan)

    def _reset_candidate(self, *, preserve_candidate: bool) -> None:
        self._stable_count = 0
        self._stable_since = None
        if preserve_candidate and self.state in {
            ScanState.WEIGHT_DETECTED,
            ScanState.WAITING_FOR_STABILITY,
        }:
            self.state = ScanState.WEIGHT_DETECTED
        elif not preserve_candidate:
            self.state = ScanState.IDLE
