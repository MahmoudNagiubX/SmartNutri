import unittest
from itertools import count

from smartnutri_backend.scale_monitor import MonitorEvent, ScaleMonitor
from smartnutri_backend.schemas import ScanState, ScaleObservation

_measurement_sequence = count(1)


def reading(weight, *, stable=True, status=None, source="hx711", simulated=False,
            session_id="unit-test-session", measurement_id=None):
    resolved_status = status or ("stable" if stable else "unstable")
    connected = resolved_status != "sensor_disconnected"
    ok = resolved_status not in {"sensor_disconnected", "sensor_error", "overload"}
    return ScaleObservation(
        ok=ok,
        source=source,
        simulated=simulated,
        sensor_connected=connected,
        tared=True,
        calibrated=True,
        stable=stable,
        status=resolved_status,
        unit="g",
        device_session_id=session_id,
        measurement_id=measurement_id or f"unit-test-{next(_measurement_sequence)}",
        weight_g=weight if ok else None,
    )


class ScaleMonitorTests(unittest.TestCase):
    def monitor(self, **overrides):
        config = {
            "placement_threshold_g": 20.0,
            "removal_threshold_g": 10.0,
            "stable_confirmations": 2,
            "removal_confirmations": 3,
            "post_stability_delay_s": 0.5,
        }
        config.update(overrides)
        return ScaleMonitor(**config)

    def test_short_unstable_spike_never_triggers_capture(self):
        monitor = self.monitor()
        monitor.observe(reading(90, stable=False), now=0.0)
        self.assertEqual(monitor.state, ScanState.WEIGHT_DETECTED)
        monitor.observe(reading(0), now=0.1)
        monitor.observe(reading(0), now=0.2)
        self.assertEqual(monitor.state, ScanState.IDLE)
        self.assertIsNone(monitor.active_placement_key)

    def test_saved_plate_mass_is_excluded_from_food_placement_threshold(self):
        monitor = self.monitor(
            saved_plate_weight_g=300,
            allow_simulated=True,
            post_stability_delay_s=0,
        )
        self.assertEqual(monitor.placement_threshold_g, 320)
        for index in range(4):
            self.assertIsNone(monitor.observe(reading(300), now=index / 10))
        self.assertEqual(monitor.state, ScanState.IDLE)
        self.assertIsNone(monitor.active_placement_key)

        monitor.observe(reading(750, stable=False), now=0.5)
        self.assertIsNone(monitor.observe(reading(750), now=0.6))
        event = monitor.observe(reading(750), now=0.7)
        self.assertEqual(event.kind, "capture")
        self.assertEqual(event.observation.weight_g, 750)
        for index in range(4):
            self.assertIsNone(monitor.observe(reading(750), now=0.8 + index / 10))

    def test_requires_consecutive_stability_and_settling_delay(self):
        monitor = self.monitor()
        monitor.observe(reading(750, stable=False), now=0.0)
        self.assertIsNone(monitor.observe(reading(750), now=0.1))
        self.assertEqual(monitor.state, ScanState.WAITING_FOR_STABILITY)
        self.assertIsNone(monitor.observe(reading(750), now=0.3))
        event = monitor.observe(reading(751), now=0.61)
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "capture")
        self.assertEqual(event.observation.weight_g, 751)
        self.assertEqual(monitor.state, ScanState.CAPTURING)

    def test_held_plate_does_not_retrigger_and_stable_removal_rearms(self):
        monitor = self.monitor(post_stability_delay_s=0)
        monitor.observe(reading(750, stable=False), now=0.0)
        monitor.observe(reading(750), now=0.1)
        trigger = monitor.observe(reading(750), now=0.2)
        self.assertEqual(trigger.kind, "capture")
        monitor.set_state(ScanState.COMPLETED)
        for step in range(5):
            self.assertIsNone(monitor.observe(reading(750), now=0.3 + step / 10))
        self.assertEqual(monitor.state, ScanState.COMPLETED)

        monitor.observe(None, now=1.0)  # disconnect does not prove plate removal
        self.assertEqual(monitor.state, ScanState.COMPLETED)
        self.assertIsNone(monitor.observe(reading(0), now=1.1))
        self.assertIsNone(monitor.observe(reading(0), now=1.2))
        removed = monitor.observe(reading(0), now=1.3)
        self.assertIsNotNone(removed)
        self.assertEqual(removed.kind, "removed")
        self.assertEqual(monitor.state, ScanState.IDLE)
        self.assertIsNone(monitor.active_placement_key)

    def test_sensor_error_and_overload_do_not_trigger_or_rearm(self):
        monitor = self.monitor()
        monitor.observe(reading(700, stable=False), now=0)
        monitor.observe(reading(None, status="overload", stable=False), now=0.1)
        self.assertEqual(monitor.state, ScanState.IDLE)
        self.assertIsNone(monitor.observe(reading(750), now=0.2))
        self.assertIsNone(monitor.observe(reading(750), now=0.3))
        self.assertEqual(monitor.state, ScanState.WAITING_FOR_STABILITY)

    def test_mock_source_is_accepted_only_when_explicitly_enabled(self):
        disabled = self.monitor()
        mock = reading(750, source="mock", simulated=True)
        disabled.observe(mock, now=0)
        self.assertEqual(disabled.state, ScanState.IDLE)

        enabled = self.monitor(allow_simulated=True, post_stability_delay_s=0)
        enabled.observe(mock, now=0)
        enabled.observe(mock, now=0.1)
        event = enabled.observe(mock, now=0.2)
        self.assertEqual(event.observation.source, "mock")
        self.assertTrue(event.observation.simulated)

    def test_live_tare_removal_accepts_negative_stable_plate_weight(self):
        monitor = self.monitor(
            post_stability_delay_s=0,
            tare_mode="live_plate",
            live_plate_weight_g=300,
        )
        monitor.observe(reading(300, stable=False), now=0)
        monitor.observe(reading(300), now=0.1)
        trigger = monitor.observe(reading(300), now=0.2)
        self.assertEqual(trigger.kind, "capture")
        monitor.set_state(ScanState.WAITING_FOR_REMOVAL)
        monitor.observe(reading(200), now=0.3)
        for step in range(3):
            self.assertIsNone(monitor.observe(reading(0), now=0.4 + step / 10))
        self.assertEqual(monitor.state, ScanState.WAITING_FOR_REMOVAL)
        monitor.observe(reading(-300), now=0.4)
        monitor.observe(reading(-300), now=0.5)
        removed = monitor.observe(reading(-300), now=0.6)
        self.assertEqual(removed.kind, "removed")
        self.assertEqual(monitor.state, ScanState.IDLE)

    def test_pending_live_tare_suppresses_empty_plate_scan_until_tare_is_confirmed(self):
        monitor = self.monitor(live_tare_pending=True, post_stability_delay_s=0)
        for index in range(5):
            self.assertIsNone(monitor.observe(reading(300), now=index / 10))
        self.assertIsNone(monitor.active_placement_key)
        self.assertEqual(monitor.state, ScanState.IDLE)

        monitor.set_live_tare(300)
        monitor.observe(reading(450, stable=False), now=0.6)
        monitor.observe(reading(450), now=0.7)
        event = monitor.observe(reading(450), now=0.8)
        self.assertEqual(event.kind, "capture")
        self.assertEqual(event.observation.weight_g, 450)

    def test_restart_latch_starts_waiting_for_removal(self):
        monitor = ScaleMonitor(
            placement_threshold_g=20,
            removal_threshold_g=10,
            stable_confirmations=2,
            removal_confirmations=2,
            post_stability_delay_s=0,
            restored_latch={"latched": True, "placement_key": "persisted-placement"},
        )
        self.assertEqual(monitor.state, ScanState.WAITING_FOR_REMOVAL)
        self.assertEqual(monitor.active_placement_key, "persisted-placement")
        self.assertIsNone(monitor.observe(reading(750), now=0))

    def test_active_placement_invalidates_weight_change_and_sensor_restart(self):
        monitor = self.monitor(post_stability_delay_s=0)
        monitor.observe(reading(750, stable=False), now=0)
        monitor.observe(reading(750), now=0.1)
        trigger = monitor.observe(reading(750), now=0.2)

        changed = monitor.observe(reading(760), now=0.3)
        self.assertEqual(changed.kind, "invalidated")
        self.assertEqual(changed.reason, "weight_changed_during_scan")
        self.assertEqual(monitor.snapshot_invalid_reason, "weight_changed_during_scan")
        self.assertIsNone(monitor.observe(reading(750), now=0.4))

        restarted = self.monitor(post_stability_delay_s=0)
        restarted.observe(reading(750, stable=False), now=0)
        restarted.observe(reading(750), now=0.1)
        restarted.observe(reading(750), now=0.2)
        event = restarted.observe(reading(750, session_id="new-device-session"), now=0.3)
        self.assertEqual(event.kind, "invalidated")
        self.assertEqual(event.reason, "scale_device_restarted")

    def test_active_placement_invalidates_repeated_stale_measurement(self):
        monitor = self.monitor(post_stability_delay_s=0, stale_reading_timeout_s=1.0)
        monitor.observe(reading(750, stable=False), now=0)
        monitor.observe(reading(750), now=0.1)
        baseline = reading(750)
        monitor.observe(baseline, now=0.2)

        stale = monitor.observe(
            reading(750, measurement_id=baseline.measurement_id), now=1.3
        )
        self.assertEqual(stale.kind, "invalidated")
        self.assertEqual(stale.reason, "scale_reading_stale")


if __name__ == "__main__":
    unittest.main()
