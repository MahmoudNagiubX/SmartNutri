import unittest

from smartnutri_backend.scale_monitor import MonitorEvent, ScaleMonitor
from smartnutri_backend.schemas import ScanState, ScaleObservation


def reading(weight, *, stable=True, status=None, source="hx711", simulated=False):
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


if __name__ == "__main__":
    unittest.main()
