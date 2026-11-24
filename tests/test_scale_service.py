import asyncio
import tempfile
import unittest
from pathlib import Path

from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.scale_service import MockScaleClient, ScaleMonitoringService
from smartnutri_backend.schemas import ScanState
from smartnutri_backend.settings import Settings


class RecordingWorkflow:
    def __init__(self, repository):
        self.repository = repository
        self.started = []
        self.cancelled = []

    async def process(self, *, scan_id, observation):
        self.started.append((scan_id, observation.weight_g, observation.simulated))
        self.repository.update_scan(scan_id, state=ScanState.COMPLETED.value)

    def cancel(self, scan_id, *, reason="plate_removed_during_scan"):
        self.cancelled.append(scan_id)

    def finish(self, scan_id):
        return None


class ScaleMonitoringServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = SQLiteRepository(Path(self.temp_dir.name) / "scale.sqlite3")
        self.settings = Settings(
            data_dir=Path(self.temp_dir.name),
            scale_mode="mock",
            auto_scan_enabled=True,
            scale_poll_interval_s=0.01,
            placement_threshold_g=20,
            removal_threshold_g=10,
            stable_confirmations=2,
            removal_confirmations=3,
            post_stability_delay_s=0.01,
        )
        self.client = MockScaleClient()
        self.workflow = RecordingWorkflow(self.repository)
        self.service = ScaleMonitoringService(
            client=self.client,
            repository=self.repository,
            workflow=self.workflow,
            settings=self.settings,
            allow_simulated=True,
        )

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    async def test_background_monitor_triggers_once_and_rearms_after_removal(self):
        async def wait_for(predicate, timeout=1.5):
            deadline = asyncio.get_running_loop().time() + timeout
            while not predicate() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            self.assertTrue(predicate(), "condition did not become true before timeout")

        await self.service.start()
        self.client.set_weight(90)
        await wait_for(lambda: len(self.workflow.started) == 1)
        await wait_for(lambda: self.service.state == ScanState.WAITING_FOR_REMOVAL)
        self.assertEqual(self.service.state, ScanState.WAITING_FOR_REMOVAL)

        await asyncio.sleep(0.04)
        self.assertEqual(len(self.workflow.started), 1)
        self.client.set_weight(0)
        await wait_for(lambda: self.service.state == ScanState.IDLE)
        self.assertEqual(
            self.service.state,
            ScanState.IDLE,
        )

        self.client.set_weight(120)
        await wait_for(lambda: len(self.workflow.started) == 2)
        self.assertEqual(len(self.workflow.started), 2)
        self.assertTrue(all(item[2] for item in self.workflow.started))
        await self.service.stop()

    async def test_restart_marks_incomplete_scan_and_waits_for_removal(self):
        scan = self.repository.begin_scan_for_placement(
            placement_key="persisted-placement",
            source="mock",
            simulated=True,
            state=ScanState.ANALYZING.value,
        )
        restarted = ScaleMonitoringService(
            client=MockScaleClient(100),
            repository=self.repository,
            workflow=self.workflow,
            settings=self.settings,
            allow_simulated=True,
        )
        persisted = self.repository.get_scan(scan["id"])
        self.assertEqual(restarted.state, ScanState.WAITING_FOR_REMOVAL)
        self.assertEqual(persisted["state"], ScanState.NEEDS_REVIEW.value)
        self.assertEqual(persisted["error_code"], "backend_restarted_during_scan")

    async def test_live_tared_plate_must_be_removed_before_platform_tare(self):
        self.repository.set_tare_configuration(
            mode="live_plate",
            live_tare_active=True,
            empty_plate_weight_g=300,
        )
        service = ScaleMonitoringService(
            client=self.client,
            repository=self.repository,
            workflow=self.workflow,
            settings=self.settings,
            allow_simulated=True,
        )
        self.assertEqual(service.monitor.live_plate_weight_g, 300)
        self.client.set_weight(0)
        with self.assertRaisesRegex(RuntimeError, "remove the live-tared empty plate"):
            service.configure_tare("platform_empty", confirmation="EMPTY_PLATFORM")

        self.client.set_weight(-300)
        configuration = service.configure_tare("platform_empty", confirmation="EMPTY_PLATFORM")
        self.assertEqual(configuration["mode"], "saved_plate")
        self.assertFalse(configuration["live_tare_active"])


if __name__ == "__main__":
    unittest.main()
