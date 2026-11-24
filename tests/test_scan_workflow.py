import asyncio
import threading
import tempfile
import time
import unittest
from pathlib import Path

from smartnutri_backend.agent import NutritionDecisionAgent
from smartnutri_backend.camera import MockCameraService
from smartnutri_backend.nutrition import NutritionService
from smartnutri_backend.portion import ExperimentalVisualProportionEstimator
from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.scale_service import MockScaleClient, ScaleMonitoringService
from smartnutri_backend.schemas import ScanState
from smartnutri_backend.settings import Settings
from smartnutri_backend.vision import MockSegmentationService
from smartnutri_backend.workflow import ScanWorkflow


class BlockingCamera(MockCameraService):
    def __init__(self):
        super().__init__(capture_retries=0)
        self.capture_started = threading.Event()
        self.release_capture = threading.Event()

    def capture_raw(self):
        self.capture_started.set()
        self.release_capture.wait(timeout=2)
        return super().capture_raw()


class BlockingInference:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.delegate = MockSegmentationService()

    def infer(self, capture, *, scan_id):
        self.started.set()
        self.release.wait(timeout=3)
        return self.delegate.infer(capture, scan_id=scan_id)


class ScanWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repository = SQLiteRepository(self.root / "scan.sqlite3")
        self.repository.create_plate_profile("fixture plate", 300, make_default=True)
        self.settings = Settings(
            data_dir=self.root,
            scale_mode="mock",
            auto_scan_enabled=False,
            stable_confirmations=2,
            removal_confirmations=3,
            post_stability_delay_s=0.01,
        )
        self.camera = BlockingCamera()
        self.workflow = ScanWorkflow(
            repository=self.repository,
            camera=self.camera,
            inference=MockSegmentationService(),
            portion_estimator=ExperimentalVisualProportionEstimator(),
            agent=NutritionDecisionAgent(),
            nutrition=NutritionService(self.repository),
            image_dir=self.root / "images",
        )
        self.scale = MockScaleClient()
        self.service = ScaleMonitoringService(
            client=self.scale,
            repository=self.repository,
            workflow=self.workflow,
            settings=self.settings,
            allow_simulated=True,
        )

    def tearDown(self):
        self.camera.release_capture.set()
        self.repository.close()
        self.temp_dir.cleanup()

    async def test_plate_removed_during_capture_aborts_before_saving_image_or_meal(self):
        observation = self.scale.set_weight(750)
        now = time.monotonic()
        await self.service.observe(observation, now=now)
        await self.service.observe(observation, now=now + 0.05)
        event = await self.service.observe(observation, now=now + 0.1)
        self.assertIsNotNone(event)

        captured_in_thread = await asyncio.wait_for(asyncio.to_thread(self.camera.capture_started.wait), timeout=1)
        self.assertTrue(captured_in_thread)
        removed = self.scale.set_weight(0)
        for _ in range(self.settings.removal_confirmations):
            await self.service.observe(removed)
        self.camera.release_capture.set()
        await asyncio.gather(*tuple(self.service._scan_tasks), return_exceptions=True)

        scan = self.repository.get_scan_by_placement(event.placement_key)
        self.assertEqual(scan["state"], ScanState.NEEDS_REVIEW.value)
        self.assertEqual(scan["error_code"], "plate_removed_during_scan")
        self.assertIsNone(scan["capture_path"])
        self.assertIsNone(self.repository.get_meal_by_scan(scan["id"]))

    async def test_weight_change_during_capture_rejects_weight_image_pair(self):
        observation = self.scale.set_weight(750)
        now = time.monotonic()
        await self.service.observe(observation, now=now)
        await self.service.observe(observation, now=now + 0.05)
        event = await self.service.observe(observation, now=now + 0.1)
        self.assertIsNotNone(event)
        self.assertTrue(await asyncio.wait_for(asyncio.to_thread(self.camera.capture_started.wait), timeout=1))

        changed = self.scale.set_weight(800)
        await self.service.observe(changed)
        self.camera.release_capture.set()
        await asyncio.gather(*tuple(self.service._scan_tasks), return_exceptions=True)

        scan = self.repository.get_scan_by_placement(event.placement_key)
        self.assertEqual(scan["error_code"], "weight_changed_during_scan")
        self.assertIsNone(scan["capture_path"])
        self.assertIsNone(self.repository.get_meal_by_scan(scan["id"]))

    async def test_weight_change_during_inference_rejects_analysis(self):
        self.camera.release_capture.set()
        inference = BlockingInference()
        self.workflow.inference = inference
        observation = self.scale.set_weight(750)
        now = time.monotonic()
        await self.service.observe(observation, now=now)
        await self.service.observe(observation, now=now + 0.05)
        event = await self.service.observe(observation, now=now + 0.1)
        self.assertIsNotNone(event)
        self.assertTrue(await asyncio.wait_for(asyncio.to_thread(inference.started.wait), timeout=1))

        changed = self.scale.set_weight(800)
        await self.service.observe(changed)
        inference.release.set()
        await asyncio.gather(*tuple(self.service._scan_tasks), return_exceptions=True)

        scan = self.repository.get_scan_by_placement(event.placement_key)
        self.assertEqual(scan["error_code"], "weight_changed_during_scan")
        self.assertIsNone(self.repository.get_meal_by_scan(scan["id"]))

    async def test_sensor_disconnect_during_inference_rejects_analysis(self):
        self.camera.release_capture.set()
        inference = BlockingInference()
        self.workflow.inference = inference
        observation = self.scale.set_weight(750)
        now = time.monotonic()
        await self.service.observe(observation, now=now)
        await self.service.observe(observation, now=now + 0.05)
        event = await self.service.observe(observation, now=now + 0.1)
        self.assertIsNotNone(event)
        self.assertTrue(await asyncio.wait_for(asyncio.to_thread(inference.started.wait), timeout=1))

        await self.service.observe(None)
        inference.release.set()
        await asyncio.gather(*tuple(self.service._scan_tasks), return_exceptions=True)

        scan = self.repository.get_scan_by_placement(event.placement_key)
        self.assertEqual(scan["error_code"], "scale_disconnected")
        self.assertIsNone(self.repository.get_meal_by_scan(scan["id"]))

    async def test_plate_removal_during_inference_rejects_analysis(self):
        self.camera.release_capture.set()
        inference = BlockingInference()
        self.workflow.inference = inference
        observation = self.scale.set_weight(750)
        now = time.monotonic()
        await self.service.observe(observation, now=now)
        await self.service.observe(observation, now=now + 0.05)
        event = await self.service.observe(observation, now=now + 0.1)
        self.assertIsNotNone(event)
        self.assertTrue(await asyncio.wait_for(asyncio.to_thread(inference.started.wait), timeout=1))

        removed = self.scale.set_weight(0)
        await self.service.observe(removed)
        inference.release.set()
        await asyncio.gather(*tuple(self.service._scan_tasks), return_exceptions=True)

        scan = self.repository.get_scan_by_placement(event.placement_key)
        self.assertEqual(scan["error_code"], "plate_removed_during_scan")
        self.assertIsNone(self.repository.get_meal_by_scan(scan["id"]))


if __name__ == "__main__":
    unittest.main()
