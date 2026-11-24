import asyncio
import io
import threading
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from smartnutri_backend.agent import NutritionDecisionAgent
from smartnutri_backend.camera import MockCameraService
from smartnutri_backend.nutrition import NutritionService
from smartnutri_backend.portion import ExperimentalVisualProportionEstimator
from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.scale_service import MockScaleClient, ScaleMonitoringService
from smartnutri_backend.schemas import FoodPrediction, ScaleObservation, ScanState
from smartnutri_backend.settings import Settings
from smartnutri_backend.vision import MockSegmentationService, SegmentationResult
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


class FixtureSegmentation:
    def __init__(self, *, inconsistent_area=False):
        self.inconsistent_area = inconsistent_area

    def infer(self, capture, *, scan_id):
        with Image.open(io.BytesIO(capture.image_bytes)) as source:
            original = source.convert("RGB")
        semantic = Image.new("L", original.size, 0)
        draw = ImageDraw.Draw(semantic)
        draw.rectangle((20, 20, 119, 69), fill=1)
        draw.rectangle((200, 200, 201, 201), fill=48)
        histogram = semantic.histogram()
        predictions = [
            FoodPrediction(
                class_id=1,
                label="rice",
                classification_confidence=0.92,
                pixel_area=histogram[1] - int(self.inconsistent_area),
                region_count=1,
                source="mock",
                simulated=True,
            ),
            FoodPrediction(
                class_id=48,
                label="chicken duck",
                classification_confidence=0.92,
                pixel_area=histogram[48],
                region_count=1,
                segmentation_quality="small_region",
                source="mock",
                simulated=True,
            ),
        ]
        output = io.BytesIO()
        semantic.save(output, format="PNG")
        return SegmentationResult(
            predictions=predictions,
            semantic_mask_png=output.getvalue(),
            annotated_image_bytes=capture.image_bytes,
            width=original.width,
            height=original.height,
            source="mock",
            simulated=True,
        )


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

    async def test_changed_scale_weight_after_inference_aborts_the_scan(self):
        class ChangeWeightDuringInference:
            def __init__(self, scale):
                self.scale = scale

            def infer(self, image, *, scan_id):
                self.scale.set_weight(780)
                return MockSegmentationService().infer(image, scan_id=scan_id)

        self.workflow.inference = ChangeWeightDuringInference(self.scale)
        observation = self.scale.set_weight(750)
        start = time.monotonic()
        await self.service.observe(observation, now=start)
        await self.service.observe(observation, now=start + 0.05)
        event = await self.service.observe(observation, now=start + 0.1)
        self.camera.release_capture.set()
        await asyncio.gather(*tuple(self.service._scan_tasks), return_exceptions=True)

        scan = self.repository.get_scan_by_placement(event.placement_key)
        self.assertEqual(scan["state"], ScanState.NEEDS_REVIEW.value)
        self.assertEqual(scan["error_code"], "placement_changed_or_unstable")
        self.assertIsNone(self.repository.get_meal_by_scan(scan["id"]))

    async def test_placement_revalidation_rejects_unstable_and_changed_readings(self):
        original = self.scale.set_weight(750)
        placement_key = "fixture-placement"
        scan = self.repository.begin_scan_for_placement(
            placement_key=placement_key,
            source="mock",
            simulated=True,
            state=ScanState.CAPTURING.value,
        )
        self.service.monitor.active_placement_key = placement_key

        self.assertIsNone(await self.service.validate_active_placement(placement_key, original))
        self.scale.set_weight(760)
        changed = await self.service.validate_active_placement(placement_key, original)
        self.assertIn("weight changed", changed)

        class UnstableScale:
            def get_weight(self):
                return ScaleObservation(
                    ok=False,
                    source="mock",
                    simulated=True,
                    sensor_connected=False,
                    tared=True,
                    calibrated=True,
                    stable=False,
                    status="unstable",
                    unit="g",
                    weight_g=750,
                )

        self.service.client = UnstableScale()
        unstable = await self.service.validate_active_placement(placement_key, original)
        self.assertIn("stale or unstable", unstable)
        self.assertEqual(scan["placement_key"], placement_key)

    async def test_minimum_region_filter_withholds_small_food_and_requests_review(self):
        self.workflow.inference = FixtureSegmentation()
        self.workflow.min_region_area_ratio = 0.01
        observation = self.scale.set_weight(750)
        start = time.monotonic()
        await self.service.observe(observation, now=start)
        await self.service.observe(observation, now=start + 0.05)
        event = await self.service.observe(observation, now=start + 0.1)
        self.camera.release_capture.set()
        await asyncio.gather(*tuple(self.service._scan_tasks), return_exceptions=True)

        scan = self.repository.get_scan_by_placement(event.placement_key)
        self.assertEqual(scan["decision"], "REVIEW_FOOD_PREDICTIONS")
        portions = scan["portions"]
        self.assertEqual([item["class_id"] for item in portions["estimates"]], [1])
        self.assertFalse(portions["allocation_complete"])
        self.assertGreater(portions["unallocated_weight_g"], 0)

    async def test_prediction_mask_conflict_blocks_portion_and_nutrition_outputs(self):
        self.workflow.inference = FixtureSegmentation(inconsistent_area=True)
        observation = self.scale.set_weight(750)
        start = time.monotonic()
        await self.service.observe(observation, now=start)
        await self.service.observe(observation, now=start + 0.05)
        event = await self.service.observe(observation, now=start + 0.1)
        self.camera.release_capture.set()
        await asyncio.gather(*tuple(self.service._scan_tasks), return_exceptions=True)

        scan = self.repository.get_scan_by_placement(event.placement_key)
        self.assertEqual(scan["error_code"], "prediction_mask_mismatch")
        self.assertEqual(scan["decision"], "REVIEW_FOOD_PREDICTIONS")
        self.assertIsNone(scan["portions"])
        self.assertIsNone(scan["nutrition_totals"])
        self.assertIsNone(self.repository.get_meal_by_scan(scan["id"]))


if __name__ == "__main__":
    unittest.main()
