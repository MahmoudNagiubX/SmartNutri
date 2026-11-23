import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from smartnutri_backend.camera import CapturedImage, MockCameraService
from smartnutri_backend.vision import (
    FoodSegLabelMap,
    MockSegmentationService,
    SegFormerInferenceService,
)


class VisionAdapterTests(unittest.TestCase):
    def test_mock_multifood_result_has_simulation_marker_and_background_exclusion(self):
        label_map = FoodSegLabelMap.mock_fixture()
        camera = MockCameraService().capture_best()
        result = MockSegmentationService(label_map).infer(camera, scan_id="mock-scan-1")
        self.assertTrue(result.simulated)
        self.assertEqual(result.source, "mock")
        self.assertEqual({item.label for item in result.predictions}, {"rice", "chicken", "fries"})
        self.assertTrue(all(1 <= item.class_id <= 103 for item in result.predictions))
        self.assertTrue(all(item.simulated for item in result.predictions))
        rice = next(item for item in result.predictions if item.label == "rice")
        self.assertEqual(rice.region_count, 2)
        self.assertIn("/mock-scan-1/mask/1", rice.mask_ref)

        with Image.open(io.BytesIO(result.semantic_mask_png)) as mask:
            self.assertEqual(mask.mode, "L")
            self.assertEqual(
                {class_id for class_id, count in enumerate(mask.histogram()) if count},
                {0, 1, 2, 3},
            )
        with Image.open(io.BytesIO(result.annotated_image_bytes)) as annotated:
            self.assertEqual(annotated.size, (640, 480))

    def test_foodseg_label_map_requires_exact_ids_and_background_zero(self):
        labels = {index: f"class-{index}" for index in range(104)}
        labels[0] = "background"
        mapping = FoodSegLabelMap(labels)
        self.assertEqual(mapping.label_for(103), "class-103")
        with self.assertRaises(ValueError):
            FoodSegLabelMap({**labels, 0: "food"})
        with self.assertRaises(ValueError):
            FoodSegLabelMap({key: value for key, value in labels.items() if key != 103})

    def test_real_inference_fails_closed_without_local_metadata_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            camera = MockCameraService().capture_best()
            service = SegFormerInferenceService(
                checkpoint_dir=Path(temp_dir) / "missing-checkpoint",
                foodseg_root=Path(temp_dir) / "missing-foodseg",
                device="cpu",
            )
            with self.assertRaises(FileNotFoundError):
                service.infer(camera, scan_id="no-weights")
            self.assertFalse(service.model_loaded)

    def test_loaded_model_mapping_must_exactly_match_foodseg103(self):
        from smartnutri_backend.vision import validate_model_label_contract

        labels = FoodSegLabelMap.mock_fixture()

        class Config:
            num_labels = 104
            id2label = {str(class_id): label for class_id, label in labels.labels.items()}
            label2id = {label: class_id for class_id, label in labels.labels.items()}

        validate_model_label_contract(Config, 104, labels)
        Config.label2id = {**Config.label2id, "rice": 5}
        with self.assertRaises(ValueError):
            validate_model_label_contract(Config, 104, labels)
        Config.label2id = {label: class_id for class_id, label in labels.labels.items()}
        with self.assertRaises(ValueError):
            validate_model_label_contract(Config, 150, labels)

    def test_captured_image_type_carries_source_provenance(self):
        capture = MockCameraService().capture_best()
        self.assertIsInstance(capture, CapturedImage)
        self.assertEqual(capture.source, "mock")
        self.assertTrue(capture.simulated)


if __name__ == "__main__":
    unittest.main()
