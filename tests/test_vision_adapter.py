import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch

from smartnutri_backend.camera import CapturedImage, MockCameraService
from smartnutri_backend.vision import (
    FoodSegLabelMap,
    MockSegmentationService,
    SegFormerInferenceService,
    _forward_segformer_logits,
    validate_foodseg103_label_anchors,
    validate_model_architecture,
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

    def test_real_foodseg_class_anchors_reject_mock_fixture_mapping(self):
        with self.assertRaises(ValueError):
            validate_foodseg103_label_anchors(FoodSegLabelMap.mock_fixture())

        labels = dict(FoodSegLabelMap.mock_fixture().labels)
        labels.update({
            0: "background",
            3: "french fries",
            24: "egg",
            25: "apple",
            29: "banana",
            48: "chicken duck",
            66: "rice",
            70: "potato",
            73: "tomato",
            87: "broccoli",
        })
        labels = FoodSegLabelMap(labels)
        validate_foodseg103_label_anchors(labels)

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

    def test_model_variant_validation_distinguishes_b0_and_b2(self):
        b2 = SimpleNamespace(
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
        )
        validate_model_architecture(b2, "b2")
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_model_architecture(b2, "b0")

    def test_horizontal_flip_tta_unflips_and_averages_logits(self):
        class FakeModel:
            def __init__(self):
                self.inputs = []

            def __call__(self, *, pixel_values):
                self.inputs.append(pixel_values.clone())
                return SimpleNamespace(logits=pixel_values[:, :1])

        values = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4).repeat(1, 3, 1, 1)
        model = FakeModel()
        logits = _forward_segformer_logits(model, values, hflip_tta=True)
        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(torch.equal(model.inputs[1], values.flip(-1)))
        self.assertTrue(torch.equal(logits, values[:, :1]))

        single_model = FakeModel()
        _forward_segformer_logits(single_model, values, hflip_tta=False)
        self.assertEqual(len(single_model.inputs), 1)

    def test_captured_image_type_carries_source_provenance(self):
        capture = MockCameraService().capture_best()
        self.assertIsInstance(capture, CapturedImage)
        self.assertEqual(capture.source, "mock")
        self.assertTrue(capture.simulated)


if __name__ == "__main__":
    unittest.main()
