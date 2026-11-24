import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from smartnutri_backend.settings import Settings


class SettingsEnvironmentTests(unittest.TestCase):
    def test_agent_thresholds_load_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "SMARTNUTRI_MIN_CLASSIFICATION_CONFIDENCE": "0.7",
                "SMARTNUTRI_PORTION_REVIEW_UNCERTAINTY_FRACTION": "0.3",
            },
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.min_classification_confidence, 0.7)
        self.assertEqual(settings.portion_review_uncertainty_fraction, 0.3)

    def test_local_checkpoint_path_selects_b0_or_b2_without_code_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            b2_checkpoint = root / "runs" / "segformer-b2" / "best_model"
            foodseg_root = root / "Datasets" / "FoodSeg103"
            with patch.dict(
                os.environ,
                {
                    "SMARTNUTRI_CV_MODE": "segformer",
                    "SMARTNUTRI_CHECKPOINT_DIR": str(b2_checkpoint),
                    "SMARTNUTRI_FOODSEG_ROOT": str(foodseg_root),
                },
            ):
                settings = Settings.from_env()

        self.assertEqual(settings.segmentation_mode, "segformer")
        self.assertEqual(settings.checkpoint_dir, b2_checkpoint)
        self.assertEqual(settings.cv_model, "b2")
        self.assertEqual(settings.selected_checkpoint_dir, b2_checkpoint)
        self.assertEqual(settings.foodseg_root, foodseg_root)

    def test_b2_is_the_local_default_and_b0_remains_selectable(self):
        default = Settings()
        b0 = Settings(cv_model="b0")

        self.assertEqual(default.cv_model, "b2")
        self.assertEqual(default.selected_checkpoint_dir.name, "best_model")
        self.assertIn("foodseg103-segformer-b2-optimized", str(default.selected_checkpoint_dir))
        self.assertIn("foodseg103-segformer-b0", str(b0.selected_checkpoint_dir))

    def test_tta_confidence_region_and_scale_tolerance_are_configurable(self):
        with patch.dict(
            os.environ,
            {
                "SMARTNUTRI_CV_MODEL": "b2",
                "SMARTNUTRI_CV_DEVICE": "cuda",
                "SMARTNUTRI_CV_HFLIP_TTA": "true",
                "SMARTNUTRI_MIN_CLASSIFICATION_CONFIDENCE": "0.7",
                "SMARTNUTRI_MIN_FOOD_REGION_AREA_RATIO": "0.002",
                "SMARTNUTRI_PLACEMENT_WEIGHT_TOLERANCE_G": "3.5",
                "SMARTNUTRI_SCALE_CONTROL_TOKEN": "scale-secret",
            },
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.cv_model, "b2")
        self.assertEqual(settings.cv_device, "cuda")
        self.assertTrue(settings.cv_hflip_tta)
        self.assertEqual(settings.min_classification_confidence, 0.7)
        self.assertEqual(settings.min_food_region_area_ratio, 0.002)
        self.assertEqual(settings.placement_weight_tolerance_g, 3.5)
        self.assertEqual(settings.scale_control_token, "scale-secret")


if __name__ == "__main__":
    unittest.main()
