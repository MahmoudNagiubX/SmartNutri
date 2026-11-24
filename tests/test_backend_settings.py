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

    def test_device_camera_and_snapshot_limits_load_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "SMARTNUTRI_SCALE_DEVICE_TOKEN": "device-secret-test",
                "SMARTNUTRI_CONTROL_TOKEN": "control-secret-test",
                "SMARTNUTRI_CAMERA_URL": "http://192.168.1.50:8080/snapshot.jpg",
                "SMARTNUTRI_CAMERA_ALLOWED_HOST": "192.168.1.50:8080",
                "SMARTNUTRI_WEIGHT_CHANGE_TOLERANCE_G": "3.5",
                "SMARTNUTRI_STALE_READING_TIMEOUT_S": "1.5",
                "SMARTNUTRI_CORRECTION_WEIGHT_TOLERANCE_G": "0.1",
            },
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.scale_device_token, "device-secret-test")
        self.assertEqual(settings.control_token, "control-secret-test")
        self.assertEqual(settings.camera_snapshot_url, "http://192.168.1.50:8080/snapshot.jpg")
        self.assertEqual(settings.camera_allowed_host, "192.168.1.50:8080")
        self.assertEqual(settings.weight_change_tolerance_g, 3.5)
        self.assertEqual(settings.stale_reading_timeout_s, 1.5)
        self.assertEqual(settings.correction_weight_tolerance_g, 0.1)
        self.assertNotIn("device-secret-test", repr(settings))
        self.assertNotIn("control-secret-test", repr(settings))

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
        self.assertEqual(settings.foodseg_root, foodseg_root)


if __name__ == "__main__":
    unittest.main()
