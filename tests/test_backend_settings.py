import os
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


if __name__ == "__main__":
    unittest.main()
