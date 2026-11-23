import tempfile
import unittest
from pathlib import Path

from smartnutri_backend.repository import SQLiteRepository


class SQLiteRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "smartnutri.sqlite3"
        self.repository = SQLiteRepository(self.database)

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def test_plate_profiles_can_be_created_selected_and_reopened(self):
        first = self.repository.create_plate_profile("Ceramic", 300.0)
        second = self.repository.create_plate_profile("Glass", 420.0, make_default=True)
        self.assertEqual(self.repository.get_default_plate()["id"], second["id"])
        self.repository.set_default_plate(first["id"])
        self.assertEqual(self.repository.get_default_plate()["id"], first["id"])

        self.repository.close()
        reopened = SQLiteRepository(self.database)
        try:
            self.assertEqual(reopened.get_default_plate()["name"], "Ceramic")
            self.assertEqual(len(reopened.list_plate_profiles()), 2)
        finally:
            reopened.close()

    def test_tare_mode_is_persisted_and_modes_are_explicit(self):
        self.repository.set_tare_configuration(
            mode="live_plate", live_tare_active=True, empty_plate_weight_g=287.5
        )
        self.repository.close()
        reopened = SQLiteRepository(self.database)
        try:
            config = reopened.get_tare_configuration()
            self.assertEqual(config["mode"], "live_plate")
            self.assertTrue(config["live_tare_active"])
            self.assertEqual(config["empty_plate_weight_g"], 287.5)
        finally:
            reopened.close()

    def test_placement_key_prevents_duplicate_scan_records(self):
        first = self.repository.create_scan(
            placement_key="placement-1", source="mock", simulated=True
        )
        second = self.repository.create_scan(
            placement_key="placement-1", source="mock", simulated=True
        )
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(self.repository.list_scans()), 1)

    def test_open_scan_latches_monitor_across_backend_restart(self):
        self.repository.create_scan(
            placement_key="placement-1", source="hx711", simulated=False
        )
        self.repository.set_monitor_latched(True, "placement-1")
        self.repository.close()
        reopened = SQLiteRepository(self.database)
        try:
            self.assertEqual(
                reopened.get_monitor_latch(),
                {"latched": True, "placement_key": "placement-1"},
            )
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
