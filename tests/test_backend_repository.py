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

    def test_live_tare_preparation_is_persisted_before_an_empty_plate_is_placed(self):
        self.repository.set_tare_configuration(
            mode="saved_plate",
            live_tare_active=False,
            live_tare_pending=True,
            empty_plate_weight_g=None,
        )
        self.repository.close()
        reopened = SQLiteRepository(self.database)
        try:
            config = reopened.get_tare_configuration()
            self.assertTrue(config["live_tare_pending"])
            self.assertFalse(config["live_tare_active"])
            self.assertEqual(config["mode"], "saved_plate")
        finally:
            reopened.close()

    def test_legacy_nutrition_rows_are_preserved_but_unverified_without_source_ids(self):
        import sqlite3

        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        connection.execute(
            """CREATE TABLE nutrition_records (
                id TEXT PRIMARY KEY,
                class_id INTEGER,
                food_identity TEXT NOT NULL,
                preparation_state TEXT NOT NULL,
                kcal_per_100g REAL NOT NULL,
                protein_g_per_100g REAL NOT NULL,
                carbohydrates_g_per_100g REAL NOT NULL,
                fat_g_per_100g REAL NOT NULL,
                source TEXT NOT NULL,
                version TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(food_identity, preparation_state, source, version)
            )"""
        )
        connection.execute(
            """INSERT INTO nutrition_records VALUES
                ('legacy-1', 66, 'legacy rice', 'cooked', 130, 2.7, 28.2, 0.3,
                 'legacy source', 'old-version', 1, '2026-01-01T00:00:00+00:00')"""
        )
        connection.commit()
        connection.close()

        migrated = SQLiteRepository(legacy_path)
        try:
            row = migrated.get_nutrition_record("legacy-1")
            self.assertEqual(row["food_identity"], "legacy rice")
            self.assertEqual(row["source_identifier"], "")
            self.assertFalse(row["verified"])
        finally:
            migrated.close()

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
