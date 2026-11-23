from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRepository:
    """Small thread-safe SQLite repository; runtime databases live outside Git."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 10000")
        if self.database_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plate_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    empty_plate_weight_g REAL NOT NULL CHECK (empty_plate_weight_g > 0),
                    reference_image_path TEXT,
                    is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nutrition_records (
                    id TEXT PRIMARY KEY,
                    class_id INTEGER CHECK (class_id BETWEEN 1 AND 103),
                    food_identity TEXT NOT NULL,
                    preparation_state TEXT NOT NULL,
                    kcal_per_100g REAL NOT NULL CHECK (kcal_per_100g >= 0),
                    protein_g_per_100g REAL NOT NULL CHECK (protein_g_per_100g >= 0),
                    carbohydrates_g_per_100g REAL NOT NULL CHECK (carbohydrates_g_per_100g >= 0),
                    fat_g_per_100g REAL NOT NULL CHECK (fat_g_per_100g >= 0),
                    source TEXT NOT NULL,
                    version TEXT NOT NULL,
                    verified INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
                    created_at TEXT NOT NULL,
                    UNIQUE(food_identity, preparation_state, source, version)
                );
                CREATE TABLE IF NOT EXISTS scans (
                    id TEXT PRIMARY KEY,
                    placement_key TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL,
                    decision TEXT,
                    source TEXT NOT NULL CHECK (source IN ('hx711', 'mock')),
                    simulated INTEGER NOT NULL CHECK (simulated IN (0, 1)),
                    tare_mode TEXT,
                    plate_profile_id TEXT REFERENCES plate_profiles(id) ON DELETE SET NULL,
                    measured_gross_weight_g REAL,
                    plate_tare_weight_g REAL,
                    measured_net_food_weight_g REAL,
                    capture_source TEXT,
                    capture_simulated INTEGER CHECK (capture_simulated IN (0, 1)),
                    capture_path TEXT,
                    annotated_path TEXT,
                    image_quality_json TEXT,
                    predictions_json TEXT,
                    portions_json TEXT,
                    nutrition_totals_json TEXT,
                    result_message TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_scans_created_at ON scans(created_at DESC);
                CREATE INDEX IF NOT EXISTS ix_scans_state ON scans(state);
                CREATE TABLE IF NOT EXISTS meals (
                    id TEXT PRIMARY KEY,
                    scan_id TEXT NOT NULL UNIQUE REFERENCES scans(id) ON DELETE CASCADE,
                    state TEXT NOT NULL,
                    measured_net_food_weight_g REAL NOT NULL CHECK (measured_net_food_weight_g > 0),
                    nutrition_estimated INTEGER NOT NULL DEFAULT 1 CHECK (nutrition_estimated IN (0, 1)),
                    kcal_total REAL,
                    protein_g_total REAL,
                    carbohydrates_g_total REAL,
                    fat_g_total REAL,
                    correction_revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meal_items (
                    id TEXT PRIMARY KEY,
                    meal_id TEXT NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
                    class_id INTEGER CHECK (class_id BETWEEN 1 AND 103),
                    food_label TEXT NOT NULL,
                    classification_confidence REAL CHECK (classification_confidence BETWEEN 0 AND 1),
                    pixel_area INTEGER CHECK (pixel_area >= 0),
                    region_count INTEGER CHECK (region_count >= 0),
                    estimated_weight_g REAL CHECK (estimated_weight_g >= 0),
                    portion_method TEXT,
                    portion_confidence REAL CHECK (portion_confidence BETWEEN 0 AND 1),
                    portion_uncertainty_g REAL CHECK (portion_uncertainty_g >= 0),
                    nutrition_record_id TEXT REFERENCES nutrition_records(id) ON DELETE SET NULL,
                    nutrient_estimate_json TEXT,
                    user_corrected INTEGER NOT NULL DEFAULT 0 CHECK (user_corrected IN (0, 1)),
                    needs_review INTEGER NOT NULL DEFAULT 0 CHECK (needs_review IN (0, 1)),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_meals_updated_at ON meals(updated_at DESC);
                CREATE INDEX IF NOT EXISTS ix_meal_items_meal_id ON meal_items(meal_id);
                """
            )
            self._set_default_setting_unlocked("tare_configuration", {
                "mode": "saved_plate",
                "live_tare_active": False,
                "empty_plate_weight_g": None,
                "live_tare_at": None,
            })
            self._set_default_setting_unlocked("monitor_latch", {
                "latched": False,
                "placement_key": None,
            })

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    @staticmethod
    def _decode_json(value: str | None) -> Any:
        return json.loads(value) if value is not None else None

    def _set_setting_unlocked(self, key: str, value: Any) -> None:
        self._connection.execute(
            """INSERT INTO app_settings(key, value_json, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
                 updated_at=excluded.updated_at""",
            (key, json.dumps(value, separators=(",", ":")), utc_now()),
        )

    def _set_default_setting_unlocked(self, key: str, value: Any) -> None:
        self._connection.execute(
            "INSERT OR IGNORE INTO app_settings(key, value_json, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, separators=(",", ":")), utc_now()),
        )

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock, self._connection:
            self._set_setting_unlocked(key, value)

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._connection.execute(
                "SELECT value_json FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return self._decode_json(row["value_json"]) if row else default

    def set_tare_configuration(
        self,
        *,
        mode: str,
        live_tare_active: bool,
        empty_plate_weight_g: float | None,
        live_tare_at: str | None = None,
    ) -> None:
        if mode not in {"saved_plate", "live_plate"}:
            raise ValueError("mode must be saved_plate or live_plate")
        if live_tare_active != (mode == "live_plate"):
            raise ValueError("live_tare_active must agree with tare mode")
        if empty_plate_weight_g is not None and (
            not math.isfinite(empty_plate_weight_g) or empty_plate_weight_g <= 0
        ):
            raise ValueError("empty plate weight must be finite and positive")
        if mode == "saved_plate" and live_tare_active:
            raise ValueError("saved plate mode cannot have an active live tare")
        self.set_setting(
            "tare_configuration",
            {
                "mode": mode,
                "live_tare_active": live_tare_active,
                "empty_plate_weight_g": empty_plate_weight_g,
                "live_tare_at": live_tare_at,
            },
        )

    def get_tare_configuration(self) -> dict[str, Any]:
        return self.get_setting("tare_configuration", {
            "mode": "saved_plate",
            "live_tare_active": False,
            "empty_plate_weight_g": None,
            "live_tare_at": None,
        })

    def set_monitor_latched(self, latched: bool, placement_key: str | None) -> None:
        self.set_setting(
            "monitor_latch",
            {"latched": bool(latched), "placement_key": placement_key if latched else None},
        )

    def get_monitor_latch(self) -> dict[str, Any]:
        return self.get_setting("monitor_latch", {"latched": False, "placement_key": None})

    def create_plate_profile(
        self,
        name: str,
        empty_plate_weight_g: float,
        reference_image_path: str | None = None,
        *,
        make_default: bool = False,
    ) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("plate name cannot be empty")
        if not math.isfinite(empty_plate_weight_g) or not 0 < empty_plate_weight_g <= 5000:
            raise ValueError("empty plate weight must be in (0, 5000] grams")
        now, profile_id = utc_now(), str(uuid.uuid4())
        with self._lock, self._connection:
            if make_default:
                self._connection.execute("UPDATE plate_profiles SET is_default = 0")
            self._connection.execute(
                """INSERT INTO plate_profiles
                   (id, name, empty_plate_weight_g, reference_image_path, is_default, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (profile_id, name, float(empty_plate_weight_g), reference_image_path,
                 int(make_default), now, now),
            )
            row = self._connection.execute(
                "SELECT * FROM plate_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
        return self._dict(row) or {}

    def list_plate_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM plate_profiles ORDER BY created_at, name"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_plate_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM plate_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
        return self._dict(row)

    def get_default_plate(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM plate_profiles WHERE is_default = 1 LIMIT 1"
            ).fetchone()
        return self._dict(row)

    def set_default_plate(self, profile_id: str) -> dict[str, Any]:
        with self._lock, self._connection:
            exists = self._connection.execute(
                "SELECT 1 FROM plate_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
            if not exists:
                raise KeyError(profile_id)
            self._connection.execute("UPDATE plate_profiles SET is_default = 0")
            self._connection.execute(
                "UPDATE plate_profiles SET is_default = 1, updated_at = ? WHERE id = ?",
                (utc_now(), profile_id),
            )
            row = self._connection.execute(
                "SELECT * FROM plate_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
        return self._dict(row) or {}

    def create_scan(
        self,
        *,
        placement_key: str,
        source: str,
        simulated: bool,
        state: str = "WEIGHT_DETECTED",
    ) -> dict[str, Any]:
        if source not in {"hx711", "mock"} or (source == "mock") != simulated:
            raise ValueError("scan source and simulated marker disagree")
        now, scan_id = utc_now(), str(uuid.uuid4())
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO scans(id, placement_key, state, source, simulated, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(placement_key) DO NOTHING""",
                (scan_id, placement_key, state, source, int(simulated), now, now),
            )
            row = self._connection.execute(
                "SELECT * FROM scans WHERE placement_key = ?", (placement_key,)
            ).fetchone()
        return self._dict(row) or {}

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM scans WHERE id = ?", (scan_id,)
            ).fetchone()
        return self._decode_scan(row)

    def get_scan_by_placement(self, placement_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM scans WHERE placement_key = ?", (placement_key,)
            ).fetchone()
        return self._decode_scan(row)

    @classmethod
    def _decode_scan(cls, row: sqlite3.Row | None) -> dict[str, Any] | None:
        result = cls._dict(row)
        if result is None:
            return None
        for key in ("image_quality_json", "predictions_json", "portions_json", "nutrition_totals_json"):
            result[key.removesuffix("_json")] = cls._decode_json(result.pop(key))
        result["simulated"] = bool(result["simulated"])
        if result["capture_simulated"] is not None:
            result["capture_simulated"] = bool(result["capture_simulated"])
        return result

    def list_scans(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_scan(row) or {} for row in rows]

    def update_scan(self, scan_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "state", "decision", "tare_mode", "plate_profile_id",
            "measured_gross_weight_g", "plate_tare_weight_g", "measured_net_food_weight_g",
            "capture_source", "capture_simulated", "capture_path", "annotated_path",
            "image_quality", "predictions", "portions", "nutrition_totals",
            "result_message", "error_code",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported scan fields: {sorted(unknown)}")
        encoded = {}
        json_keys = {"image_quality", "predictions", "portions", "nutrition_totals"}
        for key, value in values.items():
            column = key + "_json" if key in json_keys else key
            encoded[column] = json.dumps(value, separators=(",", ":")) if key in json_keys and value is not None else value
        encoded["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in encoded)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                f"UPDATE scans SET {assignments} WHERE id = ?",
                (*encoded.values(), scan_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(scan_id)
        result = self.get_scan(scan_id)
        assert result is not None
        return result

    def add_nutrition_record(self, record: dict[str, Any]) -> dict[str, Any]:
        record_id = str(uuid.uuid4())
        now = utc_now()
        values = (
            record_id,
            record.get("class_id"),
            record["food_identity"],
            record["preparation_state"],
            float(record["kcal_per_100g"]),
            float(record["protein_g_per_100g"]),
            float(record["carbohydrates_g_per_100g"]),
            float(record["fat_g_per_100g"]),
            record["source"],
            record["version"],
            int(bool(record.get("verified", False))),
            now,
        )
        with self._lock, self._connection:
            try:
                self._connection.execute(
                    """INSERT INTO nutrition_records
                       (id, class_id, food_identity, preparation_state, kcal_per_100g,
                        protein_g_per_100g, carbohydrates_g_per_100g, fat_g_per_100g,
                        source, version, verified, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("nutrition record conflicts with an existing record or schema constraint") from exc
            row = self._connection.execute(
                "SELECT * FROM nutrition_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._decode_nutrition_record(row) or {}

    @classmethod
    def _decode_nutrition_record(cls, row: sqlite3.Row | None) -> dict[str, Any] | None:
        result = cls._dict(row)
        if result is not None:
            result["verified"] = bool(result["verified"])
        return result

    def get_nutrition_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM nutrition_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._decode_nutrition_record(row)

    def list_nutrition_records(
        self,
        *,
        class_id: int | None = None,
        verified_only: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        clauses: list[str] = []
        parameters: list[Any] = []
        if class_id is not None:
            clauses.append("class_id = ?")
            parameters.append(class_id)
        if verified_only:
            clauses.append("verified = 1")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM nutrition_records{where} ORDER BY created_at DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._decode_nutrition_record(row) or {} for row in rows]

