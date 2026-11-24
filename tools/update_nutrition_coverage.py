from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from smartnutri_backend.nutrition_catalog import CURATED_NUTRITION_RECORDS


COVERAGE_PATH = PROJECT_ROOT / "docs" / "audit" / "nutrition_coverage.csv"


def _joined(records: list[dict], field: str) -> str:
    return " | ".join(str(item[field]) for item in records)


def update_coverage(path: Path = COVERAGE_PATH) -> tuple[int, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or len(rows) != 103:
        raise ValueError("coverage matrix must contain all 103 FoodSeg103 foreground classes")

    by_class: dict[int, list[dict]] = defaultdict(list)
    for record in CURATED_NUTRITION_RECORDS:
        class_id = record.get("class_id")
        if class_id is not None:
            by_class[int(class_id)].append(record)

    for row in rows:
        class_id = int(row["class_id"])
        records = by_class.get(class_id, [])
        if not records:
            continue
        row["nutrition_record_count"] = str(len(records))
        row["food_identities"] = _joined(records, "food_identity")
        row["preparation_states"] = _joined(records, "preparation_state")
        for column, source_field in (
            ("kcal_per_100g_by_record", "kcal_per_100g"),
            ("protein_g_per_100g_by_record", "protein_g_per_100g"),
            ("carbohydrates_g_per_100g_by_record", "carbohydrates_g_per_100g"),
            ("fat_g_per_100g_by_record", "fat_g_per_100g"),
        ):
            row[column] = " | ".join(
                f"{item['source_identifier']}={item[source_field]}" for item in records
            )
        row["source"] = "USDA FoodData Central"
        row["source_identifiers"] = " | ".join(item["source_identifier"] for item in records)
        row["source_version"] = "SR Legacy final release, April 2018"
        row["mapping_ambiguity"] = (
            "Source nutrient records are verified candidates; the FoodSeg103 class does not "
            "confirm the exact identity, variety, animal, cut, or preparation state."
        )
        row["verification_status"] = "AMBIGUOUS"
        row["missing_information"] = (
            "A user must confirm a matching source identity and preparation before nutrition is calculated."
        )
        row["recommended_action"] = (
            "Keep nutrients null until the user confirms the exact USDA record; do not select a candidate from pixels alone."
        )

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mapped_classes = sum(bool(items) for items in by_class.values())
    mapped_records = sum(len(items) for items in by_class.values())
    return mapped_classes, mapped_records


if __name__ == "__main__":
    classes, records = update_coverage()
    print(f"updated {COVERAGE_PATH}: {classes} candidate classes, {records} source records")
