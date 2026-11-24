from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from smartnutri_backend.nutrition_catalog import CURATED_NUTRITION_RECORDS


EXPECTED_NUTRIENTS = {
    1003: ("Protein", "G", "protein_g_per_100g"),
    1004: ("Total lipid (fat)", "G", "fat_g_per_100g"),
    1005: ("Carbohydrate, by difference", "G", "carbohydrates_g_per_100g"),
    1008: ("Energy", "KCAL", "kcal_per_100g"),
}


def _read_csv(archive: zipfile.ZipFile, suffix: str) -> list[dict[str, str]]:
    matches = [name for name in archive.namelist() if name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} inside the official SR Legacy archive")
    return list(csv.DictReader(io.TextIOWrapper(archive.open(matches[0]), encoding="latin1")))


def verify(archive_path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(archive_path) as archive:
        foods = {int(row["fdc_id"]): row for row in _read_csv(archive, "food.csv")}
        nutrients = {int(row["id"]): row for row in _read_csv(archive, "nutrient.csv")}
        for nutrient_id, (name, unit, _) in EXPECTED_NUTRIENTS.items():
            source = nutrients.get(nutrient_id)
            if source is None or source["name"] != name or source["unit_name"].upper() != unit:
                raise ValueError(f"USDA nutrient definition {nutrient_id} does not match the expected macro")

        expected_ids = {
            int(str(item["source_identifier"]).removeprefix("FDC:"))
            for item in CURATED_NUTRITION_RECORDS
            if str(item.get("source_identifier", "")).startswith("FDC:")
        }
        values: dict[int, dict[int, float]] = {}
        for row in _read_csv(archive, "food_nutrient.csv"):
            try:
                fdc_id = int(row["fdc_id"])
                nutrient_id = int(row["nutrient_id"])
                amount = float(row["amount"])
            except (TypeError, ValueError):
                continue
            if fdc_id in expected_ids and nutrient_id in EXPECTED_NUTRIENTS:
                previous = values.setdefault(fdc_id, {}).get(nutrient_id)
                if previous is not None and not math.isclose(previous, amount, abs_tol=1e-8):
                    raise ValueError(f"FDC:{fdc_id} has conflicting duplicate nutrient {nutrient_id}")
                values[fdc_id][nutrient_id] = amount

    candidate_by_id = {
        str(item["source_identifier"]): item
        for item in CURATED_NUTRITION_RECORDS
        if item["source_identifier"] not in {
            "FDC:168878", "FDC:168877", "FDC:171477", "FDC:171077", "FDC:172411",
            "FDC:170698", "FDC:168202", "FDC:173944", "FDC:169967", "FDC:170379", "FDC:170457",
        }
    }
    if expected_ids - set(foods):
        raise ValueError(f"catalog contains FDC IDs absent from archive: {sorted(expected_ids - set(foods))}")

    for record in CURATED_NUTRITION_RECORDS:
        identifier = str(record["source_identifier"])
        fdc_id = int(identifier.removeprefix("FDC:"))
        source_description = foods[fdc_id]["description"]
        if identifier in candidate_by_id and record["food_identity"] != source_description:
            raise ValueError(f"{identifier} description differs from the official USDA row")
        if identifier in candidate_by_id and not record["requires_preparation_confirmation"]:
            raise ValueError(f"{identifier} candidate must require explicit confirmation")
        source_values = values.get(fdc_id, {})
        for nutrient_id, (_, _, field) in EXPECTED_NUTRIENTS.items():
            actual = record[field]
            expected = source_values.get(nutrient_id)
            if expected is None or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"{identifier} {field}={actual} differs from official amount {expected}")

    candidate_classes = {int(item["class_id"]) for item in candidate_by_id.values()}
    return len(CURATED_NUTRITION_RECORDS), len(candidate_classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the local nutrition catalog against USDA SR Legacy CSVs.")
    parser.add_argument("--archive", required=True, type=Path, help="official FoodData Central SR Legacy ZIP")
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    record_count, candidate_classes = verify(args.archive)
    print(f"verified {record_count} USDA source records; {candidate_classes} FoodSeg103 classes have candidate mappings")


if __name__ == "__main__":
    main()
