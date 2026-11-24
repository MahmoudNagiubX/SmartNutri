"""Source-traceable USDA SR Legacy records and FoodSeg103 candidates.

Values are per 100 g. FDC identifiers are retained verbatim so every item can
be checked against the original FoodData Central record. Preparation-sensitive
FoodSeg103 classes require an explicit record selection before nutrition is
calculated automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

USDA_SOURCE = "USDA FoodData Central"
USDA_SR_LEGACY_VERSION = "SR Legacy final release, April 2018"


CURATED_NUTRITION_RECORDS: list[dict[str, object]] = [
    {
        "class_id": 66,
        "food_identity": "rice, white, long-grain, regular, enriched",
        "preparation_state": "cooked; long-grain, regular, enriched",
        "kcal_per_100g": 130.0,
        "protein_g_per_100g": 2.69,
        "carbohydrates_g_per_100g": 28.17,
        "fat_g_per_100g": 0.28,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:168878",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 66,
        "food_identity": "rice, white, long-grain, regular, enriched",
        "preparation_state": "raw; long-grain, regular, enriched",
        "kcal_per_100g": 365.0,
        "protein_g_per_100g": 7.13,
        "carbohydrates_g_per_100g": 79.95,
        "fat_g_per_100g": 0.66,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:168877",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 48,
        "food_identity": "chicken breast, meat only",
        "preparation_state": "cooked; roasted; breast, meat only",
        "kcal_per_100g": 165.0,
        "protein_g_per_100g": 31.02,
        "carbohydrates_g_per_100g": 0.0,
        "fat_g_per_100g": 3.57,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:171477",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 48,
        "food_identity": "chicken breast, meat only",
        "preparation_state": "raw; skinless, boneless, meat only",
        "kcal_per_100g": 120.0,
        "protein_g_per_100g": 22.5,
        "carbohydrates_g_per_100g": 0.0,
        "fat_g_per_100g": 2.62,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:171077",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 48,
        "food_identity": "duck, domesticated, meat only",
        "preparation_state": "cooked; roasted; domesticated, meat only",
        "kcal_per_100g": 201.0,
        "protein_g_per_100g": 23.48,
        "carbohydrates_g_per_100g": 0.0,
        "fat_g_per_100g": 11.2,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:172411",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 3,
        "food_identity": "fast foods, potato, french fried in vegetable oil",
        "preparation_state": "french-fried in vegetable oil; fast-food reference",
        "kcal_per_100g": 312.0,
        "protein_g_per_100g": 3.43,
        "carbohydrates_g_per_100g": 41.44,
        "fat_g_per_100g": 14.73,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:170698",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 25,
        "food_identity": "apple, golden delicious, with skin",
        "preparation_state": "raw; golden delicious, with skin",
        "kcal_per_100g": 57.0,
        "protein_g_per_100g": 0.28,
        "carbohydrates_g_per_100g": 13.6,
        "fat_g_per_100g": 0.15,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:168202",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 29,
        "food_identity": "banana",
        "preparation_state": "raw",
        "kcal_per_100g": 89.0,
        "protein_g_per_100g": 1.09,
        "carbohydrates_g_per_100g": 22.84,
        "fat_g_per_100g": 0.33,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:173944",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 87,
        "food_identity": "broccoli",
        "preparation_state": "cooked; boiled, drained, without salt",
        "kcal_per_100g": 35.0,
        "protein_g_per_100g": 2.38,
        "carbohydrates_g_per_100g": 7.18,
        "fat_g_per_100g": 0.41,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:169967",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 87,
        "food_identity": "broccoli",
        "preparation_state": "raw",
        "kcal_per_100g": 34.0,
        "protein_g_per_100g": 2.82,
        "carbohydrates_g_per_100g": 6.64,
        "fat_g_per_100g": 0.37,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:170379",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
    {
        "class_id": 73,
        "food_identity": "tomatoes, red, ripe",
        "preparation_state": "raw; year-round average",
        "kcal_per_100g": 18.0,
        "protein_g_per_100g": 0.88,
        "carbohydrates_g_per_100g": 3.89,
        "fat_g_per_100g": 0.2,
        "source": USDA_SOURCE,
        "source_identifier": "FDC:170457",
        "version": USDA_SR_LEGACY_VERSION,
        "verified": True,
        "requires_preparation_confirmation": True,
    },
]

_candidate_catalog_path = Path(__file__).with_name("usda_foodseg103_candidates.json")
_candidate_records = json.loads(_candidate_catalog_path.read_text(encoding="utf-8"))
if not isinstance(_candidate_records, list):
    raise ValueError("USDA FoodSeg103 candidate catalog must contain a JSON array")
if any(not item.get("requires_preparation_confirmation", False) for item in _candidate_records):
    raise ValueError("USDA FoodSeg103 candidate records must require explicit user confirmation")
CURATED_NUTRITION_RECORDS.extend(_candidate_records)
