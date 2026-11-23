from __future__ import annotations

import math
from typing import Any

from pydantic import Field

from smartnutri_backend.schemas import (
    NutrientEstimate,
    NutritionRecord,
    PortionEstimate,
    StrictModel,
)


class FoodNutritionResult(StrictModel):
    class_id: int
    label: str
    portion_weight_g: float
    nutrition_record_id: str | None = None
    food_identity: str | None = None
    preparation_state: str | None = None
    nutrients: NutrientEstimate | None = None
    mapping_status: str


class NutritionCalculation(StrictModel):
    items: list[FoodNutritionResult] = Field(default_factory=list)
    totals: NutrientEstimate | None = None
    complete: bool
    simulated: bool


class NutritionService:
    """Verified-record lookup and explicit per-100 g nutrient scaling."""

    @staticmethod
    def _record_from_row(data: dict[str, Any]) -> NutritionRecord:
        # Repository rows include storage metadata (id/created_at) that belongs
        # to the API view, not the strict create/lookup domain model.
        fields = NutritionRecord.model_fields
        return NutritionRecord.model_validate(
            {name: data[name] for name in fields if name in data}
        )

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def register(self, record: NutritionRecord) -> dict[str, Any]:
        if record.verified and record.class_id is None:
            raise ValueError("a verified nutrition record must map to a FoodSeg103 class ID")
        return self.repository.add_nutrition_record(record.model_dump())

    def list_records(self, *, class_id: int | None = None) -> list[dict[str, Any]]:
        return self.repository.list_nutrition_records(class_id=class_id)

    def get_record(self, record_id: str) -> NutritionRecord | None:
        data = self.repository.get_nutrition_record(record_id)
        return self._record_from_row(data) if data else None

    def lookup_for_class(self, class_id: int) -> tuple[str, NutritionRecord] | None:
        records = self.repository.list_nutrition_records(class_id=class_id, verified_only=True)
        if len(records) != 1:
            return None
        return records[0]["id"], self._record_from_row(records[0])

    @staticmethod
    def calculate_item(weight_g: float, record: NutritionRecord) -> NutrientEstimate:
        if isinstance(weight_g, bool) or not math.isfinite(weight_g) or weight_g < 0:
            raise ValueError("portion weight must be finite and non-negative")
        factor = weight_g / 100.0
        return NutrientEstimate(
            kcal=round(record.kcal_per_100g * factor, 2),
            protein_g=round(record.protein_g_per_100g * factor, 2),
            carbohydrates_g=round(record.carbohydrates_g_per_100g * factor, 2),
            fat_g=round(record.fat_g_per_100g * factor, 2),
            estimated=True,
            source=record.source,
            version=record.version,
        )

    def calculate(
        self,
        portions: list[PortionEstimate],
        *,
        simulated: bool | None = None,
    ) -> NutritionCalculation:
        results: list[FoodNutritionResult] = []
        simulated_output = bool(simulated) or any(item.simulated for item in portions)
        for portion in portions:
            mapping = self.lookup_for_class(portion.class_id)
            if mapping is None:
                results.append(
                    FoodNutritionResult(
                        class_id=portion.class_id,
                        label=portion.label,
                        portion_weight_g=portion.estimated_weight_g,
                        mapping_status="missing_or_ambiguous_verified_record",
                    )
                )
                continue
            record_id, record = mapping
            results.append(
                FoodNutritionResult(
                    class_id=portion.class_id,
                    label=portion.label,
                    portion_weight_g=portion.estimated_weight_g,
                    nutrition_record_id=record_id,
                    food_identity=record.food_identity,
                    preparation_state=record.preparation_state,
                    nutrients=self.calculate_item(portion.estimated_weight_g, record),
                    mapping_status="verified",
                )
            )

        complete = bool(results) and all(item.nutrients is not None for item in results)
        totals: NutrientEstimate | None = None
        if complete:
            nutrient_items = [item.nutrients for item in results if item.nutrients is not None]
            sources = {item.source for item in nutrient_items}
            versions = {item.version for item in nutrient_items}
            totals = NutrientEstimate(
                kcal=round(sum(item.kcal for item in nutrient_items), 2),
                protein_g=round(sum(item.protein_g for item in nutrient_items), 2),
                carbohydrates_g=round(sum(item.carbohydrates_g for item in nutrient_items), 2),
                fat_g=round(sum(item.fat_g for item in nutrient_items), 2),
                estimated=True,
                source=next(iter(sources)) if len(sources) == 1 else "multiple_sources",
                version=next(iter(versions)) if len(versions) == 1 else "multiple_versions",
            )
        return NutritionCalculation(
            items=results,
            totals=totals,
            complete=complete,
            simulated=simulated_output,
        )
