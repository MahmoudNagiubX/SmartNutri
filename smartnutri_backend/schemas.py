from __future__ import annotations

from enum import StrEnum
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ScanState(StrEnum):
    IDLE = "IDLE"
    WEIGHT_DETECTED = "WEIGHT_DETECTED"
    WAITING_FOR_STABILITY = "WAITING_FOR_STABILITY"
    CAPTURING = "CAPTURING"
    ANALYZING = "ANALYZING"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    COMPLETED = "COMPLETED"
    WAITING_FOR_REMOVAL = "WAITING_FOR_REMOVAL"


class AgentAction(StrEnum):
    ACCEPT_ANALYSIS = "ACCEPT_ANALYSIS"
    RETRY_WEIGHT = "RETRY_WEIGHT"
    RECAPTURE_IMAGE = "RECAPTURE_IMAGE"
    REVIEW_FOOD_PREDICTIONS = "REVIEW_FOOD_PREDICTIONS"
    REVIEW_PORTION_ESTIMATES = "REVIEW_PORTION_ESTIMATES"
    REQUEST_PLATE_TARE = "REQUEST_PLATE_TARE"
    REQUEST_FOOD_PREPARATION = "REQUEST_FOOD_PREPARATION"
    SAVE_MEAL = "SAVE_MEAL"


class TareMode(StrEnum):
    SAVED_PLATE = "saved_plate"
    LIVE_PLATE = "live_plate"


class ScaleObservation(StrictModel):
    ok: bool
    source: Literal["hx711", "mock"]
    simulated: bool
    sensor_connected: bool
    tared: bool
    calibrated: bool
    stable: bool
    status: str
    unit: Literal["g"]
    weight_g: float | None = Field(default=None, allow_inf_nan=False)
    weight_delta_g: float | None = Field(default=None, allow_inf_nan=False)
    stability_range_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    timestamp_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_source_and_status(self) -> "ScaleObservation":
        if (self.source == "mock") != self.simulated:
            raise ValueError("mock readings must be simulated and HX711 readings must be real")
        if self.status == "stable":
            if not (
                self.ok
                and self.sensor_connected
                and self.tared
                and self.calibrated
                and self.stable
                and self.weight_g is not None
            ):
                raise ValueError("stable status requires a finite, ready scale reading")
        if self.status in {"sensor_disconnected", "sensor_error", "overload"}:
            if self.weight_g is not None:
                raise ValueError("sensor error states must not expose a usable weight")
        return self

    @property
    def usable_stable(self) -> bool:
        return self.status == "stable" and self.stable and self.weight_g is not None


class FoodPrediction(StrictModel):
    class_id: int = Field(ge=1, le=103)
    label: str = Field(min_length=1, max_length=120)
    classification_confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    pixel_area: int = Field(ge=1)
    region_count: int = Field(ge=1)
    mask_ref: str | None = Field(default=None, max_length=500)
    segmentation_quality: Literal["unverified", "small_region", "fragmented"] = "unverified"
    source: Literal["segformer", "mock"] = "segformer"
    simulated: bool = False

    @model_validator(mode="after")
    def validate_simulation_source(self) -> "FoodPrediction":
        if (self.source == "mock") != self.simulated:
            raise ValueError("mock predictions must be marked simulated")
        return self


class ImageQuality(StrictModel):
    acceptable: bool
    reasons: list[str] = Field(default_factory=list)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: Literal["JPEG", "PNG", "WEBP"]
    byte_count: int = Field(gt=0)
    brightness_mean: float = Field(ge=0, le=255, allow_inf_nan=False)
    clipped_pixel_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    edge_variance: float = Field(ge=0, allow_inf_nan=False)


class PortionEstimate(StrictModel):
    class_id: int = Field(ge=1, le=103)
    label: str = Field(min_length=1, max_length=120)
    estimated_weight_g: float = Field(ge=0, allow_inf_nan=False)
    method: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    uncertainty_g: float = Field(ge=0, allow_inf_nan=False)
    review_required: bool
    simulated: bool = False


class PortionEstimationResult(StrictModel):
    measured_net_weight_g: float = Field(gt=0, allow_inf_nan=False)
    method: str = Field(min_length=1, max_length=100)
    estimates: list[PortionEstimate] = Field(default_factory=list)
    allocation_complete: bool
    unallocated_weight_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    review_required: bool
    simulated: bool = False

    @model_validator(mode="after")
    def validate_mass_conservation(self) -> "PortionEstimationResult":
        allocated = sum(item.estimated_weight_g for item in self.estimates)
        if self.allocation_complete:
            if not self.estimates or self.unallocated_weight_g not in (None, 0):
                raise ValueError("complete allocation needs food estimates and no unallocated mass")
            if abs(allocated - self.measured_net_weight_g) > 0.05:
                raise ValueError("complete portion estimates must sum to measured net weight")
        elif self.unallocated_weight_g is not None:
            if abs(allocated + self.unallocated_weight_g - self.measured_net_weight_g) > 0.05:
                raise ValueError("allocated plus unallocated mass must equal measured net weight")
        if not self.simulated and any(item.simulated for item in self.estimates):
            raise ValueError("simulated estimates cannot be mixed into a real result")
        return self


class AgentDecision(StrictModel):
    action: AgentAction
    rationale: str = Field(min_length=1, max_length=500)
    requires_user: bool = False
    retryable: bool = False


class NutritionRecord(StrictModel):
    food_identity: str = Field(min_length=1, max_length=160)
    preparation_state: str = Field(min_length=1, max_length=120)
    kcal_per_100g: float = Field(ge=0, allow_inf_nan=False)
    protein_g_per_100g: float = Field(ge=0, allow_inf_nan=False)
    carbohydrates_g_per_100g: float = Field(ge=0, allow_inf_nan=False)
    fat_g_per_100g: float = Field(ge=0, allow_inf_nan=False)
    source: str = Field(min_length=1, max_length=300)
    version: str = Field(min_length=1, max_length=100)
    class_id: int | None = Field(default=None, ge=1, le=103)
    verified: bool = False


class NutrientEstimate(StrictModel):
    kcal: float = Field(ge=0, allow_inf_nan=False)
    protein_g: float = Field(ge=0, allow_inf_nan=False)
    carbohydrates_g: float = Field(ge=0, allow_inf_nan=False)
    fat_g: float = Field(ge=0, allow_inf_nan=False)
    estimated: bool = True
    source: str
    version: str


class FoodClassMapping(StrictModel):
    class_id: int = Field(ge=0, le=103)
    label: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def background_is_not_food(self) -> "FoodClassMapping":
        if self.class_id == 0 and self.label.casefold() != "background":
            raise ValueError("FoodSeg103 class 0 must remain background")
        return self


class PlateProfileCreate(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    empty_plate_weight_g: float = Field(gt=0, le=5000, allow_inf_nan=False)
    make_default: bool = False


class PlateProfile(StrictModel):
    id: str
    name: str
    empty_plate_weight_g: float
    reference_image_path: str | None = None
    is_default: bool
    created_at: datetime
    updated_at: datetime


class PlateDefaultRequest(StrictModel):
    plate_profile_id: str = Field(min_length=1, max_length=64)


class ScaleStatusResponse(StrictModel):
    configured: bool
    status: str
    source: Literal["hx711", "mock"] | None = None
    simulated: bool = False
    sensor_connected: bool = False
    stable: bool = False
    tared: bool = False
    calibrated: bool = False
    unit: Literal["g"] = "g"
    weight_g: float | None = Field(default=None, allow_inf_nan=False)
    state: ScanState = ScanState.IDLE
    last_error: str | None = None


class TareConfigurationResponse(StrictModel):
    mode: TareMode
    live_tare_active: bool
    empty_plate_weight_g: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    live_tare_at: datetime | None = None
