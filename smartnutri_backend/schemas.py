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
    device_session_id: str = Field(min_length=1, max_length=80)
    measurement_id: str = Field(min_length=1, max_length=120)
    weight_g: float | None = Field(default=None, allow_inf_nan=False)
    weight_delta_g: float | None = Field(default=None, allow_inf_nan=False)
    stability_range_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    timestamp_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_source_and_status(self) -> "ScaleObservation":
        if (self.source == "mock") != self.simulated:
            raise ValueError("mock readings must be simulated and HX711 readings must be real")
        if self.status == "stable":
            real_ready = (
                self.source == "hx711"
                and self.sensor_connected
                and self.tared
                and self.calibrated
            )
            simulated_ready = self.source == "mock" and self.simulated
            if not (self.ok and (real_ready or simulated_ready) and self.stable and self.weight_g is not None):
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
    source_identifier: str | None = Field(default=None, min_length=1, max_length=160)
    version: str = Field(min_length=1, max_length=100)
    class_id: int | None = Field(default=None, ge=1, le=103)
    verified: bool = False
    requires_preparation_confirmation: bool = False

    @model_validator(mode="after")
    def verified_records_need_traceable_source(self) -> "NutritionRecord":
        if self.verified and not self.source_identifier:
            raise ValueError("verified nutrition records require a source identifier")
        return self


class NutrientEstimate(StrictModel):
    kcal: float = Field(ge=0, allow_inf_nan=False)
    protein_g: float = Field(ge=0, allow_inf_nan=False)
    carbohydrates_g: float = Field(ge=0, allow_inf_nan=False)
    fat_g: float = Field(ge=0, allow_inf_nan=False)
    estimated: bool = True
    source: str
    source_identifier: str | None = None
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
    reference_image_path: str | None = Field(default=None, max_length=500)


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
    live_tare_pending: bool = False
    empty_plate_weight_g: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    live_tare_at: datetime | None = None


class TareRequest(StrictModel):
    mode: Literal["platform_empty", "empty_plate", "prepare_empty_plate"]
    confirm: Literal["EMPTY_PLATFORM", "EMPTY_PLATE", "PREPARE"]

    @model_validator(mode="after")
    def require_action_specific_confirmation(self) -> "TareRequest":
        expected = {
            "prepare_empty_plate": "PREPARE",
            "empty_plate": "EMPTY_PLATE",
            "platform_empty": "EMPTY_PLATFORM",
        }[self.mode]
        if self.confirm != expected:
            raise ValueError(f"mode {self.mode} requires confirm={expected}")
        return self


class ScaleCalibrationRequest(StrictModel):
    known_weight_g: float = Field(gt=0, le=5000, allow_inf_nan=False)
    confirm: Literal["CALIBRATE"]


class MockPlacementRequest(StrictModel):
    gross_weight_g: float = Field(gt=0, le=5000, allow_inf_nan=False)


class MealItemCorrection(StrictModel):
    class_id: int = Field(ge=1, le=103)
    label: str = Field(min_length=1, max_length=120)
    weight_g: float = Field(gt=0, allow_inf_nan=False)
    nutrition_record_id: str | None = Field(default=None, min_length=1, max_length=64)
    preparation_state: str | None = Field(default=None, min_length=1, max_length=120)


class MealCorrectionRequest(StrictModel):
    items: list[MealItemCorrection] = Field(min_length=1, max_length=104)


class MealItemResponse(StrictModel):
    id: str
    meal_id: str
    class_id: int | None = Field(default=None, ge=1, le=103)
    food_label: str
    classification_confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    pixel_area: int | None = Field(default=None, ge=0)
    region_count: int | None = Field(default=None, ge=0)
    estimated_weight_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    portion_method: str | None = None
    portion_confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    portion_uncertainty_g: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    nutrition_record_id: str | None = None
    nutrients: NutrientEstimate | None = None
    user_corrected: bool
    needs_review: bool
    created_at: datetime


class MealResponse(StrictModel):
    id: str
    scan_id: str
    state: str
    measured_net_food_weight_g: float = Field(gt=0, allow_inf_nan=False)
    nutrition_estimated: bool
    kcal_total: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    protein_g_total: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    carbohydrates_g_total: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    fat_g_total: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    correction_revision: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    simulated: bool
    items: list[MealItemResponse] = Field(default_factory=list)


class ScanResponse(StrictModel):
    id: str
    placement_key: str
    state: ScanState
    decision: AgentAction | None = None
    source: Literal["hx711", "mock"]
    simulated: bool
    device_session_id: str | None = None
    measurement_id: str | None = None
    measurement_timestamp_ms: int | None = Field(default=None, ge=0)
    stable_weight_g: float | None = Field(default=None, allow_inf_nan=False)
    measurement_received_at: datetime | None = None
    verified_measurement_id: str | None = None
    verified_weight_g: float | None = Field(default=None, allow_inf_nan=False)
    verified_timestamp_ms: int | None = Field(default=None, ge=0)
    verified_received_at: datetime | None = None
    capture_placement_key: str | None = None
    tare_mode: TareMode | None = None
    plate_profile_id: str | None = None
    measured_gross_weight_g: float | None = Field(default=None, allow_inf_nan=False)
    plate_tare_weight_g: float | None = Field(default=None, allow_inf_nan=False)
    measured_net_food_weight_g: float | None = Field(default=None, allow_inf_nan=False)
    capture_source: str | None = None
    capture_simulated: bool | None = None
    image_quality: ImageQuality | None = None
    predictions: list[FoodPrediction] | None = None
    portions: PortionEstimationResult | None = None
    nutrition_totals: NutrientEstimate | None = None
    result_message: str | None = None
    error_code: str | None = None
    created_at: datetime
    updated_at: datetime
    capture_url: str | None = None
    annotated_url: str | None = None
    mask_url: str | None = None
    meal: MealResponse | None = None


class HealthResponse(StrictModel):
    status: str
    service: str
    version: str
    database: str
    auto_scan_enabled: bool
    scale_configured: bool
    camera_mode: str
    segmentation_mode: str
    scan_state: ScanState


class ScaleReadingResponse(StrictModel):
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
    stability_threshold_g: float = Field(ge=0, allow_inf_nan=False)
    sample_count: int = Field(ge=0)
    window_size: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    device_session_id: str = Field(min_length=1, max_length=80)
    measurement_id: str = Field(min_length=1, max_length=120)
    operation: str = "none"
    operation_status: str = "idle"
    operation_ok: bool | None = None
    operation_message: str = "idle"


class ScaleCalibrationResponse(StrictModel):
    reading: ScaleReadingResponse
    simulated: bool


class CameraStatusResponse(StrictModel):
    configured: bool
    source: str
    simulated: bool
    transport: str | None = None
    reachable: bool | None = None
    file_available: bool | None = None


class FoodPredictionsResponse(StrictModel):
    scan_id: str
    predictions: list[FoodPrediction]
    source: str
    simulated: bool


class PortionResponse(StrictModel):
    scan_id: str
    portions: PortionEstimationResult | None
    simulated: bool


class ScanNutritionResponse(StrictModel):
    scan_id: str
    measured_net_food_weight_g: float | None = Field(default=None, allow_inf_nan=False)
    totals: NutrientEstimate | None = None
    meal: MealResponse | None = None
    estimated: bool
    simulated: bool


class NutritionRecordResponse(NutritionRecord):
    id: str
    created_at: datetime


class AgentDecisionResponse(StrictModel):
    scan_id: str
    action: AgentAction | None = None
    rationale: str | None = None
    state: ScanState
    simulated: bool


class MockPlacementResponse(StrictModel):
    scan: ScanResponse
    simulated: Literal[True]
    simulation_notice: str
