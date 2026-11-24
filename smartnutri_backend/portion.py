from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from PIL import Image

from smartnutri_backend.schemas import FoodPrediction, PortionEstimate, PortionEstimationResult


class PortionEstimator(Protocol):
    def estimate(
        self,
        measured_net_weight_g: float,
        predictions: Sequence[FoodPrediction],
        *,
        semantic_mask: Image.Image | None = None,
        unidentified_area_px: int = 0,
        prior_weights_g: Mapping[int, float] | None = None,
        model_estimates_g: Mapping[int, float] | None = None,
        simulated: bool | None = None,
    ) -> PortionEstimationResult: ...


class ExperimentalVisualProportionEstimator:
    """Uncalibrated visual-area baseline for experiments only; it requires user review."""

    method_name = "experimental_visual_area_proportion_v1"

    def estimate(
        self,
        measured_net_weight_g: float,
        predictions: Sequence[FoodPrediction],
        *,
        semantic_mask: Image.Image | None = None,
        unidentified_area_px: int = 0,
        prior_weights_g: Mapping[int, float] | None = None,
        model_estimates_g: Mapping[int, float] | None = None,
        simulated: bool | None = None,
    ) -> PortionEstimationResult:
        if isinstance(measured_net_weight_g, bool) or not math.isfinite(measured_net_weight_g):
            raise ValueError("measured net food weight must be finite")
        if measured_net_weight_g <= 0:
            raise ValueError("measured net food weight must be positive")
        if unidentified_area_px < 0:
            raise ValueError("unidentified area cannot be negative")
        if prior_weights_g is not None or model_estimates_g is not None:
            raise NotImplementedError(
                "this experimental baseline does not consume learned priors or model estimates"
            )

        by_class: dict[int, FoodPrediction] = {}
        for prediction in predictions:
            if prediction.class_id in by_class:
                raise ValueError(f"duplicate semantic class {prediction.class_id}")
            by_class[prediction.class_id] = prediction

        areas = {class_id: prediction.pixel_area for class_id, prediction in by_class.items()}
        if semantic_mask is not None:
            if semantic_mask.mode != "L":
                raise ValueError("semantic mask must be a single-channel class-ID image")
            histogram = semantic_mask.histogram()
            if any(histogram[class_id] for class_id in range(104, 255)):
                raise ValueError("semantic mask contains values outside FoodSeg103 IDs 0..103")
            for class_id in by_class:
                areas[class_id] = histogram[class_id]

        by_class = {
            class_id: prediction
            for class_id, prediction in by_class.items()
            if areas[class_id] > 0
        }
        areas = {class_id: areas[class_id] for class_id in by_class}

        if not by_class or sum(areas.values()) <= 0:
            simulation = bool(simulated) or any(item.simulated for item in predictions)
            return PortionEstimationResult(
                measured_net_weight_g=measured_net_weight_g,
                method="unavailable",
                estimates=[],
                allocation_complete=False,
                unallocated_weight_g=round(measured_net_weight_g, 2),
                review_required=True,
                simulated=simulation,
            )

        known_area = sum(areas.values())
        total_visible_area = known_area + unidentified_area_px
        if total_visible_area <= 0:
            raise ValueError("visible food area must be positive")
        is_simulated = bool(simulated) or any(item.simulated for item in predictions)
        unallocated_weight = (
            round(measured_net_weight_g * unidentified_area_px / total_visible_area, 2)
            if unidentified_area_px
            else 0.0
        )
        known_weight = measured_net_weight_g - unallocated_weight
        ordered = sorted(by_class.values(), key=lambda item: item.class_id)
        target_cents = max(0, int(round(known_weight * 100)))
        raw_cents = {
            item.class_id: target_cents * areas[item.class_id] / known_area
            for item in ordered
        }
        portion_cents = {class_id: math.floor(value) for class_id, value in raw_cents.items()}
        cents_left = target_cents - sum(portion_cents.values())
        for class_id in sorted(raw_cents, key=lambda value: raw_cents[value] - portion_cents[value], reverse=True)[:cents_left]:
            portion_cents[class_id] += 1
        estimates: list[PortionEstimate] = []
        for prediction in ordered:
            weight_g = portion_cents[prediction.class_id] / 100.0
            uncertainty = round(max(weight_g * 0.5, measured_net_weight_g * 0.25), 2)
            estimates.append(
                PortionEstimate(
                    class_id=prediction.class_id,
                    label=prediction.label,
                    estimated_weight_g=max(0.0, weight_g),
                    method=self.method_name,
                    confidence=0.25,
                    uncertainty_g=uncertainty,
                    review_required=True,
                    simulated=is_simulated,
                )
            )

        complete = unidentified_area_px == 0
        return PortionEstimationResult(
            measured_net_weight_g=measured_net_weight_g,
            method=self.method_name,
            estimates=estimates,
            allocation_complete=complete,
            unallocated_weight_g=None if complete else unallocated_weight,
            review_required=True,
            simulated=is_simulated,
        )
