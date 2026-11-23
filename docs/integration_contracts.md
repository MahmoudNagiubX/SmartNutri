# SmartNutri vision, scale, and nutrition contracts

These contracts describe the boundary for a future laptop/backend Nutrition Agent. They do not claim that a FastAPI service, model-serving endpoint, nutrition database, agent, or meal-state store exists today. Current repository code trains/evaluates/visualizes SegFormer offline and provides a separate ESP32 scale client.

## Food prediction contract (future CV service)

One prediction represents a food candidate from one camera frame or crop:

| Field | Type / rule |
| --- | --- |
| `prediction_id` | Unique string for this inference result. |
| `frame_id` | String identifying the smartphone-camera frame/crop. |
| `class_id` | Integer `1..103`; `0` is FoodSeg103 background and is not a food item. |
| `label` | Exact name looked up from FoodSeg103's `id2label` mapping for `class_id`. |
| `confidence` | Finite float in `[0, 1]`; the serving implementation must define whether this is pixel, region, or crop confidence. |
| `mask_ref` | Optional reference to the corresponding segmentation mask/region. Do not send a large mask inline unless the API explicitly supports it. |
| `created_at` | UTC timestamp in ISO-8601 form. |

The current SegFormer scripts produce semantic class predictions for offline evaluation and visualization. They do not expose this inference contract or a calibrated confidence value. Do not infer a confidence threshold from the current segmentation metrics; choose one using held-out/validation evidence and document the aggregation method.

## Scale reading contract (implemented firmware API)

The ESP32 `GET /weight` response is parsed by `smartnutri_weight.WeightReading`. A reading may be used as a physical mass only when all are true:

- `source == "hx711"` and `simulated == false`;
- `sensor_connected`, `tared`, and `calibrated` are true;
- `status == "stable"`, `stable == true`, `ok == true`;
- `unit == "g"` and `weight_g` is finite and non-null.

The Python client exposes `get_stable_weight()` to wait for that condition. It fails on mock readings, sensor errors, overload, disconnected, un-tared, or un-calibrated states. The signed `weight_delta_g` is change from the immediately previous accepted scale sample; it is not automatically an ingredient's portion mass.

## Sequential ingredient addition contract (future agent/backend)

Track each addition as its own event. A meal item should retain at least:

| Field | Type / rule |
| --- | --- |
| `item_id` | Unique string within the meal. |
| `prediction_id` | Reference to the accepted food prediction. |
| `class_id`, `label` | FoodSeg103 ID/name pair; must match the mapping. |
| `confidence` | The prediction confidence and its aggregation method. |
| `confirmation` | `not_required` or `user_confirmed`; low-confidence acceptance must record explicit user confirmation. |
| `baseline_weight_g` | Stable real scale reading immediately before the single addition. |
| `final_weight_g` | Stable real scale reading after the addition settles. |
| `portion_weight_g` | `final_weight_g - baseline_weight_g`; finite and positive before recording. |
| `scale_timestamps` | Timestamps from the baseline and final readings. |
| `nutrient_source`, `nutrient_version` | Traceable source/version for validated nutrition data. |

Expected event order: capture a baseline stable weight; identify the candidate ingredient from the camera; request user confirmation if the serving policy considers its confidence low; ask the user to add one ingredient; wait for a new stable reading; compute the difference between the two stable readings; reject non-positive or invalid differences; associate that difference with the accepted ingredient; then advance the baseline for the next addition. `weight_delta_g` alone is insufficient if frames/requests arrive between the baseline and final readings.

## Nutrition contract (future data lookup and meal state)

Nutrient records must be keyed by a validated food identity and report values per 100 g with an explicit source, version/date, and units:

- energy in `kcal_per_100g`;
- protein, carbohydrate, and fat in `g_per_100g`;
- optional fiber and other nutrients only with the same explicit unit convention.

For an accepted item of `portion_weight_g`, compute each nutrient as `per_100g_value * portion_weight_g / 100`. Store the source record and calculation result with the item so later corrections are auditable. FoodSeg103 labels are visual classes, not guaranteed one-to-one matches to a nutrition database; ambiguous ingredients require user selection or a documented mapping.

## FastAPI boundary recommendation

When a FastAPI service is added, validate incoming/outgoing objects at the API boundary and keep responsibilities separate:

1. CV inference returns `FoodPrediction` candidates and mask references.
2. The scale client returns validated `WeightReading` objects; only stable, non-simulated readings are eligible for meal changes.
3. The agent applies confidence/confirmation and one-addition sequencing rules.
4. A nutrition repository supplies validated per-100 g values.
5. Meal state stores the selected prediction, associated portion mass, source record, and calculated nutrients.

No agent decision, nutrition lookup, calibration, or meal mutation should run on the ESP32. The camera, ring light, scale, and laptop have independent power/data roles; this contract does not specify a camera transport or model-serving endpoint.
