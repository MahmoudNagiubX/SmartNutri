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

## BE-01 v2 whole-plate workflow (current backend contract)

The sequential-addition design above is retained as historical context only. BE-01 v2 replaces that workflow with automatic whole-plate analysis: one stable plate placement triggers one overhead RGB capture, and semantic segmentation returns every visible FoodSeg103 class in the image. A plate may contain several ingredients before the first capture; no per-food addition events are required.

The backend uses either a saved empty-plate profile (`net_food_weight_g = gross_weight_g - saved_plate_weight_g`) or an explicitly user-confirmed live tare performed while the plate is empty (`net_food_weight_g = post-tare reading`). The same tare is never subtracted twice. Without a known tare, the system requests plate registration. A reading is eligible only when it is stable, in grams, finite, and from a connected/tared/calibrated HX711, except that simulated test readings are separately marked and never presented as physical measurements.

Semantic class IDs remain `1..103` for food and `0` for background. FoodSeg103 `id2label` and checkpoint metadata are loaded locally for real inference. Classification confidence, segmentation quality, and portion-estimation uncertainty are distinct values. The initial visual-area portion baseline is experimental, requires review, and does not claim that pixel area equals mass. Nutrition estimates require an operator-maintained verified per-100 g identity/preparation/source/version record; missing or ambiguous mappings yield no invented totals.

See [automatic_scan_flow.md](automatic_scan_flow.md), [portion_estimation.md](portion_estimation.md), [ai_agent.md](ai_agent.md), and [api_reference.md](api_reference.md) for the implemented state machine, review policy, and routes.

## AUDIT-02 placement snapshot and device security

Every accepted placement scan persists its unique `placement_key`, firmware `device_session_id`, `measurement_id`, device `timestamp_ms`, stable placement weight, and backend receive time. The camera capture is tagged with that same placement key. Before capture, after capture, after inference, and before persisting the analysis, the backend requires a new stable sample from the same device session within the configured weight tolerance. Scale disconnects, repeated/stale samples, unstable or overload states, device reboot, confirmed removal, weight changes beyond tolerance, or placement mismatch reject the scan. `verified_measurement_id`, `verified_weight_g`, `verified_timestamp_ms`, and `verified_received_at` expose the most recent successful revalidation. Revalidation parameters are software defaults only and have not been physically calibrated.

The ESP32 JSON reading includes a random boot `device_session_id` and a monotonically increasing sample `measurement_id`. `GET /weight` remains read-only and unauthenticated. ESP32 `POST /tare` and `/calibrate` require `X-SmartNutri-Device-Token` plus exact confirmations (`EMPTY` and `CALIBRATE`); a missing device secret fails closed. The backend additionally requires its distinct `X-SmartNutri-Control-Token` for real hardware writes and validates an operator's tare mode (`EMPTY_PLATFORM`, `PREPARE`, or `EMPTY_PLATE`) before forwarding a tare. Neither secret belongs in a URL, response, or committed file. The firmware uses plain HTTP: use a trusted, firewalled LAN only and never expose or port-forward the device.

Phone snapshot mode requires both `SMARTNUTRI_CAMERA_URL` and an exact `SMARTNUTRI_CAMERA_ALLOWED_HOST` value containing a private RFC1918 IPv4 address and port. The URL's host/port must match; DNS names, public hosts, redirects, and environment proxies are disabled. Snapshot responses are byte-limited and then decoded/validated as supported images. These restrictions support the Android phone's private-LAN snapshot endpoint; configure a reserved private address and do not expose the camera service publicly.
