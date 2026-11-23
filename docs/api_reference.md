# SmartNutri Backend API v1 — React dashboard contract

Base URL for the default local process: `http://127.0.0.1:8000`. The launcher binds to loopback by default. For a phone or browser on trusted Wi-Fi, set `SMARTNUTRI_API_HOST` to the laptop's private IPv4 address and scope the host firewall. Responses use JSON except image resources. CORS is disabled by default; configure explicit browser origins for a React development server. Image requests are capped at 8 MiB by default and checked for format, dimensions, brightness, clipping, and sharpness before analysis. FastAPI publishes typed schemas at `/openapi.json`.

The API supports a small mobile-first dashboard. The UI can poll scale and scan status, show the selected image and segmentation, and review or correct a saved meal. It does not orchestrate segmentation or capture before the scale has stabilized.

## Scale, tare, and plates

| Method and path | Purpose |
| --- | --- |
| `GET /api/v1/health` | Service/database health, modes, scan state, auto-scan setting. |
| `GET /api/v1/scale/status` | Latest validated scale state; mock readings expose `simulated=true`. |
| `GET /api/v1/scale/tare-config` | Persisted `saved_plate` or `live_plate` configuration. |
| `POST /api/v1/scale/tare` | Explicitly confirmed initial platform tare, live-tare preparation, or empty-plate live tare. |
| `POST /api/v1/scale/calibrate` | Explicit known-weight hardware calibration; real scale only. |
| `GET /api/v1/plates` | List profiles. |
| `POST /api/v1/plates` | Register an empty plate. |
| `GET /api/v1/plates/default` | Read selected profile. |
| `PUT /api/v1/plates/default` | Select a profile with `{ "plate_profile_id": "..." }`. |

Plate creation may include an optional `reference_image_path`, but it must name an existing JPEG/PNG/WEBP inside the backend runtime image directory and remain under the configured image-size limit. No arbitrary filesystem path or upload is accepted.

Example saved plate registration:

```json
POST /api/v1/plates
{
  "name": "Ceramic dinner plate",
  "empty_plate_weight_g": 300,
  "make_default": true
}
```

Example explicit live plate tare:

```json
POST /api/v1/scale/tare
{
  "mode": "empty_plate",
  "confirm": "TARE"
}
```

For two-step live tare, first prepare while the platform is empty, then place an empty plate and explicitly tare it:

```json
POST /api/v1/scale/tare
{ "mode": "prepare_empty_plate", "confirm": "PREPARE" }

POST /api/v1/scale/tare
{ "mode": "empty_plate", "confirm": "TARE" }
```

`mode` is `platform_empty`, `prepare_empty_plate`, or `empty_plate`. A human must ensure the platform or plate is actually empty before confirming. Preparation persists `live_tare_pending=true` and suppresses scans while the empty plate is placed. The empty-plate action requires a stable calibrated reading, records the plate's pre-tare mass, and asks the ESP32 to tare. Subsequent readings are net food mass; the backend does not subtract the plate again. Platform-empty tare can establish the initial zero for a connected but not-yet-tared HX711. Calibration requires `{ "known_weight_g": 500, "confirm": "CALIBRATE" }`; it is never called automatically.

## Automatic scan, segmentation, and portions

| Method and path | Purpose |
| --- | --- |
| `GET /api/v1/camera/status` | Current mock, file, phone snapshot, or unconfigured camera status. |
| `GET /api/v1/scans/active` | Current placement scan, if latched. |
| `GET /api/v1/scans?limit=50` | Recent scans, maximum 200. |
| `GET /api/v1/scans/{scan_id}` | Scan result and persisted meal draft/result. |
| `GET /api/v1/scans/{scan_id}/foods` | FoodSeg class predictions, region/pixel stats, confidence, source. |
| `GET /api/v1/scans/{scan_id}/portions` | Per-food estimated grams, method, uncertainty, review/simulation markers. |
| `GET /api/v1/scans/{scan_id}/nutrition` | Measured net weight, estimated totals, and associated meal. |
| `GET /api/v1/scans/{scan_id}/mask.png` | Semantic PNG mask; each pixel retains class ID `0..103`. |
| `GET /api/v1/scans/{scan_id}/mask/{class_id}` | Binary mask for one food class `1..103`. |
| `GET /api/v1/scans/{scan_id}/capture.jpg` | Selected RGB capture. |
| `GET /api/v1/scans/{scan_id}/annotated.jpg` | Display overlay for the segmentation output. |
| `POST /api/v1/scans/{scan_id}/recapture` | Re-capture only a scan awaiting camera retry while same stable plate remains. |
| `POST /api/v1/camera/capture/{scan_id}` | Alias of the guarded scan recapture action. |
| `POST /api/v1/scans/mock-placement` | Mock-only automatic-pipeline demonstration. |

Example mock request:

```json
POST /api/v1/scans/mock-placement
{ "gross_weight_g": 750 }
```

This endpoint returns `simulated=true` and a notice that all scale values are synthetic. In saved-plate mode, a test profile at 300 g yields a synthetic 450 g net mass. In live-plate mode the `gross_weight_g` request field represents the simulated post-tare scale reading, so the returned net weight is that reading and the plate tare is not subtracted again. The mock endpoint is absent when `SMARTNUTRI_SCALE_MODE` is not `mock`.

## Nutrition and meals

| Method and path | Purpose |
| --- | --- |
| `GET /api/v1/nutrition/records?class_id=66` | Read the seeded catalog and operator-added records, optionally filtered by class. |
| `POST /api/v1/nutrition/records` | Add a food/preparation record with per-100 g macros, source identifier, source, and version. |
| `GET /api/v1/meals?limit=50` | Recent saved and review-draft meals. |
| `GET /api/v1/meals/{meal_id}` | One meal, measured total, food estimates/corrections, and nutrient values. |
| `POST /api/v1/meals/{meal_id}/corrections` | Replace per-food portion weights and recalculate from verified mappings. |
| `GET /api/v1/agents/decisions/{scan_id}` | Agent action, rationale, state, and scale simulation marker. |

Nutrition record example (only use after verifying the source data independently):

```json
POST /api/v1/nutrition/records
{
  "class_id": 66,
  "food_identity": "verified food identity",
  "preparation_state": "documented preparation",
  "kcal_per_100g": 0,
  "protein_g_per_100g": 0,
  "carbohydrates_g_per_100g": 0,
  "fat_g_per_100g": 0,
  "source": "publisher",
  "source_identifier": "publisher record ID",
  "version": "release/date",
  "verified": true,
  "requires_preparation_confirmation": true
}
```

These zero values illustrate schema units only; they are not SmartNutri food records. Startup idempotently seeds the USDA SR Legacy catalog documented in [nutrition catalog](nutrition_catalog.md). A verified item is calculated as `per_100g × estimated_portion_g / 100`; all nutrient values remain marked estimated and retain source/source identifier/version. Raw and cooked foods are separate. FoodSeg103 class 48 is the joint “chicken duck” label, so a prediction alone cannot identify chicken versus duck. Preparation-sensitive records require explicit user choice. Missing or ambiguous mappings return null item nutrients. Meal totals remain null when a food mapping is missing or when portion allocation leaves measured mass unidentified.

Correction example:

The following 450 g portion values are illustrative correction inputs. Resolve the record IDs first using `GET /api/v1/nutrition/records?class_id=66` and `?class_id=48`; they are generated by the local database and differ between installations.

```json
POST /api/v1/meals/{meal_id}/corrections
{
  "items": [
    { "class_id": 66, "label": "rice", "weight_g": 150, "nutrition_record_id": "<FDC-168878-local-record-id>" },
    { "class_id": 48, "label": "chicken breast, meat only", "weight_g": 200, "nutrition_record_id": "<FDC-171477-local-record-id>" },
    { "class_id": 3, "label": "french fries", "weight_g": 100 }
  ]
}
```

Corrected weights must be positive, unique by FoodSeg class, and sum to measured net weight within 0.05 g. `nutrition_record_id` selects a verified preparation/source record when class alone cannot identify the food; it must be verified and map to the same class ID (or have no class mapping when manually entered). For chicken/duck class 48, identify the animal and preparation before selecting a matching record. If any verified nutrient mapping is missing, the correction is saved as a review draft and totals remain null.

## Dashboard request sequence

1. Load `/health`, `/scale/status`, `/scale/tare-config`, `/plates`, `/plates/default`, and `/camera/status` under `/api/v1`.
2. Poll scale status and show `status`, `stable`, `weight_g`, `simulated`, and scan `state`. Mark synthetic readings visibly as simulated.
3. Register/select a saved plate, or use the two-step live-tare flow above. Show `live_tare_pending=true` as “place an empty plate, then confirm tare.”
4. Let the monitor scan automatically. Poll `/scans/active`; `null` means there is no active placement scan. Render the scan state, decision, and result message for progress or user actions.
5. Show `capture_url` (or `/scans/{id}/capture.jpg`) and `annotated_url` (or `/scans/{id}/annotated.jpg`). A food prediction includes class ID, label, classification confidence, pixel area, region count, segmentation quality, source, and simulation provenance.
6. Show estimated grams, method, confidence, uncertainty, and review flags from the scan or `/portions`. The experimental area baseline is an estimate, never a measured portion.
7. Display only returned nutrition values. Preserve null totals as a review state; retain `estimated=true` and source provenance.
8. Submit optional corrections to `/meals/{meal_id}/corrections`; use `/meals?limit=50` and `/meals/{meal_id}` for history and detail.

### Core response shapes

The following examples are synthetic contract examples. They demonstrate the `simulated` marker and must be rendered as simulated test data:

```json
GET /api/v1/scale/status
{
  "configured": true,
  "status": "stable",
  "source": "mock",
  "simulated": true,
  "sensor_connected": true,
  "stable": true,
  "tared": true,
  "calibrated": true,
  "unit": "g",
  "weight_g": 750.0,
  "state": "WAITING_FOR_REMOVAL",
  "last_error": null
}
```

```json
GET /api/v1/scans/active
{
  "id": "<scan-id>",
  "placement_key": "<placement-key>",
  "state": "NEEDS_REVIEW",
  "decision": "REVIEW_PORTION_ESTIMATES",
  "source": "mock",
  "simulated": true,
  "tare_mode": "saved_plate",
  "plate_profile_id": "<plate-profile-id>",
  "measured_gross_weight_g": 750.0,
  "plate_tare_weight_g": 300.0,
  "measured_net_food_weight_g": 450.0,
  "capture_source": "mock",
  "capture_simulated": true,
  "created_at": "2026-11-23T12:00:00+00:00",
  "updated_at": "2026-11-23T12:00:01+00:00",
  "capture_url": "/api/v1/scans/<scan-id>/capture.jpg",
  "annotated_url": "/api/v1/scans/<scan-id>/annotated.jpg",
  "mask_url": "/api/v1/scans/<scan-id>/mask.png",
  "predictions": [
    {
      "class_id": 1,
      "label": "rice",
      "classification_confidence": 0.96,
      "pixel_area": 1000,
      "region_count": 1,
      "segmentation_quality": "unverified",
      "source": "mock",
      "simulated": true
    }
  ],
  "portions": {
    "measured_net_weight_g": 450.0,
    "method": "experimental_visual_area_proportion_v1",
    "allocation_complete": true,
    "review_required": true,
    "simulated": true
  }
}
```

The abbreviated scan omits nullable fields and other response members; the complete strict schema is available through `/openapi.json`. In real inference, FoodSeg103 IDs are preserved (`0` is background and not returned as food), and every returned prediction must have `source="segformer"`, `simulated=false`. The UI should use scan `decision`/`state` for review actions and never infer that a complete allocation is physically accurate.

For a React browser client on a separate development origin, set `SMARTNUTRI_CORS_ORIGINS` to the exact origin (for example, `http://localhost:5173`). Do not bundle the control token into a LAN-visible browser app. A trusted local mediation layer is required before a browser UI may call privileged real-hardware controls.

## Control and deployment

Real `POST /api/v1/scale/tare`, `/api/v1/scale/calibrate`, scan/camera recapture, nutrition-record writes, and meal corrections require `X-SmartNutri-Control-Token: <secret>`. Set `SMARTNUTRI_CONTROL_TOKEN` outside the repository; real controls fail closed when it is unset. Mock-mode control calls are explicitly simulated and never actuate hardware. The supported server launcher defaults to `127.0.0.1`; for a trusted Wi-Fi client, set `SMARTNUTRI_API_HOST` to the laptop's explicit RFC1918 IPv4 address and allow only the Android device or trusted home subnet through the Windows Private-profile firewall. Do not bind to `0.0.0.0` or expose the API publicly. CORS is disabled by default; configure `SMARTNUTRI_CORS_ORIGINS` only for an intentional browser client. Do not commit tokens, camera credentials, runtime images, database files, or nutrient-source credentials. See [Flutter Android integration](flutter_android_integration.md) for an existing mobile client integration, and use the contract above for a future React dashboard.

Common environment configuration:

| Variable | Meaning |
| --- | --- |
| `SMARTNUTRI_DATA_DIR` | Runtime database and image root; defaults outside the checkout under `%LOCALAPPDATA%\SmartNutri`. |
| `SMARTNUTRI_SCALE_MODE` | `real`, `mock`, or `disabled`. |
| `SMARTNUTRI_SCALE_URL` | ESP32 base URL for the existing `GET /weight` and `POST /tare` client. |
| `SMARTNUTRI_CAMERA_MODE` | `mock`, `file`, or `ip_snapshot`. |
| `SMARTNUTRI_CAMERA_FILE` / `SMARTNUTRI_CAMERA_URL` | Test-image path or private phone snapshot URL. |
| `SMARTNUTRI_CV_MODE` | `mock` or `segformer`. |
| `SMARTNUTRI_CHECKPOINT_DIR` / `SMARTNUTRI_FOODSEG_ROOT` | Local checkpoint and FoodSeg103 label metadata for real inference. Files are never downloaded by the service. |
| `SMARTNUTRI_AUTO_SCAN` | `true` to start background polling; default is `false`. |
| `SMARTNUTRI_MIN_CLASSIFICATION_CONFIDENCE` | Minimum food-class confidence before area allocation; default is `0.55`. |
| `SMARTNUTRI_PORTION_REVIEW_UNCERTAINTY_FRACTION` | Maximum per-item uncertainty as a fraction of total net weight before agent review; default is `0.20`. |
| `SMARTNUTRI_CONTROL_TOKEN` | Secret required for real write/control operations. |
| `SMARTNUTRI_API_HOST` / `SMARTNUTRI_API_PORT` | Server launcher address and port; defaults to `127.0.0.1:8000`, and non-private/public binds are rejected. |
| `SMARTNUTRI_CORS_ORIGINS` | Optional comma-separated explicit browser origins; empty by default and wildcards are rejected. |

For hardware, use `SMARTNUTRI_SCALE_MODE=real`, set the ESP32 base URL, use the phone app's private snapshot URL, and keep auto-scan disabled until tare, calibration, camera, and local inference have been checked individually. Set `SMARTNUTRI_AUTO_SCAN=true` only after those checks. Real SegFormer mode requires an already-present local FoodSeg103 checkpoint with training metadata; if missing or inconsistent, inference fails closed for review.
