# Automatic whole-plate scan flow

## Workflow

The user places a plate containing any number of visible foods. The backend polls the ESP32 scale, waits for a stable reading, applies the configured tare strategy, waits through the monitor's settling window, and then captures a phone snapshot. The backend checks image format, size, exposure, sharpness, and orientation; it selects a usable frame, runs segmentation and portion estimation, evaluates the nutrition policy, and stores a meal or review draft in SQLite. No item-by-item food addition is required.

## State machine

```text
IDLE
  -> WEIGHT_DETECTED         reading reaches placement threshold
  -> WAITING_FOR_STABILITY   stable samples and settling delay are gathered
  -> CAPTURING               one unique placement is latched
  -> ANALYZING               image quality passes and CV starts
  -> COMPLETED               agent accepts and saves, or
     NEEDS_REVIEW            tare, image, food, portion, or nutrient correction needed
  -> WAITING_FOR_REMOVAL     plate remains present; no repeat capture
  -> IDLE                    stable removal is confirmed
```

The monitor applies configurable thresholds (`placement_threshold_g=20`, `removal_threshold_g=10` by default), consecutive stable confirmations (2), removal confirmations (3), polling interval (0.25 seconds), and post-stability delay (1 second). HX711 readings must already report `stable`; backend confirmations and the additional delay add another guard. Invalid, unstable, overload, disconnected, and missing readings do not trigger a scan. A temporary network failure resets the candidate stability streak without turning the error into a weight.

Each scan has a unique `placement_key`. The database inserts the scan and persisted latch atomically. When the backend restarts with a latched placement, it enters `WAITING_FOR_REMOVAL`, marks an interrupted in-progress scan for review, and does not retrigger on a plate that stayed in place. If removal is confirmed during capture or inference, a cancellation event prevents the in-flight scan from saving a new image or meal.

## Tare modes

### Saved plate profile

The empty platform must first be tared at the HX711. Register a default plate profile with its measured empty mass. The stable reading includes the plate and food:

```text
net_food_weight_g = measured_gross_weight_g - saved_plate_weight_g
```

If no default profile exists, or gross weight is less than or equal to the profile weight, the agent requests plate registration/correction. It does not create negative mass or take a photo for an invalid net result.
If a profile is wrong but still lighter than the loaded gross weight, weight arithmetic alone cannot detect that bias; re-weigh the empty plate and correct the profile.

### Live plate tare

Tare the empty platform, calibrate the scale, then place a known-empty plate. The user explicitly confirms the tare request. The backend records the stable pre-tare plate reading, asks the ESP32 to tare, and persists `live_plate` mode. Subsequent scale readings are already net food mass; the backend does not subtract the plate again. A live tare is never issued automatically. Food must not be on the plate at the time of this explicit operation. When changing back to an empty-platform tare, the service requires the live-tared plate's negative stable reading to confirm removal.

If the scale is not yet tared, `platform_empty` can issue the initial explicit tare only when the HX711 is connected and the caller confirms the platform is empty. An empty plate live tare requires a stable calibrated reading. After a live-tared scan, a near-zero reading means the empty plate may still be on the scale; the monitor re-arms only after stable negative weight is consistent with the recorded empty-plate tare mass. This prevents removing food while leaving the plate from triggering another scan.

## Camera and analysis coordination

The capture call occurs only after the weight monitor emits the stable-placement event and settling interval. The camera adapter supports:

- `mock`: synthetic plate image, always marked simulated;
- `file`: one configured test image, marked simulated;
- `ip_snapshot`: an HTTP(S) snapshot URL on a private/loopback/`.local` host for an Android phone camera app.

The browser is not assumed to remotely invoke the phone camera. The IP adapter performs a bounded snapshot request, blocks public hosts and public redirects, checks content size, decodes the image, applies EXIF orientation and converts to RGB. Configurable defaults are 8 MiB maximum, 320×240 minimum, 4096 maximum dimension, mean brightness 18–238, maximum clipped-pixel fraction 0.65, and edge-variance minimum 4. The camera tries the configured small burst and bounded retries; a failed final capture becomes `NEEDS_REVIEW` and can be recaptured only while the same stable plate remains on the scale.

Only the selected image, semantic class-ID PNG, and annotated JPEG are retained for a scan under the runtime data directory. There is no continuous video storage.

## Multi-food vision contract

The real SegFormer adapter is lazy, local-only, CPU-default, and validates the checkpoint metadata, label mapping, class count, padding label, and preprocessing metadata before inference. It outputs semantic class IDs `0..103`; background class 0 is excluded from food predictions. Each predicted food class reports its ID/name, per-pixel probability summary, visible pixel area, connected-region count, mask URL, and separate coarse segmentation-quality marker. Multiple disconnected regions of one class share one semantic class prediction; they are not guaranteed to be separate food instances.

The mock adapter returns Rice, Chicken, and Fries labels for deterministic API-path tests, all with `simulated=true`. These are fixture outputs, not FoodSeg103 mapping claims and not model accuracy.

## Review and failure states

The agent requests a plate tare for missing/inconsistent tare; a capture retry for unacceptable image quality; food review for low classification confidence; preparation/mapping input for missing or ambiguous nutritional data; and portion review for uncertain or incomplete masses. A review draft is still stored with the measured total and all estimates that are available. Missing nutrient values and totals remain null. After a user correction, the service recalculates nutrition against the corrected masses and verified mappings.

See `docs/api_reference.md` for calls and `tests/test_automatic_scan_api.py`, `tests/test_scale_service.py`, and `tests/test_scan_workflow.py` for synthetic coverage.
