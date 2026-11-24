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

The monitor applies configurable thresholds (`placement_threshold_g=20` minimum net-food grams and `removal_threshold_g=10` by default), consecutive stable confirmations (2), removal confirmations (3), polling interval (0.25 seconds), and post-stability delay (1 second). Active scan verification also uses `SMARTNUTRI_WEIGHT_CHANGE_TOLERANCE_G=5` and `SMARTNUTRI_STALE_READING_TIMEOUT_S=2` by default; these are software safeguards and are not hardware-calibrated. In saved-plate mode with a selected profile, it adds the profile's empty weight to the placement threshold; in live-tare mode the reading is already net and only the configured minimum applies. HX711 readings must already report `stable`; backend confirmations and the additional delay add another guard. Invalid, unstable, overload, disconnected, and missing readings do not trigger a scan. A temporary network failure resets the candidate stability streak before placement; during an active scan it invalidates that snapshot, so the pipeline cannot accept results using an unverified scale state.

Each scan has a unique `placement_key` and initial weight snapshot: the ESP32 boot `device_session_id`, `measurement_id`, device timestamp, stable weight, and backend receive time. Capture is associated with the same placement key. Before capture, after capture, after inference, and immediately before persisting analysis, the backend reads and validates another scale sample. A changed device session, repeated/stale sample, disconnect, sensor error, overload, unstable state, confirmed removal, placement mismatch, or weight delta above `SMARTNUTRI_WEIGHT_CHANGE_TOLERANCE_G` rejects the scan. Successful revalidation stores the latest verified measurement ID, weight, device timestamp, and backend receive time. Defaults are engineering safeguards only; physical tolerances remain unvalidated.

The database inserts the scan and persisted latch atomically. When the backend restarts with a latched placement, it enters `WAITING_FOR_REMOVAL`, marks an interrupted in-progress scan for review, and does not retrigger on a plate that stayed in place. If removal or another invalidating scale state occurs during capture or inference, a cancellation event prevents the in-flight scan from accepting analysis or saving a meal. A confirmed removal is required before another placement can trigger a scan.

## Tare modes

### Saved plate profile

The empty platform must first be tared at the HX711. Register a default plate profile with its measured empty mass. The stable reading includes the plate and food:

```text
net_food_weight_g = measured_gross_weight_g - saved_plate_weight_g
```

If no default profile exists, or gross weight is less than or equal to the profile weight, the agent requests plate registration/correction. It does not create negative mass or take a photo for an invalid net result.
If a profile is wrong but still lighter than the loaded gross weight, weight arithmetic alone cannot detect that bias; re-weigh the empty plate and correct the profile.

In saved-plate mode, placing the empty plate by itself does not trigger a scan. The monitor waits until stable gross weight reaches `saved_plate_weight_g + placement_threshold_g`; if the user adds food while the plate remains on the scale, that stable increase triggers one scan. The full gross reading is passed to the workflow and the saved plate mass is subtracted once. The monitor then remains latched until confirmed removal, preventing repeated scans while the loaded plate stays in place. A wrong or overweight plate profile can hold the trigger threshold too high, so the dashboard must show the selected profile and live gross reading clearly.

### Live plate tare

Tare the empty platform and calibrate the scale, then ask the backend to prepare live tare while the scale is stably empty. The backend persists `live_tare_pending`, suppresses scans, and waits for an empty plate. The user places the empty plate and explicitly confirms `empty_plate`; the backend records its stable pre-tare reading, asks the ESP32 to tare, clears the pending state, and persists `live_plate` mode. Subsequent scale readings are already net food mass; the backend does not subtract the plate again. A live tare is never issued automatically. Food must not be on the plate at the time of this explicit operation. When changing back to an empty-platform tare, the service requires the live-tared plate's negative stable reading to confirm removal.

If the scale is not yet tared, `platform_empty` can issue the initial explicit tare only when the HX711 is connected and the caller confirms the platform is empty. An empty plate live tare requires a stable calibrated reading. After a live-tared scan, a near-zero reading means the empty plate may still be on the scale; the monitor re-arms only after stable negative weight is consistent with the recorded empty-plate tare mass. This prevents removing food while leaving the plate from triggering another scan.

## Camera and analysis coordination

The capture call occurs only after the weight monitor emits the stable-placement event and settling interval, with a fresh scale revalidation immediately before capture. The camera adapter supports:

- `mock`: synthetic plate image, always marked simulated;
- `file`: one configured test image, marked simulated;
- `ip_snapshot`: an HTTP snapshot URL whose literal RFC1918 IPv4 address and port exactly match `SMARTNUTRI_CAMERA_ALLOWED_HOST` for an Android phone camera app. Public, loopback, link-local, multicast, DNS, `.local`, and other unapproved hosts are rejected; redirects and environment proxies are disabled.

The browser is not assumed to remotely invoke the phone camera. Configure a reserved phone IP and exact snapshot port; never include credentials in the URL. Snapshot responses are byte-limited and checked for image content before decoding; EXIF orientation is applied and output is converted to RGB. Configurable defaults are 8 MiB maximum, 320x240 minimum, 4096 maximum dimension, mean brightness 18-238, maximum clipped-pixel fraction 0.65, and edge-variance minimum 4. The camera tries the configured small burst and bounded retries; a failed final capture becomes NEEDS_REVIEW and can be recaptured only while the same stable plate remains on the scale.

The phone adapter and ESP32 use plain HTTP on the private LAN. HTTP provides no transport encryption or server authentication. Restrict devices with router/firewall rules and do not expose or port-forward either service. `SMARTNUTRI_CONTROL_TOKEN` protects backend writes; the separate `SMARTNUTRI_SCALE_DEVICE_TOKEN` protects ESP32 tare/calibration. Missing device credentials fail closed. Tare requires an operator-confirmed empty platform/plate and is never triggered automatically from weight detection.

The browser is not assumed to remotely invoke the phone camera. Configure a reserved phone IP and exact snapshot port; never include credentials in the URL. Snapshot responses are byte-limited and checked for image content before decoding; EXIF orientation is applied and output is converted to RGB. Configurable defaults are 8 MiB maximum, 320x240 minimum, 4096 maximum dimension, mean brightness 18-238, maximum clipped-pixel fraction 0.65, and edge-variance minimum 4. The camera tries the configured small burst and bounded retries; a failed final capture becomes NEEDS_REVIEW and can be recaptured only while the same stable plate remains on the scale.

Only the selected image, semantic class-ID PNG, and annotated JPEG are retained for a scan under the runtime data directory. There is no continuous video storage.

## Multi-food vision contract

The real SegFormer adapter is lazy, local-only, CPU-default, and checks the FoodSeg103 label anchors and complete local class map, 104-channel model head, and preprocessing metadata before inference. The current checkpoint has 104 model classes; its older training metadata does not include optional dataset/count/padding-label fields, so the adapter rejects conflicts when those fields exist and validates class mapping against the loaded model config. The existing preprocessing configuration uses mask padding `ignore_index=255` during training; inference creates a dummy all-background mask and applies the deterministic image transform. Real inference outputs semantic class IDs `0..103`; background class 0 is excluded from food predictions. Each predicted food class reports its ID/name, confidence summary, visible pixel area, connected-region count, mask URL, and a coarse segmentation-quality marker. Multiple disconnected regions of one class share one semantic class prediction; they are not guaranteed to be separate food instances.

The mock adapter returns Rice, Chicken, and Fries labels for deterministic API-path tests using fixture IDs 1, 2, and 3, all with `simulated=true`. The real adapter rejects that arbitrary map using canonical FoodSeg103 anchors (including rice 66, chicken/duck 48, and fries 3); mock class IDs never enter real inference. Fixture outputs are not model accuracy.

## Local checkpoint smoke test

When a checkpoint already exists locally and no training process is using the GPU, run the standalone smoke test on CPU:

```powershell
.\.venv\Scripts\python.exe .\tools\smoke_test_segformer.py `
  --checkpoint-dir .\runs\foodseg103-segformer-b2-optimized-20261124_075454\best_model `
  --foodseg-root ..\Datasets\FoodSeg103
```

It loads the local checkpoint only, performs one CPU forward pass, prints class IDs/labels and provenance, and writes no checkpoint or dataset files. With no `--image`, the input is a synthetic mock-camera image: `input_simulated=true` while `model_output_simulated=false` indicates real model computation on synthetic pixels, not a real food photograph. Predicted labels are not accuracy measurements. The script does not download weights or data. The latest recorded smoke result used the existing SegFormer-B2 checkpoint, emitted 104 classes on CPU, and is not a validation score.

The latest performance evidence is [the CV model evaluation report](cv_model_evaluation_report.md), which supersedes the older [B0 baseline evaluation report](baseline_evaluation_report.md). On all 2,135 official validation images, B2 measured 0.364898 foreground mIoU and 0.494406 foreground Dice in one pass; horizontal-flip test-time augmentation measured 0.369469 and 0.498852 at about 1.81x latency. These are dataset-level semantic segmentation results, not food identity, physical portion mass, or calorie accuracy for a new plate. The report identifies rare-class misses and meat-class confusion. Pixel accuracy and uncalibrated per-image softmax values are not calibrated ingredient probabilities.

## Review and failure states

The agent requests a plate tare for missing/inconsistent tare; a capture retry for unacceptable image quality; food review for low classification confidence; preparation/mapping input for missing or ambiguous nutritional data; and portion review for uncertain or incomplete masses. A review draft is still stored with the measured total and all estimates that are available. Missing nutrient values and totals remain null. After a user correction, the service recalculates nutrition against the corrected masses and verified mappings.

See `docs/api_reference.md` for calls and `tests/test_automatic_scan_api.py`, `tests/test_scale_service.py`, and `tests/test_scan_workflow.py` for synthetic coverage.
