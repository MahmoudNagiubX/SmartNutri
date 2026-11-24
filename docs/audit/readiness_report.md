# SmartNutri Readiness Gate

Historical readiness snapshot. Its open P1 items were reassessed and partially addressed in the [final backend readiness report](final_backend_readiness.md); use that follow-up for the current status. This file is retained to preserve the original audit record.

Assessment date: 2026-11-24. Ratings reflect current evidence and unresolved defects, not a roadmap promise.

## Readiness verdicts

| Gate | Verdict | Evidence and boundary |
|---|---|---|
| A. Ready for design system | **GREEN** | Product direction and Flutter target are documented; versioned API and mock workflow provide a stable basis for design exploration. No visual design system or frontend was created. This gate does not imply the backend is ready for production. |
| B. Ready for Flutter implementation | **AMBER** | Flutter API contracts, OpenAPI schemas, control-token expectations, mock mode, meal history and correction paths exist. Before relying on real readings, fix/guard stale weight-to-image association and address the direct unauthenticated ESP32 tare endpoint. A real phone/LAN integration test is still required. |
| C. Ready for mock end-to-end demo | **AMBER** | A real loopback FastAPI/SQLite HTTP demo passed the 300 g saved plate + 750 g synthetic gross = 450 g net case, correction, provenance and duplicate rejection. 120 backend and 39 CV contract tests passed. The requested A–H cases were not all exercised through a live server, and synthetic fixtures are not live device/model evidence. |
| D. Ready for real hardware demo | **RED** | No physical test has occurred. The ESP32 builds, but calibration, HX711 electrical readings, actual stable/removal thresholds, camera capture, Wi-Fi behavior and restart pairing are unverified. Direct firmware POST /tare is not authenticated or explicitly confirmed. |
| E. Ready for reliable nutrition estimation | **RED** | Only 1 of 103 foreground classes has a currently verified unambiguous catalog mapping; 6 are ambiguous and 96 missing. The only portion estimator is an uncalibrated pixel-area allocation that deliberately requires review. There is no real multi-food mass or nutrition validation. |

There is no unconditional overall green light. Mock software exploration can proceed; real-device and reliable-nutrition claims cannot.

## Progression conditions

### Before a trusted real-hardware demo

1. Ensure scanning uses one immutable placement snapshot and rechecks scale freshness/weight before save; cancel or reacquire if load changes during capture/inference.
2. Protect direct ESP32 tare with an explicit trusted control mechanism and require an empty-platform/empty-plate confirmation policy. Test it on a local device only after authorization.
3. Physically verify wiring, HX711 3.3 V levels, tare/calibration, stability threshold, removal detection, overload, sensor disconnection and Wi-Fi recovery. Do not use compile success as measurement evidence.
4. Validate camera endpoint allowlisting, real phone snapshot latency/retries and quality thresholds under controlled lighting.

### Before reliable per-food nutrition

1. Expand the catalog only with authoritative, traceable records and explicit species/cut/preparation state; keep unsupported values null.
2. Collect controlled multi-food reference meals with each component independently weighed, repeated placements, standardized camera geometry and saved ground truth.
3. Replace or calibrate the area baseline and report held-out per-food error, uncertainty calibration, mass conservation and failure rates.
4. Evaluate FoodSeg103 rare classes and confusions on actual SmartNutri camera images; its benchmark mIoU is not a nutrition metric.

### Before Flutter relies on real devices

1. Freeze a versioned OpenAPI contract and stable API error/scan-state semantics.
2. Add a transaction boundary for meal correction plus scan-state update.
3. Test Flutter over the actual trusted Wi-Fi route and keep control credentials out of the mobile package.
4. Preserve provenance for synthetic, file, phone-camera, mock-CV and real-model outputs in every UI surface.

## Recommended priority

| Priority | Work | Why |
|---|---|---|
| P1 | Prevent stale/mismatched weight and image association | Avoids recording mass for a different plate state than the captured image |
| P1 | Secure and confirm firmware tare operations | Prevents a remote tare from silently changing the measurement zero |
| P1 | Keep all estimated portions/nutrients explicitly reviewed until validated | Current area-to-mass output is experimental; automated save is blocked by policy by design |
| P1 | Expand verified nutrition coverage and resolve class/preparation ambiguity | Sparse catalog cannot support general meal totals |
| P2 | Validate camera allowlist after DNS resolution and redirect handling | Reduces server-side request risks when snapshot URL is configured |
| P2 | Make correction updates cross-table atomic | Prevents meal and scan state divergence after an interrupted write |
| P2 | Calibrate quality/confidence signals and exclude letterbox padding from class confidence | Avoids overinterpreting current uncalibrated confidence summaries |
| P2 | Maintain dependency audit and add a supported CUDA-aware advisory workflow | Current scanner cannot resolve the pinned local CUDA Torch wheels from PyPI |

## Non-goals completed

- No Flutter app or UI was implemented.
- No design system or design.md was created.
- No model was trained, resumed, stopped or changed.
- No dataset or checkpoint was modified or overwritten.
- No physical hardware was flashed, calibrated or measured.
