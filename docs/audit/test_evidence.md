# SmartNutri Audit Test Evidence

Audit date: 2026-11-24. The counts below are observed executions, not claims about physical accuracy.

## Summary

| Check | Result | Evidence |
|---|---|---|
| Backend/Python unit and contract suites | PASS — 120 tests | Backend-only temporary Python environment; 17 modules |
| CV/data/training contract suites | PASS — 39 tests | Existing CUDA project environment; no model training invoked |
| Dataset integrity scan | PASS | 7,118 FoodSeg103 image/mask records and 3,490 Nutrition5k RGB/metadata pairs read without modifying data |
| Coverage matrix validation | PASS | 103 unique IDs, IDs 1–103, exact source-label strings match local `id2label.json` |
| Nutrition values against USDA SR Legacy source | PASS after two precision corrections | All 11 current records, source IDs and per-100 g nutrient fields cross-checked |
| Existing local model smoke | PASS, limited | CPU-only SegFormer-B2 load/forward on synthetic image; 104 logits; not a real-food accuracy test |
| FastAPI over HTTP mock demonstration | PASS, Scenario A | Temporary SQLite, loopback Uvicorn, 300 g saved tare + 750 g synthetic gross = 450 g net, correction, simulation provenance, duplicate 409 |
| ESP32 PlatformIO build | PASS | Temporary copy, `esp32dev`, firmware binary produced; no flash |
| PlatformIO cppcheck check | FAIL / inconclusive | `pio check -e esp32dev` installed cppcheck but returned “cppcheck failed to perform check”; toolchain defines warnings preceded it. No static findings can be concluded |
| Dependency vulnerability audit | PARTIAL | Backend requirement file: no known vulnerabilities in resolved app dependencies. Local pip/setuptools issues and CUDA wheel audit limitations are listed below |
| Physical ESP32/HX711, smartphone camera, or food-portion validation | NOT TESTED | No physical hardware session or actual meal was measured |

## Python test suites

Backend environment: temporary isolated Python 3.12 environment. The CV `.venv` was not modified. Each module was run with Python unittest discovery; all module commands exited zero.

| Module | Tests | Result |
|---|---:|---|
| `test_agent_policy.py` | 8 | PASS |
| `test_automatic_scan_api.py` | 12 | PASS |
| `test_backend_api.py` | 5 | PASS |
| `test_backend_contracts.py` | 6 | PASS |
| `test_backend_repository.py` | 6 | PASS |
| `test_backend_settings.py` | 2 | PASS |
| `test_camera_service.py` | 8 | PASS |
| `test_flutter_contracts.py` | 6 | PASS |
| `test_nutrition_catalog.py` | 8 | PASS |
| `test_nutrition_service.py` | 6 | PASS |
| `test_portion_estimator.py` | 7 | PASS |
| `test_scale_monitor.py` | 9 | PASS |
| `test_scale_service.py` | 3 | PASS |
| `test_scan_workflow.py` | 1 | PASS |
| `test_server.py` | 4 | PASS |
| `test_vision_adapter.py` | 6 | PASS |
| `test_weight_api.py` | 23 | PASS |
| **Backend total** | **120** | **PASS** |

CV/project environment: `test_image_processing.py` 18, `test_class_weighting.py` 7, `test_optimized_training.py` 10, `test_training_experiment.py` 3, and `test_segmentation_comparison.py` 1. **39 passed, 0 failed.** The two experiment test files and their related untracked configuration/tooling were the user's existing working-tree work; they were run but not staged or changed.

Warnings observed: PyTorch reports that distributed output redirection is not supported on Windows; Starlette reports the current httpx TestClient integration is deprecated. Both test suites passed. No dependency upgrade was performed in response.

## HTTP and workflow evidence

A real Uvicorn process was started on loopback with mock scale/camera/CV and a unique temporary data directory. It was stopped after the test and is no longer listening.

1. Health and scale status returned HTTP 200.
2. A synthetic 300 g saved plate and three synthetic test-only nutrition fixtures were created.
3. POST mock placement with synthetic gross 750 g returned HTTP 200, net food 450 g, and `simulated=true`.
4. GET food predictions returned HTTP 200 and preserved `simulated=true` even when the inference adapter reported a SegFormer-like real-model source. This reproduced and verified the API provenance fix.
5. GET meals showed exactly one record. A correction to 100 g rice + 200 g chicken + 150 g fries persisted as revision 1, recomputed synthetic fixture nutrition, retained the 450 g measured total and simulated provenance.
6. A repeated placement returned HTTP 409, preventing another meal for the same latched placement.

Nutrition figures in this HTTP run used explicitly synthetic fixtures and are not USDA records or real nutritional results. The route-level test suite separately exercises live-tare no-double-subtraction, empty saved-plate placement followed by food, missing/wrong profile, unknown/incomplete mass, camera error, correction and duplicate placement. Restart, sensor, rearming, ambiguous class, and fruit-mapping behavior are covered by focused unit/API tests, not all by the live server run.

## Model and dataset checks

- `tools/smoke_test_segformer.py` loaded the pre-existing local B2 checkpoint on CPU, verified the FoodSeg103 mapping and returned 104 model classes for a synthetic mock-camera input. The source pixels are synthetic; `model_output_simulated=false` means the model computation was real. This is not a scene accuracy result.
- The complete local FoodSeg103 scan found 4,983 train and 2,135 validation images, no decode or invalid-mask records, no mismatch between image classes and indexed masks, no exact image bytes crossing splits, and all 103 foreground classes in both splits. Four raw image/mask dimension mismatches were resolved by the dataset's EXIF orientation when transformed; zero corrected mismatches remained.
- Nutrition5k RGB scan found 3,490 matched image/metadata pairs, no image decode or numeric validity problems, and 46 ingredient-mass sums over 1 g away from total dish mass. The source files were not altered.
- The 103-row nutrition matrix was rechecked against the local mapping after preserving leading whitespace for source labels 62 and 91. IDs and labels match exactly.

## Firmware and build checks

- Command: `pio run -e esp32dev` from a temporary copy of `firmware/weight_scale` with PlatformIO 6.2.0, platform espressif32 7.0.1, Arduino ESP32 framework 3.20017.241212. Result: PASS after 339.32 s. Reported static allocation: 45,312 bytes RAM (13.8%) and 792,493 bytes flash (60.5%).
- Command: `pio check -e esp32dev`. Result: FAIL to produce a cppcheck result. The check runner reported failure to extract toolchain defines and ended “cppcheck failed to perform check.” Standalone cppcheck was not available at the expected tool path. No source edits or hardware flash followed.
- Firmware C++ scale-logic tests, electrical signal timing, load-cell linearity, calibration, disconnect detection, and overload response on hardware: NOT TESTED.

## Security and dependency checks

- Git signature scan checked the current worktree and 228 unique historical blobs for common GitHub, Google, AWS, private-key and token patterns. No matches; no historical blob or tracked file exceeded 1 MB. Gitleaks was unavailable, so this is a limited signature check, not a full secret scanner.
- `.gitignore` checks passed for local `.venv`, `runs/` checkpoints, `.pio` build output, ignored ESP32 `include/secrets.h`, `.env.*`, Parquet, ZIP and PDF paths. Dataset directories are siblings of the repository and not in Git.
- `pip-audit` on `smartnutri_backend/requirements.txt`: PASS, no known vulnerable resolved application dependencies at audit time.
- Local root CV environment: pip-audit found vulnerable setuptools metadata; requirements resolution reports setuptools 81.0.0 with PYSEC-2026-3447 and GHSA-xmj3-9gf7-h52w. The installed root environment also reported setuptools 78.1.0 and three unique advisories (PYSEC-2025-49, PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w). The backend temporary environment has pip 25.0.1 with six unique PYSEC advisories. These are local build/install tools; no version was changed to preserve the CUDA environment and avoid blind upgrades.
- pip-audit could not resolve the pinned CUDA packages `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` from PyPI, so their vulnerability status is NOT TESTED by that scan. Runtime packages are not claimed vulnerability-free on that basis.
- Ruff, mypy, Bandit, Semgrep and Gitleaks were not installed. The PlatformIO cppcheck runner did not complete successfully.

## Not tested

- Physical ESP32/HX711 measurements, wiring voltage, 5 kg overload, tare/calibration accuracy, or Wi-Fi disconnect/reconnect on a real device.
- Smartphone IP snapshot capture, real lighting/exposure/sharpness thresholds, redirects/DNS behavior against a real phone, or phone-local permission behavior.
- Real food predictions, cooking-state recognition, measured multi-food portion accuracy, or nutrition accuracy in the final application.
- Every audit scenario A–H as a live socket-level server scenario. The documented live HTTP run covers Scenario A; other failure paths are covered by relevant unit/API tests, and some full interactions remain partial.
