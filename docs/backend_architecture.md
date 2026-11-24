# SmartNutri BE-01 v2 backend architecture

## Scope

The backend coordinates whole-plate weighing, phone-camera capture, FoodSeg103 semantic segmentation, experimental per-food portion estimates, nutritional lookup, decision policy, and SQLite persistence. It runs on the laptop. The ESP32 only reports scale readings. The repository includes a small traceable USDA SR Legacy catalog for demonstrations, but it does not add a frontend, a trained nutrition agent, or a validated portion-mass model. The API does not load a checkpoint during startup.

## Components and data flow

```text
ESP32 + HX711 -- GET /weight --> ScaleMonitoringService
                                      |
Android phone -- private snapshot --> CameraService
                                      |
                          ScanWorkflow (laptop)
                            |          |         |
                       SegFormer   Portion     Rule agent
                       adapter     estimator       |
                            |          |      verified nutrient records
                            +----------+---------+  |
                                       SQLite <-----+
                                         |
                                    FastAPI v1
```

- `smartnutri_weight` is the existing HTTP client for the ESP32. The backend polling worker uses bounded GET retries from that client and continues polling after temporary errors.
- `scale_service.py` contains the background monitor, mock scale, source-aware observation conversion, explicit tare handling, and placement dispatch. When a saved plate profile is selected, placement detection adds the empty plate's weight to the minimum net-food trigger; live-tare readings use the minimum directly.
- `camera.py` provides mock, file-test, and private-LAN IP snapshot adapters. It validates size, format, exposure, sharpness, and EXIF orientation. It normalizes captured frames to RGB JPEG and saves only one selected frame per scan.
- `vision.py` has the multi-food mock predictor and lazy local-only SegFormer adapter. The real adapter checks canonical FoodSeg103 label anchors and exact class IDs `0..103`, verifies the loaded model's complete `id2label`/`label2id` and 104-channel head, and uses preprocessing metadata stored with the checkpoint. It defaults to CPU and never downloads a checkpoint. The mock fixture's IDs 1/2/3 cannot pass real-label anchor validation.
- `portion.py` exposes the `PortionEstimator` interface. Its current estimator is an explicitly experimental pixel-area baseline; it always asks for review.
- `agent.py` implements a transparent policy. It is not an LLM or a separately trained model.
- `nutrition.py` calculates nutrients only from verified records with a food identity, preparation state, source, version, and source identifier. Preparation-sensitive or multi-record class mappings require explicit record selection. Mock fixture records are isolated from USDA records by source identifier and scan provenance.
- `nutrition_catalog.py` contains 11 USDA SR Legacy records covering eight food identities. The catalog and source IDs are documented in [nutrition catalog](nutrition_catalog.md); preparation variants are not collapsed.
- `repository.py` persists the scale/tare configuration (including pending live tare), placement latch, scans, plate profiles, nutrition records, meals, and meal items in SQLite. Existing databases are migrated additively; legacy verified rows without source IDs are preserved but marked unverified.
- `workflow.py` coordinates one placement through capture, CV, portions, nutrition, agent decision, and persistence.
- `api.py` provides versioned `/api/v1` routes and lifespan-managed background polling.

## Persistence model

SQLite enables foreign keys and WAL mode for file-backed databases. The default database and runtime images are stored under `%LOCALAPPDATA%\SmartNutri`, outside the Git checkout.

| Table | Purpose and key fields |
| --- | --- |
| `app_settings` | JSON settings for tare mode and the monitor placement latch. |
| `plate_profiles` | Plate ID, name, measured empty weight, optional reference-image path, default flag, and timestamps. |
| `nutrition_records` | Optional FoodSeg103 class ID, food identity, preparation state, kcal/protein/carbohydrate/fat per 100 g, source, source identifier, version, verified flag, and preparation-confirmation flag. The USDA SR Legacy catalog is seeded idempotently at startup. |
| `scans` | Unique placement key; scan state and decision; scale source/simulation marker; tare mode and plate profile; measured scale/gross/net values; camera provenance and quality; predictions, portion result, nutrition totals, and safe error code. |
| `meals` | One record per scan; state, measured net food weight, estimated nutrient totals, correction revision, and timestamps. Review drafts are persisted as `needs_review`. |
| `meal_items` | FoodSeg class, label, classification confidence, pixel area/region count, estimated grams, portion method/confidence/uncertainty, nutrition record, calculated nutrients, correction marker, and review marker. |

Both measured net meal weight and per-food estimates are retained. If a portion or nutrient mapping is incomplete, the total measured weight remains stored and the corresponding totals remain null. Simulation is derived from the scale, camera, CV, and portion provenance; a simulated value is never relabelled as a physical reading.

## Runtime and security defaults

Start the API through `python -m smartnutri_backend.server`, which binds to `127.0.0.1` by default. A LAN deployment must set `SMARTNUTRI_API_HOST` to one explicit RFC1918 IPv4 address; wildcard, public, hostname, and IPv6 binds are rejected. Auto-scan defaults off and must be explicitly enabled after the source configuration is ready. CORS is disabled by default; optional explicit origins are only for a separate browser client. The camera adapter accepts only private, loopback, or `.local` snapshot hosts and rejects redirects outside that set. Captured images are size-limited and stored outside Git. Real tare, calibration, recapture, nutrient-record writes, and meal corrections require `X-SmartNutri-Control-Token`; real control requests fail closed if no token is configured. Mock mode is explicitly simulated and does not call ESP32 hardware.

For a trusted-Wi-Fi client, bind only to the laptop's Wi-Fi IPv4 and scope inbound TCP/8000 to the Private firewall profile and the phone or trusted subnet. Do not add router port forwarding or expose the local API to the public internet. The REST/JSON and image contracts for the future Flutter application are documented in [API reference](api_reference.md) and [Flutter Android integration](flutter_android_integration.md). Camera capture remains a separate camera-service adapter.

## Research relationship and evidence boundaries

1. **FoodSeg103 benchmark** motivates the ingredient semantic-segmentation stage and exact label/mask handling. A semantic mask identifies classes; it does not supply physical mass.
2. **Nutrition5k** motivates the separate future nutrition/mass-learning stage. It is not read or modified by this backend, and no Nutrition5k model is implemented here.
3. **Automated Food Weight and Content Estimation Using Computer Vision and AI Algorithms: Phase 2** motivates keeping portion estimation as a separate research problem and replaceable interface.
4. **Health to Eat: A Smart Plate with Food Recognition, Classification, and Weight Measurement for Type-2 Diabetic Mellitus Patients’ Nutrition Control** motivates combining visual recognition and measured weight as different inputs.
5. **SCALEeat: Vision-Guided Food Scale for Automated Macronutrient Estimation** motivates an end-to-end vision/scale/nutrition workflow and explicit validation of each stage.

These papers inform the architecture; they do not validate this implementation. Mock end-to-end results validate code paths only. The local CPU checkpoint smoke test validates that the adapter loads the present checkpoint and emits a structurally valid 104-class result using synthetic camera input; it provides no accuracy evidence. Real-scene inference, physical weighing, cooking/preparation resolution, and per-food mass accuracy remain unvalidated.

## Audit verification snapshot

On 2026-11-24, the backend, repository, scale-client, camera, agent, nutrition, scan, model-contract, and Flutter-contract suites passed **120 tests** in the isolated backend environment. The project environment passed **39 CV/data/training-experiment contract tests** (18 image processing, 7 class weighting, 10 optimized-training contracts, and 4 local experiment tests). A CPU-only smoke test loaded the existing local SegFormer-B2 checkpoint and emitted a structurally valid 104-class result for a synthetic image. An isolated FastAPI/SQLite HTTP demonstration passed saved-plate subtraction (750 g gross minus 300 g tare = 450 g net), correction/recalculation, simulated provenance, and duplicate-placement checks. These results validate software paths only; no physical scale/camera test or real-scene accuracy evaluation was performed. Mock placement and portion values are synthetic.

## Running tests

Install `smartnutri_backend/requirements.txt` into a backend-only environment and run the backend suite:

```powershell
python -m unittest tests.test_agent_policy tests.test_automatic_scan_api tests.test_backend_api tests.test_backend_contracts tests.test_backend_repository tests.test_backend_settings tests.test_camera_service tests.test_flutter_contracts tests.test_nutrition_catalog tests.test_nutrition_service tests.test_portion_estimator tests.test_scale_monitor tests.test_scale_service tests.test_scan_workflow tests.test_server tests.test_vision_adapter tests.test_weight_api -v
```

Use the project environment for preprocessing tests and the standalone CPU model smoke check. The current smoke evidence used the existing B2 checkpoint:

```powershell
python -m unittest tests.test_image_processing -v
python .\tools\smoke_test_segformer.py --checkpoint-dir .\runs\foodseg103-segformer-b2-optimized-20261124_075454\best_model --foodseg-root ..\Datasets\FoodSeg103
```

The integration fixtures are synthetic and marked as such. Unit tests do not start model training. The smoke script uses only already-present local files and does not write to the checkpoint or dataset. The B2 official-validation metrics are in [the latest CV model evaluation report](cv_model_evaluation_report.md), which supersedes the older B0 baseline as the newest performance evidence.
