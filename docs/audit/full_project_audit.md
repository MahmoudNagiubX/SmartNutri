# SmartNutri Full Project Audit

Task: AUDIT-01 — pre-design and pre-frontend quality gate  
Audit date: 2026-11-24  
Repository: `https://github.com/MahmoudNagiubX/SmartNutri` (public by user intent)  
Checkout: `feature/automatic-backend`, starting commit `5164eeb982c0d0b20d4e0c36eaac1cab5aadc6f3`

## 1. Executive summary

The repository has a coherent whole-plate backend, a tested FoodSeg103 semantic-segmentation pipeline, working mock scale/camera adapters, typed FastAPI contracts, SQLite persistence, and independently compiled ESP32 firmware. The implementation is substantially more complete than a mock-only skeleton, but it is not ready to claim reliable real meal nutrition.

The highest-impact unresolved issues are: (1) a stable reading is captured at the start of a scan but the workflow does not verify that the same load remains during camera capture/inference; (2) firmware HTTP POST /tare is unauthenticated and has no confirmation field; (3) the portion estimator is experimental area allocation and every estimate requires user review; and (4) only 1 of 103 foreground FoodSeg103 classes has a currently verified, unambiguous nutrition mapping. No physical device, phone camera, or actual meal-mass validation was performed.

Audit evidence includes 120 passing backend tests, 39 passing CV/data/experiment contract tests, a 7,118-row FoodSeg103 integrity scan, a 3,490-image Nutrition5k RGB scan, an existing B2 CPU inference smoke test on synthetic pixels, a real loopback HTTP mock meal/correction run, and a PlatformIO ESP32 compile. PlatformIO cppcheck did not complete. Dependency security scanning is partial because the CUDA PyTorch wheels were not resolvable from PyPI and the local environments contain vulnerable packaging tools.

Small fixes applied: two USDA nutrient values were corrected to the archived source precision; a food-prediction API response now preserves simulated capture provenance; focused regression tests were added; the 103-class nutrition coverage CSV now preserves exact local label strings; and stale React/B0-only documentation was corrected. No data, checkpoints, training scripts, firmware or design files were changed.

Detailed inventory: [component_inventory.md](component_inventory.md). Test evidence: [test_evidence.md](test_evidence.md). Readiness gates: [readiness_report.md](readiness_report.md). Full class matrix: [nutrition_coverage.csv](nutrition_coverage.csv).

## 2. Product-vision alignment

| Product requirement | Actual state | Audit result |
|---|---|---|
| One complete multi-food plate triggers one scan | Monitor has explicit states, stability confirmations, settling delay, unique placement keys and removal rearming | Implemented in software; automatic scanning defaults off, and live sensor freshness needs improvement |
| Saved plate tare or explicitly confirmed empty live tare | Both persisted modes exist; saved plate subtracts once; live tare is not subtracted a second time | Unit/API tests pass; wrong/missing profile is rejected; direct firmware tare endpoint is a safety gap |
| Capture only after stable placement | Camera runs after monitor trigger and workflow receives the triggering observation | Trigger ordering is correct; the initial reading is not revalidated during the full capture/inference window |
| Multi-food recognition and regions | SegFormer adapter emits semantic classes, class areas, connected-region counts and class confidence summaries | Implemented; semantic segmentation does not establish instance identity or physical mass |
| Per-food portion masses | Experimental area-proportion estimator conserves total known mass and leaves unidentified mass unallocated | Not physically validated; every output is marked for review |
| Nutrition from verified per-100 g data | USDA-backed catalog preserves source IDs, preparation and version; missing totals stay null | Correct safety behavior, but coverage is far too sparse for general meals |
| Rule-based AI Nutrition Agent | Deterministic policy routes invalid scale, camera, low confidence, preparation/mapping, and portion uncertainty to actions | Useful transparent decision policy, not a trained neural agent; current estimator prevents automatic SAVE_MEAL |
| Persist and expose results to future app | SQLite and typed REST/OpenAPI routes include meals, history and corrections | Present; Flutter application intentionally absent |
| Minimal interaction | Mock and high-confidence paths exist, but catalog ambiguity and every area estimate require user review | Product goal is not yet met for real general meals; safety checks should not be weakened |

## 3. Inventory, branches and local work

The source inventory and component roles are listed in [component_inventory.md](component_inventory.md). In summary, the root contains the CV pipeline and experiments; `smartnutri_backend/` contains API, workflow, camera, vision, agent, portion, nutrition and repository modules; `smartnutri_weight/` is the ESP32 client/logic package; and `firmware/weight_scale/` contains Arduino/PlatformIO firmware.

The checkout started on `feature/automatic-backend`, equal to its origin tracking head. `main`, `audit/esp32-weight-integration`, and feature branches exist. PR #1 and PR #2 are open drafts; neither was merged or changed. The public repository contains no tracked file above 1 MB and the scanned history has no blob above 1 MB. Common credential-pattern scanning found no matches in the current files or 228 unique historical blobs; Gitleaks was unavailable, so this is not a full secret scan.

Unrelated user work was present in `train.py` and untracked CV experiment files under `configs/losses/`, `docs/experiment_2_class_imbalance.md`, `tests/test_segmentation_comparison.py`, `tests/test_training_experiment.py`, and `tools/`. It was preserved and excluded from audit commits. Local checkpoints, runs, `.venv`, PlatformIO build files, PDFs and datasets are ignored or outside the repository. No `design.md` or Flutter application exists.

## 4. Dataset integrity

### FoodSeg103

Local source: `../Datasets/FoodSeg103`. The official train partition has 4,983 records across three equal 1,661-row Parquet shards; validation has 2,135 rows in one shard. Each includes embedded JPEG image bytes, indexed PNG semantic masks and class annotations. The locally supplied label map contains IDs 0–103, with ID 0 background and 103 foreground categories. There is no official test split in the local source.

The read-only full scan covered all 7,118 pairs: zero image decode errors, zero invalid mask values, zero mismatch between `classes_on_image` and mask IDs, and 103 foreground classes present in both splits. Four train images had raw dimensions differing from masks; all four were corrected by EXIF orientation handling and none remained mismatched. Exact image hashes did not cross train and validation. There are 42 repeated image-byte occurrences within train and 6 within validation; numeric row IDs overlap all 2,135 validation rows because IDs are split-local. Preserve split membership and never treat a row ID as a global key.

Class pixel counts are strongly imbalanced: the smallest train foreground class has 3,670 pixels and the largest 244,699,757. This affects rare-class learning and makes pixel accuracy especially easy to overread.

### Nutrition5k RGB subset

Local source: `../Datasets/Nutrition5k_RGB`. The local manifest has 3,490 unique dish IDs and 3,490 matching decodable 640×480 PNGs, with no missing/extra image IDs or invalid/missing/negative total nutrition fields. It records total mass/calories/macros and semicolon-delimited ingredient names and ingredient masses, with 240 distinct ingredient names and up to 34 ingredients per dish. Forty-six ingredient-mass sums differ from the recorded total mass by more than 1 g; cause was not established and the source was left unchanged. No split column exists.

This RGB subset can support a carefully designed separate mass/nutrition experiment if identity mapping, independent split design, and source provenance are established. It is not a pixel-segmentation dataset and cannot supervise FoodSeg103 masks. The local subset lacks the full Nutrition5k video/depth observations, and it is not evidence of generalization to the SmartNutri overhead camera.

## 5. Computer vision and image-processing review

`foodseg_data.py`, `image_processing.py`, the baseline and B2 training/evaluation modules, `metrics.py`, `visualize.py`, and their tests preserve the exact 104-class head and FoodSeg103 IDs. Image processing corrects EXIF, converts RGB, letterboxes while preserving aspect ratio, pads masks with 255, uses nearest-neighbor mask resize, synchronizes geometry, restricts random augmentation to training, and normalizes once. Validation is deterministic. Data loading uses official train/validation partitions and does not reduce labels.

The real inference adapter reads the local FoodSeg103 map and checkpoint metadata, fails closed when mappings or checkpoints do not agree, emits 104 logits, ignores background when reporting foreground predictions, returns a semantic PNG and annotated image, and keeps mock fixture IDs out of real inference. SegFormer output is semantic, not instance segmentation or portion estimation. Class confidence is a softmax summary, not calibrated probability.

The latest evaluation evidence is `docs/cv_model_evaluation_report.md`, not the older B0 report. Existing B2 evaluated over all 2,135 official validation images measured 0.364898 foreground mIoU, 0.494406 foreground Dice and 0.781938 pixel accuracy in one pass. Horizontal-flip TTA measured 0.369469 mIoU, 0.498852 Dice and 0.784023 pixel accuracy at roughly 1.81× latency. B2 was selected on a held-out split carved from the official train partition; its official validation metrics were not used for selection. B0 checkpoint selection did use official validation, so B0/B2 comparison is not fully symmetric.

B2 predicts 101 of 103 foreground classes and has at least one correct pixel in 99. Several rare classes remain near zero, and large confusions occur among meat classes including chicken/duck, pork and steak. These metrics support segmentation research only; they are not plate-level identity precision, portion error, or nutrient accuracy. The existing B0 baseline foreground mIoU is approximately 0.1918, and the newer B2 report supersedes it as the latest evidence.

The existing local B2 model was loaded on CPU using a synthetic image; the smoke test returned 104 classes and did not write to data or checkpoints. It does not demonstrate recognition of real food. A provenance caveat remains: the B2 run metadata describes safetensors initialization although the pinned upstream model artifact observed during earlier inspection was PyTorch format; exact original weight artifact cannot be proven from available historical metadata. No model was trained or modified during this audit.

## 6. Nutrition catalog and coverage

The live code contains 11 USDA FoodData Central SR Legacy records across eight identities and seven FoodSeg103 classes: rice (raw and cooked), chicken (raw and cooked), duck (cooked), fries, apple, banana, broccoli (raw and cooked), and tomato. Every record's source identifier and kcal/protein/carbohydrate/fat values were cross-checked against the official SR Legacy downloadable source; all amounts are per 100 g edible portion. The source is the USDA SR Legacy final release from April 2018, so its age and reference-food descriptions are retained rather than hidden. See the [USDA dataset download page](https://fdc.nal.usda.gov/download-datasets/) and the record-level audit matrix.

The exact 103 foreground-class matrix has these statuses:

| Status | Count | Interpretation |
|---|---:|---|
| VERIFIED | 1 | Class 3 French fries maps to an exact fast-food/french-fried USDA reference; real recipes and oil absorption can still differ |
| AMBIGUOUS | 6 | Classes 25 apple, 29 banana, 48 chicken/duck, 66 rice, 73 tomato, and 87 broccoli have a source record but class label alone does not resolve identity/preparation safely |
| MISSING | 96 | No locally curated verified record |

The class 48 label explicitly combines chicken and duck. It must require species, cut/edible portion and raw/cooked selection; the code does not silently decide this. The same explicit-selection policy applies to rice and preparation-sensitive records. Strawberry, cherry and blueberry remain null with no fabricated nutrition. No data was inserted to increase coverage.

Two displayed values were corrected from rounded guesses to exact archived source precision: duck protein FDC:172411 is 23.48 g rather than 23.5; Golden Delicious apple carbohydrate FDC:168202 is 13.60 rather than 13.61. A new multi-item test verifies the formula nutrient = grams × per-100-g value / 100 with cooked rice, roasted chicken and fries fixtures, totaling 837 kcal, 69.51 g protein, 83.70 g carbohydrate and 22.29 g fat at the service's two-decimal rounding. Another test ensures explicit duck record selection and leaves missing fruit values unavailable. The current catalog is still insufficient for general meal coverage.

## 7. Firmware and hardware

The firmware targets ESP32 DevKit V1/ESP32-WROOM-32 through PlatformIO `esp32dev`. Pins and documented assumptions match the target: DOUT GPIO32, SCK GPIO33, HX711 VCC 3.3 V and common ground. Firmware bit-bangs the signed 24-bit reading, sets the 25th clock pulse for channel A/gain 128, applies a five-sample median and EMA, polls about every 100 ms, uses a 10-reading/2 g stable window, reports stale/disconnected sensor states, and rejects overload/out-of-range conditions. Wi-Fi retries and HTTP serving are cooperative/nonblocking at the application loop level. Preferences stores calibration/tare state; a calibration request requires a known positive mass in range and the exact confirmation token. USB Serial provides status and calibration fallback.

PlatformIO compilation passed on a temporary copy for `esp32dev`: 45,312 bytes RAM (13.8%) and 792,493 bytes flash (60.5%). No firmware was flashed. A requested `pio check -e esp32dev` cppcheck run failed before producing an analysis because toolchain define extraction/check execution failed. Physical analog accuracy, HX711 timing/voltage, load cell capacity, stable threshold, calibration, Wi-Fi recovery, and disconnect behavior remain NOT TESTED.

Confirmed safety gap: firmware `POST /tare` immediately starts tare without an authorization token or explicit confirmation; the serial `tare` command also starts it directly. This differs from calibration, which requires explicit confirmation. Backend control-token protection does not protect a direct HTTP request to the ESP32. A local network client could tare a loaded platform and corrupt later measurements. Firmware should require a trusted control mechanism and empty-platform confirmation before a real demo; do not expose the ESP32 endpoint on an untrusted network.

Calibration/tare reads are averaged but are not gated by the same stable-window test used for measurements. This is a physical reliability limitation requiring device tests, not compile evidence.

## 8. Backend, scan state, plate modes and persistence

The backend separates scale monitoring, capture, inference, portion estimation, policy and persistence. The state machine rejects unstable spikes, uses saved plate mass to set the placement threshold, triggers once per placement key, latches while a plate remains, and requires confirmed removal before rearming. Backend restart restores an open placement as latched/waiting-for-removal. Saved plate net mass is gross minus profile mass once; live-tare mode accepts already-tared food net and does not subtract plate weight again. Empty-plate-first and live-tare tests pass.

The highest backend integrity risk is that the trigger observation is captured once and passed into `workflow._process_once`; the workflow stores net weight before camera capture and inference. During a scan, the scale monitor only cancels on the configured removal condition. A material food addition/removal or plate exchange that remains above that threshold is not detected as a changed load; the saved weight and later image can represent different plate states. Add a placement snapshot/freshness check and cancel/reacquire on material drift before accepting analysis. This was not changed because it crosses monitor/workflow orchestration and needs a focused design/test.

`SMARTNUTRI_AUTO_SCAN` defaults false, so actual automatic trigger behavior requires explicit operator configuration. This is a safe development default but means a default launch is not an automatic scale until configured.

SQLite uses foreign keys, WAL, uniqueness constraints on placement key and one meal per scan, and checks for nutrition fields. Tables include app settings, plate profiles, nutrition records, scans, meals and meal items. Repository writes use transactions and an in-process lock. One correction route updates the meal and scan decision in two separate repository operations; a process crash between them can leave meal state and scan state inconsistent. Make that correction boundary atomic before relying on recovery.

## 9. Camera, CV adapter and image association

Mock, configured-file and IP snapshot adapters exist. The service validates image type/size/dimensions, EXIF and RGB conversion, exposure/clipping, sharpness, retry count and optional burst selection. Unit tests cover mock/file behavior, EXIF correction, blur/exposure rejection, retry selection, size/format rejection and blocking public redirects. Camera failure produces review/error state rather than fake successful results.

The phone snapshot service rejects credentials in URLs and public target/redirect hostnames, but `.local` names are allowed without resolving/pinning the final IP address. DNS rebinding or a misconfigured private name is not ruled out. Keep the URL operator-controlled; for stronger SSRF protection resolve and validate the address at connect time, restrict ports/addresses, and prevent redirects from changing the allowed peer. No real phone snapshot was captured and quality thresholds were not calibrated for the actual lighting setup.

The vision adapter returns class-level confidence and connected-component counts. In one code path, confidence is averaged over low-resolution model output before the semantic mask is cropped to the unpadded image area, so letterbox predictions can influence a foreground class confidence even when those predicted pixels are later removed from the mask. This is a non-blocking confidence-quality issue; crop the confidence selection with the same content window before using it for decision thresholds.

## 10. Portion estimation and AI Nutrition Agent

The `ExperimentalVisualProportionEstimator` distributes known total grams by semantic visible pixel area. It excludes background, preserves total known mass using cent-level allocation, keeps unknown visible area unallocated, rejects invalid totals and duplicate semantic classes, and writes uncertainty/review metadata. The synthetic fixture with known 450 g totals reports 13.33 g MAE against its hand-authored portion distribution. That is arithmetic self-consistency against one synthetic fixture, not measured physical accuracy.

One overhead RGB image and one total weight cannot identify every food's mass without learned/calibrated priors or extra geometry. The current area estimator explicitly sets each estimate and the aggregate to review-required. The agent checks this state and high relative uncertainty, so ordinary current area estimates cannot reach automatic SAVE_MEAL. This is a deliberate correctness guard, but it conflicts with minimal interaction and blocks reliable automatic nutrition until the estimator is researched and calibrated.

The agent is a deterministic rule-based policy with scale status, known tare, camera quality, prediction confidence, mapping availability, portion completeness/uncertainty and current acceptance as percepts. Its actions map to retries, capture/food/portion review, preparation/tare request or save. Tests cover invalid scale, missing tare, camera failure, low confidence, missing nutrition, uncertain portions and acceptable analysis. It is not a trained AI model and does not use an LLM. The implementation handles uncertainty conservatively, but the currently available real catalog/portion evidence means user corrections are necessary for many real plates.

## 11. API, database and future Flutter contract

Versioned API groups include health; scale status/tare configuration/tare/calibration; plate profiles/default; camera status/capture/recapture; scans/active/detail/mock placement; foods/masks/images/portions/nutrition; verified nutrition records; agent decisions; meal list/detail/correction. Pydantic schemas and OpenAPI are exposed at `/openapi.json`. Meal results preserve measured total separately from per-food estimates; incomplete nutrition stays null. The mock-placement route is explicitly simulated and cannot actuate a real scale.

The launcher defaults to loopback and rejects wildcard/public bind hosts. CORS is disabled unless exact origins are configured. Real backend controls require `X-SmartNutri-Control-Token`; read endpoints are unauthenticated, and trusted-Wi-Fi use therefore depends on LAN firewall scoping. The API is a reasonable mock Flutter contract; keep the control token out of the app and test Android local-network permissions and image requests on a real device later. The approved frontend is Flutter; no React or Flutter application is implemented.

## 12. Security and dependencies

`.gitignore` covers dataset directories, raw/processed data, Parquet/archive formats, model/checkpoint files, runs/caches, virtual environments, secrets, runtime SQLite/image data, PlatformIO output and PDFs. Git checks confirmed local examples for `.venv`, runs, `.pio`, firmware secrets, `.env`, Parquet, ZIP and PDFs are ignored. The public repo has no credential-pattern match in the current tree or scanned history; no dataset, model, runtime image or database is tracked.

`pip-audit` found no known vulnerabilities in resolved backend application requirements. It found vulnerable setuptools/pip bootstrap packages in local environments. The root requirements audit could not resolve the pinned CUDA Torch/Torchvision builds from PyPI, so their advisory status was not verified. No dependency upgrades were made; the CUDA environment currently reports PyTorch 2.11.0+cu128 with CUDA 12.8 available. Ruff, mypy, Bandit, Semgrep and Gitleaks were unavailable. The source credential scan is a limited regular-expression check.

## 13. Documentation consistency and research papers

The supplied research directory contains five papers: the FoodSeg103 benchmark, Nutrition5k, a 2025/2026 food weight/content estimation paper, Health to Eat smart plate, and SCALEeat (2026). Their opening sections motivate semantic food masks, ingredient mass/nutrition labels, weighing plus vision, and the need for separate mass estimation. They do not establish that this specific camera, classes, estimator, catalog or firmware achieves those papers' results.

Two stale references were corrected: React dashboard wording was replaced with the approved future Flutter contract, and documentation that named the B0 baseline as the only/latest model evidence now points to the B2 evaluation. The historical sequential-addition integration contract remains labeled as superseded. The source dataset's class IDs 62 and 91 have leading spaces in raw labels; loaders trim display labels without changing IDs. No source dataset file was edited.

## 14. Confirmed issues and fixes

| ID | Severity | Location and evidence | Impact / root cause | Action and status |
|---|---|---|---|---|
| AUD-01 | P1 — fixed | `smartnutri_backend/api.py` food-prediction response; `tests/test_automatic_scan_api.py` provenance regression | A real model output on a simulated camera capture was reported as non-simulated on the `/foods` contract because it inspected model/prediction flags but omitted scan/capture provenance | Response now ORs scan/capture provenance with inference provenance. Regression test passes. |
| AUD-02 | P2 — fixed | `smartnutri_backend/nutrition_catalog.py`; `tests/test_nutrition_catalog.py`; `docs/nutrition_catalog.md` | Two nutrient values had rounding inconsistent with exact USDA SR Legacy source values | Duck protein changed 23.5→23.48 g; apple carbohydrate 13.61→13.60 g. Source cross-check and calculation tests pass. |
| AUD-03 | P2 — fixed | `docs/backend_architecture.md`, `docs/api_reference.md`, `docs/portion_estimation.md`, `docs/automatic_scan_flow.md` | Stale React reference and B0-only/latest performance claims contradicted approved Flutter direction and newer B2 report | Documentation corrected to Flutter and B2 evidence; no frontend added. |
| AUD-04 | P1 — open | `smartnutri_backend/scale_monitor.py:206-243`; `smartnutri_backend/workflow.py:70-160` | The scan uses one initial stable observation, while monitoring only identifies removal and not material load changes during capture/inference | Add weight freshness/drift check bound to immutable placement snapshot and cancel/retry on drift. Blocks reliable real scan integrity. |
| AUD-05 | P1 — open | `firmware/weight_scale/src/main.cpp:394-399,418-423,469-471` | Direct HTTP and serial tare start without auth or explicit confirmation; backend token does not guard direct device API | Require trusted device authorization plus explicit empty-platform confirmation; verify locally before hardware demo. Firmware not changed in this review. |
| AUD-06 | P1 — open | `smartnutri_backend/portion.py:84,121,133`; `smartnutri_backend/agent.py:115-130` | Experimental visible-area weights always request review; policy refuses automatic save. A single image cannot establish physical mass split. | Keep review gate. Develop controlled weighed-meal dataset and calibrated estimator before automatic nutrition claims. |
| AUD-07 | P1 — open | `docs/audit/nutrition_coverage.csv`; `smartnutri_backend/nutrition_catalog.py` | Of 103 foreground classes, 96 lack records and 6 have ambiguous identities/preparation; only fries is exact reference match | Curate source-backed data with species/preparation confirmation and leave unsupported classes null. Reliable general nutrition is blocked. |
| AUD-08 | P2 — open | `smartnutri_backend/settings.py:41,139`; `scale_service.py:124-126` | Auto scan defaults disabled | Safe development default, but operator must explicitly enable after configuration; make setup state explicit in eventual client. |
| AUD-09 | P2 — open | `smartnutri_backend/camera.py:274-307` | `.local` hostname is allowed without connect-time IP/peer pinning | Resolve/validate private peer and redirect destination at connection time; no real phone test yet. |
| AUD-10 | P2 — open | `smartnutri_backend/api.py:667-691`; `repository.py:685` | Correction persists meal then scan in distinct transactions | Use one repository transaction to avoid divergence on crash between writes. |
| AUD-11 | P2 — open | `smartnutri_backend/vision.py:341-349` | Confidence averages low-resolution output before letterbox crop is applied to confidence cells | Apply the content-window mask to confidence calculation; confidence is presently uncalibrated. |
| AUD-12 | P2 — open | `firmware/weight_scale/src/main.cpp` tare/calibration sampling paths | Average samples are not guarded by a stable-window assertion | Add/verify stability guard after the physical sensor is characterized; do not treat compilation as calibration evidence. |
| AUD-13 | P2 — open | Local `../Datasets/FoodSeg103/id2label.json`, IDs 62 and 91 | Raw labels carry leading whitespace; loaders currently trim, IDs remain stable | No data modification. Normalize only display strings and retain original map provenance. Coverage CSV now preserves exact raw labels. |
| AUD-14 | P2 — open | Local `../Datasets/Nutrition5k_RGB/metadata.csv` | 46 of 3,490 ingredient-mass totals differ from dish total by >1 g; no split column | Investigate source/cooking/rounding before using as portion ground truth; data was left untouched. |
| AUD-15 | P2 — partial | Local Python environments and CUDA requirement resolution | Vulnerable setuptools/pip bootstrap packages reported; CUDA Torch wheels were skipped by pip-audit | Maintain an isolated CUDA-aware advisory workflow. No package update was attempted. |

No P0 issue was confirmed in the audited checkout. P1 issues AUD-04 through AUD-07 block reliable real-world automation/nutrition readiness.

## 15. Applied fixes

1. Preserved the `simulated` flag at the food endpoint even when a real-model adapter analyzes a simulated camera source; added regression coverage.
2. Corrected USDA SR Legacy record precision for duck protein and apple carbohydrates; added explicit multi-food arithmetic, duck/chicken ambiguity and missing-fruit assertions.
3. Corrected stale React/B0-only documentation to Flutter and current B2 evidence.
4. Created a 103-class coverage matrix and preserved exact class label strings, including raw leading whitespace for IDs 62 and 91.
5. No dataset, training code, firmware, checkpoint, environment dependency, frontend or design artifact was changed.

## 16. Remaining limitations and next steps

Prioritize placement weight/image consistency, direct tare protection, controlled component-by-component weighing trials, verified nutrition coverage, confidence/quality calibration, atomic corrections, and actual phone/ESP32 testing. Keep unsupported foods and unresolved masses nullable and reviewed. Use `readiness_report.md` for gating. Do not cite mock values, segmentation mIoU, or compile output as evidence of physical portion or nutrition accuracy.

## 17. Explicit scope confirmations

- Model training was not started, resumed, interrupted, or modified during the audit.
- No datasets were modified, downloaded, or overwritten.
- No checkpoints were overwritten or downloaded.
- The CUDA environment was not replaced or upgraded; it still reports CUDA available.
- No Flutter application was implemented.
- No `design.md` or design system was created.
- No firmware was flashed and no physical tare/calibration was performed.
