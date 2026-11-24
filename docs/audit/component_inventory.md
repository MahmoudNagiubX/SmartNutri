# SmartNutri Component Inventory

Audit date: 2026-11-24
Checkout: `feature/automatic-backend` at `5164eeb982c0d0b20d4e0c36eaac1cab5aadc6f3` before the audit commits
Remote: `origin` → `https://github.com/MahmoudNagiubX/SmartNutri`

## System map

| Component | Active implementation | Responsibility | Audit classification |
|---|---|---|---|
| FoodSeg103 CV | `foodseg_data.py`, `image_processing.py`, `train.py`, `evaluate.py`, `visualize.py`, `metrics.py` | Read official segmentation splits, synchronized preprocessing, SegFormer training/evaluation, metrics and visualizations | Active; training files and local run artifacts preserved |
| CV experiments | `class_weighting.py`, `training_losses.py`, `train_optimized.py`, `evaluate_optimized.py`, `configs/training`, `configs/losses`, `tools/compare_segmentation_runs.py`, `tools/compute_training_class_weights.py` | B0 imbalance and B2 experiments, ablations and comparison utilities | Research/experimental; no experiment started during this audit |
| Backend API | `smartnutri_backend/api.py`, `schemas.py`, `server.py`, `settings.py` | Versioned FastAPI routes, typed contracts, local startup and runtime configuration | Active |
| Scan orchestration | `scale_monitor.py`, `scale_service.py`, `workflow.py`, `agent.py` | Stable placement state machine, placement-key persistence, capture/inference/nutrition workflow and review decisions | Active; weight freshness during analysis remains unresolved |
| Camera | `camera.py` | Mock, configured-file and private-network snapshot capture, retries, image validation and quality reporting | Active; no physical-phone validation |
| Vision | `vision.py` | Mock adapter and local-only SegFormer adapter; FoodSeg103 class IDs, masks, regions and confidence summaries | Active; semantic masks do not give instance identity or mass |
| Portion/nutrition | `portion.py`, `nutrition.py`, `nutrition_catalog.py` | Experimental visible-area baseline, verified per-100 g lookup, macro calculations | Active; insufficient for reliable automatic nutrition |
| Persistence | `repository.py` | SQLite schema, plate/configuration, scan/meal records, correction data and restart latch | Active |
| Scale client | `smartnutri_weight/client.py`, `logic.py` | ESP32 HTTP contract, stable-read logic, mock/real source validation, calibration/tare client operations | Active |
| ESP32 firmware | `firmware/weight_scale/src/main.cpp`, `include/scale_logic.h`, `platformio.ini` | HX711 acquisition/filtering, Wi-Fi, calibration/tare, HTTP and serial interfaces | Active; compiled, not flashed or physically tested |
| Future mobile app | No Flutter source tree present | Native Android dashboard consuming documented REST/OpenAPI contracts | Missing by design; do not interpret docs as an implemented application |

## Repository files

### Active source and configuration

- Root CV pipeline: `foodseg_data.py`, `image_processing.py`, `train.py`, `evaluate.py`, `visualize.py`, and `metrics.py`.
- Optimized CV tools: `train_optimized.py`, `evaluate_optimized.py`, `class_weighting.py`, `training_losses.py`, and their JSON profiles under `configs/`.
- Backend Python package: 15 source modules plus a separate dependency file under `smartnutri_backend/`.
- Scale client package: two source modules under `smartnutri_weight/`.
- Firmware: source, scale logic header, example secret header, and PlatformIO config under `firmware/weight_scale/`.
- Image-processing profiles: baseline, safe augmentation, optional CLAHE/gamma/denoising ablations, and the B2 profile under `configs/image_processing/`.
- Root `requirements.txt` is the pinned CV/CUDA environment; backend dependencies are intentionally separate in `smartnutri_backend/requirements.txt`.

### Tests

The repository contains 21 test modules covering image processing, training contracts, API/schema/repository behavior, camera quality, scale client and monitor logic, scan flow, vision mapping, portion estimation, nutrition, and Flutter API contracts. Four current working-tree experiment test/config files belong to the user and were not staged or changed: `tests/test_segmentation_comparison.py`, `tests/test_training_experiment.py`, `configs/losses/`, and associated experiment tooling/docs.

### Documentation

Active documentation includes the README, image processing, model evaluation/training, backend architecture, automatic scan flow, AI agent, API reference, Flutter Android integration contract, scale wiring/operation, nutrition catalog, portion estimation, and integration contracts. The sequential-addition contract is explicitly retained as historical context and superseded by the whole-plate workflow. The older B0 baseline report is retained as historical evidence; `docs/cv_model_evaluation_report.md` is newer. The audit added `docs/audit/` and corrected stale React/B0-only language. No `design.md` exists.

### Generated and local-only artifacts

- `runs/` contains B0/B2 checkpoints, training metadata, evaluation tables/confusion matrices, visualizations and experiment outputs. These are local generated artifacts, not source files, and are ignored by Git.
- `.venv/`, `__pycache__/`, `firmware/weight_scale/.pio/`, local model weights, databases, captures, and caches are generated/local-only. None is tracked.
- No dataset or dataset archive is inside the repository. The sibling datasets are outside the repository root and are ignored by project naming patterns if copied under it.
- Five supplied PDF papers are outside the repository under `../Research Papers/`; `.pdf` is ignored. Their titles/opening sections were inspected for scope, not treated as validation evidence.

## Local datasets and research material

| Local path | Contents and count | Supported use | Limitations / integrity results |
|---|---|---|---|
| `../Datasets/FoodSeg103` | Official train: 4,983 rows across 3 Parquet shards; validation: 2,135 rows in 1 shard; 103 foreground classes plus background | Semantic segmentation; grayscale indexed masks | Full read-only scan: 7,118 decodable image/mask pairs, no invalid masks or class-list mismatch, no exact image hash shared across splits; 4 raw dimension mismatches all resolved by EXIF orientation; no official test split; IDs are split-local and overlap numerically |
| `../Datasets/Nutrition5k_RGB` | 3,490 metadata rows and matching 3,490 readable 640×480 PNGs; 240 ingredient names; no dataset split column | Separate future mass/nutrition research | Has dish total calories/mass plus ingredient names/mass strings; not pixel masks; 46 ingredient mass sums differ from metadata total by more than 1 g; RGB subset lacks the full dataset's videos/depth inputs |
| `../Research Papers` | Five PDFs: FoodSeg103 benchmark; Nutrition5k; Automated Food Weight and Content Estimation, Phase 2; Health to Eat smart plate; SCALEeat | Background research for system separation and future experiments | Literature motivates the architecture but published scores or hardware results do not validate SmartNutri |

FoodSeg103 raw labels for IDs 62 and 91 contain leading whitespace in the supplied `id2label.json`; the CV loader trims label strings while preserving IDs. The coverage matrix preserves exact raw strings. Normalizing whitespace for display is safe; remapping or reducing class IDs is not.

## Git and branch state

- Current branch: `feature/automatic-backend`; it matched `origin/feature/automatic-backend` before audit commits.
- Local branches include `main`, `audit/esp32-weight-integration`, and `feature/automatic-backend`; no branch was switched or reset during the audit.
- Open drafts: PR #1 `audit/esp32-weight-integration` → `main`; PR #2 `feature/automatic-backend` → `audit/esp32-weight-integration`. Neither was merged. Audit commits are intended for the existing feature branch and will update its draft PR.
- Before audit edits, `git ls-files` reported 83 tracked paths; no tracked file exceeded 1 MB, and the historical object scan found no blob above 1 MB. A signature scan of 228 unique historical blobs and the current working tree found no common credential patterns. This is not a substitute for GitHub secret scanning or Gitleaks.
- The user's modified `train.py` and untracked imbalance experiment files were preserved and excluded from audit commits.

## Dependency and runtime separation

- The active CV environment is the existing project `.venv`; it reports Python 3.12, PyTorch 2.11.0+cu128, CUDA 12.8 available, one device. It was not upgraded or replaced.
- Backend tests and server smoke ran in a separate temporary backend-only Python environment; its dependency ranges are separate from the training environment.
- PlatformIO 6.2.0 and the ESP32 build platform were installed under temporary directories for the compile. The repository firmware source and active training environment were not changed.
- A temporary audit-tool environment was used for pip-audit. Do not copy its tooling packages into the project environment.

## Missing, experimental, or unresolved components

- No Flutter application, visual design system, or UI is present or authorized in this task.
- No portion model with physical validation or documented learned priors exists; the current estimator is experimental visible-area allocation.
- A full 103-class verified nutrition catalog does not exist. Only 11 traceable USDA records are present.
- No physical scale calibration, smartphone camera, controlled meal portion, or hardware/network validation has been performed.
- The real camera snapshot adapter and local checkpoint adapter exist, but a real phone capture and real food-image prediction were not run in the hardware environment.
- The full source-tree, backend module and API responsibilities are documented in `full_project_audit.md`; specific passing and untested checks are in `test_evidence.md`.
