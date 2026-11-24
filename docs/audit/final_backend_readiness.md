# SmartNutri Final Backend Readiness

Assessment: 2026-11-24. SegFormer-B2 is frozen as the selected local inference model. No training was started, resumed, or modified during this readiness workflow.

## Readiness summary

| Area | Status | Evidence and remaining boundary |
| --- | --- | --- |
| Food segmentation | **READY for local software integration** | B2 loaded through the FastAPI scan route with a local FoodSeg103 validation image. The adapter verifies the B0/B2 architecture, all 104 class IDs and labels, checkpoint preprocessing metadata, and semantic-mask output. FoodSeg103 validation scores are not real-world camera accuracy. |
| Uncertainty and portion policy | **READY for reviewed software workflow** | Configurable confidence and minimum-region thresholds, ambiguous-class review, mask/prediction consistency checks, same-placement scale revalidation, and mass-conserving corrections are in place. Pixel-area mass remains experimental and cannot be accepted automatically. |
| Nutrition catalog | **PARTIAL** | 92 exact USDA SR Legacy records provide candidates for 68/103 foreground labels. All candidates require explicit identity/preparation confirmation, so automatic class-only lookup resolves zero classes. Recipe foods and some FoodSeg103 classes remain unsupported. |
| Flutter API contract | **READY for client implementation** | The 26-path OpenAPI v1 schema is checked in and regression-tested. A full simulated API workflow exercised scan, image, food, portion, nutrition, meal, decision, history, and correction routes. There is no Flutter app or phone-to-backend test in this repository. |
| Real hardware | **NOT READY for physical claims** | Firmware builds; Python failure fixtures pass; no board was flashed and no scale, camera, calibration, tare, or Wi-Fi behavior was physically measured. |

## Selected model and CV results

The selected model is the completed local SegFormer-B2 checkpoint at `runs/foodseg103-segformer-b2-optimized-20261124_075454/best_model`. It completed 33 epochs, selected epoch 23 on a 498-image internal training-derived split, and reached internal foreground mIoU 0.360011, foreground Dice 0.468488, and pixel accuracy 0.786231. Training is closed for this project milestone.

On all 2,135 official FoodSeg103 validation images, using the matched 512-pixel deterministic preprocessing and foreground scoring protocol:

| Model / mode | Foreground mIoU | Foreground Dice | Pixel accuracy | Macro presence precision / recall / F1 |
| --- | ---: | ---: | ---: | ---: |
| B0, one pass | 0.191775 | 0.269975 | 0.733852 | 0.3376 / 0.3936 / 0.2687 |
| **B2, one pass** | **0.364898** | **0.494406** | **0.781938** | **0.3300 / 0.7020 / 0.4233** |
| B0, horizontal-flip TTA | 0.195610 | 0.274161 | 0.738893 | 0.3658 / 0.3932 / 0.2757 |
| B2, horizontal-flip TTA | 0.369469 | 0.498852 | 0.784023 | 0.3452 / 0.6975 / 0.4368 |

Original-resolution B2 scoring was 0.340845 mIoU, 0.466678 Dice, and 0.791797 pixel accuracy. HFlip TTA remains optional and disabled by default: for B2 it added 0.004571 mIoU and 0.013507 macro presence F1, at roughly 1.81x forward latency.

**Split disclosure:** the existing B0 used all official validation images for checkpoint selection. B2 used a 10% internal split of the official training partition and did not use the official validation set for selection. The matched comparison is useful for deployment choice, but the official validation result is not an untouched estimate for B0. Treat the B2 score as the less selection-biased estimate.

### Architecture, processing, and training record

FoodSeg103 has 4,983 official training examples and 2,135 official validation examples, indexed masks for 104 IDs (background plus 103 ingredients), and no separate official test split. B2 uses a MiT-B2 encoder with stage depths 3/4/6/3 and widths 64/128/320/512, plus a 768-wide segmentation decoder adapted to 104 labels. The selected initialization was the pinned NVIDIA ADE20K SegFormer-B2 checkpoint with its PyTorch weights; the FoodSeg103 head was newly initialized. The upstream license restricts this checkpoint to non-commercial research/evaluation use.

Image processing applies EXIF orientation, RGB conversion, aspect-preserving 512×512 letterboxing, bilinear image resize, nearest-neighbor mask resize, and ImageNet normalization. Padding labels are `255` and are ignored by loss and metrics. Validation is deterministic. Training synchronizes image/mask horizontal flips (p=0.5), random scale 0.9–1.1 (p=0.5), rotations up to ±5° (p=0.25), and mild brightness/contrast/saturation jitter (p=0.5); low-probability blur and low-amplitude noise are each p=0.05.

The objective was 0.5 class-weighted focal loss (gamma 2) plus 0.5 soft Dice loss with background excluded from Dice. Moderate square-root frequency weights came only from the 4,485 gradient-training images, were mean-normalized, and capped at 2.0; this avoids combining strong inverse-frequency weights with focal loss. Training used AdamW, initial learning rate 6e-5, weight decay 0.01, 5% linear warmup and polynomial decay, seed 42, effective batch 8 (microbatch 2 with accumulation 4), mixed precision, up to 40 epochs, and early-stopping patience 10. Actual saved-run curves are [shown here](cv_metrics/training_curves.png). The [CV model evaluation report](../cv_model_evaluation_report.md) documents the research references and detailed protocol.

B2 produced predictions for 101 of 103 foreground classes present in ground truth and had at least one correctly labeled pixel for 99 classes. For the bottom quartile of 26 labels by ground-truth pixel support, it predicted 24 and had true-positive pixels for 22. Best class IoU: broccoli 0.8592, corn 0.8552, carrot 0.8095, strawberry 0.7939, and green beans 0.7933. Weakest supported classes: bamboo shoots, pudding, kelp, and salad at 0; king oyster mushroom 0.0001; hamburg 0.0044. Notable confusions were steak→pork (3.06M pixels), chicken/duck→pork (2.72M), and pork→steak (2.17M).

Batch-size-1 RTX 4050 Laptop measurements at 512×512, forward only (preprocessing and host/device transfer excluded): B0 33.05 ms mean and B2 71.57 ms mean. B2 p50/p95 was 71.50/92.89 ms; B2 HFlip-TTA mean was 129.73 ms. Peak PyTorch allocation was 653 MiB for B2; system GPU use during evaluation was approximately 2.2–2.5 GiB including desktop/context use. These are model-path measurements, not a phone, network, scale, or full-server latency guarantee.

Compact per-class scores, presence tables, confusion matrices, and the actual training curves are included under [the CV metric artifacts](cv_metrics/README.md). The 30 B0/B2 source-image panels and contact sheet remain in the ignored local evaluation run under `runs/segmentation-comparison-20261124_160934/visual_analysis/` so dataset imagery stays outside Git. The detailed report is [CV model evaluation](../cv_model_evaluation_report.md). Datasets and checkpoints are excluded from Git.

## Deployment configuration

B2 is the preferred real CV model. The API retains mock mode as its safe demo default; enabling the local model is explicit. From the repository root in PowerShell:

```powershell
$env:SMARTNUTRI_CV_MODE = 'segformer'
$env:SMARTNUTRI_CV_MODEL = 'b2'
$env:SMARTNUTRI_CV_DEVICE = 'auto'
$env:SMARTNUTRI_FOODSEG_ROOT = (Resolve-Path '..\Datasets\FoodSeg103').Path
$env:SMARTNUTRI_SCALE_MODE = 'mock'
$env:SMARTNUTRI_CAMERA_MODE = 'mock'
.\.venv\Scripts\python.exe -m smartnutri_backend.server
```

The checkpoint path can be omitted for the frozen local B2 path; `SMARTNUTRI_CHECKPOINT_DIR` can override it for a different local checkpoint. The backend does not download weights. B0 remains selectable by setting `SMARTNUTRI_CV_MODEL=b0`. HFlip TTA is controlled by `SMARTNUTRI_CV_HFLIP_TTA` and is off by default. The local model returns the same FoodSeg103 IDs `0..103` (background `0`; food `1..103`) and masks consumed by the existing backend contract.

The API smoke used a local FoodSeg103 validation image and mock scale. It loaded the actual B2 checkpoint through FastAPI, verified the emitted labels against the local 104-class mapping, fetched the PNG mask through the backend, and returned a review-state scan with no nutrition totals. Because the test scale was simulated, it did not claim physical mass or nutrition.

## Food records and agent policy

The catalog contains 92 USDA SR Legacy source rows with exact FDC identifiers, source descriptions, per-100 g energy/macros, and source preparation descriptors. The values were verified against the official USDA SR Legacy CSV archive; the archive is local audit input and is not part of the repository. The added records expand FoodSeg103 candidate coverage to 68/103 foreground labels; 35 have no catalog candidate.

Candidate coverage includes common ingredients used in Egyptian meals (rice, egg, dates, tilapia, shrimp, pita, eggplant, potato, garlic, okra, green beans, onion, and pepper). These are generic USDA entries, not Egyptian recipes. The FoodSeg103-to-FDC relationship remains a candidate association requiring explicit identity and preparation confirmation. No dish-level recipe values are invented. Consequently, all 92 rows require confirmation and automatic lookup by segmentation class alone returns no record. The backend keeps missing or ambiguous item nutrients and meal totals null. See [catalog provenance and coverage](../nutrition_catalog.md).

The transparent nutrition agent is a policy layer, not a trained model. It asks for review when class confidence is below the configurable default 0.55, regions are below the configurable default 0.1% of image area or are fragmented, the prediction list conflicts with mask IDs/areas, or the class itself remains ambiguous (including chicken/duck, cilantro/mint, and pepper varieties). Uncertain masks are withheld from portion allocation and represented as unallocated mass. User corrections must conserve the measured total within 0.05 g and keep explicit source IDs; no inferred macros fill missing data.

## Portions and placement safety

The only automatic multi-food estimator remains `experimental_visual_area_proportion_v1`. It allocates net scale mass by pixel area, labels each item for review, and is not a measured per-food mass. Background and uncertain predicted regions do not receive confirmed portions; their corresponding mass remains unallocated. The API supports a user-confirmed single-food correction equal to the scale-measured net total and reviewed multi-food corrections that sum to that total. Operators can enter sequentially weighed ingredients through the same correction API; a guided sequential-weighing UI and hardware validation are not implemented.

For each scan, the workflow rechecks the active placement and obtains a fresh stable scale reading after capture and after inference. It rejects a changed placement key, source, unstable/missing reading, or weight change above the configurable 5 g default. Tests cover changed/unstable readings, camera failure fixtures, and prediction-mask mismatch. This narrows association risk but does not replace physical timing and weighing tests.

Backend and direct ESP32 tare/calibration calls require control credentials in real mode. Tare also requires action-specific confirmation; the firmware rejects an empty or shorter than 32-character control secret and uses `X-SmartNutri-Control-Token`. The firmware compiled in a unique temporary directory with PlatformIO, Espressif32 7.0.1 and Arduino-ESP32 2.0.17; no firmware was flashed. HTTP remains unencrypted, so the scale must stay on a trusted private LAN.

## Flutter and software workflow

The checked-in [OpenAPI v1 snapshot](../contracts/openapi-v1.json) freezes all 26 API paths and is compared against the running FastAPI schema. The test workflow creates a plate profile, starts a clearly marked synthetic placement, reads scan/food/portion/nutrition/meal/decision/image resources, and applies a mass-conserving single-item mock correction. It asserts that mock input and missing nutrition keep totals null. It tests the backend contract for a future Flutter app; the repository contains no Flutter project, generated client, or on-device UI to run.

The existing guide [Flutter Android integration](../flutter_android_integration.md) documents route polling, binary masks/images, token handling, private-LAN binding, and Android networking. A physical phone-to-laptop run remains outstanding.

## Test evidence and remaining blockers

The final backend/API/agent/nutrition/portion/camera/scale/vision tests passed: **136 tests across 17 modules**. `compileall` passed for the backend, weight client, and tools. The USDA verifier passed for all **92 source records**; the coverage generator reported 68 candidate classes. FastAPI/httpx and their small dependencies were installed temporarily in the local CV environment to run the API tests and B2 smoke, then removed. Torch `2.11.0+cu128` and torchvision `0.26.0+cu128` were confirmed unchanged. The full CV benchmark, visual panels, and training curves were already generated from the completed run; no model evaluation requiring additional training was performed in this integration workflow.

Two API smoke runs loaded B2 on CUDA through the default local checkpoint selection and completed the first mock placement workflow in 8.73–12.88 seconds, including lazy model loading; this is a cold-start integration range, not steady-state inference latency. The firmware build succeeded in 52.33 seconds in a unique temporary copy using PlatformIO Core 6.1.19, Espressif32 7.0.1, and Arduino-ESP32 2.0.17. PlatformIO emitted an obsolete-Core notice, but compilation and linking succeeded. No board was flashed. The firmware source was not built in place, and the test suite did not invoke any training or full-evaluation command.

Still required before real-user claims:

1. Physical ESP32/HX711 calibration and repeatability checks with at least two verified masses, tare confirmation, plate removal, and sensor/network failure/recovery.
2. Phone overhead snapshot capture over the trusted LAN, lighting/quality checks, placement timing, and a separate manually labeled camera-domain test set.
3. Repeated single-food measurements and independently weighed mixed plates before claiming per-item mass accuracy. Pixel-area allocation is not a calibrated mass model.
4. A Flutter client implementation and on-device contract/error-state exercise; current readiness covers the frozen backend contract only.
5. Food identity/preparation review for each candidate USDA mapping before that record is used in a meal calculation. Composite Egyptian dishes remain unsupported without reviewed recipe data.

### Smartphone camera and LAN test instructions

1. Mount the phone overhead at the intended fixed height and angle. Keep original camera files and use a separate capture log. Do not fine-tune on this test set.
2. Bind the backend only to the laptop's current private Wi-Fi IPv4 address; allow the phone through the Windows Private firewall only. Configure `SMARTNUTRI_CAMERA_MODE=ip_snapshot` and `SMARTNUTRI_CAMERA_URL` to the phone camera app's private snapshot URL. Keep CORS disabled for a native client.
3. Test `GET /api/v1/camera/status`, the snapshot endpoint, `GET /api/v1/scale/status`, and tare/calibration individually. Use one shared, private control secret for backend and ESP32; never place it in a Flutter build.
4. Start with auto-scan disabled. Capture an empty plate, one known food, then a mixed plate. Verify the image belongs to the same stable placement, the net scale total is sensible, and any unstable/changed reading stops the scan for review.
5. Weigh each food independently for the reference record, then verify that corrected item masses sum to the measured plate net within 0.05 g. Review masks manually and label a held-out subset with IDs from local `id2label.json` before computing camera-domain metrics.
6. Enable auto-scan only after independent camera and scale checks. Keep all captured images/masks outside Git and report this test separately from FoodSeg103 validation.

## Research and data sources

- [Official FoodSeg103 benchmark](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1)
- [FoodSeg103 SegFormer-B2 comparative study](https://github.com/mahendran-jinachandran/FoodSeg103Benchmarking)
- [FoodSeg103 SegFormer-B0/B1 study and implementation](https://github.com/joneshshrestha/ingredient-level-segmentation)
- [FoodSeg103 food-specific checkpoint collection](https://huggingface.co/mericberktas/foodseg-models) (reviewed but not selected because split, class mapping, reported score, and checkpoint use rights could not be independently verified)
- [Pinned NVIDIA ADE20K SegFormer-B2 PyTorch checkpoint](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/tree/de01bae28967510f9ddd496c60a969357195400c) and [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
- [USDA FoodData Central download archive](https://fdc.nal.usda.gov/download-datasets/) for the SR Legacy source rows used in the nutrition candidate catalog
