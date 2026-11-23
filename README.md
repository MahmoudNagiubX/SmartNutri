# SmartNutri — FoodSeg103 segmentation

This folder contains SmartNutri's first model stage: ingredient-level semantic segmentation on FoodSeg103. Nutrition5k_RGB remains separate for later food-mass and nutrient experiments.

## Inspected datasets

The scripts use datasets already stored beside this project:

- `../Datasets/FoodSeg103`: 4,983 official train rows and 2,135 official validation rows in Parquet shards. Each row contains a JPEG image and a single-channel PNG mask. `id2label.json` maps mask values `0..103` to background and 103 ingredient classes.
- `../Datasets/Nutrition5k_RGB`: 3,490 RGB PNGs and 3,490 matching CSV rows, all 640×480. It has no supplied train/validation split and is not read by these segmentation scripts.

The full local audit decoded all 7,118 FoodSeg103 image-mask pairs and all 3,490 Nutrition5k images. FoodSeg masks use valid class IDs and agree with `classes_on_image`. Four image-mask pairs whose raw dimensions differed were explained by JPEG EXIF orientation tags 6 or 8; the loader applies EXIF orientation before resizing. The numeric row ids are split-local and repeat across splits, so the loader does not treat them as global identifiers. No exact image bytes cross the official train/validation boundary. There are 42 duplicate occurrences within train and 6 within validation; the pipeline preserves the supplied rows.

## Pipeline choices

- Reads Parquet row groups directly with PyArrow; it does not extract, rewrite, or copy either dataset.
- Keeps the official FoodSeg103 train and validation split.
- Runs a shared pipeline from `image_processing.py`: EXIF orientation, RGB conversion, aspect-preserving 512×512 letterboxing, bilinear image resize, nearest-neighbor mask resize, SegFormer/ImageNet normalization, and ignore label `255` for mask padding.
- Validates the 104 FoodSeg103 labels (`0..103`) plus padding label `255`. Geometric transforms are synchronized between image and mask; photometric transforms touch only training images. Validation remains deterministic.
- Uses SegFormer-B0 initialized from `nvidia/segformer-b0-finetuned-ade-512-512`; the 104-class output layer is adapted for FoodSeg103. If those pretrained model weights are not already cached, Hugging Face downloads them the first time training starts. The FoodSeg103 and Nutrition5k datasets are never downloaded by these scripts.
- Defaults to at most 20 epochs, early-stopping patience 6, batch size 2, gradient accumulation 4, CUDA FP16 mixed precision, seed 42, and zero data-loader workers for Windows. This is a conservative starting point for a 6 GB GPU; peak VRAM has not been measured by a training run.
- Selects the best checkpoint by validation foreground mean IoU. Evaluation reports pixel accuracy, mIoU, and Dice both with and without background, plus per-class metrics.

Preprocessing experiment profiles are in `configs/image_processing/`:

- `baseline.json`: deterministic resize/pad and normalization only; this is the default.
- `safe_augmentation.json`: adds training-only flips, small rotations, mild color changes, optional blur, and low-level Gaussian noise.
- `clahe_ablation.json`, `gamma_ablation.json`, and `denoising_ablation.json`: separate optional image-only experiments. Compare each against baseline using validation results.

FoodSeg103 has no separate official test split. Validation metrics are model-selection results and should be reported as validation results, not as a held-out test score. Ingredient segmentation alone does not produce calories or portion weights.

## Environment

The existing `.venv` was verified with Python 3.12.10, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Transformers 5.24.0, PyArrow 26.0.1, NumPy 2.5.4, and Pillow 13.0.0. A real CUDA matrix multiplication succeeded on the NVIDIA GeForce RTX 4050 Laptop GPU. These scripts do not install or upgrade packages.

Keep the existing CUDA-enabled PyTorch build when adding packages. A generic dependency install can replace it with a CPU-only Torch wheel.

## Train

From PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe .\train.py
```

Defaults are 20 epochs, 512-pixel input, batch size 2, accumulation 4, learning rate `6e-5`, seed 42, and output under `runs\foodseg103-segformer-b0`. Training evaluates the official validation split after every epoch and saves the best checkpoint under `best_model`.

Useful overrides:

```powershell
.\.venv\Scripts\python.exe .\train.py --epochs 10 --batch-size 1 --gradient-accumulation-steps 8
.\.venv\Scripts\python.exe .\train.py --foodseg-root 'D:\Datasets\FoodSeg103' --output-dir '.\runs\foodseg103-trial-2'
```

Training refuses to reuse a non-empty output directory. Choose a new `--output-dir` for each run.

The default profile is baseline. To select safe augmentation, use:

```powershell
.\.venv\Scripts\python.exe .\train.py --preprocessing-config .\configs\image_processing\safe_augmentation.json --output-dir .\runs\foodseg103-safe-augmentation
```

## Evaluate

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --model-dir '.\runs\foodseg103-segformer-b0\best_model'
```

The default evaluates the official validation split. It writes `metrics.json` and `per_class_metrics.csv` beside the checkpoint. The saved training metadata supplies the image size automatically.

## Visualize

```powershell
.\.venv\Scripts\python.exe .\visualize.py --model-dir '.\runs\foodseg103-segformer-b0\best_model' --start-index 0 --count 4
```

This saves side-by-side source image, ground-truth overlay, and prediction overlay PNGs under the checkpoint's `visualizations` folder.

## Automatic multi-food backend

The laptop-side BE-01 backend now provides a versioned FastAPI API, background scale monitor, saved-plate and explicit live-plate tare modes, phone/file/mock camera adapters, local-only SegFormer inference, an experimental portion estimator, a transparent nutrition decision policy, verified-record nutrition calculations, and SQLite meal history/corrections. The API and runtime are independent of the existing training process. No frontend, production nutrition dataset, or validated portion-weight model is included.

Read the implementation notes in [backend architecture](docs/backend_architecture.md), [automatic scan flow](docs/automatic_scan_flow.md), [portion estimation](docs/portion_estimation.md), [agent policy](docs/ai_agent.md), [API reference](docs/api_reference.md), and the [Flutter Android integration contract](docs/flutter_android_integration.md). BE-02 prepares typed REST/OpenAPI integration contracts only; the Flutter app is not implemented here. These whole-plate contracts supersede the earlier sequential-addition proposal in [integration_contracts.md](docs/integration_contracts.md). The independent ESP32 module remains documented in [weight_scale.md](docs/weight_scale.md).

Install the backend into a separate Python environment so the CUDA training environment is not changed. From PowerShell:

```powershell
$backendEnv = Join-Path $env:LOCALAPPDATA 'SmartNutriBackendVenv'
python -m venv $backendEnv
& (Join-Path $backendEnv 'Scripts\python.exe') -m pip install -r .\smartnutri_backend\requirements.txt
$env:SMARTNUTRI_SCALE_MODE = 'mock'
$env:SMARTNUTRI_AUTO_SCAN = 'false'
& (Join-Path $backendEnv 'Scripts\python.exe') -m smartnutri_backend.server
```

The mock scan endpoint accepts only synthetic test values and labels its responses. Real scale control requires `SMARTNUTRI_CONTROL_TOKEN`; set it outside Git. The launcher defaults to loopback, CORS is disabled unless explicit browser origins are configured, and the backend uses a mock camera/CV adapter with auto-scan disabled. For Flutter over trusted Wi-Fi, bind to the laptop's exact private IPv4 address and scope the Windows Private-profile firewall to the phone or home subnet; the launcher rejects wildcard and public binds. Use local checkpoint metadata and `SMARTNUTRI_CV_MODE=segformer` for real inference; model loading is lazy and local-only. See the Flutter Android integration guide for polling, camera-service boundaries, and Android network permissions.

To regenerate the actual-source preprocessing figure without a model or training:

```powershell
.\.venv\Scripts\python.exe .\tools\visualize_preprocessing.py
```

It reads local validation Parquet rows and writes `docs/assets/preprocessing_before_after.png`; it does not modify the dataset.

## Tests

Run the lightweight test suite with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Research references

The five supplied papers informed the project boundary and staged design:

1. *A Large-Scale Benchmark for Food Image Segmentation* — ingredient labels and pixel-level masks for this stage.
2. *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food* — later nutrition prediction using richer mass and nutrient labels.
3. *Automated Food Weight and Content Estimation Using Computer Vision and AI Algorithms: Phase 2* — food identification and portion estimation as related tasks.
4. *Health to Eat: A Smart Plate with Food Recognition, Classification, and Weight Measurement for Type-2 Diabetic Mellitus Patients’ Nutrition Control* — recognition, measured weight, and nutrient information as distinct inputs.
5. *SCALEeat: Vision-Guided Food Scale for Automated Macronutrient Estimation* — recognition combined with measured weight and food-composition data.

Nutrition5k is reserved for the later nutrition stage; the current model trains only on FoodSeg103 masks.

To recreate the current runtime, install the pinned packages from this folder with `python -m pip install -r requirements.txt`. Its Torch entries target the verified CUDA 12.8 build.
