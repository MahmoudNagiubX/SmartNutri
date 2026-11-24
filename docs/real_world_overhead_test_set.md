# SmartNutri overhead-camera test set

This set measures transfer from FoodSeg103 images to the actual SmartNutri overhead smartphone camera. It must remain independent from training and checkpoint selection. Captures should be taken after the capture setup is fixed; do not fine-tune on these images before reporting the final metrics.

## Capture protocol

1. Mount the smartphone directly above the plate and keep camera mode, distance, angle, focus, and resolution consistent with the intended SmartNutri use.
2. Capture distinct meals and capture sessions. Avoid near-duplicate frames from a video or burst; keep repeated meals together as a single capture group.
3. Include common single ingredients, mixed dishes, overlaps, sauces, different plate colors, and realistic lighting/background variation. Record capture session, dish ID, and conditions in a separate CSV so results can be grouped by meal/session.
4. Keep original camera files unchanged. Do not crop, enhance, denoise, or color-correct the evaluation images. Preprocessing in the evaluation script performs the deterministic 512 letterbox and ImageNet normalization.
5. Create one single-channel indexed PNG mask for each image. Use the exact FoodSeg103 IDs 0–103 and use 255 only for pixels that are genuinely unlabeled/uncertain. Label with the local `Datasets/FoodSeg103/id2label.json` mapping; do not infer extra classes from nutrition catalog names.
6. Keep the set on local storage outside Git. Do not put these images or masks under a tracked source directory.

## Folder layout

```text
Datasets/SmartNutriOverheadTest/
  images/
    meal_0001.jpg
    meal_0002.jpg
  masks/
    meal_0001.png
    meal_0002.png
```

Image and mask stems must match. The evaluator rejects missing or unpaired files and invalid mask IDs.

## Evaluation command

After the model is selected, evaluate the B0 baseline and B2 on the identical camera images. The optional original-resolution and horizontal-flip TTA results are reported separately.

```powershell
$b2Run = Get-ChildItem .\runs -Directory -Filter 'foodseg103-segformer-b2-optimized-*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe .\evaluate_optimized.py --model-dirs .\runs\foodseg103-segformer-b0\best_model (Join-Path $b2Run.FullName 'best_model') --split external --external-root ..\Datasets\SmartNutriOverheadTest --original-resolution --hflip-tta
```

Review the run's `metrics.json`, per-class CSVs, presence precision/recall, and confusion matrices together. Report image counts and capture-session coverage with the metrics; this small project test set is a camera-domain check, not a replacement for FoodSeg103 benchmark results.
