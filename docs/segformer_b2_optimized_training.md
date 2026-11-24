# SmartNutri SegFormer-B2 training

## Training outcome

The optimized run `runs/foodseg103-segformer-b2-optimized-20261124_075454` completed 33 epochs and early-stopped with the best checkpoint at epoch 23. Its internal validation foreground mIoU was 0.360011, foreground Dice 0.468488, and pixel accuracy 0.786231. Final comparison on all 2,135 official validation images is recorded in [the CV model evaluation report](cv_model_evaluation_report.md). No additional training was started during final evaluation.

## Checkpoint decision

| Candidate | Evidence | Decision |
| --- | --- | --- |
| FoodSeg B2, `mericberktas/foodseg-models` | The model card lists MiT-B2, 104 FoodSeg103 classes, epoch 40, and mIoU 0.661 over classes present. It says the scores were not recomputed for upload. The `.pth` artifact is 329,635,790 bytes (about 314 MiB). The model card declares no weights license; it does not identify the exact split or provide a reproducible evaluation report. The config is listed but its class map could not be independently checked from the published card and file endpoint. | Not selected. The score, split integrity, class map, and redistribution/use rights cannot be verified well enough for this project. |
| NVIDIA ADE20K SegFormer-B2 | `nvidia/segformer-b2-finetuned-ade-512-512`, pinned to revision `de01bae28967510f9ddd496c60a969357195400c`; use the official `pytorch_model.bin` (110,000,905 bytes, about 105 MiB; SHA-256 `187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3`). The model card identifies B2 and 512×512 ADE20K fine-tuning. The 150-class decoder is replaced with a freshly initialized FoodSeg103 104-class head. | Selected fallback. It is version-pinned, PyTorch-weight loading is explicit, and the architecture is checked at load time. NVIDIA's SegFormer license limits use to non-commercial research/evaluation. |

The FoodSeg103 data package maps IDs 0–103 to background and the 103 ingredient labels. The training loader strips incidental whitespace from the source names and validates IDs 0–103. Every saved model is checked against that exact local mapping before evaluation.

The public FoodSeg B2 card and repository are available at [foodseg-models](https://huggingface.co/mericberktas/foodseg-models). Its reported 0.661 mIoU is not used as a target or a verified result. The cited [B2 comparative study](https://github.com/mahendran-jinachandran/FoodSeg103Benchmarking) reports B2 experiment scores and ADE20K pretraining, but its repository tree contains only a README; the README points trained weights to Google Drive and does not provide versioned training/evaluation code. The [B0/B1 study](https://github.com/joneshshrestha/ingredient-level-segmentation) includes code and configs, reports its split method and training settings, and is MIT-licensed. The [official benchmark](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1) and local dataset card identify FoodSeg103 and its 104 label IDs.

The ADE20K model is published with license metadata `other`; the upstream [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) permits research/evaluation use and restricts commercial use. The FoodSeg103 data is separately distributed under Apache-2.0 per the official benchmark.

**Pretrained-file provenance note:** the completed run's historical `run_config.json` records `model.safetensors`, `use_safetensors=true`, and a byte count matching the official PyTorch file, although the pinned upstream revision currently exposes `pytorch_model.bin` at that size. The exact initial artifact used by the completed run cannot be independently established from the saved fine-tuned checkpoint. The tracked training config now selects the official pinned `pytorch_model.bin` and records its expected size and SHA-256; the trainer checks that its filename matches `use_safetensors=false`. Historical run metadata is unchanged, and the recorded SHA-256 is not independently rechecked at runtime.

## Split integrity

The local Parquet package has 4,983 `train` rows and 2,135 `validation` rows. The cited B0/B1 implementation treats the 2,135-image partition as the held-out test split and carves 498 validation images from the official 4,983-image training split. The new B2 trainer follows that clean selection pattern:

- A seed-42 ID split makes 4,485 gradient-training rows and 498 internal model-selection rows.
- Class frequencies and capped class weights are computed from the 4,485 gradient-training rows only.
- The 2,135-row external partition is not loaded by the trainer. It remains for final B0/B2 comparison and is also the split the existing B0 run used for checkpoint selection.

The existing B0 run config confirms that B0 selected checkpoints using all 2,135 validation rows. Because B0 used that partition during selection, the matched comparison is useful for project decisions but is not a pristine held-out comparison for B0. B2's checkpoint selection used only 498 examples carved from the official training split; the official validation partition was not used for its training or selection.

## Training settings

`configs/training/segformer_b2_foodseg103.json` is the single comprehensive configuration.

| Setting | Value |
| --- | --- |
| Architecture | SegFormer-B2 / MiT-B2; 104 output classes |
| Input | 512×512; aspect-preserving bilinear image resize, nearest-neighbor mask resize, mean-color letterbox |
| Training | 40 maximum epochs; early-stop patience 10 on internal foreground mIoU |
| Optimizer | AdamW; learning rate 6e-5; weight decay 0.01 |
| Schedule | 5% linear warmup, then polynomial decay with power 1 |
| Batch | Microbatch 2 × accumulation 4 = effective batch 8 |
| Precision | BF16 AMP when CUDA supports it, otherwise FP16 AMP; scaler state is saved when enabled |
| Seed | 42 |
| Loss | 0.5 focal (gamma 2) + 0.5 soft Dice; Dice ignores background by default |
| Imbalance | Square-root frequency weights from gradient-training masks, mean-normalized and capped at 2.0 |
| Augmentation | Synchronized horizontal flip, 0.9–1.1 scale, ±5° rotation, mild color jitter; very mild blur/noise each at 5% probability |
| Validation | No augmentation; deterministic preprocessing; 255 padding ignored in loss and metrics |

The B0/B1 study reports a 512 input, AdamW, 6e-5 learning rate, 0.01 weight decay, polynomial schedule, seed 42, batch 8, 40 epochs, and a 10% internal validation split. Its best checkpoints were at epochs 34 and 37, supporting a 40-epoch ceiling and patience 10. B2 uses the same effective batch through accumulation. The evaluation tool reports metrics at both input and original-mask resolution.

Augmentation is train-only. Photometric transforms never touch masks; all geometry transforms use bilinear image interpolation, nearest-neighbor mask interpolation, and 255 for newly exposed mask pixels. Brightness, contrast, saturation, blur, and noise stay mild. CLAHE, gamma correction, and denoising remain off.

## GPU, disk, and time

On the current RTX 4050 Laptop GPU (6 GiB), a synthetic B2 microbatch-2 BF16 forward/backward with the implemented focal+Dice loss completed successfully at 2.01 GiB peak allocated and 2.84 GiB peak reserved. Three warmed synthetic steps averaged 0.203 seconds per microbatch of two. A separate 20-image real-data preprocessing sample averaged 14.2 images/second. These are readiness measurements, not a full-run benchmark.

At those rates, the model-step component is roughly 5.1 hours for 40 epochs. Sequential CPU preprocessing, internal validation, and checkpoint writes add time; plan roughly 8–10 hours for the full 40-epoch ceiling, with early stopping potentially shorter. The run writes one best Hugging Face model (about 105 MiB) plus a resumable model/AdamW state (roughly 315 MiB, depending on serialization) and JSON history. Run output and weights are covered by `.gitignore`.

Runs get unique timestamped directories under `runs/`. Existing paths are never reused. `last_training_state.pt` records model, optimizer, scheduler, scaler, epoch, history, and RNG states. Resume with:

```powershell
.\.venv\Scripts\python.exe .\train_optimized.py --config .\configs\training\segformer_b2_foodseg103.json --resume .\runs\<run-directory>\last_training_state.pt
```

## Evaluation and integration

`evaluate_optimized.py` evaluates one or more local checkpoints with the same rows, input preprocessing, batch size 1, and device settings. It writes aggregate metrics, all 104 per-class metrics, pixel confusion matrices, image-level ingredient-presence precision/recall, rare-class coverage, and mean/p50/p95 model-forward latency. The rare-class report labels the bottom quartile by ground-truth pixel support on the evaluated split. Original-resolution evaluation is optional; it removes letterbox padding, then uses tiled bilinear logit resizing for large masks. Horizontal-flip TTA is optional and written as a separate result.

The final same-protocol evaluation used:

```powershell
$b2Run = Get-ChildItem .\runs -Directory -Filter 'foodseg103-segformer-b2-optimized-*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe .\evaluate_optimized.py --model-dirs .\runs\foodseg103-segformer-b0\best_model (Join-Path $b2Run.FullName 'best_model') --split validation --original-resolution --hflip-tta
```

The best checkpoint is saved in standard Hugging Face `SegformerForSemanticSegmentation` format with the same 104 class IDs and output-channel contract as B0. Each `best_model/` also receives the `training_metadata.json` required by `smartnutri_backend.vision.SegFormerInferenceService`, including the deterministic preprocessing profile and ignore label. Set `SMARTNUTRI_CV_MODE=segformer` and point `SMARTNUTRI_CHECKPOINT_DIR` at either B0 or B2 to select a local checkpoint; the backend loads weights locally only.

## Manual training command

This is the original training entry point for reproducibility. It is included for reference only; final evaluation did not run it.

```powershell
.\.venv\Scripts\python.exe .\train_optimized.py --config .\configs\training\segformer_b2_foodseg103.json
```
