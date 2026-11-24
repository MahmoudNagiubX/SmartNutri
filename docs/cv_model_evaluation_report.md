# SmartNutri CV Model Evaluation

## Scope and split integrity

This report compares the existing SegFormer-B0 model and the completed SegFormer-B2 model on the same 2,135 official FoodSeg103 validation images. Both use the baseline deterministic 512-pixel letterbox and ImageNet normalization, ignore padded target pixels (`255`), and use batch-size-1 CUDA FP16 inference. Pixel metrics are aggregated over the full split. Original-resolution metrics remove letterbox padding and resize logits before class selection; they are supplementary. Horizontal-flip test-time augmentation (TTA) is reported separately from single-pass inference.

The B2 run used a seed-42 split of the official 4,983-image training partition: 4,485 images for gradient updates and 498 for checkpoint selection. Its official-validation mIoU was not used for training or checkpoint selection. The existing B0 run did use all 2,135 official validation images for checkpoint selection. Therefore, the matched comparison is useful for model choice but is not a pristine held-out comparison for B0; the B2 result is the more independent estimate.

## Dataset and processing

FoodSeg103 contains 4,983 training and 2,135 official validation examples with indexed masks for 104 IDs: background `0` plus 103 ingredient labels. The loader applies EXIF orientation to source images, preserves aspect ratio in a 512×512 letterbox, resizes images bilinearly and masks with nearest-neighbor interpolation, and marks padding as ignore label `255`. Validation applies no augmentation. Model inference uses RGB tensors normalized by ImageNet mean and standard deviation.

Training-only geometry and appearance changes are synchronized with the indexed mask: horizontal flip (0.5), random scale 0.9–1.1 (0.5), rotation up to ±5° (0.25), and mild brightness/contrast/saturation jitter (0.5). Blur and low-amplitude image noise each have 0.05 probability. Images use bilinear geometry interpolation, masks use nearest-neighbor, and newly exposed mask pixels become `255`. CLAHE, gamma changes, and denoising are disabled.

## Architecture and training

The model is SegFormer-B2 with a MiT-B2 encoder (stage depths 3/4/6/3; hidden widths 64/128/320/512) and 768-wide segmentation decoder. The ADE20K 150-class head is replaced with a newly initialized 104-class FoodSeg103 head. Training used 512×512 input, AdamW (learning rate 6e-5, weight decay 0.01), 5% linear warmup followed by polynomial decay, effective batch size 8 (microbatch 2 with accumulation 4), automatic mixed precision, seed 42, and at most 40 epochs with patience 10.

The objective combined class-weighted focal loss (weight 0.5, gamma 2) and soft Dice loss (weight 0.5; background excluded from Dice). Class frequencies came only from the 4,485 gradient-training examples. Square-root frequency tempering, mean normalization, and a maximum weight of 2.0 provide moderate imbalance correction without stacking aggressive weights on focal loss. `255` is excluded from both losses and metrics.

The run completed 33 epochs and early-stopped. The best checkpoint was epoch 23 with internal foreground mIoU **0.360011**, foreground Dice **0.468488**, and pixel accuracy **0.786231**.

## Same-split comparison

| Model / protocol | Foreground mIoU | Foreground Dice | Pixel accuracy | Result |
| --- | ---: | ---: | ---: | --- |
| SegFormer-B0, 512 single pass | 0.191775 | 0.269975 | 0.733852 | 2,135 images |
| SegFormer-B2, 512 single pass | **0.364898** | **0.494406** | **0.781938** | 2,135 images |
| SegFormer-B0, 512 flip TTA | 0.195610 | 0.274161 | 0.738893 | 2,135 images |
| SegFormer-B2, 512 flip TTA | **0.369469** | **0.498852** | **0.784023** | 2,135 images |

The single-pass B2 gain over the synchronized B0 run is +0.173123 foreground mIoU, +0.224430 Dice, and +4.81 percentage points pixel accuracy. The existing saved B0 baseline metrics are 0.191785 mIoU, 0.269989 Dice, and 0.733854 accuracy; the synchronized FP16 evaluation differs by at most 0.000014 on these metrics. The table uses the fresh, identically configured B0 run for direct comparison.

| Model / protocol | Original-resolution foreground mIoU | Dice | Pixel accuracy |
| --- | ---: | ---: | ---: |
| B0 single pass | 0.185919 | 0.262675 | 0.751500 |
| B2 single pass | **0.340845** | **0.466678** | **0.791797** |
| B0 flip TTA | 0.189305 | 0.266072 | 0.756607 |
| B2 flip TTA | **0.345639** | **0.470916** | **0.793970** |

For ingredient presence across foreground classes, B0 single-pass macro precision/recall/F1 was 0.3376/0.3936/0.2687; B2 was 0.3300/0.7020/0.4233. B2 flip TTA was 0.3452/0.6975/0.4368. Flip TTA therefore adds 0.004571 input-resolution mIoU and 0.013507 macro presence F1 for B2, at about 1.81× the measured forward time.

B2 produced predictions for 101 of 103 foreground classes present in ground truth; 99 classes had at least one correctly labelled pixel. B0 predicted 69 classes and had true-positive pixels for 62. For the 26 rarest labels by ground-truth pixel support, B2 predicted 24 and had true-positive pixels for 22; B0 predicted only one rare label with a true-positive pixel. The full B0/B2 per-class scores, presence metrics, and labelled 104×104 confusion matrices are stored as CSVs outside Git under `runs/segmentation-comparison-20261124_160934/`.

## Per-class and visual analysis

Best B2 classes by IoU were broccoli (0.8592), corn (0.8552), carrot (0.8095), strawberry (0.7939), and green beans (0.7933). The worst supported classes were bamboo shoots, pudding, kelp, and salad (IoU 0); king oyster mushroom (0.0001) and hamburg (0.0044) were also very weak. Meat confusions are concentrated in steak→pork (3.06 million pixels), chicken duck→pork (2.72 million), and pork→steak (2.17 million). For example, B2's full per-class metrics, labelled confusion matrix, and image-presence metrics are `foodseg103-segformer-b2-optimized-20261124_075454__single__512_per_class.csv`, `foodseg103-segformer-b2-optimized-20261124_075454__single__512_confusion.csv`, and `foodseg103-segformer-b2-optimized-20261124_075454__single__presence_per_class.csv` under the evaluation output directory. Rare-class coverage is defined as the bottom quartile of foreground labels by ground-truth pixel support on this validation split.

Thirty source/ground-truth/B0/B2 comparison panels and a contact sheet are generated from the same official validation examples. Selection includes successful predictions, difficult examples, meat confusions, rare labels, and mixed-food plates. The saved B0 and B2 epoch histories are plotted as actual training curves; B0's and B2's epoch counts and metrics are not inferred from synthetic data. The files are under `runs/segmentation-comparison-20261124_160934/visual_analysis/`.

## Speed and GPU memory

On the RTX 4050 Laptop GPU, batch-size-1 forward-only latency (preprocessing and host/device transfer excluded) was B0 mean/p50/p95 33.05/30.99/46.04 ms and B2 71.57/71.50/92.89 ms. Flip TTA was 64.64/63.31/88.15 ms for B0 and 129.73/129.85/167.12 ms for B2. Peak PyTorch CUDA allocation was 358 MiB for B0 and 653 MiB for B2; peak reserved memory was about 943 MiB for each. `nvidia-smi` showed about 2.2–2.5 GiB total GPU use during B2 evaluation, including the desktop and other CUDA context overhead. These measurements are for this 512×512 evaluation path, not end-to-end backend latency.

## Backend integration

The SmartNutri adapter loads local Hugging Face checkpoints using their saved 104-class `id2label`/`label2id` contract and `training_metadata.json` preprocessing profile. Select B0 or B2 with `SMARTNUTRI_CV_MODE=segformer`, `SMARTNUTRI_CHECKPOINT_DIR=<local best_model directory>`, and `SMARTNUTRI_FOODSEG_ROOT=<local FoodSeg103 directory>`. The service uses local files only. The local-image adapter smoke passed for both B0 and B2 on a 485×512 FoodSeg validation fixture, checking mask dimensions, all emitted class IDs/names, and ingredient predictions. With no scale reading, the nutrition agent returned `RETRY_WEIGHT`; no portion or nutrition values were created. These are fixture-based adapter checks, not phone-camera or physical-scale tests.

## Verification

The targeted CV/backend unit suites passed (49 tests), followed by the backend API, automatic-scan, agent-policy, scale-workflow, and server suites in an isolated temporary backend environment (29 tests). The active CUDA training environment was left unchanged. Python compilation passed for the B2 trainer/evaluator, loss and weighting helpers, visualization generator, and backend smoke tool. The adapter smokes passed for both checkpoints; full training was not run.

## Pretrained source and research

- [Official FoodSeg103 benchmark repository](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1)
- [FoodSeg103 SegFormer-B2 comparative study](https://github.com/mahendran-jinachandran/FoodSeg103Benchmarking)
- [FoodSeg103 SegFormer-B0/B1 study and code](https://github.com/joneshshrestha/ingredient-level-segmentation)
- [Food-specific FoodSeg103 model collection](https://huggingface.co/mericberktas/foodseg-models)
- [Pinned NVIDIA ADE20K SegFormer-B2 checkpoint tree](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/tree/de01bae28967510f9ddd496c60a969357195400c)
- [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)

The FoodSeg103-specific public B2 score was not adopted because its card does not establish a reproducible split, recomputed metrics, class mapping, or checkpoint use license. The selected initialization was the pinned NVIDIA ADE20K B2 checkpoint. Its upstream license restricts the project to non-commercial research/evaluation use.

The completed run's historical `run_config.json` records `model.safetensors` and `use_safetensors=true`, while the pinned revision's published artifact is `pytorch_model.bin` (110,000,905 bytes). The byte count and file name in that historical record conflict; the exact initialization artifact used cannot be independently proven from the fine-tuned checkpoint. The tracked config now selects the official PyTorch `.bin`, records its expected size and SHA-256, and the trainer enforces that the artifact name matches `use_safetensors=false`. Historical run metadata was preserved unchanged.

## Limitations and deployment

The FoodSeg103 validation distribution does not represent the fixed overhead smartphone-camera domain. A separately captured and manually indexed overhead-camera test set remains necessary before claiming real-world performance. Segmentation identifies pixel labels only; portion mass requires the scale workflow, and nutrition lookup requires validated class/portion inputs. This report does not estimate either from masks.

The evaluation adapter produces a semantic mask and ingredient-presence predictions. In deployment, `SMARTNUTRI_CHECKPOINT_DIR` selects the chosen local B0 or B2 checkpoint; local metadata controls deterministic preprocessing and the backend preserves the existing nutrition agent and scale decision flow. For an independent camera-domain evaluation, follow [the overhead-camera test-set protocol](real_world_overhead_test_set.md) and report it separately from FoodSeg103 results.
