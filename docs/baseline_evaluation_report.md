# SmartNutri Baseline Evaluation and Visual Analysis

| Field | Value |
|---|---|
| Evaluation date | 2026-11-23 |
| Checkpoint | `runs/foodseg103-segformer-b0/best_model` (best epoch 20) |
| Dataset | FoodSeg103 official validation split, 2,135 images |
| Status | Inference and analysis only. **No training was started or resumed.** The dataset and checkpoint were not modified. |

## Setup and reproducibility

The run used the existing `.venv` environment, PyTorch 2.11.0+cu128, CUDA 12.8, and the NVIDIA GeForce RTX 4050 Laptop GPU. Evaluation used CUDA automatic mixed precision, batch size 2, and the checkpoint's recorded 512-pixel `baseline` preprocessing profile. The profile uses EXIF orientation correction, RGB conversion, aspect-preserving letterboxing, bilinear image resizing, nearest-neighbor mask resizing, ImageNet normalization, and ignore label 255 for padding. Validation augmentation and image enhancement were disabled.

Both CLI help commands exited successfully. `evaluate.py` exposes `--model-dir`, `--foodseg-root`, `--split`, `--image-size`, `--preprocessing-config`, `--batch-size`, `--output-dir`, `--device`, and `--no-amp`. `visualize.py` exposes `--model-dir`, `--foodseg-root`, `--start-index`, `--count`, `--image-size`, `--preprocessing-config`, `--output-dir`, `--device`, and `--no-amp`.

Commands executed:

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --model-dir .\runs\foodseg103-segformer-b0\best_model --foodseg-root ..\Datasets\FoodSeg103 --split validation --batch-size 2 --output-dir .\runs\baseline-evaluation --device cuda
.\.venv\Scripts\python.exe .\visualize.py --model-dir .\runs\foodseg103-segformer-b0\best_model --foodseg-root ..\Datasets\FoodSeg103 --start-index 0 --count 20 --output-dir .\runs\baseline-visualizations --device cuda
.\.venv\Scripts\python.exe .\runs\baseline-evaluation\_diagnostics.py
```

The last local, ignored helper made a second inference-only pass to collect the full confusion matrix and image-level class support. It also rewrote the 20 CLI-selected PNGs to show the four requested panels: original image, standalone ground-truth mask, standalone predicted mask, and prediction overlay. The existing `visualize.py` output itself contains three panels (original, ground-truth overlay, prediction overlay), so this extra rendering was needed to include standalone masks without changing tracked source code.

All evaluation CSV/JSON files, the diagnostic helper, and 20 PNGs are in `runs/`. The existing `.gitignore` excludes `runs/`, checkpoints, and generated visualizations; these outputs are intentionally not committed.

## Evaluation results

| Metric | Validation result |
|---|---:|
| Pixel accuracy (includes background) | 0.733854 |
| Mean IoU including background | 0.198639 |
| **Foreground mean IoU** | **0.191785** |
| Mean Dice including background | 0.276526 |
| **Foreground mean Dice** | **0.269989** |
| Evaluated non-padding pixels | 443,268,608 |
| Classes with ground-truth support | 104 / 104 (background + 103 food classes) |

The scores match both the saved checkpoint metadata (foreground mIoU 0.191785) and the epoch-20 training history. Pixel accuracy includes background and should not be read as ingredient segmentation quality; foreground mIoU and Dice are the more relevant summary measures here.

## Per-class metrics and support

Support is measured on the deterministically preprocessed 512-pixel validation masks. `GT pixels` excludes padding label 255; `images` is the number of validation images containing that class after preprocessing. IoU and Dice are defined for every class because each class has ground-truth support. Values below 0.05 are marked `low`; values exactly zero are marked `zero`.

| ID | Food class | GT pixels | Images | Predicted pixels | IoU | Dice | Flag |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | background | 207,441,268 | ? | 209,062,635 | 0.904542 | 0.949879 |  |
| 1 | candy | 330,065 | 11 | 0 | 0.000000 | 0.000000 | zero |
| 2 | egg tart | 28,270 | 1 | 0 | 0.000000 | 0.000000 | zero |
| 3 | french fries | 3,396,590 | 78 | 3,094,437 | 0.390758 | 0.561936 |  |
| 4 | chocolate | 636,280 | 18 | 3 | 0.000000 | 0.000000 | zero |
| 5 | biscuit | 3,908,474 | 77 | 5,416,055 | 0.352830 | 0.521617 |  |
| 6 | popcorn | 197,510 | 4 | 0 | 0.000000 | 0.000000 | zero |
| 7 | pudding | 22,121 | 1 | 0 | 0.000000 | 0.000000 | zero |
| 8 | ice cream | 6,736,661 | 277 | 7,380,851 | 0.368229 | 0.538256 |  |
| 9 | cheese butter | 2,409,596 | 120 | 115,625 | 0.004559 | 0.009076 | low |
| 10 | cake | 7,327,663 | 126 | 6,804,494 | 0.357855 | 0.527089 |  |
| 11 | wine | 604,276 | 41 | 113 | 0.000000 | 0.000000 | zero |
| 12 | milkshake | 1,419,872 | 27 | 1,737,179 | 0.400870 | 0.572316 |  |
| 13 | coffee | 817,028 | 57 | 623,930 | 0.276086 | 0.432708 |  |
| 14 | juice | 1,851,408 | 54 | 2,420,430 | 0.391945 | 0.563161 |  |
| 15 | milk | 994,766 | 28 | 224,805 | 0.177123 | 0.300942 |  |
| 16 | tea | 65,768 | 6 | 0 | 0.000000 | 0.000000 | zero |
| 17 | almond | 606,932 | 24 | 6,095 | 0.000936 | 0.001869 | low |
| 18 | red beans | 598,699 | 14 | 0 | 0.000000 | 0.000000 | zero |
| 19 | cashew | 147,936 | 10 | 0 | 0.000000 | 0.000000 | zero |
| 20 | dried cranberries | 207,857 | 12 | 0 | 0.000000 | 0.000000 | zero |
| 21 | soy | 335,568 | 14 | 250,652 | 0.378884 | 0.549551 |  |
| 22 | walnut | 354,489 | 22 | 0 | 0.000000 | 0.000000 | zero |
| 23 | peanut | 67,240 | 2 | 0 | 0.000000 | 0.000000 | zero |
| 24 | egg | 2,170,688 | 69 | 2,325,554 | 0.253652 | 0.404661 |  |
| 25 | apple | 1,364,499 | 52 | 1,379,062 | 0.185464 | 0.312897 |  |
| 26 | date | 16,789 | 2 | 0 | 0.000000 | 0.000000 | zero |
| 27 | apricot | 116,118 | 8 | 0 | 0.000000 | 0.000000 | zero |
| 28 | avocado | 414,171 | 24 | 0 | 0.000000 | 0.000000 | zero |
| 29 | banana | 1,555,873 | 56 | 1,530,930 | 0.349343 | 0.517797 |  |
| 30 | strawberry | 4,794,047 | 172 | 5,645,814 | 0.643002 | 0.782716 |  |
| 31 | cherry | 855,041 | 53 | 646,378 | 0.233892 | 0.379112 |  |
| 32 | blueberry | 890,367 | 58 | 1,485,099 | 0.436152 | 0.607390 |  |
| 33 | raspberry | 454,305 | 22 | 107,666 | 0.144015 | 0.251771 |  |
| 34 | mango | 354,482 | 12 | 0 | 0.000000 | 0.000000 | zero |
| 35 | olives | 342,874 | 15 | 0 | 0.000000 | 0.000000 | zero |
| 36 | peach | 550,874 | 18 | 532,828 | 0.182817 | 0.309122 |  |
| 37 | lemon | 3,130,578 | 172 | 3,081,061 | 0.524320 | 0.687939 |  |
| 38 | pear | 466,120 | 14 | 5,856 | 0.000000 | 0.000000 | zero |
| 39 | fig | 225,514 | 5 | 0 | 0.000000 | 0.000000 | zero |
| 40 | pineapple | 1,199,872 | 36 | 911,187 | 0.112753 | 0.202657 |  |
| 41 | grape | 729,307 | 35 | 809,810 | 0.425957 | 0.597433 |  |
| 42 | kiwi | 257,460 | 13 | 11,962 | 0.043802 | 0.083928 | low |
| 43 | melon | 115,242 | 4 | 0 | 0.000000 | 0.000000 | zero |
| 44 | orange | 1,555,091 | 70 | 1,556,816 | 0.301560 | 0.463382 |  |
| 45 | watermelon | 399,660 | 8 | 1,856 | 0.001886 | 0.003766 | low |
| 46 | steak | 14,321,713 | 337 | 16,919,120 | 0.377314 | 0.547899 |  |
| 47 | pork | 8,582,751 | 195 | 11,943,887 | 0.147749 | 0.257458 |  |
| 48 | chicken duck | 18,216,598 | 394 | 19,458,349 | 0.324306 | 0.489776 |  |
| 49 | sausage | 1,443,507 | 53 | 1,881,163 | 0.156236 | 0.270249 |  |
| 50 | fried meat | 3,164,634 | 84 | 1,171,955 | 0.037253 | 0.071831 | low |
| 51 | lamb | 1,603,272 | 28 | 90,988 | 0.000130 | 0.000261 | low |
| 52 | sauce | 6,712,502 | 327 | 6,548,861 | 0.333231 | 0.499884 |  |
| 53 | crab | 412,603 | 6 | 0 | 0.000000 | 0.000000 | zero |
| 54 | fish | 4,899,065 | 95 | 4,722,084 | 0.215747 | 0.354921 |  |
| 55 | shellfish | 729,487 | 13 | 371,357 | 0.330171 | 0.496434 |  |
| 56 | shrimp | 2,380,297 | 45 | 2,974,014 | 0.337235 | 0.504377 |  |
| 57 | soup | 770,731 | 22 | 471,056 | 0.283860 | 0.442198 |  |
| 58 | bread | 19,955,677 | 414 | 21,674,245 | 0.485814 | 0.653937 |  |
| 59 | corn | 3,840,263 | 145 | 3,851,983 | 0.757750 | 0.862181 |  |
| 60 | hamburg | 39,948 | 1 | 0 | 0.000000 | 0.000000 | zero |
| 61 | pizza | 735,286 | 13 | 599,922 | 0.120729 | 0.215447 |  |
| 62 | hanamaki baozi | 329,711 | 9 | 0 | 0.000000 | 0.000000 | zero |
| 63 | wonton dumplings | 610,660 | 10 | 0 | 0.000000 | 0.000000 | zero |
| 64 | pasta | 2,061,522 | 48 | 1,785,216 | 0.363090 | 0.532745 |  |
| 65 | noodles | 3,432,541 | 75 | 4,235,978 | 0.512236 | 0.677455 |  |
| 66 | rice | 7,178,669 | 206 | 7,413,713 | 0.568770 | 0.725116 |  |
| 67 | pie | 8,590,869 | 161 | 9,145,063 | 0.271181 | 0.426660 |  |
| 68 | tofu | 675,976 | 20 | 1,439 | 0.000000 | 0.000000 | zero |
| 69 | eggplant | 249,613 | 7 | 0 | 0.000000 | 0.000000 | zero |
| 70 | potato | 9,301,120 | 306 | 12,195,584 | 0.447139 | 0.617963 |  |
| 71 | garlic | 234,124 | 20 | 0 | 0.000000 | 0.000000 | zero |
| 72 | cauliflower | 1,376,396 | 70 | 728,756 | 0.255176 | 0.406598 |  |
| 73 | tomato | 6,231,882 | 349 | 6,243,938 | 0.582810 | 0.736424 |  |
| 74 | kelp | 50,140 | 2 | 0 | 0.000000 | 0.000000 | zero |
| 75 | seaweed | 157,108 | 5 | 0 | 0.000000 | 0.000000 | zero |
| 76 | spring onion | 566,574 | 57 | 31 | 0.000000 | 0.000000 | zero |
| 77 | rape | 637,175 | 21 | 7,504 | 0.000000 | 0.000000 | zero |
| 78 | ginger | 86,355 | 8 | 0 | 0.000000 | 0.000000 | zero |
| 79 | okra | 104,954 | 2 | 0 | 0.000000 | 0.000000 | zero |
| 80 | lettuce | 4,007,561 | 171 | 4,816,929 | 0.354584 | 0.523532 |  |
| 81 | pumpkin | 597,497 | 16 | 682,121 | 0.251011 | 0.401293 |  |
| 82 | cucumber | 4,262,172 | 164 | 5,357,600 | 0.452409 | 0.622977 |  |
| 83 | white radish | 384,944 | 16 | 0 | 0.000000 | 0.000000 | zero |
| 84 | carrot | 8,975,693 | 398 | 9,353,732 | 0.723739 | 0.839732 |  |
| 85 | asparagus | 2,934,822 | 77 | 3,640,069 | 0.576826 | 0.731629 |  |
| 86 | bamboo shoots | 67,197 | 3 | 0 | 0.000000 | 0.000000 | zero |
| 87 | broccoli | 8,255,775 | 309 | 8,787,619 | 0.799362 | 0.888495 |  |
| 88 | celery stick | 1,123,598 | 65 | 1,417,275 | 0.387861 | 0.558933 |  |
| 89 | cilantro mint | 4,193,172 | 264 | 4,656,880 | 0.390426 | 0.561592 |  |
| 90 | snow peas | 874,823 | 24 | 598 | 0.000018 | 0.000037 | low |
| 91 | cabbage | 871,888 | 33 | 61,771 | 0.028484 | 0.055391 | low |
| 92 | bean sprouts | 456,516 | 13 | 0 | 0.000000 | 0.000000 | zero |
| 93 | onion | 2,343,844 | 158 | 3,484,562 | 0.192324 | 0.322603 |  |
| 94 | pepper | 2,334,940 | 131 | 2,631,043 | 0.232047 | 0.376686 |  |
| 95 | green beans | 1,310,305 | 63 | 1,222,980 | 0.722360 | 0.838803 |  |
| 96 | French beans | 3,320,067 | 121 | 4,053,649 | 0.584653 | 0.737894 |  |
| 97 | king oyster mushroom | 45,779 | 3 | 0 | 0.000000 | 0.000000 | zero |
| 98 | shiitake | 1,496,722 | 54 | 323,099 | 0.061910 | 0.116600 |  |
| 99 | enoki mushroom | 44,253 | 3 | 0 | 0.000000 | 0.000000 | zero |
| 100 | oyster mushroom | 20,277 | 2 | 0 | 0.000000 | 0.000000 | zero |
| 101 | white button mushroom | 1,208,393 | 37 | 1,173,232 | 0.149270 | 0.259765 |  |
| 102 | salad | 207,810 | 10 | 0 | 0.000000 | 0.000000 | zero |
| 103 | other ingredients | 1,781,528 | 72 | 60 | 0.000000 | 0.000000 | zero |

The class distribution is strongly imbalanced. Foreground ground-truth support ranges from 16,789 pixels (`date`, 2 images) to 19,955,677 pixels (`bread`, 414 images), a 1,188.6× range; the median is 817,028 pixels and 28 images. Fifteen classes occur in at most five validation images, and 25 occur in at most ten. Pixel-support quartiles are 330,065 / 817,028 / 2,934,822; image-support quartiles are 11 / 28 / 78.

The five highest-IoU classes are broccoli (0.7994), corn (0.7577), carrot (0.7237), green beans (0.7224), and strawberry (0.6430). Forty-one classes have zero IoU, and 49/103 foreground classes have IoU below 0.05. The zero-IoU classes are: candy, egg tart, chocolate, popcorn, pudding, wine, tea, red beans, cashew, dried cranberries, walnut, peanut, date, apricot, avocado, mango, olives, pear, fig, melon, crab, hamburg, hanamaki baozi, wonton dumplings, tofu, eggplant, garlic, kelp, seaweed, spring onion, rape, ginger, okra, white radish, bamboo shoots, bean sprouts, king oyster mushroom, enoki mushroom, oyster mushroom, salad, and other ingredients. Eight more classes have nonzero IoU below 0.05: snow peas, lamb, almond, watermelon, cheese butter, cabbage, fried meat, and kiwi. Exact class IoU, Dice, support, and prediction counts are in the table above and the local CSV.

Although every food class occurs in the ground truth, the model predicts pixels for only 69 of the 103 foreground classes; 34 classes have zero predicted pixels. A zero-IoU class can also have false-positive predictions with no correct overlap (for example, `other ingredients` has 1.78 million ground-truth pixels across 72 images but zero IoU). The long tail is a major factor, but support alone does not explain all failures.

## Confusion and visual error analysis

The largest foreground-to-foreground confusions in the full validation pass are:

| Ground truth | Predicted as | Confused pixels | Share of that class's GT pixels |
|---|---|---:|---:|
| steak | pork | 3,196,154 | 22.3% |
| pork | steak | 2,456,065 | 28.6% |
| chicken duck | pork | 2,278,962 | 12.5% |
| pork | chicken duck | 1,988,998 | 23.2% |
| fish | chicken duck | 1,607,854 | 32.8% |
| cake | bread | 1,582,793 | 21.6% |
| pie | bread | 1,421,241 | 16.5% |
| cilantro mint | lettuce | 669,755 | 16.0% |
| lamb | steak | 640,298 | 39.9% |

There were 9,627,176 background pixels predicted as food and 11,248,543 foreground pixels predicted as background. These totals show both spurious foreground regions and missed food pixels.

The visual examples below are local-only links into the ignored `runs/` directory. Each image has the same class colors in the raw masks and overlay; black is background and gray is ignored letterbox padding.

### Larger regions with useful agreement, alongside class confusion

Validation index 12 shows broad vegetable regions such as broccoli matching reasonably well, while the mixed meat area is fragmented and confused. It is a representative partial success, not a claim that all classes in the image are correct.

![Validation index 12: large-region agreement and meat-class errors](../runs/baseline-visualizations/validation_00012.png)

### Similar-ingredient confusion

Validation index 4 contains fish and meat-like regions. The prediction assigns parts of the fish region to chicken duck/steak-like classes, consistent with the measured fish-to-chicken-duck and meat confusions.

![Validation index 4: fish and meat-class confusion](../runs/baseline-visualizations/validation_00004.png)

### Fragmentation and boundary spill

Validation index 9 is a cluttered multi-ingredient plate. The prediction contains fragmented regions and spills across adjacent items; this boundary assessment is qualitative because no boundary-specific metric was computed.

![Validation index 9: fragmented regions and boundary spill](../runs/baseline-visualizations/validation_00009.png)

### Small regions

Using a strict threshold of at most 262 pixels (0.1% of a 512×512 canvas) for a class's total region in one image, four class-image regions occur in validation: apple, corn, carrot, and onion. All four have zero correctly predicted pixels. This threshold only captures the very smallest regions; the PNGs also show smaller garnish/fragment misses, but those visual observations are not a separate exhaustive object-instance metric.

### Mapping and preprocessing checks

The checkpoint's `id2label` mapping matches the local FoodSeg103 mapping exactly, all 104 IDs are present, and the evaluation's class metrics reproduce the first evaluation run. The checkpoint metadata and baseline config both specify 512-pixel baseline preprocessing, and evaluation/visualization load that metadata. No label-ID or preprocessing mismatch was observed. The 512-pixel input may still erase very small structures, and some ingredient labels may be visually ambiguous; those are hypotheses to test, not confirmed defects.

## Recommended next experiment

**Prioritize C — class-imbalance handling.** This is the strongest measured signal: 41 classes have zero IoU, 34 are never predicted, foreground support spans 1,188.6×, and the class image-support median is only 28. The FoodSeg103 benchmark paper also describes the task's long-tailed ingredient distribution, overlapping foods, and substantial appearance variation. That supports testing class-aware training, but it does not establish that a particular weighting scheme will improve this checkpoint.

For a controlled next run, first compute class frequencies from the **training split only**, then change one imbalance mechanism (for example, capped, tempered class-weighted cross-entropy). Keep the image pipeline, initialization, seed, and other training settings fixed. Compare foreground mIoU/Dice, the per-class table, predicted-class coverage, and regressions on frequent classes; do not derive weights from validation labels.

The run is not a strong case for blindly extending training. Validation foreground mIoU rose from 0.189672 at epoch 18 to 0.191288 at epoch 19 and 0.191785 at epoch 20, while train loss continued to fall from 0.6081 to 0.5853. That is continued but small improvement; the epoch-20 learning rate was only 3.04×10⁻⁸. A short lower-learning-rate continuation with early stopping is a reasonable later check, but it will not directly address the 34 unpredicted classes. After the imbalance ablation, targeted fine-tuning on the meat and baked-food confusion groups is the next priority. Safe augmentation is also a later controlled ablation; no gain is assumed. Better preprocessing is lower priority because label mapping and the configured pipeline checked out, though a resolution ablation could test the small-region hypothesis.

## Research references and scope

These papers inform the task and system boundaries. They are not evidence that class weighting, longer training, or augmentation will improve this local model; those claims require SmartNutri validation experiments.

1. [A Large-Scale Benchmark for Food Image Segmentation (FoodSeg103)](https://arxiv.org/abs/2105.05409) — fine-grained ingredient masks; discusses appearance variation, overlap, and long-tail ingredient distribution.
2. [Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food](https://openaccess.thecvf.com/content/CVPR2021/html/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.html) — visual, depth, component-weight, and nutrition annotations for a later nutrition-estimation stage.
3. [Automated Food Weight and Content Estimation Using Computer Vision and AI Algorithms: Phase 2](https://www.mdpi.com/1424-8220/26/1/76) — combines detection/segmentation with monocular depth and geometric volume estimation; a 2D mask alone is not a weight measurement.
4. [Health to Eat: A Smart Plate with Food Recognition, Classification, and Weight Measurement](https://pmc.ncbi.nlm.nih.gov/articles/PMC9920985/) — treats recognition, measured weight, and nutrition information as connected but distinct parts of a system.
5. [SCALEeat: Vision-Guided Food Scale for Automated Macronutrient Estimation](https://www.mdpi.com/2673-4591/134/1/83) — combines visual food recognition with load-cell mass and food-composition data.

## Local outputs

- `runs/baseline-evaluation/metrics.json` — aggregate validation metrics.
- `runs/baseline-evaluation/per_class_metrics.csv` — evaluation-script per-class metrics.
- `runs/baseline-evaluation/class_support.csv` — pixel support, image support, and small-region counts.
- `runs/baseline-evaluation/top_confusions.csv` and `diagnostics.json` — confusion details and summary.
- `runs/baseline-visualizations/validation_00000.png` through `validation_00019.png` — 20 four-panel visualizations; `visualizations.json` maps indices and source names.

These outputs are ignored by Git and are not part of the documentation commit.
