# SmartNutri image-processing pipeline

## Scope

This module prepares FoodSeg103 image and semantic-mask pairs for SegFormer-B0. It does not extract, rewrite, or copy the dataset. The five supplied papers motivate the distinction between ingredient segmentation and later food-mass or nutrient estimation; they are not evidence that a particular enhancement or augmentation improves this model. Enhancement profiles are experiments and must be judged on the official validation split.

The implementation lives in `image_processing.py`. `foodseg_data.prepare_example` keeps its three-value return contract and delegates to this module. Training, evaluation, and prediction visualization use the same preprocessing configuration. The configuration used for a checkpoint is stored in `training_metadata.json` so later evaluation can reproduce it.

## Deterministic preprocessing order

| Operation | Implementation and default | Purpose and possible benefit | Limitation or risk |
|---|---|---|---|
| EXIF orientation | `ImageOps.exif_transpose` before size checks | Makes the decoded pixels agree with the orientation viewers display; avoids rejecting or misaligning images whose JPEG dimensions differ from their masks because of orientation tags. | Corrects orientation metadata only. It cannot repair a truly mismatched image and mask; such pairs raise an error. |
| RGB conversion | Convert the oriented image to 3-channel RGB | Gives the model a consistent channel order and shape across source formats. | Discards alpha and other source color channels. FoodSeg103 RGB photos do not require them. |
| Optional enhancement | Disabled in baseline; separate profiles described below | Tests controlled contrast, brightness-response, or denoising hypotheses without changing labels. | Can remove useful texture or amplify noise. It changes the image distribution and is not presumed beneficial. |
| Aspect-preserving resize | Target `512 × 512`; scale by `min(target / width, target / height)`; round dimensions; bilinear interpolation for RGB and nearest-neighbor for masks | Keeps food geometry from being stretched. Bilinear interpolation is suitable for continuous image intensities; nearest-neighbor keeps mask IDs discrete. | Letterboxing uses some pixels that do not contain food. A smaller target can lose fine boundaries; a larger target does not add source detail. |
| Padding | Center the resized pair in a square. RGB fill is the configured mean color; mask fill is `255`. | Mean-color fill becomes approximately zero after normalization. Ignore label `255` prevents padded pixels from contributing to segmentation loss or metrics. | The semantic mask must retain its indexed, single-channel values. RGB or interpolated masks are rejected or would be invalid. |
| SegFormer normalization | Convert RGB bytes to `[0, 1]`, then `(pixel - mean) / std` once. Defaults: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. | Matches the ImageNet normalization used by the selected SegFormer-B0 checkpoint and centers the mean-colored padding near zero. | Do not pass these tensors through another image processor that normalizes them again. Model code receives `pixel_values` directly. |

Mask values are checked before resizing. The valid semantic IDs are `0..103` (background plus 103 food classes); `255` is accepted for ignored pixels. Any other value raises an error. Padding and rotation fill use `255`. No photometric function is called on a mask.

The preprocessing display image returned for visual review is the RGB view before normalization. The model tensor is normalized; the returned display image is not a second model input.

## Training-only augmentation profile

`configs/image_processing/safe_augmentation.json` contains the suggested safe profile. Random transforms are gated by `training=True`; validation, evaluation, and the default visualization path pass `training=False`.

| Transform | Parameters | Purpose and possible benefit | Limitation or risk |
|---|---|---|---|
| Horizontal flip | Probability `0.5` | Adds left-right variation without changing food class identity. | Not appropriate for every task or asymmetric context; FoodSeg103 ingredient labels are not directional. |
| Small rotation | Probability `0.25`, uniform angle from `-5°` to `+5°` | Adds tolerance to small camera and plate rotations. | Larger rotations can create unrealistic plate geometry. Image uses bilinear interpolation; mask uses nearest-neighbor; exposed corners are ignored. |
| Brightness and contrast | Together with color jitter, probability `0.5`; each factor is uniform within `1 ± 0.10` | Models modest lighting and exposure differences. | Strong shifts can hide low-contrast ingredients or change apparent color cues. |
| Saturation | Same `0.5` photometric event; factor within `1 ± 0.08` | Adds mild camera color variation. | Food color can carry class information, so the range stays small. |
| Gaussian blur | Probability `0.1`; radius uniform from `0` to `0.8` pixels | Simulates mild focus softness. | Can erase fine boundaries and small ingredients. |
| Gaussian noise | Probability `0.1`; standard deviation `3` on the 8-bit `0..255` scale (about `0.012` on `[0,1]`) | Simulates mild sensor noise. | Synthetic independent noise does not model all camera noise and can obscure small regions. |

Every sampled flip or rotation is applied to both image and mask with the same parameters. Blur, noise, brightness, contrast, and saturation are image-only. Randomness comes from the dataset's seeded Python RNG (`seed + epoch`); the noise generator is seeded from that RNG. The validation branch never samples these transforms, so the same validation pair and config produce the same output.

## Optional image-enhancement ablations

Each profile changes one image-only operation and leaves the other enhancements at baseline values. Choose a new output directory for each run.

| Profile | Parameters and operation | Possible benefit | Limitation or risk |
|---|---|---|---|
| `clahe_ablation.json` | CLAHE on the Y channel of YCbCr; clip limit `2.0`, `8 × 8` tiles; local tile mappings are interpolated across tile boundaries. | May reveal local ingredient detail in uneven illumination. | Can exaggerate texture, noise, or compression artifacts; YCbCr Y is a luminance approximation, not a calibrated light measurement. |
| `gamma_ablation.json` | Gamma `1.15`, applied as `pixel ** gamma` on normalized byte intensities before resize. | Tests whether slightly darker midtones improve separation under bright exposure. | Darkens the image and can hide dark ingredients; the chosen direction may not suit all lighting. |
| `denoising_ablation.json` | `3 × 3` median filter on RGB before resize. | Can suppress isolated pixel noise while retaining some edges. | May remove thin boundaries and small food regions; it is not a general camera-noise model. |

Enhancements are deterministic and apply to both training and validation when selected, which makes a validation score a comparison of a full preprocessing profile. They never apply to masks. For interpretable comparisons, first establish the baseline, then run safe augmentation, then run each enhancement profile independently with the same split, seed, model initialization, and optimizer settings. Do not enable multiple enhancement profiles together and call that a single-operation ablation.

## Experiment configurations

| Configuration | Use |
|---|---|
| `configs/image_processing/baseline.json` | Recommended initial run; resize, pad, validate labels, and normalize without random augmentation or enhancements. |
| `configs/image_processing/safe_augmentation.json` | Train-only augmentation experiment; validation remains deterministic. |
| `configs/image_processing/clahe_ablation.json` | Optional CLAHE comparison. |
| `configs/image_processing/gamma_ablation.json` | Optional gamma comparison. |
| `configs/image_processing/denoising_ablation.json` | Optional median-denoising comparison. |

Run a profile with `train.py --preprocessing-config <path>`. The default is baseline. Profile details are saved with the best checkpoint and reused by `evaluate.py` and `visualize.py`, unless a caller explicitly supplies a different profile.

## Relationship to the five SmartNutri papers

1. **[A Large-Scale Benchmark for Food Image Segmentation (FoodSeg103)](https://arxiv.org/abs/2105.05409).** This is the direct task/data source for this stage: fine-grained ingredient labels and pixel masks define what the model learns. The local FoodSeg103 label map has 104 IDs including background. Aspect-preserving image resize, nearest-neighbor mask resize, strict ID validation, and ignored padding protect that supervision. The paper motivates the segmentation task; it does not establish that the optional enhancement profiles improve performance.
2. **[Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food](https://openaccess.thecvf.com/content/CVPR2021/html/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.html).** Nutrition5k includes dish video, depth, component weights, and nutritional annotations. It motivates the later nutrition stage. The current FoodSeg103 mask pipeline produces ingredient regions only; it does not estimate mass, calories, or nutrients.
3. **[Automated Food Weight and Content Estimation Using Computer Vision and AI Algorithms: Phase 2](https://www.mdpi.com/1424-8220/26/1/76).** This work combines ingredient detection and segmentation with monocular depth to estimate volume and then weight. It motivates keeping content segmentation, geometric measurement, and weight estimation as distinct steps. A FoodSeg103 mask alone is not a mass measurement.
4. **[Health to Eat: A Smart Plate with Food Recognition, Classification, and Weight Measurement for Type-2 Diabetic Mellitus Patients' Nutrition Control](https://pmc.ncbi.nlm.nih.gov/articles/PMC9920985/).** Its smart-plate scope combines food recognition with hardware weight measurement and nutrition outputs. This module supplies visual ingredient regions; measured weight and a nutrition mapping are separate inputs to portion and nutrient estimates.
5. **[SCALEeat: Vision-Guided Food Scale for Automated Macronutrient Estimation](https://www.mdpi.com/2673-4591/134/1/83).** SCALEeat combines camera-based food recognition with a load cell and food-composition data to calculate macronutrients from measured mass. SegFormer segmentation is only the visual branch of that wider measurement system.

The papers inform the system boundary and intended downstream uses. They are not cited here as evidence for any claimed augmentation or enhancement gain; those claims require SmartNutri validation experiments.

## Real-source visual example

`docs/assets/preprocessing_before_after.png` is generated from the local FoodSeg103 validation rows by `tools/visualize_preprocessing.py`. It displays the decoded source RGB, the EXIF-corrected/resized/padded view, and the aligned ground-truth overlay. The asset is a display rendering before tensor normalization. The script reads the Parquet rows and label map only; it writes the figure under `docs/` and does not write to the source dataset.
