# Portion Estimation Engine

## Contract and current implementation

The `PortionEstimator` interface receives measured net food mass, the detected FoodSeg103 classes, semantic masks/pixel-region statistics, optional prior/model estimates, and simulation provenance. It returns per-class grams, method, confidence, uncertainty, review flag, unallocated mass, mass-conservation status, and simulation marker.

The current `ExperimentalVisualProportionEstimator` is a research baseline named `experimental_visual_area_proportion_v1`. It computes:

```text
known food area = sum(pixel area for detected food classes)
estimated grams for class i = allocatable mass × area_i / known food area
```

When unidentified area is supplied, it is kept unallocated using its visible-area fraction. Background pixels never receive food mass. If no food regions are found, the entire measured mass remains unallocated. The output model enforces nonnegative allocations and conservation of allocated plus unallocated mass against measured mass within 0.05 g.
In the workflow, detections below the agent's confidence threshold are withheld from the known-food allocation and their mask pixels are passed as unidentified area. Background and truly unrecognized food remain difficult to distinguish without a dedicated plate/foreground model; this limitation is another reason all baseline results remain under review.

## Why results require review

Pixel area is not physical mass. Food height, density, overlap, camera angle, plate depth, occlusion, and preparation state change the relationship between a top-down semantic mask and mass. The baseline always sets confidence to 0.25, assigns broad uncertainty `max(50% of its estimated class mass, 25% of total mass)`, and marks every per-food estimate for review. It is an experimental comparison point; it must not be described as accurate mass estimation.

The scale-measured total is still useful and is persisted independently. When estimates cannot support a safe per-food calculation, the agent keeps a review draft, leaves missing nutrition totals null, and invites optional correction. No calories or macros are invented.

## Synthetic error check

`test_synthetic_known_multi_food_portions_report_absolute_error` uses only a synthetic 450 g fixture with known portions Rice 110 g, Chicken 220 g, Fries 120 g and mask areas in a 20:50:30 ratio. The baseline produces 90 g, 225 g, and 135 g; absolute errors are 20 g, 5 g, and 15 g, so fixture MAE is 13.33 g. This validates deterministic arithmetic against a hand-authored fixture only. It is not real-world validation, CV performance, or evidence of portion accuracy.

## Nutrition calculation

For each recognized portion with a verified per-100 g record:

```text
estimated nutrient = nutrient_per_100g × portion_weight_g / 100
```

Energy is reported in kcal, macros in grams. Each recognized item keeps its source/version and `estimated=true`, but a meal-level total is emitted only when the portion allocation covers the complete measured net weight and every represented food has exactly one verified mapping. If any food mass is unidentified, only recognized-item estimates may be shown and the meal totals remain null. Multiple preparation/source mappings are treated as ambiguous and require selection. FoodSeg103 labels do not establish cooking method or recipe composition.

## Research relationship and next evaluation

FoodSeg103 provides semantic classes/masks; the Nutrition5k paper illustrates a separate path with richer mass/nutrition labels; the food-weight estimation Phase 2 paper and SCALEeat motivate calibrated portion-estimation research; Health to Eat motivates combining image recognition and physical scale values. None of these references validate the area baseline.

To replace this estimator, keep the interface stable and evaluate on real multi-food plates with independent known weights. Record plate/camera geometry, per-class mass error, meal-level mass conservation, confidence calibration, and errors by food/preparation class. Candidate successors include a learned model with documented data provenance or a calibrated geometric/volume method. Do not train priors on the evaluation set.

Nutrition lookups use the traceable records in [nutrition catalog](nutrition_catalog.md) or separately reviewed operator records. Per-food estimates retain source, source identifier, and version. Preparation-sensitive class mappings such as rice (class 66), broccoli (class 87), and the joint chicken/duck class (class 48) require explicit record selection. See [API reference](api_reference.md) for the React dashboard correction contract.
