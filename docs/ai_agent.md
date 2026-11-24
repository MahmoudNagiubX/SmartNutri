# SmartNutri Nutrition Agent policy

## Agent type and purpose

`NutritionDecisionAgent` is an explicit rule-based policy in the laptop backend. It is not a separately trained deep-learning model or an LLM. It coordinates scale, camera, CV, portion, and nutrition evidence while avoiding unsupported meal claims.

## Percepts

The agent receives an `AgentContext` containing:

- a source-aware scale observation (status, stability, unit, simulated flag);
- whether a plate tare is known and the derived positive net weight;
- camera quality result and remaining capture attempts;
- all detected non-background foods with classification confidence;
- portion estimates, uncertainty, allocation completeness, and review flags;
- the set of FoodSeg103 classes with exactly one verified nutrition mapping;
- current meal state and whether analysis has already been accepted.

## Internal state and goals

Persistent state lives in scan/meal rows, not hidden agent memory. The scale monitor owns placement and removal state; a scan record owns the image, prediction, and analysis state; a meal row owns measured net weight and item corrections. The policy goals are to preserve measured mass, keep mock and physical readings distinct, avoid subtracting tare twice, avoid food/background confusion, avoid unverified nutrition, and request user input only when evidence is insufficient.

## Decision policy

Rules are evaluated in order:

1. Missing, unstable, uncalibrated, disconnected, overload, or otherwise invalid scale reading → `RETRY_WEIGHT` (severe hardware errors require user attention).
2. Missing/invalid tare or nonpositive net mass → `REQUEST_PLATE_TARE`.
3. Missing or unacceptable capture quality → `RECAPTURE_IMAGE`; adapter retries happen first, then the agent requests a user recapture.
4. No detected food regions → `REQUEST_FOOD_PREPARATION`.
5. Any classification confidence below configurable default 0.55 → `REVIEW_FOOD_PREDICTIONS`.
6. Missing or ambiguous verified food/preparation/source mapping → `REQUEST_FOOD_PREPARATION`.
7. Incomplete, uncertain, review-required, non-conserving portions, or per-item uncertainty above the configurable default 20% of total net food mass → `REVIEW_PORTION_ESTIMATES`.
8. Complete inputs → `ACCEPT_ANALYSIS`, then `SAVE_MEAL` after the accepted-state transition.

The current experimental area estimator always requires review, so production auto-scans normally save a `needs_review` draft instead of marking a meal final. The draft preserves the total weight, visible classes, experimental estimates, and any complete estimated nutrition. User corrections recalculate totals from corrected portions. No correction or missing ingredient is silently fabricated.

## Actions and failure handling

Actions use the `AgentAction` enum: `ACCEPT_ANALYSIS`, `RETRY_WEIGHT`, `RECAPTURE_IMAGE`, `REVIEW_FOOD_PREDICTIONS`, `REVIEW_PORTION_ESTIMATES`, `REQUEST_PLATE_TARE`, `REQUEST_FOOD_PREPARATION`, and `SAVE_MEAL`. Decisions carry a rationale and user/retry flags; scans persist the action and reason. Image, sensor, model-loading, nutrient-map, and restart failures are saved as safe state/error codes instead of leaking exception details or marking a scan complete.

Future policy extensions can add validated evidence, thresholds, or user goals behind `AgentContext`; a future learned or LLM component would need explicit evaluation and may not bypass the physical-measurement, tare, source, and nutrition constraints.

## Research relationship

The five supplied papers motivate separating visual segmentation, weight measurement, portion estimation, and nutrient lookup. FoodSeg103 is not a cooking-state database; Nutrition5k provides a research direction for nutrition/mass data; Health to Eat and SCALEeat motivate sensor fusion. The current rule policy and mock demonstrations are engineering artifacts, not published-model reproductions or clinical recommendations.
