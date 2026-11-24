# FoodSeg103 final metric tables

These compact CSVs are copied from the completed comparison run at `runs/segmentation-comparison-20261124_160934/`. They contain numeric evaluation results and the 104 label names; they contain no FoodSeg103 images, masks, logits, or model weights.

- `*_single_512_per_class.csv`: per-class pixel counts, IoU, Dice, precision, recall, and rare-support marker for the deterministic 512×512 protocol.
- `*_single_512_confusion.csv`: pixel confusion matrix; each row is a true class ID/name and each predicted column is one class ID. Counts are aggregated over all 2,135 official validation images.
- `*_single_512_presence_per_class.csv`: per-class image-level presence precision, recall, and F1.
- `*_hflip_512_*.csv`: the same tables for optional original-plus-horizontal-flip logit averaging.
- `training_curves.png`: actual B0 and B2 loss and internal-validation histories saved by their completed runs. The B0 and B2 internal splits differ; these curves are not a matched generalization comparison.

Background is class 0 in the full confusion matrix. The headline segmentation metrics in [the CV evaluation report](../../cv_model_evaluation_report.md) use foreground classes 1–103. B0 used the official validation split for checkpoint selection; B2 selected on its internal split of the official training partition. The direct B0/B2 comparison therefore has this selection-bias limitation.
