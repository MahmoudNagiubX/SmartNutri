from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from nutrition5k import (
    DishRecord,
    NutritionImageTransform,
    PositiveDensityModel,
    evaluation_report,
    fit_target_scales,
    load_partitions,
    mean_training_density,
    validation_score,
)
from train_nutrition5k import _train_epoch


def _dish_id(day: int, minute: int) -> str:
    timestamp = int(datetime(2019, 5, day, 12, minute, tzinfo=timezone.utc).timestamp())
    return f"dish_{timestamp}"


class Nutrition5kTests(unittest.TestCase):
    def test_partitioning_excludes_unmatched_and_keeps_day_groups_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "nutrition"
            image_dir = root / "images"
            split_dir = root / "dish_ids" / "splits"
            image_dir.mkdir(parents=True)
            split_dir.mkdir(parents=True)
            train_ids = [
                _dish_id(14, 1), _dish_id(14, 2),
                _dish_id(15, 1), _dish_id(15, 2),
                _dish_id(16, 1), _dish_id(16, 2),
            ]
            test_id = _dish_id(17, 1)
            unmatched_id = _dish_id(18, 1)
            all_ids = train_ids + [test_id, unmatched_id]
            for dish_id in all_ids:
                Image.new("RGB", (16, 12), (80, 120, 40)).save(image_dir / f"{dish_id}.png")
            with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as sink:
                writer = csv.DictWriter(
                    sink,
                    fieldnames=[
                        "dish_id", "total_calories", "total_mass_g", "fat_g",
                        "carb_g", "protein_g", "ingredients",
                    ],
                )
                writer.writeheader()
                for index, dish_id in enumerate(all_ids):
                    writer.writerow(
                        {
                            "dish_id": dish_id,
                            "total_calories": 100 + index,
                            "total_mass_g": 100,
                            "fat_g": 10 + index,
                            "carb_g": 20 + index,
                            "protein_g": 15 + index,
                            "ingredients": "test:100",
                        }
                    )
            (split_dir / "rgb_train_ids.txt").write_text("\n".join(train_ids), encoding="utf-8")
            (split_dir / "rgb_test_ids.txt").write_text(test_id, encoding="utf-8")

            partitions = load_partitions(root, seed=42, validation_fraction=0.25)

        self.assertEqual(len(partitions.training) + len(partitions.validation), 6)
        self.assertEqual([record.dish_id for record in partitions.test], [test_id])
        self.assertEqual(partitions.excluded_ids, [unmatched_id])
        self.assertFalse(
            {record.group_id for record in partitions.training}
            & {record.group_id for record in partitions.validation}
        )
        self.assertTrue(
            {record.dish_id for record in partitions.training + partitions.validation}.isdisjoint(
                {record.dish_id for record in partitions.test}
            )
        )

    def test_target_scales_and_mean_baseline_are_derived_from_given_training_rows(self):
        records = [
            DishRecord("dish_1557835200", Path("one.png"), "2019-05-14", 100,
                       (100, 10, 20, 5), (1.0, 0.1, 0.2, 0.05)),
            DishRecord("dish_1557921600", Path("two.png"), "2019-05-15", 200,
                       (400, 40, 60, 10), (2.0, 0.2, 0.3, 0.05)),
            DishRecord("dish_1558008000", Path("three.png"), "2019-05-16", 100,
                       (900, 20, 40, 20), (9.0, 0.2, 0.4, 0.2)),
        ]

        scales = fit_target_scales(records[:2])
        mean_density = mean_training_density(records[:2])

        np.testing.assert_allclose(scales, [1.5, 0.15, 0.25, 0.05])
        np.testing.assert_allclose(mean_density, [1.5, 0.15, 0.25, 0.05])
        self.assertNotEqual(float(scales[0]), float(np.median([1.0, 2.0, 9.0])))

    def test_evaluation_reports_totals_and_density_errors_against_baseline(self):
        record = DishRecord(
            "dish_1557835200", Path("one.png"), "2019-05-14", 100,
            (100, 10, 20, 5), (1.0, 0.1, 0.2, 0.05),
        )
        report = evaluation_report(
            [record],
            np.asarray([[1.2, 0.08, 0.25, 0.04]], dtype=np.float32),
            np.asarray([1.0, 0.1, 0.2, 0.05], dtype=np.float32),
        )

        calories = report["nutrients"]["calories"]
        self.assertAlmostEqual(calories["model_total"]["mae"], 20.0, places=5)
        self.assertAlmostEqual(calories["model_total"]["normalized_mae"], 0.2, places=5)
        self.assertAlmostEqual(calories["model_total"]["signed_bias"], 20.0, places=5)
        self.assertAlmostEqual(calories["mean_density_baseline_total"]["mae"], 0.0, places=5)

    def test_validation_score_is_scale_normalized_and_zero_for_exact_prediction(self):
        record = DishRecord(
            "dish_1557835200", Path("one.png"), "2019-05-14", 100,
            (100, 10, 20, 5), (1.0, 0.1, 0.2, 0.05),
        )
        score = validation_score(
            [record], np.asarray([[1.0, 0.1, 0.2, 0.05]]),
            np.asarray([1.0, 0.1, 0.2, 0.05]),
        )
        self.assertEqual(score, 0.0)

    def test_image_transform_preserves_rgb_shape_and_normalizes(self):
        transform = NutritionImageTransform(image_size=64, training=False)
        pixels = transform(Image.new("RGB", (80, 60), (100, 120, 140)))
        self.assertEqual(tuple(pixels.shape), (3, 64, 64))
        self.assertTrue(torch.isfinite(pixels).all())

    def test_model_wrapper_outputs_four_nonnegative_values(self):
        backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, 4)
        )
        model = PositiveDensityModel(backbone)
        prediction = model(torch.zeros(2, 3, 8, 8))
        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertTrue(torch.all(prediction >= 0))

    def test_cpu_training_epoch_backpropagates_and_updates_after_accumulation(self):
        backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, 4)
        )
        model = PositiveDensityModel(backbone)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        samples = [
            {
                "pixel_values": torch.randn(3, 8, 8),
                "density": torch.tensor([1.0, 0.1, 0.2, 0.05]),
            }
            for _ in range(4)
        ]
        loader = torch.utils.data.DataLoader(samples, batch_size=2, shuffle=False)
        before = model.backbone[-1].weight.detach().clone()

        result = _train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device=torch.device("cpu"),
            amp_dtype=None,
            target_scales=torch.tensor([1.0, 0.1, 0.2, 0.05]),
            training_config={
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
                "smooth_l1_beta": 1.0,
            },
        )

        self.assertEqual(result["examples_seen"], 4)
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertTrue(np.isfinite(result["mean_normalized_smooth_l1"]))
        self.assertFalse(torch.equal(before, model.backbone[-1].weight.detach()))


if __name__ == "__main__":
    unittest.main()
