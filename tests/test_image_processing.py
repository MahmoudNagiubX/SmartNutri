from __future__ import annotations

import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageOps

from image_processing import (
    AugmentationConfig,
    EnhancementConfig,
    PreprocessingConfig,
    load_preprocessing_config,
    preprocess_pair,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def make_pair(width: int = 32, height: int = 32) -> tuple[Image.Image, Image.Image]:
    labels = np.zeros((height, width), dtype=np.uint8)
    labels[:, width // 2 :] = 2
    labels[height // 4 : height // 2, : width // 2] = 103
    intensities = labels.astype(np.uint8) * 2
    rgb = np.repeat(intensities[..., None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB"), Image.fromarray(labels, mode="L")


def base_config(image_size: int = 32) -> PreprocessingConfig:
    return PreprocessingConfig(image_size=image_size)


class ImageProcessingTests(unittest.TestCase):
    def test_exif_orientation_is_applied_before_image_mask_alignment(self) -> None:
        raw = np.zeros((2, 4, 3), dtype=np.uint8)
        raw[0] = (240, 10, 10)
        raw[1] = (10, 20, 240)
        source = Image.fromarray(raw, mode="RGB")
        exif = Image.Exif()
        exif[274] = 6
        encoded = io.BytesIO()
        source.save(encoded, format="PNG", exif=exif)

        with Image.open(io.BytesIO(encoded.getvalue())) as oriented_source:
            expected_resized = ImageOps.exif_transpose(oriented_source).convert("RGB")
            expected_resized = expected_resized.resize(
                (16, 32), Image.Resampling.BILINEAR
            )
        expected = Image.new("RGB", (32, 32), (124, 116, 104))
        expected.paste(expected_resized, (8, 0))
        mask = Image.fromarray(np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint8))

        _, labels, display_image = preprocess_pair(
            Image.open(io.BytesIO(encoded.getvalue())), mask, base_config()
        )

        self.assertEqual(display_image.size, (32, 32))
        np.testing.assert_array_equal(np.asarray(display_image), np.asarray(expected))
        self.assertEqual(labels[0, 8].item(), 1)
        self.assertEqual(labels[0, 23].item(), 2)
        self.assertEqual(labels[31, 8].item(), 7)
        self.assertEqual(labels[31, 23].item(), 8)

    def test_letterbox_padding_uses_ignore_index_and_normalized_mean_fill(self) -> None:
        image, mask = make_pair(width=32, height=16)

        pixel_values, labels, _ = preprocess_pair(image, mask, base_config())

        self.assertEqual(tuple(pixel_values.shape), (3, 32, 32))
        self.assertTrue(torch.all(labels[:8] == 255))
        self.assertTrue(torch.all(labels[8:24] != 255))
        self.assertTrue(torch.all(labels[24:] == 255))
        self.assertTrue(torch.allclose(pixel_values[:, :8], torch.zeros_like(pixel_values[:, :8]), atol=0.01))

    def test_mask_resize_uses_nearest_neighbor_and_preserves_semantic_ids(self) -> None:
        labels = np.zeros((8, 8), dtype=np.uint8)
        labels[2:6, 2:6] = 103
        image = Image.fromarray(np.repeat(labels[..., None], 3, axis=2), mode="RGB")

        _, resized_labels, _ = preprocess_pair(
            image, Image.fromarray(labels, mode="L"), base_config(image_size=64)
        )

        self.assertEqual(resized_labels.max().item(), 103)
        self.assertTrue(set(resized_labels.unique().tolist()).issubset({0, 103, 255}))

    def test_every_foodseg103_class_id_survives_preprocessing(self) -> None:
        labels = np.tile(np.arange(104, dtype=np.uint8), (4, 1))
        image = Image.fromarray(np.repeat(labels[..., None], 3, axis=2), mode="RGB")

        _, resized_labels, _ = preprocess_pair(
            image, Image.fromarray(labels, mode="L"), base_config(image_size=208)
        )

        self.assertTrue(set(range(104)).issubset(set(resized_labels.unique().tolist())))

    def test_invalid_mask_labels_are_rejected_instead_of_wrapped(self) -> None:
        image, _ = make_pair()
        invalid = np.zeros((32, 32), dtype=np.uint8)
        invalid[0, 0] = 104

        with self.assertRaisesRegex(ValueError, "FoodSeg103 labels"):
            preprocess_pair(image, Image.fromarray(invalid, mode="L"), base_config())

    def test_training_flip_is_synchronized_and_disabled_for_validation(self) -> None:
        image, mask = make_pair()
        config = PreprocessingConfig(
            image_size=32,
            augmentation=AugmentationConfig(horizontal_flip_probability=1.0),
        )
        baseline = preprocess_pair(image, mask, config, training=False, rng=random.Random(2))
        validation_again = preprocess_pair(
            image, mask, config, training=False, rng=random.Random(999)
        )
        flipped = preprocess_pair(image, mask, config, training=True, rng=random.Random(2))

        torch.testing.assert_close(baseline[0], validation_again[0], rtol=0, atol=0)
        torch.testing.assert_close(baseline[1], validation_again[1], rtol=0, atol=0)
        torch.testing.assert_close(flipped[1], torch.flip(baseline[1], dims=(-1,)))
        np.testing.assert_array_equal(
            np.asarray(flipped[2]),
            np.asarray(baseline[2].transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
        )

    def test_photometric_augmentation_changes_only_the_training_image(self) -> None:
        image, mask = make_pair()
        config = PreprocessingConfig(
            image_size=32,
            augmentation=AugmentationConfig(brightness_delta=0.4),
        )

        validation = preprocess_pair(image, mask, config, training=False)
        augmented = preprocess_pair(image, mask, config, training=True, rng=random.Random(7))

        self.assertFalse(torch.equal(validation[0], augmented[0]))
        torch.testing.assert_close(validation[1], augmented[1], rtol=0, atol=0)

    def test_validation_preprocessing_is_deterministic_with_safe_augmentation_config(self) -> None:
        image, mask = make_pair()
        config = load_preprocessing_config(
            PROJECT_DIR / "configs" / "image_processing" / "safe_augmentation.json"
        )

        first = preprocess_pair(image, mask, config, training=False, rng=random.Random(1))
        second = preprocess_pair(image, mask, config, training=False, rng=random.Random(900))

        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(first[2]), np.asarray(second[2]))

    def test_enhancement_changes_image_but_never_changes_mask(self) -> None:
        image, mask = make_pair()
        config = PreprocessingConfig(
            image_size=32,
            enhancements=EnhancementConfig(gamma=1.8),
        )

        baseline = preprocess_pair(image, mask, base_config())
        enhanced = preprocess_pair(image, mask, config, training=False)

        self.assertFalse(torch.equal(baseline[0], enhanced[0]))
        torch.testing.assert_close(baseline[1], enhanced[1], rtol=0, atol=0)

    def test_clahe_and_denoising_profiles_execute_without_touching_masks(self) -> None:
        image, mask = make_pair()
        config_dir = PROJECT_DIR / "configs" / "image_processing"
        for name in ("clahe_ablation.json", "denoising_ablation.json"):
            with self.subTest(config=name):
                config = load_preprocessing_config(config_dir / name)
                baseline = preprocess_pair(image, mask, base_config(config.image_size))
                enhanced = preprocess_pair(image, mask, config, training=False)
                self.assertFalse(torch.equal(baseline[0], enhanced[0]))
                torch.testing.assert_close(baseline[1], enhanced[1], rtol=0, atol=0)

    def test_segformer_normalization_is_applied_once(self) -> None:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:] = (128, 64, 192)
        image = Image.fromarray(rgb, mode="RGB")
        mask = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")

        pixel_values, _, _ = preprocess_pair(image, mask, base_config())

        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        expected = (rgb[0, 0].astype(np.float32) / 255.0 - mean) / std
        np.testing.assert_allclose(pixel_values[:, 16, 16].numpy(), expected, rtol=0, atol=1e-6)

    def test_existing_prepare_example_interface_remains_usable(self) -> None:
        from foodseg_data import prepare_example

        image, mask = make_pair()
        image_bytes = io.BytesIO()
        mask_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        mask.save(mask_bytes, format="PNG")
        row = {
            "id": 12,
            "image": {"bytes": image_bytes.getvalue(), "path": "sample.png"},
            "label": {"bytes": mask_bytes.getvalue()},
        }

        pixel_values, labels, display_image = prepare_example(row, 32)

        self.assertEqual(tuple(pixel_values.shape), (3, 32, 32))
        self.assertEqual(tuple(labels.shape), (32, 32))
        self.assertEqual(display_image.size, (32, 32))

    def test_iterable_dataset_reads_a_temporary_parquet_sample(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from foodseg_data import FoodSegParquetDataset

        image, mask = make_pair()
        image_bytes = io.BytesIO()
        mask_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        mask.save(mask_bytes, format="PNG")
        label_map = {"0": "background", **{str(i): f"class_{i}" for i in range(1, 104)}}

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "data").mkdir()
            (root / "id2label.json").write_text(json.dumps(label_map), encoding="utf-8")
            table = pa.Table.from_pylist(
                [
                    {
                        "id": 2,
                        "image": {"bytes": image_bytes.getvalue(), "path": "sample.png"},
                        "label": {"bytes": mask_bytes.getvalue()},
                    }
                ]
            )
            pq.write_table(table, root / "data" / "train-00000-of-00001.parquet")
            dataset = FoodSegParquetDataset(root, "train", 32, training=False)
            example = next(iter(dataset))

        self.assertEqual(tuple(example["pixel_values"].shape), (3, 32, 32))
        self.assertEqual(tuple(example["labels"].shape), (32, 32))

    def test_rotation_keeps_image_and_mask_landmarks_aligned(self) -> None:
        labels = np.zeros((32, 32), dtype=np.uint8)
        labels[6:18, 4:20] = 1
        rgb = np.repeat((labels * 255)[..., None], 3, axis=2)
        config = PreprocessingConfig(
            image_size=32,
            augmentation=AugmentationConfig(
                rotation_probability=1.0,
                rotation_degrees=7.0,
            ),
        )
        _, transformed_labels, transformed_image = preprocess_pair(
            Image.fromarray(rgb),
            Image.fromarray(labels),
            config,
            training=True,
            rng=random.Random(11),
        )

        mask_y, mask_x = np.where(transformed_labels.numpy() == 1)
        image_y, image_x = np.where(np.asarray(transformed_image).mean(axis=2) > 127)
        self.assertGreater(mask_x.size, 0)
        self.assertGreater(image_x.size, 0)
        self.assertAlmostEqual(float(mask_x.mean()), float(image_x.mean()), delta=0.75)
        self.assertAlmostEqual(float(mask_y.mean()), float(image_y.mean()), delta=0.75)

    def test_training_defaults_match_recommended_readiness_settings(self) -> None:
        import train

        with patch("sys.argv", ["train.py"]):
            args = train.parse_args()

        self.assertEqual(args.epochs, 20)
        self.assertEqual(args.early_stopping_patience, 6)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.gradient_accumulation_steps, 4)
        self.assertEqual(args.seed, 42)
        self.assertFalse(args.no_amp)

    def test_existing_training_evaluation_and_metrics_modules_import_without_training(self) -> None:
        import evaluate  # noqa: F401
        import metrics  # noqa: F401
        import train  # noqa: F401
        import visualize  # noqa: F401

    def test_all_preprocessing_experiment_configs_are_loadable(self) -> None:
        config_dir = PROJECT_DIR / "configs" / "image_processing"
        names = {
            "baseline.json",
            "safe_augmentation.json",
            "clahe_ablation.json",
            "gamma_ablation.json",
            "denoising_ablation.json",
        }

        for name in names:
            with self.subTest(config=name):
                self.assertEqual(load_preprocessing_config(config_dir / name).image_size, 512)


if __name__ == "__main__":
    unittest.main()
