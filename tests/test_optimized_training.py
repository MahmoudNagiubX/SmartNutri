from __future__ import annotations

import io
import random
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

from class_weighting import compute_class_weights
from evaluate_optimized import resize_logits_argmax, unletterbox_logits_argmax
from foodseg_data import ParquetSplit
from image_processing import AugmentationConfig, PreprocessingConfig, preprocess_pair
from metrics import update_image_presence_confusion
from train_optimized import (
    DEFAULT_CONFIG,
    create_timestamped_run_dir,
    deterministic_train_validation_split,
    checkpoint_provenance,
    load_foodseg_b2_checkpoint,
    load_training_config,
    optimizer_parameter_groups,
)
from training_losses import FocalDiceLoss


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class OptimizedTrainingTests(unittest.TestCase):
    def test_finetune_config_uses_lower_separate_learning_rates(self):
        config, _, _ = load_training_config(
            DEFAULT_CONFIG.with_name("segformer_b2_foodseg103_finetune.json")
        )

        self.assertEqual(config["training"]["encoder_learning_rate"], 1e-5)
        self.assertEqual(config["training"]["decoder_learning_rate"], 2e-5)
        self.assertEqual(config["training"]["max_epochs"], 25)
        self.assertEqual(config["training"]["early_stopping_patience"], 7)

    def test_optimizer_parameter_groups_cover_encoder_and_decoder_once(self):
        class TinySegFormer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.segformer = torch.nn.Linear(2, 2)
                self.decode_head = torch.nn.Linear(2, 3)

        model = TinySegFormer()
        groups = optimizer_parameter_groups(
            model,
            {"encoder_learning_rate": 1e-5, "decoder_learning_rate": 2e-5},
        )

        self.assertEqual([group["group_name"] for group in groups], ["encoder", "decoder"])
        self.assertEqual([group["lr"] for group in groups], [1e-5, 2e-5])
        grouped_parameters = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(grouped_parameters), len(list(model.parameters())))
        self.assertEqual({id(p) for p in grouped_parameters}, {id(p) for p in model.parameters()})

    def test_local_checkpoint_loader_never_resolves_remote_weights(self):
        id2label = {class_id: f"class {class_id}" for class_id in range(104)}
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={str(key): value for key, value in id2label.items()},
                label2id={value: key for key, value in id2label.items()},
                num_labels=104,
                depths=[3, 4, 6, 3],
                hidden_sizes=[64, 128, 320, 512],
                decoder_hidden_size=768,
            ),
            decode_head=SimpleNamespace(
                classifier=SimpleNamespace(out_channels=104)
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            with patch(
                "train_optimized.SegformerForSemanticSegmentation.from_pretrained",
                return_value=fake_model,
            ) as from_pretrained:
                loaded = load_foodseg_b2_checkpoint(checkpoint_dir, id2label)

        self.assertIs(loaded, fake_model)
        self.assertEqual(fake_model.config.semantic_loss_ignore_index, 255)
        from_pretrained.assert_called_once_with(
            str(checkpoint_dir.resolve()),
            local_files_only=True,
        )

    def test_checkpoint_provenance_hashes_existing_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "model.safetensors").write_bytes(b"model-bytes")

            identity = checkpoint_provenance(checkpoint_dir)

        self.assertEqual(identity["weights_file"], "model.safetensors")
        self.assertEqual(
            identity["weights_sha256"],
            "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
        )

    def test_pretrained_config_selects_pinned_official_pytorch_weights(self):
        config, _, _ = load_training_config(DEFAULT_CONFIG)

        self.assertEqual(
            config["pretrained"]["revision"],
            "de01bae28967510f9ddd496c60a969357195400c",
        )
        self.assertEqual(config["pretrained"]["artifact_file"], "pytorch_model.bin")
        self.assertFalse(config["pretrained"]["use_safetensors"])

    def test_internal_validation_split_is_deterministic_disjoint_and_498_images(self):
        sample_ids = list(range(4983))

        train_ids, validation_ids = deterministic_train_validation_split(
            sample_ids, validation_fraction=0.10, seed=42
        )
        repeated_train_ids, repeated_validation_ids = deterministic_train_validation_split(
            sample_ids, validation_fraction=0.10, seed=42
        )

        self.assertEqual(len(train_ids), 4485)
        self.assertEqual(len(validation_ids), 498)
        self.assertFalse(set(train_ids) & set(validation_ids))
        self.assertEqual(set(train_ids) | set(validation_ids), set(sample_ids))
        self.assertEqual(train_ids, repeated_train_ids)
        self.assertEqual(validation_ids, repeated_validation_ids)

    def test_parquet_subset_reads_only_selected_training_ids(self):
        image = Image.new("RGB", (8, 8), color=(30, 60, 90))
        mask = Image.new("L", (8, 8), color=3)
        image_value = {"bytes": _png_bytes(image), "path": None}
        mask_value = {"bytes": _png_bytes(mask), "path": None}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            table = pa.table(
                {
                    "id": [10, 11, 12],
                    "image": [image_value, image_value, image_value],
                    "label": [mask_value, mask_value, mask_value],
                }
            )
            pq.write_table(table, data / "train-00000-of-00001.parquet", row_group_size=2)
            source = ParquetSplit(root, "train")

            rows = list(
                source.iter_rows(
                    shuffle=False, rng=random.Random(0), include_ids={"11", "12"}
                )
            )

        self.assertEqual([row["id"] for row in rows], [11, 12])

    def test_training_scale_uses_synchronized_mask_geometry(self):
        labels = np.zeros((64, 64), dtype=np.uint8)
        labels[20:40, 20:40] = 7
        rgb = np.repeat((labels * 30)[..., None], 3, axis=2)
        config = PreprocessingConfig(
            image_size=64,
            augmentation=AugmentationConfig(
                scale_probability=1.0,
                scale_min=0.75,
                scale_max=0.75,
            ),
        )

        _, transformed_labels, transformed_image = preprocess_pair(
            Image.fromarray(rgb),
            Image.fromarray(labels),
            config,
            training=True,
            rng=random.Random(3),
        )

        mask_y, mask_x = np.where(transformed_labels.numpy() == 7)
        image_values = np.asarray(transformed_image).mean(axis=2)
        image_y, image_x = np.where(image_values > 100)
        self.assertGreater(mask_x.size, 0)
        self.assertGreater(image_x.size, 0)
        self.assertAlmostEqual(float(mask_x.mean()), float(image_x.mean()), delta=1.5)
        self.assertAlmostEqual(float(mask_y.mean()), float(image_y.mean()), delta=1.5)
        self.assertTrue(set(np.unique(transformed_labels.numpy())).issubset({0, 7, 255}))

    def test_focal_dice_loss_ignores_padding_and_backpropagates(self):
        logits = torch.randn(2, 104, 12, 12, requires_grad=True)
        labels = torch.randint(0, 104, (2, 12, 12))
        labels[:, 0, 0] = 255
        weights = compute_class_weights(np.ones(104, dtype=np.int64), max_weight=2.0)
        criterion = FocalDiceLoss(class_weights=weights, ignore_index=255)

        loss = criterion(logits, labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad[:, :, 0, 0], torch.zeros_like(logits.grad[:, :, 0, 0])))

    def test_focal_dice_loss_returns_differentiable_zero_for_all_ignored_batch(self):
        logits = torch.randn(1, 104, 4, 4, requires_grad=True)
        labels = torch.full((1, 4, 4), 255, dtype=torch.long)
        criterion = FocalDiceLoss(ignore_index=255)

        loss = criterion(logits, labels)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_class_presence_confusion_counts_food_classes_per_image(self):
        predictions = torch.tensor(
            [
                [[0, 2], [0, 0]],
                [[0, 2], [0, 0]],
            ]
        )
        labels = torch.tensor(
            [
                [[0, 2], [0, 0]],
                [[0, 3], [0, 0]],
            ]
        )
        confusion = torch.zeros((104, 2, 2), dtype=torch.int64)

        update_image_presence_confusion(confusion, predictions, labels, ignore_index=255)

        self.assertEqual(confusion[2].tolist(), [[0, 1], [0, 1]])
        self.assertEqual(confusion[3].tolist(), [[1, 0], [1, 0]])

    def test_run_directory_creation_never_reuses_a_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = create_timestamped_run_dir(
                Path(temporary), "segformer-b2", timestamp="20261124_120000"
            )
            second = create_timestamped_run_dir(
                Path(temporary), "segformer-b2", timestamp="20261124_120000"
            )

        self.assertNotEqual(first, second)
        self.assertIn("20261124_120000", first.name)
        self.assertIn("20261124_120000", second.name)
        self.assertTrue(second.name.endswith("-01"))

    def test_tiled_original_resolution_resize_matches_full_bilinear_argmax(self):
        logits = torch.randn(1, 104, 7, 9)
        expected = F.interpolate(
            logits,
            size=(17, 21),
            mode="bilinear",
            align_corners=False,
        )[0].argmax(dim=0)

        actual = resize_logits_argmax(
            logits,
            (17, 21),
            tile_rows=4,
            max_full_logits_values=1,
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_original_resolution_resize_removes_letterbox_padding(self):
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 1] = 10.0
        logits[:, 0, 1:3, :] = 20.0

        prediction = unletterbox_logits_argmax(
            logits,
            (2, 4),
            image_size=4,
        )

        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertTrue(torch.equal(prediction, torch.zeros_like(prediction)))


if __name__ == "__main__":
    unittest.main()
