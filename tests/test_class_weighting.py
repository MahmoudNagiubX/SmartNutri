import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from class_weighting import (
    compute_class_weights,
    count_training_pixel_frequencies,
    load_class_weight_config,
    weighted_cross_entropy_loss,
)
from image_processing import PreprocessingConfig


def _png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _write_split(
    root: Path,
    split: str,
    mask: Image.Image,
    *,
    row_id: int,
    image_bytes: bytes | None = None,
) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "id": row_id,
        "image": {
            "bytes": image_bytes or _png_bytes(Image.new("RGB", mask.size, (120, 100, 80))),
            "path": None,
        },
        "label": {"bytes": _png_bytes(mask), "path": None},
    }
    table = pa.Table.from_pylist([row])
    pq.write_table(table, data_dir / f"{split}-00000-of-00001.parquet")


class ClassWeightingTests(unittest.TestCase):
    def test_training_pixel_frequencies_use_only_train_and_exclude_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = {str(class_id): ("background" if class_id == 0 else f"food_{class_id}") for class_id in range(104)}
            (root / "id2label.json").write_text(json.dumps(labels), encoding="utf-8")

            train_mask = np.zeros((8, 8), dtype=np.uint8)
            train_mask[:, 4:6] = 1
            train_mask[:, 6:7] = 2
            train_mask[:, 7:8] = 255
            _write_split(root, "train", Image.fromarray(train_mask, mode="L"), row_id=1)

            validation_mask = Image.new("L", (8, 8), color=88)
            _write_split(root, "validation", validation_mask, row_id=1)

            result = count_training_pixel_frequencies(
                root,
                PreprocessingConfig(image_size=32),
            )

        self.assertEqual(result.image_count, 1)
        self.assertEqual(result.class_pixel_counts.shape, (104,))
        self.assertEqual(int(result.class_pixel_counts[0]), 512)
        self.assertEqual(int(result.class_pixel_counts[1]), 256)
        self.assertEqual(int(result.class_pixel_counts[2]), 128)
        self.assertEqual(int(result.class_pixel_counts[88]), 0)
        self.assertEqual(result.ignored_padding_pixels, 128)
        self.assertEqual(result.valid_pixel_count, 896)

    def test_frequency_resize_uses_exif_corrected_image_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = {str(class_id): ("background" if class_id == 0 else f"food_{class_id}") for class_id in range(104)}
            (root / "id2label.json").write_text(json.dumps(labels), encoding="utf-8")
            mask_values = np.zeros((8, 4), dtype=np.uint8)
            mask_values[:, 2:] = 1
            exif = Image.Exif()
            exif[274] = 6
            image_stream = io.BytesIO()
            Image.new("RGB", (8, 4), (80, 100, 120)).save(
                image_stream,
                format="JPEG",
                exif=exif,
            )
            _write_split(
                root,
                "train",
                Image.fromarray(mask_values, mode="L"),
                row_id=1,
                image_bytes=image_stream.getvalue(),
            )

            result = count_training_pixel_frequencies(
                root,
                PreprocessingConfig(image_size=32),
            )

        self.assertEqual(int(result.class_pixel_counts[0]), 256)
        self.assertEqual(int(result.class_pixel_counts[1]), 256)
        self.assertEqual(result.valid_pixel_count, 512)
        self.assertEqual(result.ignored_padding_pixels, 512)

    def test_weights_keep_all_classes_finite_mean_one_and_under_cap(self):
        counts = torch.zeros(104, dtype=torch.int64)
        counts[0] = 10_000
        counts[1] = 1_000
        counts[2] = 100
        counts[103] = 1

        weights = compute_class_weights(counts, tempering_exponent=0.5, max_weight=5.0)

        self.assertEqual(tuple(weights.shape), (104,))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(weights > 0))
        self.assertLessEqual(float(weights.max()), 5.0 + 1e-6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[103]), float(weights[2]))
        self.assertGreater(float(weights[2]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_weighted_loss_aligns_spatial_shapes_and_backpropagates(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 104, 4, 4, requires_grad=True)
        labels = torch.randint(0, 104, (2, 16, 16), dtype=torch.int64)
        labels[:, 0, 0] = 255
        weights = torch.ones(104)
        weights[1] = 3.0

        loss = weighted_cross_entropy_loss(logits, labels, weights)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_ignore_index_has_zero_gradient_and_all_ignored_batch_is_finite(self):
        logits = torch.randn(1, 104, 2, 2, requires_grad=True)
        labels = torch.tensor([[[0, 1], [255, 1]]], dtype=torch.int64)
        weights = torch.ones(104)

        loss = weighted_cross_entropy_loss(logits, labels, weights)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(logits.grad[0, :, 1, 0], torch.zeros(104)))

        ignored_logits = torch.randn(1, 104, 2, 2, requires_grad=True)
        ignored_labels = torch.full((1, 2, 2), 255, dtype=torch.int64)
        ignored_loss = weighted_cross_entropy_loss(ignored_logits, ignored_labels, weights)
        ignored_loss.backward()
        self.assertEqual(float(ignored_loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(ignored_logits.grad).all())
        self.assertEqual(float(ignored_logits.grad.abs().sum()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_weighted_loss_cuda_backward_uses_the_same_104_class_contract(self):
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        # SegFormer-B0 emits stride-4 logits for 512-pixel inputs. Exercise the
        # real batch/target shape without loading a model or starting training.
        logits = torch.randn(2, 104, 128, 128, device=device, requires_grad=True)
        labels = torch.randint(0, 104, (2, 512, 512), dtype=torch.int64, device=device)
        labels[:, :16, :16] = 255
        weights = torch.ones(104, device=device)

        loss = weighted_cross_entropy_loss(logits, labels, weights)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertLess(torch.cuda.max_memory_allocated(device), 2 * 1024**3)

    def test_experiment_config_preserves_foodseg_label_and_ignore_contract(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "losses"
            / "experiment_2_class_weighted_ce.json"
        )
        config = load_class_weight_config(config_path)
        self.assertEqual(config.num_classes, 104)
        self.assertEqual(config.ignore_index, 255)
        self.assertEqual(config.frequency_split, "train")
        self.assertEqual(config.tempering_exponent, 0.5)
        self.assertEqual(config.max_weight, 5.0)


if __name__ == "__main__":
    unittest.main()
