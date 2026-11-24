from __future__ import annotations

import json
import random
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from image_processing import (
    IGNORE_INDEX,
    IMAGE_MEAN,
    IMAGE_STD,
    PreprocessingConfig,
    preprocess_pair,
)


class ParquetSplit:
    """Read a FoodSeg103 split directly from its local Parquet row groups."""

    def __init__(self, foodseg_root: str | Path, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")

        self.root = Path(foodseg_root)
        self.split = split
        self.files = sorted((self.root / "data").glob(f"{split}-*.parquet"))
        if not self.files:
            raise FileNotFoundError(
                f"No {split} Parquet shards found under {self.root / 'data'}"
            )

        self.row_groups: list[tuple[int, int, int]] = []
        self.num_rows = 0
        for file_index, path in enumerate(self.files):
            parquet_file = pq.ParquetFile(path)
            for group_index in range(parquet_file.metadata.num_row_groups):
                row_count = parquet_file.metadata.row_group(group_index).num_rows
                self.row_groups.append((file_index, group_index, row_count))
                self.num_rows += row_count

    def list_ids(self) -> list[int | str]:
        """Read only sample IDs, avoiding image and mask payloads."""
        sample_ids: list[int | str] = []
        for file_index, group_index, _ in self.row_groups:
            rows = pq.ParquetFile(self.files[file_index]).read_row_group(
                group_index, columns=["id"]
            ).to_pylist()
            sample_ids.extend(row["id"] for row in rows)
        return sample_ids

    def iter_rows(
        self,
        *,
        shuffle: bool,
        rng: random.Random,
        include_ids: Iterable[int | str] | None = None,
    ) -> Iterator[dict]:
        order = list(range(len(self.row_groups)))
        if shuffle:
            rng.shuffle(order)
        selected_ids = None if include_ids is None else {str(value) for value in include_ids}

        readers: dict[int, pq.ParquetFile] = {}
        for order_index in order:
            file_index, group_index, _ = self.row_groups[order_index]
            parquet_file = readers.get(file_index)
            if parquet_file is None:
                parquet_file = pq.ParquetFile(self.files[file_index])
                readers[file_index] = parquet_file

            rows = parquet_file.read_row_group(
                group_index, columns=["image", "label", "id"]
            ).to_pylist()
            if selected_ids is not None:
                rows = [row for row in rows if str(row["id"]) in selected_ids]
            if shuffle:
                rng.shuffle(rows)
            yield from rows

    def row_at(self, index: int) -> dict:
        if index < 0 or index >= self.num_rows:
            raise IndexError(f"index {index} is outside 0..{self.num_rows - 1}")

        offset = 0
        for file_index, group_index, row_count in self.row_groups:
            if index < offset + row_count:
                parquet_file = pq.ParquetFile(self.files[file_index])
                return (
                    parquet_file.read_row_group(
                        group_index, columns=["image", "label", "id"]
                    )
                    .slice(index - offset, 1)
                    .to_pylist()[0]
                )
            offset += row_count
        raise IndexError(index)


def load_label_map(foodseg_root: str | Path) -> dict[int, str]:
    path = Path(foodseg_root) / "id2label.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels = {int(key): str(value).strip() for key, value in raw.items()}
    expected_ids = set(range(104))
    if set(labels) != expected_ids:
        raise ValueError(
            f"Expected FoodSeg103 ids 0..103 in {path}; found {len(labels)} ids"
        )
    if labels[0].lower() != "background":
        raise ValueError(f"Expected label 0 to be background in {path}")
    return labels


def _image_bytes(row: dict, key: str) -> bytes:
    value = row.get(key)
    if not value or value.get("bytes") is None:
        raise ValueError(f"Parquet row {row.get('id')} has no embedded {key} bytes")
    return value["bytes"]


def prepare_example(
    row: dict,
    image_size: int,
    *,
    horizontal_flip: bool | None = None,
    processing_config: PreprocessingConfig | None = None,
    training: bool = False,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """Decode and preprocess one pair while keeping the historical return shape."""
    if image_size < 32:
        raise ValueError("image_size must be at least 32 pixels")

    config = processing_config or PreprocessingConfig(image_size=image_size)
    if config.image_size != image_size:
        config = replace(config, image_size=image_size)

    with Image.open(BytesIO(_image_bytes(row, "image"))) as source_image:
        with Image.open(BytesIO(_image_bytes(row, "label"))) as source_mask:
            source_image.load()
            source_mask.load()
            return preprocess_pair(
                source_image,
                source_mask,
                config,
                training=training,
                rng=rng,
                force_horizontal_flip=horizontal_flip,
            )


class FoodSegParquetDataset(IterableDataset):
    """Streaming, row-group-shuffled FoodSeg103 dataset for a single DataLoader worker."""

    def __init__(
        self,
        foodseg_root: str | Path,
        split: str,
        image_size: int | None = None,
        *,
        training: bool = False,
        seed: int = 42,
        processing_config: PreprocessingConfig | None = None,
        include_ids: Iterable[int | str] | None = None,
    ) -> None:
        super().__init__()
        self.source = ParquetSplit(foodseg_root, split)
        self.processing_config = processing_config or PreprocessingConfig(
            image_size=image_size or 512
        )
        if image_size is not None and self.processing_config.image_size != image_size:
            self.processing_config = replace(self.processing_config, image_size=image_size)
        self.image_size = self.processing_config.image_size
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.include_ids = (
            None if include_ids is None else frozenset(str(value) for value in include_ids)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.source.num_rows if self.include_ids is None else len(self.include_ids)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random.Random(self.seed + self.epoch)
        for row in self.source.iter_rows(
            shuffle=self.training,
            rng=rng,
            include_ids=self.include_ids,
        ):
            pixel_values, labels, _ = prepare_example(
                row,
                self.image_size,
                processing_config=self.processing_config,
                training=self.training,
                rng=rng,
            )
            yield {"pixel_values": pixel_values, "labels": labels}
