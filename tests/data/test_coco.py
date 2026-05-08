"""Unit tests for COCODataset."""

import unittest.mock

import pytest
import torch

from mole_jepa import config as config_module
from mole_jepa.data import coco

_C, _H, _W = 3, 224, 224
_MAX_LENGTH = 32
_N_IMAGES = 4
_CAPTIONS_PER_IMAGE = 5


def _make_config() -> config_module.DataConfig:
    return config_module.DataConfig(max_seq_length=_MAX_LENGTH)


def _make_hf_dataset(
    n_images: int = _N_IMAGES,
    captions_per_image: int = _CAPTIONS_PER_IMAGE,
    caption_field: str = "captions",
) -> list[dict]:
    return [
        {
            "image": unittest.mock.MagicMock(),
            caption_field: [
                f"image {i} caption {j}" for j in range(captions_per_image)
            ],
        }
        for i in range(n_images)
    ]


@pytest.fixture
def dataset() -> coco.COCODataset:
    hf_dataset = _make_hf_dataset()
    image_transform = unittest.mock.MagicMock(return_value=torch.randn(_C, _H, _W))
    tokenize = unittest.mock.MagicMock(
        return_value=(
            torch.randint(0, 100, (_MAX_LENGTH,)),
            torch.ones(_MAX_LENGTH, dtype=torch.long),
        )
    )

    with (
        unittest.mock.patch(
            "mole_jepa.data.transforms.build_image_transform",
            return_value=image_transform,
        ),
        unittest.mock.patch(
            "mole_jepa.data.transforms.build_tokenizer",
            return_value=tokenize,
        ),
    ):
        return coco.COCODataset(hf_dataset, _make_config())  # type: ignore[arg-type]


class TestCOCODataset:
    def test_n_images(self, dataset: coco.COCODataset) -> None:
        assert dataset.n_images == _N_IMAGES

    def test_captions_per_image(self, dataset: coco.COCODataset) -> None:
        assert dataset.captions_per_image == _CAPTIONS_PER_IMAGE

    def test_total_captions(self, dataset: coco.COCODataset) -> None:
        # Caption TensorDataset should have N * captions_per_image entries.
        assert len(dataset._captions) == _N_IMAGES * _CAPTIONS_PER_IMAGE

    def test_image_loader_batch_shape(self, dataset: coco.COCODataset) -> None:
        batch_size = 2
        loader = dataset.image_loader(batch_size=batch_size)
        (pixel_values,) = next(iter(loader))
        assert pixel_values.shape == (batch_size, _C, _H, _W)

    def test_caption_loader_batch_shape(self, dataset: coco.COCODataset) -> None:
        batch_size = 3
        loader = dataset.caption_loader(batch_size=batch_size)
        input_ids, attention_mask = next(iter(loader))
        assert input_ids.shape == (batch_size, _MAX_LENGTH)
        assert attention_mask.shape == (batch_size, _MAX_LENGTH)

    def test_image_loader_covers_all_images(self, dataset: coco.COCODataset) -> None:
        total = sum(pv.shape[0] for (pv,) in dataset.image_loader(batch_size=1))
        assert total == _N_IMAGES

    def test_caption_loader_covers_all_captions(
        self, dataset: coco.COCODataset
    ) -> None:
        total = sum(ids.shape[0] for ids, _ in dataset.caption_loader(batch_size=1))
        assert total == _N_IMAGES * _CAPTIONS_PER_IMAGE

    def test_custom_captions_per_image(self) -> None:
        hf_dataset = _make_hf_dataset(n_images=3, captions_per_image=10)
        image_transform = unittest.mock.MagicMock(return_value=torch.randn(_C, _H, _W))
        tokenize = unittest.mock.MagicMock(
            return_value=(
                torch.randint(0, 100, (_MAX_LENGTH,)),
                torch.ones(_MAX_LENGTH, dtype=torch.long),
            )
        )

        with (
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_image_transform",
                return_value=image_transform,
            ),
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_tokenizer",
                return_value=tokenize,
            ),
        ):
            ds = coco.COCODataset(
                hf_dataset,  # type: ignore[arg-type]
                _make_config(),
                captions_per_image=2,
            )

        assert ds.n_images == 3
        assert len(ds._captions) == 3 * 2
