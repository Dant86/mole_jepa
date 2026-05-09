"""Unit tests for CC3MDataset."""

import collections.abc
import typing
import unittest.mock

import pytest
import torch

from mole_jepa import config as config_module
from mole_jepa.data import cc3m

_C, _H, _W = 3, 224, 224
_MAX_LENGTH = 32
_N_EXAMPLES = 5


def _make_config() -> config_module.DataConfig:
    return config_module.DataConfig(max_seq_length=_MAX_LENGTH)


def _make_example(caption: str = "a photo") -> dict:
    return {"image": unittest.mock.MagicMock(), "caption": caption}


def _make_transform_mocks(
    n: int,
) -> tuple[unittest.mock.MagicMock, unittest.mock.MagicMock]:
    image_transform = unittest.mock.MagicMock(return_value=torch.randn(_C, _H, _W))
    tokenize = unittest.mock.MagicMock(
        return_value=(
            torch.randint(0, 100, (_MAX_LENGTH,)),
            torch.ones(_MAX_LENGTH, dtype=torch.long),
        )
    )
    return image_transform, tokenize


@pytest.fixture
def dataset() -> collections.abc.Iterator[cc3m.CC3MDataset]:
    examples = [_make_example(f"caption {i}") for i in range(_N_EXAMPLES)]
    hf_dataset = unittest.mock.MagicMock()
    hf_dataset.__iter__ = unittest.mock.Mock(return_value=iter(examples))

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
        yield cc3m.CC3MDataset(hf_dataset, _make_config())


class TestCC3MDataset:
    def test_yields_correct_number_of_items(self, dataset: cc3m.CC3MDataset) -> None:
        items = list(dataset)
        assert len(items) == _N_EXAMPLES

    def test_item_shapes(self, dataset: cc3m.CC3MDataset) -> None:
        pixel_values, input_ids, attention_mask = next(iter(dataset))
        assert pixel_values.shape == (_C, _H, _W)
        assert input_ids.shape == (_MAX_LENGTH,)
        assert attention_mask.shape == (_MAX_LENGTH,)

    def test_malformed_examples_are_skipped(self) -> None:
        # Two valid examples, one that raises during image transform.
        bad_transform = unittest.mock.MagicMock(
            side_effect=[
                torch.randn(_C, _H, _W),
                RuntimeError("bad image"),
                torch.randn(_C, _H, _W),
            ]
        )
        tokenize = unittest.mock.MagicMock(
            return_value=(
                torch.randint(0, 100, (_MAX_LENGTH,)),
                torch.ones(_MAX_LENGTH, dtype=torch.long),
            )
        )
        examples = [_make_example() for _ in range(3)]
        hf_dataset = unittest.mock.MagicMock()
        hf_dataset.__iter__ = unittest.mock.Mock(return_value=iter(examples))

        with (
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_image_transform",
                return_value=bad_transform,
            ),
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_tokenizer",
                return_value=tokenize,
            ),
        ):
            ds = cc3m.CC3MDataset(hf_dataset, _make_config())
            assert len(list(ds)) == 2

    def test_worker_init_fn_shards_dataset(self, dataset: cc3m.CC3MDataset) -> None:
        # Save a reference before worker_init_fn reassigns dataset._dataset.
        original_hf_dataset = typing.cast(unittest.mock.MagicMock, dataset._dataset)

        worker_info = unittest.mock.MagicMock()
        worker_info.num_workers = 4
        worker_info.id = 2
        worker_info.dataset = dataset

        with unittest.mock.patch(
            "torch.utils.data.get_worker_info", return_value=worker_info
        ):
            cc3m.CC3MDataset.worker_init_fn(worker_id=2)

        original_hf_dataset.shard.assert_called_once_with(num_shards=4, index=2)

    def test_worker_init_fn_noop_outside_worker(
        self, dataset: cc3m.CC3MDataset
    ) -> None:
        mock_dataset = typing.cast(unittest.mock.MagicMock, dataset._dataset)

        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=None):
            cc3m.CC3MDataset.worker_init_fn(worker_id=0)

        mock_dataset.shard.assert_not_called()
