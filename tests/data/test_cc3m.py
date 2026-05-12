"""Unit tests for CC3MDataset."""

import collections.abc
import io
import typing
import unittest.mock
import urllib.error

import PIL.Image
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
    # Field names match the DataConfig defaults (pixparse/cc3m-wds layout).
    return {"jpg": unittest.mock.MagicMock(), "txt": caption}


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

    def test_worker_init_fn_shards_dataset(self, dataset: cc3m.CC3MDataset) -> None:
        # 4 workers, 4 shards — every worker gets one shard.
        original_hf_dataset = typing.cast(unittest.mock.MagicMock, dataset._dataset)
        original_hf_dataset.n_shards = 4

        worker_info = unittest.mock.MagicMock()
        worker_info.num_workers = 4
        worker_info.id = 2
        worker_info.dataset = dataset

        with unittest.mock.patch(
            "torch.utils.data.get_worker_info", return_value=worker_info
        ):
            cc3m.CC3MDataset.worker_init_fn(worker_id=2)

        original_hf_dataset.shard.assert_called_once_with(num_shards=4, index=2)

    def test_worker_init_fn_caps_to_available_shards(
        self, dataset: cc3m.CC3MDataset
    ) -> None:
        # 4 workers but only 2 shards — sharding is capped, excess workers no-op.
        original_hf_dataset = typing.cast(unittest.mock.MagicMock, dataset._dataset)
        original_hf_dataset.n_shards = 2

        worker_info = unittest.mock.MagicMock()
        worker_info.num_workers = 4
        worker_info.id = 1  # within range — should get shard 1 of 2
        worker_info.dataset = dataset

        with unittest.mock.patch(
            "torch.utils.data.get_worker_info", return_value=worker_info
        ):
            cc3m.CC3MDataset.worker_init_fn(worker_id=1)

        original_hf_dataset.shard.assert_called_once_with(num_shards=2, index=1)

    def test_worker_init_fn_noop_for_excess_worker(
        self, dataset: cc3m.CC3MDataset
    ) -> None:
        # Worker id beyond available shards — shard() must not be called.
        mock_dataset = typing.cast(unittest.mock.MagicMock, dataset._dataset)
        mock_dataset.n_shards = 1

        worker_info = unittest.mock.MagicMock()
        worker_info.num_workers = 4
        worker_info.id = 2  # beyond n_shards=1
        worker_info.dataset = dataset

        with unittest.mock.patch(
            "torch.utils.data.get_worker_info", return_value=worker_info
        ):
            cc3m.CC3MDataset.worker_init_fn(worker_id=2)

        mock_dataset.shard.assert_not_called()

    def test_worker_init_fn_noop_outside_worker(
        self, dataset: cc3m.CC3MDataset
    ) -> None:
        mock_dataset = typing.cast(unittest.mock.MagicMock, dataset._dataset)

        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=None):
            cc3m.CC3MDataset.worker_init_fn(worker_id=0)

        mock_dataset.shard.assert_not_called()

    def test_url_image_is_fetched(self) -> None:
        """When the image field is a URL string, _fetch_image is called."""
        example = {"jpg": "http://example.com/img.jpg", "txt": "a photo"}
        hf_dataset = unittest.mock.MagicMock()
        hf_dataset.__iter__ = unittest.mock.Mock(return_value=iter([example]))

        fake_image = PIL.Image.new("RGB", (_H, _W))
        image_transform = unittest.mock.MagicMock(return_value=torch.randn(_C, _H, _W))
        tokenize = unittest.mock.MagicMock(
            return_value=(
                torch.randint(0, 100, (_MAX_LENGTH,)),
                torch.ones(_MAX_LENGTH, dtype=torch.long),
            )
        )
        with (
            unittest.mock.patch(
                "mole_jepa.data.cc3m._fetch_image", return_value=fake_image
            ) as mock_fetch,
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_image_transform",
                return_value=image_transform,
            ),
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_tokenizer", return_value=tokenize
            ),
        ):
            dataset = cc3m.CC3MDataset(hf_dataset, _make_config())
            items = list(dataset)

        mock_fetch.assert_called_once_with("http://example.com/img.jpg")
        assert len(items) == 1

    def test_url_fetch_failure_skips_example(self) -> None:
        """A URLError from _fetch_image causes the example to be skipped."""
        pil_example = {"jpg": PIL.Image.new("RGB", (_H, _W)), "txt": "good"}
        url_example = {"jpg": "http://example.com/dead.jpg", "txt": "bad"}
        hf_dataset = unittest.mock.MagicMock()
        hf_dataset.__iter__ = unittest.mock.Mock(
            return_value=iter([url_example, pil_example])
        )

        image_transform = unittest.mock.MagicMock(return_value=torch.randn(_C, _H, _W))
        tokenize = unittest.mock.MagicMock(
            return_value=(
                torch.randint(0, 100, (_MAX_LENGTH,)),
                torch.ones(_MAX_LENGTH, dtype=torch.long),
            )
        )
        with (
            unittest.mock.patch(
                "mole_jepa.data.cc3m._fetch_image",
                side_effect=urllib.error.URLError("dead link"),
            ),
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_image_transform",
                return_value=image_transform,
            ),
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_tokenizer", return_value=tokenize
            ),
        ):
            dataset = cc3m.CC3MDataset(hf_dataset, _make_config())
            items = list(dataset)

        # URL example was skipped; only the PIL example is yielded.
        assert len(items) == 1


class TestFetchImage:
    def test_returns_rgb_pil_image(self) -> None:
        """_fetch_image reads bytes from a URL and returns an RGB PIL Image."""
        buf = io.BytesIO()
        PIL.Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buf, "JPEG")
        fake_jpeg = buf.getvalue()

        mock_resp = unittest.mock.MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = unittest.mock.Mock(return_value=False)
        mock_resp.read.return_value = fake_jpeg

        with unittest.mock.patch("urllib.request.urlopen", return_value=mock_resp):
            img = cc3m._fetch_image("http://example.com/img.jpg")

        assert isinstance(img, PIL.Image.Image)
        assert img.mode == "RGB"
