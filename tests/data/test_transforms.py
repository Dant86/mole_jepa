"""Unit tests for data transforms."""

import unittest.mock

import numpy as np
import PIL.Image
import pytest
import torch

from mole_jepa.data import transforms

_MODEL_NAME = "stub-model"
_MAX_LENGTH = 32
_C, _H, _W = 3, 224, 224


@pytest.fixture
def mock_image_processor() -> unittest.mock.MagicMock:
    processor = unittest.mock.MagicMock()
    # The new torchvision-based transform reads these config attributes rather
    # than calling the processor as a function.
    processor.size = {"height": _H, "width": _W}
    processor.crop_size = {"height": _H, "width": _W}
    processor.image_mean = [0.5, 0.5, 0.5]
    processor.image_std = [0.5, 0.5, 0.5]
    return processor


@pytest.fixture
def dummy_image() -> PIL.Image.Image:
    """A small random RGB PIL image for transform testing."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (_H, _W, _C), dtype=np.uint8)
    return PIL.Image.fromarray(arr, mode="RGB")


@pytest.fixture
def mock_tokenizer() -> unittest.mock.MagicMock:
    tokenizer = unittest.mock.MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 100, (1, _MAX_LENGTH)),
        "attention_mask": torch.ones(1, _MAX_LENGTH, dtype=torch.long),
    }
    return tokenizer


def _patch_processor(processor: unittest.mock.MagicMock) -> unittest.mock.patch:  # type: ignore[type-arg]
    """Patch AutoImageProcessor.from_pretrained to return *processor*."""
    p = unittest.mock.patch("mole_jepa.data.transforms.transformers.AutoImageProcessor")
    mock_cls = p.start()
    mock_cls.from_pretrained.return_value = processor
    return p


class TestBuildImageTransform:
    def test_returns_callable(
        self, mock_image_processor: unittest.mock.MagicMock
    ) -> None:
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_image_processor
            transform = transforms.build_image_transform(_MODEL_NAME)
        assert callable(transform)

    def test_shortest_edge_config(self, dummy_image: PIL.Image.Image) -> None:
        """size={'shortest_edge': N} is resolved to an int resize target."""
        processor = unittest.mock.MagicMock()
        processor.size = {"shortest_edge": _H}
        processor.crop_size = {"height": _H, "width": _W}
        processor.image_mean = [0.5, 0.5, 0.5]
        processor.image_std = [0.5, 0.5, 0.5]
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = processor
            transform = transforms.build_image_transform(_MODEL_NAME)
        assert transform(dummy_image).shape == (_C, _H, _W)

    def test_no_crop_size_attr_with_tuple_resize(
        self, dummy_image: PIL.Image.Image
    ) -> None:
        """crop_size absent + height/width resize → crop_to inherits resize_to tuple."""
        processor = unittest.mock.MagicMock()
        processor.size = {"height": _H, "width": _W}
        processor.crop_size = None  # absent → getattr returns None
        processor.image_mean = [0.5, 0.5, 0.5]
        processor.image_std = [0.5, 0.5, 0.5]
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = processor
            transform = transforms.build_image_transform(_MODEL_NAME)
        assert transform(dummy_image).shape == (_C, _H, _W)

    def test_no_crop_size_attr_with_int_resize(
        self, dummy_image: PIL.Image.Image
    ) -> None:
        """crop_size absent + shortest_edge resize → crop_to is (N, N) tuple."""
        processor = unittest.mock.MagicMock()
        processor.size = {"shortest_edge": _H}
        processor.crop_size = None
        processor.image_mean = [0.5, 0.5, 0.5]
        processor.image_std = [0.5, 0.5, 0.5]
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = processor
            transform = transforms.build_image_transform(_MODEL_NAME)
        assert transform(dummy_image).shape == (_C, _H, _W)

    def test_output_shape(
        self,
        mock_image_processor: unittest.mock.MagicMock,
        dummy_image: PIL.Image.Image,
    ) -> None:
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_image_processor
            transform = transforms.build_image_transform(_MODEL_NAME)

        result = transform(dummy_image)
        assert result.shape == (_C, _H, _W)

    def test_output_is_tensor(
        self,
        mock_image_processor: unittest.mock.MagicMock,
        dummy_image: PIL.Image.Image,
    ) -> None:
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_image_processor
            transform = transforms.build_image_transform(_MODEL_NAME)

        result = transform(dummy_image)
        assert isinstance(result, torch.Tensor)

    def test_val_transform_is_deterministic(
        self,
        mock_image_processor: unittest.mock.MagicMock,
        dummy_image: PIL.Image.Image,
    ) -> None:
        """train=False uses CenterCrop — two calls on the same image must be equal."""
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_image_processor
            transform = transforms.build_image_transform(_MODEL_NAME, train=False)

        result1 = transform(dummy_image)
        result2 = transform(dummy_image)
        assert result1.shape == (_C, _H, _W)
        assert torch.equal(result1, result2)

    def test_train_transform_output_shape(
        self,
        mock_image_processor: unittest.mock.MagicMock,
        dummy_image: PIL.Image.Image,
    ) -> None:
        """train=True (default) still produces the expected spatial dimensions."""
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_image_processor
            transform = transforms.build_image_transform(_MODEL_NAME, train=True)

        result = transform(dummy_image)
        assert result.shape == (_C, _H, _W)


class TestPreprocess:
    def test_output_shapes(
        self,
        dummy_image: PIL.Image.Image,
    ) -> None:
        """preprocess() returns correctly shaped tensors with a leading batch dim."""
        fake_pixels = torch.randn(_C, _H, _W)
        fake_ids = torch.randint(0, 100, (_MAX_LENGTH,))
        fake_mask = torch.ones(_MAX_LENGTH, dtype=torch.long)

        with (
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_image_transform",
                return_value=lambda img: fake_pixels,
            ),
            unittest.mock.patch(
                "mole_jepa.data.transforms.build_tokenizer",
                return_value=lambda text: (fake_ids, fake_mask),
            ),
        ):
            from mole_jepa import config as config_module

            cfg = config_module.DataConfig(max_seq_length=_MAX_LENGTH)
            pixel_values, input_ids, attention_mask = transforms.preprocess(
                dummy_image, "a caption", cfg
            )

        assert pixel_values.shape == (1, _C, _H, _W)
        assert input_ids.shape == (1, _MAX_LENGTH)
        assert attention_mask.shape == (1, _MAX_LENGTH)


class TestBuildTokenizer:
    def test_returns_callable(self, mock_tokenizer: unittest.mock.MagicMock) -> None:
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoTokenizer"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tokenizer
            tokenize = transforms.build_tokenizer(_MODEL_NAME, _MAX_LENGTH)
        assert callable(tokenize)

    def test_output_shapes(self, mock_tokenizer: unittest.mock.MagicMock) -> None:
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoTokenizer"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tokenizer
            tokenize = transforms.build_tokenizer(_MODEL_NAME, _MAX_LENGTH)

        input_ids, attention_mask = tokenize("hello world")
        assert input_ids.shape == (_MAX_LENGTH,)
        assert attention_mask.shape == (_MAX_LENGTH,)

    def test_output_are_tensors(self, mock_tokenizer: unittest.mock.MagicMock) -> None:
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoTokenizer"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tokenizer
            tokenize = transforms.build_tokenizer(_MODEL_NAME, _MAX_LENGTH)

        input_ids, attention_mask = tokenize("hello world")
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
