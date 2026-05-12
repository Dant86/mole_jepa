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
