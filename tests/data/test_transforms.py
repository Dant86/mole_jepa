"""Unit tests for data transforms."""

import unittest.mock

import pytest
import torch

from mole_jepa.data import transforms

_MODEL_NAME = "stub-model"
_MAX_LENGTH = 32
_C, _H, _W = 3, 224, 224


@pytest.fixture
def mock_image_processor() -> unittest.mock.MagicMock:
    processor = unittest.mock.MagicMock()
    processor.return_value = {
        "pixel_values": torch.randn(1, _C, _H, _W),
    }
    return processor


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

    def test_output_shape(self, mock_image_processor: unittest.mock.MagicMock) -> None:
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_image_processor
            transform = transforms.build_image_transform(_MODEL_NAME)

        result = transform(unittest.mock.MagicMock())
        assert result.shape == (_C, _H, _W)

    def test_output_is_tensor(
        self, mock_image_processor: unittest.mock.MagicMock
    ) -> None:
        with unittest.mock.patch(
            "mole_jepa.data.transforms.transformers.AutoImageProcessor"
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = mock_image_processor
            transform = transforms.build_image_transform(_MODEL_NAME)

        result = transform(unittest.mock.MagicMock())
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
