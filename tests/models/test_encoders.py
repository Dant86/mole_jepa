"""Unit tests for ImageEncoder, TextEncoder, and _mean_pool."""

import unittest.mock

import pytest
import torch

from mole_jepa.models import encoders

_HIDDEN_SIZE = 32
_EMBED_DIM = 16
_B = 4
_T = 12


@pytest.fixture
def mock_pretrained() -> unittest.mock.MagicMock:
    mock_model = unittest.mock.MagicMock()
    mock_model.config.hidden_size = _HIDDEN_SIZE
    with unittest.mock.patch(
        "transformers.AutoModel.from_pretrained", return_value=mock_model
    ):
        yield mock_model


class TestMeanPool:
    def test_uniform_mask_returns_mean(self) -> None:
        hidden = torch.ones(_B, _T, _HIDDEN_SIZE)
        mask = torch.ones(_B, _T)
        result = encoders._mean_pool(hidden, mask)
        assert result.shape == (_B, _HIDDEN_SIZE)
        assert torch.allclose(result, torch.ones(_B, _HIDDEN_SIZE))

    def test_partial_mask_ignores_padding(self) -> None:
        # hidden[0] = [[0, 1], [2, 3], [4, 5]]; only first token unmasked.
        hidden = torch.arange(6, dtype=torch.float).reshape(1, 3, 2)
        mask = torch.tensor([[1.0, 0.0, 0.0]])
        result = encoders._mean_pool(hidden, mask)
        assert torch.allclose(result, torch.tensor([[0.0, 1.0]]))

    def test_all_masked_no_nan(self) -> None:
        # Denominator is clamped to 1e-9; result should be finite.
        hidden = torch.randn(_B, _T, _HIDDEN_SIZE)
        mask = torch.zeros(_B, _T)
        result = encoders._mean_pool(hidden, mask)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestImageEncoder:
    def test_output_shape(self, mock_pretrained: unittest.mock.MagicMock) -> None:
        mock_pretrained.return_value.last_hidden_state = torch.randn(
            _B, 197, _HIDDEN_SIZE
        )
        encoder = encoders.ImageEncoder(embed_dim=_EMBED_DIM)
        z = encoder(torch.randn(_B, 3, 224, 224))
        assert z.shape == (_B, _EMBED_DIM)

    def test_uses_cls_token_not_mean(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        # Set CLS (position 0) to a known value; all other positions to zero.
        # The projection of a zero vector is zero, so any non-zero output
        # must come from the CLS token.
        last_hidden = torch.zeros(_B, 10, _HIDDEN_SIZE)
        last_hidden[:, 0, :] = 1.0
        mock_pretrained.return_value.last_hidden_state = last_hidden

        encoder = encoders.ImageEncoder(embed_dim=_EMBED_DIM)
        torch.nn.init.constant_(encoder.projection.weight, 1.0)
        torch.nn.init.zeros_(encoder.projection.bias)

        z = encoder(torch.randn(_B, 3, 224, 224))
        # Output should equal projection(ones) = _HIDDEN_SIZE for every sample.
        expected = torch.full((_B, _EMBED_DIM), float(_HIDDEN_SIZE))
        assert torch.allclose(z, expected)

    def test_gradient_flows(self, mock_pretrained: unittest.mock.MagicMock) -> None:
        mock_pretrained.return_value.last_hidden_state = torch.randn(
            _B, 197, _HIDDEN_SIZE
        )
        encoder = encoders.ImageEncoder(embed_dim=_EMBED_DIM)
        encoder(torch.randn(_B, 3, 224, 224)).sum().backward()
        assert encoder.projection.weight.grad is not None
        assert torch.isfinite(encoder.projection.weight.grad).all()


class TestTextEncoder:
    def test_output_shape(self, mock_pretrained: unittest.mock.MagicMock) -> None:
        mock_pretrained.return_value.last_hidden_state = torch.randn(
            _B, _T, _HIDDEN_SIZE
        )
        encoder = encoders.TextEncoder(embed_dim=_EMBED_DIM)
        z = encoder(
            torch.randint(0, 100, (_B, _T)),
            torch.ones(_B, _T, dtype=torch.long),
        )
        assert z.shape == (_B, _EMBED_DIM)

    def test_padding_tokens_excluded(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        # Set padded positions to a large value; if they were included in the
        # mean the output would differ from the masked-only result.
        last_hidden = torch.zeros(_B, _T, _HIDDEN_SIZE)
        last_hidden[:, 0, :] = 1.0  # only first token is "real"
        last_hidden[:, 1:, :] = 1e6  # padding — should be zeroed out
        mock_pretrained.return_value.last_hidden_state = last_hidden

        encoder = encoders.TextEncoder(embed_dim=_EMBED_DIM)
        mask = torch.zeros(_B, _T, dtype=torch.long)
        mask[:, 0] = 1  # only first token unmasked

        z_masked = encoder(torch.randint(0, 100, (_B, _T)), mask)
        # With all tokens unmasked the 1e6 padding would dominate.
        mask_all = torch.ones(_B, _T, dtype=torch.long)
        z_all = encoder(torch.randint(0, 100, (_B, _T)), mask_all)
        assert not torch.allclose(z_masked, z_all)

    def test_gradient_flows(self, mock_pretrained: unittest.mock.MagicMock) -> None:
        mock_pretrained.return_value.last_hidden_state = torch.randn(
            _B, _T, _HIDDEN_SIZE
        )
        encoder = encoders.TextEncoder(embed_dim=_EMBED_DIM)
        encoder(
            torch.randint(0, 100, (_B, _T)),
            torch.ones(_B, _T, dtype=torch.long),
        ).sum().backward()
        assert encoder.projection.weight.grad is not None
        assert torch.isfinite(encoder.projection.weight.grad).all()
