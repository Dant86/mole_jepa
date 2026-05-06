"""Unit tests for MoLeJEPA."""

import unittest.mock

import pytest
import torch

from mole_jepa.models import encoders
from mole_jepa.models import mole_jepa as mole_jepa_module
from mole_jepa.models import predictor as predictor_module

_HIDDEN_SIZE = 32
_EMBED_DIM = 16
_B = 4
_T = 8


@pytest.fixture
def mock_pretrained() -> unittest.mock.MagicMock:
    mock_model = unittest.mock.MagicMock()
    mock_model.config.hidden_size = _HIDDEN_SIZE
    with unittest.mock.patch(
        "transformers.AutoModel.from_pretrained", return_value=mock_model
    ):
        yield mock_model


@pytest.fixture
def model(mock_pretrained: unittest.mock.MagicMock) -> mole_jepa_module.MoLeJEPA:
    return mole_jepa_module.MoLeJEPA(
        image_encoder=encoders.ImageEncoder(embed_dim=_EMBED_DIM),
        text_encoder=encoders.TextEncoder(embed_dim=_EMBED_DIM),
        predictor=predictor_module.Predictor(
            embed_dim=_EMBED_DIM, hidden_dim=64
        ),
    )


def _set_hidden(mock_pretrained: unittest.mock.MagicMock) -> None:
    mock_pretrained.return_value.last_hidden_state = torch.randn(
        _B, _T, _HIDDEN_SIZE
    )


class TestMoLeJEPA:
    def test_output_type(
        self,
        model: mole_jepa_module.MoLeJEPA,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        _set_hidden(mock_pretrained)
        output = model(
            torch.randn(_B, 3, 224, 224),
            torch.randint(0, 100, (_B, _T)),
            torch.ones(_B, _T, dtype=torch.long),
        )
        assert isinstance(output, mole_jepa_module.MoLeJEPAOutput)

    def test_output_shapes(
        self,
        model: mole_jepa_module.MoLeJEPA,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        _set_hidden(mock_pretrained)
        output = model(
            torch.randn(_B, 3, 224, 224),
            torch.randint(0, 100, (_B, _T)),
            torch.ones(_B, _T, dtype=torch.long),
        )
        assert output.z_v.shape == (_B, _EMBED_DIM)
        assert output.z_hat_t.shape == (_B, _EMBED_DIM)
        assert output.z_t.shape == (_B, _EMBED_DIM)

    def test_gradient_flows_to_all_components(
        self,
        model: mole_jepa_module.MoLeJEPA,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        _set_hidden(mock_pretrained)
        output = model(
            torch.randn(_B, 3, 224, 224),
            torch.randint(0, 100, (_B, _T)),
            torch.ones(_B, _T, dtype=torch.long),
        )
        (output.z_v + output.z_hat_t + output.z_t).sum().backward()
        assert model.image_encoder.projection.weight.grad is not None
        assert model.text_encoder.projection.weight.grad is not None
        assert model.predictor.net[0].weight.grad is not None
