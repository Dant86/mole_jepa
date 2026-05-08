"""Unit tests for ModelConfig and build."""

import collections.abc
import unittest.mock

import pytest

from mole_jepa import config as config_module
from mole_jepa import losses, models

_HIDDEN_SIZE = 32


@pytest.fixture
def mock_pretrained() -> collections.abc.Iterator[unittest.mock.MagicMock]:
    mock_model = unittest.mock.MagicMock()
    mock_model.config.hidden_size = _HIDDEN_SIZE
    with unittest.mock.patch(
        "transformers.AutoModel.from_pretrained", return_value=mock_model
    ):
        yield mock_model


class TestBuild:
    def test_default_returns_mole_jepa_and_jepa_loss(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        model, loss = config_module.build(config_module.ModelConfig())
        assert isinstance(model, models.MoLeJEPA)
        assert isinstance(loss, losses.JEPALoss)

    def test_contrastive_returns_info_nce_loss(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        model, loss = config_module.build(config_module.ModelConfig(contrastive=True))
        assert isinstance(model, models.MoLeJEPA)
        assert isinstance(loss, losses.InfoNCELoss)

    def test_embed_dim_propagates_to_all_components(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        embed_dim = 64
        model, _ = config_module.build(config_module.ModelConfig(embed_dim=embed_dim))
        assert model.image_encoder.projection.out_features == embed_dim
        assert model.text_encoder.projection.out_features == embed_dim
        assert model.predictor.net[0].in_features == embed_dim
        assert model.predictor.net[-1].out_features == embed_dim

    def test_predictor_hidden_dim_propagates(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        hidden_dim = 128
        model, _ = config_module.build(
            config_module.ModelConfig(predictor_hidden_dim=hidden_dim)
        )
        # First linear projects embed_dim → hidden_dim.
        assert model.predictor.net[0].out_features == hidden_dim

    def test_info_nce_temperature_propagates(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        temperature = 0.5
        _, loss = config_module.build(
            config_module.ModelConfig(
                contrastive=True, info_nce_temperature=temperature
            )
        )
        assert isinstance(loss, losses.InfoNCELoss)
        assert loss.temperature == pytest.approx(temperature)

    def test_jepa_reg_weight_propagates(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        reg_weight = 2.5
        _, loss = config_module.build(
            config_module.ModelConfig(jepa_reg_weight=reg_weight)
        )
        assert isinstance(loss, losses.JEPALoss)
        assert loss.reg_weight == pytest.approx(reg_weight)


class TestSerialize:
    def test_deterministic(self) -> None:
        cfg = config_module.ModelConfig()
        assert cfg.serialize() == cfg.serialize()

    def test_differs_for_different_configs(self) -> None:
        assert config_module.ModelConfig(contrastive=False).serialize() != (
            config_module.ModelConfig(contrastive=True).serialize()
        )

    def test_returns_64_char_hex(self) -> None:
        result = config_module.ModelConfig().serialize()
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
