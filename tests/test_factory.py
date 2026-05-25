"""Unit tests for factory.build — wires ModelConfig into model + loss."""

import collections.abc
import unittest.mock

import pytest

from mole_jepa import config as config_module
from mole_jepa import factory as factory_module
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
        model, loss = factory_module.build(config_module.ModelConfig())
        assert isinstance(model, models.MoLeJEPA)
        assert isinstance(loss, losses.JEPALoss)

    def test_contrastive_returns_info_nce_loss(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        model, loss = factory_module.build(config_module.ModelConfig(contrastive=True))
        assert isinstance(model, models.MoLeJEPA)
        assert isinstance(loss, losses.InfoNCELoss)

    def test_embed_dim_propagates_to_all_components(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        embed_dim = 64
        model, _ = factory_module.build(config_module.ModelConfig(embed_dim=embed_dim))
        assert model.image_encoder.projection.out_features == embed_dim
        assert model.text_encoder.projection.out_features == embed_dim
        assert model.predictor.net[0].in_features == embed_dim
        assert model.predictor.net[-1].out_features == embed_dim

    def test_predictor_hidden_dim_propagates(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        hidden_dim = 128
        model, _ = factory_module.build(
            config_module.ModelConfig(predictor_hidden_dim=hidden_dim)
        )
        # First linear projects embed_dim → hidden_dim.
        assert model.predictor.net[0].out_features == hidden_dim

    def test_info_nce_temperature_propagates(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        temperature = 0.5
        cfg = config_module.ModelConfig(
            contrastive=True, info_nce_temperature=temperature
        )
        _, loss = factory_module.build(cfg)
        assert isinstance(loss, losses.InfoNCELoss)
        assert loss.temperature == pytest.approx(temperature)

    def test_jepa_lam_propagates(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        lam_image, lam_text = 0.1, 0.4
        _, loss = factory_module.build(
            config_module.ModelConfig(jepa_lam_image=lam_image, jepa_lam_text=lam_text)
        )
        assert isinstance(loss, losses.JEPALoss)
        assert loss.lam_image == pytest.approx(lam_image)
        assert loss.lam_text == pytest.approx(lam_text)

    def test_freeze_image_encoder_freezes_vit(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        model, _ = factory_module.build(
            config_module.ModelConfig(freeze_image_encoder=True)
        )
        mock_pretrained.requires_grad_.assert_called_once_with(False)

    def test_freeze_image_encoder_false_does_not_freeze(
        self, mock_pretrained: unittest.mock.MagicMock
    ) -> None:
        factory_module.build(config_module.ModelConfig(freeze_image_encoder=False))
        mock_pretrained.requires_grad_.assert_not_called()
