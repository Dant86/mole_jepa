"""Tests for mole_jepa.checkpoint."""

import collections.abc
import pathlib
import unittest.mock

import pytest
import torch

from mole_jepa import checkpoint as ckpt_module
from mole_jepa import config as config_module
from mole_jepa import models

_HIDDEN = 32
_EMBED = 16


@pytest.fixture
def mock_pretrained() -> collections.abc.Iterator[unittest.mock.MagicMock]:
    mock_model = unittest.mock.MagicMock()
    mock_model.config.hidden_size = _HIDDEN
    with unittest.mock.patch(
        "transformers.AutoModel.from_pretrained", return_value=mock_model
    ):
        yield mock_model


@pytest.fixture
def small_config() -> config_module.ModelConfig:
    return config_module.ModelConfig(embed_dim=_EMBED, predictor_n_layers=2)


@pytest.fixture
def model_and_optimizer(
    mock_pretrained: unittest.mock.MagicMock,
    small_config: config_module.ModelConfig,
) -> tuple[models.MoLeJEPA, torch.optim.Optimizer]:
    from mole_jepa import factory

    model, _ = factory.build(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


class TestLocation:
    def test_returns_path_under_checkpoint_dir(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        loc = ckpt_module.location(tmp_path, small_config)
        assert loc.parent == tmp_path
        assert loc.name == small_config.serialize()

    def test_different_configs_give_different_locations(
        self, tmp_path: pathlib.Path
    ) -> None:
        loc_a = ckpt_module.location(tmp_path, config_module.ModelConfig(embed_dim=64))
        loc_b = ckpt_module.location(tmp_path, config_module.ModelConfig(embed_dim=128))
        assert loc_a != loc_b


class TestSaveLoad:
    def test_round_trip_preserves_weights(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        model, optimizer = model_and_optimizer
        ckpt_module.save(
            model, optimizer, epoch=3, config=small_config, checkpoint_dir=tmp_path
        )
        loaded_model, epoch = ckpt_module.load(small_config, tmp_path)

        assert epoch == 3
        for p_orig, p_loaded in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_save_writes_config_json(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, optimizer = model_and_optimizer
        ckpt_module.save(
            model, optimizer, epoch=0, config=small_config, checkpoint_dir=tmp_path
        )
        loc = ckpt_module.location(tmp_path, small_config)
        assert (loc / "config.json").exists()

    def test_save_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        """Tmp file is cleaned up; only the final checkpoint.pt remains."""
        model, optimizer = model_and_optimizer
        ckpt_module.save(
            model, optimizer, epoch=0, config=small_config, checkpoint_dir=tmp_path
        )
        loc = ckpt_module.location(tmp_path, small_config)
        assert (loc / "checkpoint.pt").exists()
        assert not (loc / "checkpoint.pt.tmp").exists()

    def test_load_raises_if_no_checkpoint(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            ckpt_module.load(small_config, tmp_path)


class TestListCheckpoints:
    def test_empty_dir_returns_empty_list(self, tmp_path: pathlib.Path) -> None:
        assert ckpt_module.list_checkpoints(tmp_path) == []

    def test_nonexistent_dir_returns_empty_list(self, tmp_path: pathlib.Path) -> None:
        assert ckpt_module.list_checkpoints(tmp_path / "nonexistent") == []

    def test_finds_saved_checkpoint(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, optimizer = model_and_optimizer
        ckpt_module.save(
            model, optimizer, epoch=7, config=small_config, checkpoint_dir=tmp_path
        )
        results = ckpt_module.list_checkpoints(tmp_path)
        assert len(results) == 1
        assert results[0].epoch == 7
        assert results[0].config == small_config
        assert results[0].config_hash == small_config.serialize()

    def test_sorted_by_epoch(
        self,
        mock_pretrained: unittest.mock.MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        from mole_jepa import factory

        for embed_dim, epoch in [(16, 5), (32, 2), (64, 9)]:
            cfg = config_module.ModelConfig(embed_dim=embed_dim, predictor_n_layers=2)
            model, _ = factory.build(cfg)
            opt = torch.optim.AdamW(model.parameters())
            ckpt_module.save(
                model, opt, epoch=epoch, config=cfg, checkpoint_dir=tmp_path
            )

        results = ckpt_module.list_checkpoints(tmp_path)
        assert [r.epoch for r in results] == [2, 5, 9]
