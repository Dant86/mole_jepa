"""Tests for mole_jepa.model_io."""

import collections.abc
import pathlib
import unittest.mock

import pytest
import torch

from mole_jepa import config as config_module
from mole_jepa import model_io as mio
from mole_jepa import models

_HIDDEN = 32
_EMBED = 16


# ── fixtures ──────────────────────────────────────────────────────────────────


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


# ── TestModelDir ──────────────────────────────────────────────────────────────


class TestModelDir:
    def test_returns_path_under_root(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        d = mio.model_dir(tmp_path, small_config)
        assert d.parent == tmp_path
        assert d.name == small_config.serialize()

    def test_different_configs_give_different_paths(
        self, tmp_path: pathlib.Path
    ) -> None:
        d1 = mio.model_dir(tmp_path, config_module.ModelConfig(embed_dim=64))
        d2 = mio.model_dir(tmp_path, config_module.ModelConfig(embed_dim=128))
        assert d1 != d2


# ── TestSaveLoadModel ─────────────────────────────────────────────────────────


class TestSaveLoadModel:
    def test_save_writes_model_and_config(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, small_config, tmp_path)
        loc = mio.model_dir(tmp_path, small_config)
        assert (loc / "model.pt").exists()
        assert (loc / "config.json").exists()

    def test_save_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, small_config, tmp_path)
        loc = mio.model_dir(tmp_path, small_config)
        assert not (loc / "model.pt.tmp").exists()

    def test_save_does_not_overwrite_existing_config(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        """Second save_model call must not clobber an existing config.json."""
        model, _ = model_and_optimizer
        mio.save_model(model, small_config, tmp_path)
        # Spy on write_text — it must not be called on the second save.
        with unittest.mock.patch("pathlib.Path.write_text") as mock_write:
            mio.save_model(model, small_config, tmp_path)
        mock_write.assert_not_called()

    def test_model_pt_contains_only_state_dict(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, small_config, tmp_path)
        loc = mio.model_dir(tmp_path, small_config)
        contents = torch.load(loc / "model.pt", weights_only=True)
        # State dict is a flat mapping of str → Tensor, not a nested dict
        # with "model_state_dict", "epoch", etc.
        assert isinstance(contents, dict)
        assert all(isinstance(v, torch.Tensor) for v in contents.values())

    def test_load_model_round_trip(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, small_config, tmp_path)
        loaded = mio.load_model(small_config, tmp_path)
        for p_orig, p_loaded in zip(model.parameters(), loaded.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_load_model_weights_into_existing(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        from mole_jepa import factory

        model, _ = model_and_optimizer
        mio.save_model(model, small_config, tmp_path)
        target, _ = factory.build(small_config)
        mio.load_model_weights(target, small_config, tmp_path)
        for p_orig, p_loaded in zip(model.parameters(), target.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_load_raises_if_missing(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            mio.load_model(small_config, tmp_path)

    def test_load_model_weights_raises_if_missing(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        with pytest.raises(FileNotFoundError):
            mio.load_model_weights(model, small_config, tmp_path)


# ── TestTrainState ────────────────────────────────────────────────────────────


class TestTrainState:
    def test_has_train_state_false_initially(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        assert not mio.has_train_state(small_config, tmp_path)

    def test_has_train_state_true_after_save(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=3, config=small_config, root=tmp_path)
        assert mio.has_train_state(small_config, tmp_path)

    def test_round_trip_preserves_epoch(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=7, config=small_config, root=tmp_path)
        result = mio.load_train_state(small_config, tmp_path)
        assert result is not None
        _, epoch = result
        assert epoch == 7

    def test_round_trip_preserves_optimizer_state(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        original_state = optimizer.state_dict()
        mio.save_train_state(optimizer, epoch=0, config=small_config, root=tmp_path)
        result = mio.load_train_state(small_config, tmp_path)
        assert result is not None
        opt_state, _ = result
        assert opt_state["param_groups"] == original_state["param_groups"]

    def test_load_returns_none_when_missing(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        assert mio.load_train_state(small_config, tmp_path) is None

    def test_save_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=0, config=small_config, root=tmp_path)
        loc = mio.model_dir(tmp_path, small_config)
        assert not (loc / "train_state.pt.tmp").exists()

    def test_cleanup_removes_train_state(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=0, config=small_config, root=tmp_path)
        assert mio.has_train_state(small_config, tmp_path)
        mio.cleanup_train_state(small_config, tmp_path)
        assert not mio.has_train_state(small_config, tmp_path)

    def test_cleanup_is_safe_when_missing(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        mio.cleanup_train_state(small_config, tmp_path)  # should not raise


# ── TestListModels ────────────────────────────────────────────────────────────


class TestListModels:
    def test_empty_root_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert mio.list_models(tmp_path) == []

    def test_nonexistent_root_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert mio.list_models(tmp_path / "nonexistent") == []

    def test_finds_saved_model(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, small_config, tmp_path)
        results = mio.list_models(tmp_path)
        assert len(results) == 1
        assert results[0].config == small_config
        assert results[0].config_hash == small_config.serialize()

    def test_skips_plain_files_in_root(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        """Non-directory entries in the root (e.g. stray files) are ignored."""
        model, _ = model_and_optimizer
        (tmp_path / "stray.txt").write_text("garbage")
        mio.save_model(model, small_config, tmp_path)
        results = mio.list_models(tmp_path)
        assert len(results) == 1

    def test_skips_dir_missing_model_pt(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        # Only write config.json, no model.pt
        import dataclasses
        import json

        loc = mio.model_dir(tmp_path, small_config)
        loc.mkdir(parents=True)
        (loc / "config.json").write_text(json.dumps(dataclasses.asdict(small_config)))
        assert mio.list_models(tmp_path) == []

    def test_skips_dir_missing_config_json(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, small_config, tmp_path)
        (mio.model_dir(tmp_path, small_config) / "config.json").unlink()
        assert mio.list_models(tmp_path) == []

    def test_filter_by_field(
        self,
        mock_pretrained: unittest.mock.MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        from mole_jepa import factory

        for embed_dim in [16, 32, 64]:
            cfg = config_module.ModelConfig(embed_dim=embed_dim, predictor_n_layers=2)
            model, _ = factory.build(cfg)
            mio.save_model(model, cfg, tmp_path)

        results = mio.list_models(tmp_path, embed_dim=32)
        assert len(results) == 1
        assert results[0].config.embed_dim == 32

    def test_filter_multiple_fields(
        self,
        mock_pretrained: unittest.mock.MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        from mole_jepa import factory

        cfgs = [
            config_module.ModelConfig(
                embed_dim=16, contrastive=False, predictor_n_layers=2
            ),
            config_module.ModelConfig(
                embed_dim=16, contrastive=True, predictor_n_layers=2
            ),
            config_module.ModelConfig(
                embed_dim=32, contrastive=True, predictor_n_layers=2
            ),
        ]
        for cfg in cfgs:
            model, _ = factory.build(cfg)
            mio.save_model(model, cfg, tmp_path)

        results = mio.list_models(tmp_path, embed_dim=16, contrastive=True)
        assert len(results) == 1
        assert results[0].config.embed_dim == 16
        assert results[0].config.contrastive is True

    def test_unknown_filter_field_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="Unknown ModelConfig field"):
            mio.list_models(tmp_path, nonexistent_field=42)

    def test_sorted_most_recent_first(
        self,
        mock_pretrained: unittest.mock.MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        import time

        from mole_jepa import factory

        saved_configs = []
        for embed_dim in [16, 32, 64]:
            cfg = config_module.ModelConfig(embed_dim=embed_dim, predictor_n_layers=2)
            model, _ = factory.build(cfg)
            mio.save_model(model, cfg, tmp_path)
            saved_configs.append(cfg)
            time.sleep(0.01)  # ensure distinct mtimes

        results = mio.list_models(tmp_path)
        assert results[0].config.embed_dim == 64  # last saved → most recent
        assert results[-1].config.embed_dim == 16  # first saved → oldest


# ── TestExplicitPath (_at variants) ──────────────────────────────────────────


class TestExplicitPath:
    """Tests for the explicit checkpoint_dir variants of save/load functions."""

    def test_save_model_at_writes_model_pt(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model_at(model, tmp_path)
        assert (tmp_path / "model.pt").exists()

    def test_save_model_at_writes_config_json_when_provided(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model_at(model, tmp_path, config=small_config)
        assert (tmp_path / "config.json").exists()

    def test_save_model_at_no_config_json_without_config(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model_at(model, tmp_path)
        assert not (tmp_path / "config.json").exists()

    def test_save_model_at_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model_at(model, tmp_path)
        assert not (tmp_path / "model.pt.tmp").exists()

    def test_save_model_at_does_not_overwrite_existing_config_json(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model_at(model, tmp_path, config=small_config)
        with unittest.mock.patch("pathlib.Path.write_text") as mock_write:
            mio.save_model_at(model, tmp_path, config=small_config)
        mock_write.assert_not_called()

    def test_load_model_weights_at_round_trip(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model_at(model, tmp_path)
        # Build a fresh model with the same architecture via mock.
        target = models.MoLeJEPA(
            image_encoder=model.image_encoder,
            text_encoder=model.text_encoder,
            predictor=model.predictor,
        )
        mio.load_model_weights_at(target, tmp_path)
        for p_orig, p_loaded in zip(model.parameters(), target.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_load_model_weights_at_raises_if_missing(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        model, _ = model_and_optimizer
        with pytest.raises(FileNotFoundError):
            mio.load_model_weights_at(model, tmp_path)

    def test_save_train_state_at_writes_file(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state_at(optimizer, epoch=5, checkpoint_dir=tmp_path)
        assert (tmp_path / "train_state.pt").exists()

    def test_save_train_state_at_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state_at(optimizer, epoch=0, checkpoint_dir=tmp_path)
        assert not (tmp_path / "train_state.pt.tmp").exists()

    def test_load_train_state_at_round_trip(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state_at(optimizer, epoch=9, checkpoint_dir=tmp_path)
        result = mio.load_train_state_at(tmp_path)
        assert result is not None
        _, epoch = result
        assert epoch == 9

    def test_load_train_state_at_returns_none_when_absent(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        assert mio.load_train_state_at(tmp_path) is None

    def test_has_train_state_at_false_initially(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        assert not mio.has_train_state_at(tmp_path)

    def test_has_train_state_at_true_after_save(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state_at(optimizer, epoch=0, checkpoint_dir=tmp_path)
        assert mio.has_train_state_at(tmp_path)

    def test_cleanup_train_state_at_removes_file(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        tmp_path: pathlib.Path,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state_at(optimizer, epoch=0, checkpoint_dir=tmp_path)
        assert mio.has_train_state_at(tmp_path)
        mio.cleanup_train_state_at(tmp_path)
        assert not mio.has_train_state_at(tmp_path)

    def test_cleanup_train_state_at_safe_when_absent(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        mio.cleanup_train_state_at(tmp_path)  # must not raise
