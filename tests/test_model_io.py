"""Tests for mole_jepa.model_io."""

import collections.abc
import pathlib
import unittest.mock

import pytest
import torch

from mole_jepa import config as config_module
from mole_jepa import model_io as mio
from mole_jepa import models, nfs_registry

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


@pytest.fixture
def registry(
    tmp_path: pathlib.Path,
    small_config: config_module.ModelConfig,
) -> tuple[pathlib.Path, str]:
    """Register a model under 'test_model' and return (registry_dir, name)."""
    reg_dir = tmp_path / "registry"
    ckpt_base = tmp_path / "checkpoints"
    name = "test_model"
    nfs_registry.register(
        name,
        config=small_config,
        checkpoint_dir=ckpt_base,
        registry_dir=reg_dir,
    )
    return reg_dir, name


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
        registry: tuple[pathlib.Path, str],
    ) -> None:
        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        ckpt = nfs_registry.get_checkpoint_dir(name, reg_dir)
        assert (ckpt / "model.pt").exists()
        assert (ckpt / "config.json").exists()

    def test_save_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        ckpt = nfs_registry.get_checkpoint_dir(name, reg_dir)
        assert not (ckpt / "model.pt.tmp").exists()

    def test_save_does_not_overwrite_existing_config(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        """Second save_model call must not clobber an existing config.json."""
        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        # Spy on write_text — it must not be called on the second save.
        with unittest.mock.patch("pathlib.Path.write_text") as mock_write:
            mio.save_model(model, name, registry_dir=reg_dir)
        mock_write.assert_not_called()

    def test_model_pt_contains_only_state_dict(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        ckpt = nfs_registry.get_checkpoint_dir(name, reg_dir)
        contents = torch.load(ckpt / "model.pt", weights_only=True)
        assert isinstance(contents, dict)
        assert all(isinstance(v, torch.Tensor) for v in contents.values())

    def test_load_model_round_trip(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        loaded = mio.load_model(name, registry_dir=reg_dir)
        for p_orig, p_loaded in zip(model.parameters(), loaded.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_load_model_weights_into_existing(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        from mole_jepa import factory

        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        cfg = nfs_registry.get_config(name, reg_dir)
        target, _ = factory.build(cfg)
        mio.load_model_weights(target, name, registry_dir=reg_dir)
        for p_orig, p_loaded in zip(model.parameters(), target.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_load_raises_if_missing(
        self,
        registry: tuple[pathlib.Path, str],
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        reg_dir, name = registry
        with pytest.raises(FileNotFoundError):
            mio.load_model(name, registry_dir=reg_dir)

    def test_load_model_weights_raises_if_missing(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        model, _ = model_and_optimizer
        reg_dir, name = registry
        with pytest.raises(FileNotFoundError):
            mio.load_model_weights(model, name, registry_dir=reg_dir)


# ── TestTrainState ────────────────────────────────────────────────────────────


class TestTrainState:
    def test_has_train_state_false_initially(
        self,
        registry: tuple[pathlib.Path, str],
    ) -> None:
        reg_dir, name = registry
        assert not mio.has_train_state(name, registry_dir=reg_dir)

    def test_has_train_state_true_after_save(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        _, optimizer = model_and_optimizer
        reg_dir, name = registry
        mio.save_train_state(optimizer, epoch=3, name=name, registry_dir=reg_dir)
        assert mio.has_train_state(name, registry_dir=reg_dir)

    def test_round_trip_preserves_epoch(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        _, optimizer = model_and_optimizer
        reg_dir, name = registry
        mio.save_train_state(optimizer, epoch=7, name=name, registry_dir=reg_dir)
        result = mio.load_train_state(name, registry_dir=reg_dir)
        assert result is not None
        _, epoch, _ = result
        assert epoch == 7

    def test_round_trip_preserves_optimizer_state(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        _, optimizer = model_and_optimizer
        reg_dir, name = registry
        original_state = optimizer.state_dict()
        mio.save_train_state(optimizer, epoch=0, name=name, registry_dir=reg_dir)
        result = mio.load_train_state(name, registry_dir=reg_dir)
        assert result is not None
        opt_state, _, _ = result
        assert opt_state["param_groups"] == original_state["param_groups"]

    def test_round_trip_preserves_scheduler_state(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        _, optimizer = model_and_optimizer
        reg_dir, name = registry
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: 1.0
        )
        # Advance the scheduler a few steps so its state is non-trivial.
        for _ in range(5):
            scheduler.step()
        original_sched_state = scheduler.state_dict()
        mio.save_train_state(
            optimizer, epoch=0, name=name, scheduler=scheduler, registry_dir=reg_dir
        )
        result = mio.load_train_state(name, registry_dir=reg_dir)
        assert result is not None
        _, _, sched_state = result
        assert sched_state is not None
        assert sched_state["last_epoch"] == original_sched_state["last_epoch"]

    def test_round_trip_scheduler_none_when_not_saved(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        """When saved without a scheduler, the loaded scheduler state is None."""
        _, optimizer = model_and_optimizer
        reg_dir, name = registry
        mio.save_train_state(optimizer, epoch=0, name=name, registry_dir=reg_dir)
        result = mio.load_train_state(name, registry_dir=reg_dir)
        assert result is not None
        _, _, sched_state = result
        assert sched_state is None

    def test_load_returns_none_when_missing(
        self,
        registry: tuple[pathlib.Path, str],
    ) -> None:
        reg_dir, name = registry
        assert mio.load_train_state(name, registry_dir=reg_dir) is None

    def test_save_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        _, optimizer = model_and_optimizer
        reg_dir, name = registry
        mio.save_train_state(optimizer, epoch=0, name=name, registry_dir=reg_dir)
        ckpt = nfs_registry.get_checkpoint_dir(name, reg_dir)
        assert not (ckpt / "train_state.pt.tmp").exists()

    def test_cleanup_removes_train_state(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        _, optimizer = model_and_optimizer
        reg_dir, name = registry
        mio.save_train_state(optimizer, epoch=0, name=name, registry_dir=reg_dir)
        assert mio.has_train_state(name, registry_dir=reg_dir)
        mio.cleanup_train_state(name, registry_dir=reg_dir)
        assert not mio.has_train_state(name, registry_dir=reg_dir)

    def test_cleanup_is_safe_when_missing(
        self,
        registry: tuple[pathlib.Path, str],
    ) -> None:
        reg_dir, name = registry
        mio.cleanup_train_state(name, registry_dir=reg_dir)  # should not raise


# ── TestListModels ────────────────────────────────────────────────────────────


class TestListModels:
    def test_empty_root_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert mio.list_models(tmp_path) == []

    def test_nonexistent_root_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert mio.list_models(tmp_path / "nonexistent") == []

    def test_finds_saved_model(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
        small_config: config_module.ModelConfig,
    ) -> None:
        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        ckpt_root = nfs_registry.get_checkpoint_dir(name, reg_dir).parent
        results = mio.list_models(ckpt_root)
        assert len(results) == 1
        assert results[0].config == small_config
        assert results[0].config_hash == small_config.serialize()

    def test_skips_plain_files_in_root(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registry: tuple[pathlib.Path, str],
    ) -> None:
        """Non-directory entries in the root (e.g. stray files) are ignored."""
        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        ckpt_root = nfs_registry.get_checkpoint_dir(name, reg_dir).parent
        (ckpt_root / "stray.txt").write_text("garbage")
        results = mio.list_models(ckpt_root)
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
        registry: tuple[pathlib.Path, str],
    ) -> None:
        model, _ = model_and_optimizer
        reg_dir, name = registry
        mio.save_model(model, name, registry_dir=reg_dir)
        ckpt = nfs_registry.get_checkpoint_dir(name, reg_dir)
        (ckpt / "config.json").unlink()
        ckpt_root = ckpt.parent
        assert mio.list_models(ckpt_root) == []

    def test_filter_by_field(
        self,
        mock_pretrained: unittest.mock.MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        from mole_jepa import factory

        ckpt_root = tmp_path / "ckpts"
        reg_dir = tmp_path / "registry"
        for embed_dim in [16, 32, 64]:
            cfg = config_module.ModelConfig(embed_dim=embed_dim, predictor_n_layers=2)
            model, _ = factory.build(cfg)
            name = f"model_{embed_dim}"
            nfs_registry.register(
                name, config=cfg, checkpoint_dir=ckpt_root, registry_dir=reg_dir
            )
            mio.save_model(model, name, registry_dir=reg_dir)

        results = mio.list_models(ckpt_root, embed_dim=32)
        assert len(results) == 1
        assert results[0].config.embed_dim == 32

    def test_filter_multiple_fields(
        self,
        mock_pretrained: unittest.mock.MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        from mole_jepa import factory

        ckpt_root = tmp_path / "ckpts"
        reg_dir = tmp_path / "registry"
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
        for i, cfg in enumerate(cfgs):
            model, _ = factory.build(cfg)
            name = f"model_{i}"
            nfs_registry.register(
                name, config=cfg, checkpoint_dir=ckpt_root, registry_dir=reg_dir
            )
            mio.save_model(model, name, registry_dir=reg_dir)

        results = mio.list_models(ckpt_root, embed_dim=16, contrastive=True)
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

        ckpt_root = tmp_path / "ckpts"
        reg_dir = tmp_path / "registry"
        for embed_dim in [16, 32, 64]:
            cfg = config_module.ModelConfig(embed_dim=embed_dim, predictor_n_layers=2)
            model, _ = factory.build(cfg)
            name = f"model_{embed_dim}"
            nfs_registry.register(
                name, config=cfg, checkpoint_dir=ckpt_root, registry_dir=reg_dir
            )
            mio.save_model(model, name, registry_dir=reg_dir)
            time.sleep(0.01)  # ensure distinct mtimes

        results = mio.list_models(ckpt_root)
        assert results[0].config.embed_dim == 64  # last saved → most recent
        assert results[-1].config.embed_dim == 16  # first saved → oldest
