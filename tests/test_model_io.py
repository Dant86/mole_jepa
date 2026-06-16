"""Tests for mole_jepa.model_io."""

import collections.abc
import pathlib
import unittest.mock

import pytest
import torch

from mole_jepa import config as config_module
from mole_jepa import model_io as mio
from mole_jepa import models
from mole_jepa import registry as reg_module

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
def registered_model(
    tmp_path: pathlib.Path,
    small_config: config_module.ModelConfig,
) -> str:
    """Register 'test_model' in the fake Supabase registry; return its name.

    The checkpoint root is set to a temp directory so model_io can actually
    write files during tests.
    """
    name = "test_model"
    reg_module.register(
        name,
        config=small_config,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    return name


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
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        assert (ckpt / "model.pt").exists()
        assert (ckpt / "config.json").exists()

    def test_save_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        assert not (ckpt / "model.pt.tmp").exists()

    def test_save_does_not_overwrite_existing_config(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        """Second save_model call must not clobber an existing config.json."""
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        with unittest.mock.patch("pathlib.Path.write_text") as mock_write:
            mio.save_model(model, registered_model)
        mock_write.assert_not_called()

    def test_model_pt_contains_only_state_dict(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        contents = torch.load(ckpt / "model.pt", weights_only=True)
        assert isinstance(contents, dict)
        assert all(isinstance(v, torch.Tensor) for v in contents.values())

    def test_epoch_rotation_backs_up_previous_model(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        """save_model(epoch=N) copies the *previous* model.pt to model_NNNN.pt."""
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)  # no epoch -> no backup yet
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        assert not (ckpt / "model_0001.pt").exists()

        mio.save_model(model, registered_model, epoch=1)
        assert (ckpt / "model_0001.pt").exists()
        assert (ckpt / "model.pt").exists()

    def test_first_epoch_save_has_no_prior_model_to_back_up(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        """epoch=0 on a fresh checkpoint dir: no model.pt exists yet to copy."""
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model, epoch=0)
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        assert (ckpt / "model.pt").exists()
        assert not (ckpt / "model_0000.pt").exists()

    def test_keep_last_prunes_oldest_epoch_checkpoints(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        mio.save_model(model, registered_model)  # seed model.pt
        for epoch in range(1, 5):
            mio.save_model(model, registered_model, epoch=epoch, keep_last=2)

        epoch_files = sorted(ckpt.glob("model_[0-9]*.pt"))
        assert [f.name for f in epoch_files] == ["model_0003.pt", "model_0004.pt"]

    def test_keep_last_one_retains_only_most_recent(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        mio.save_model(model, registered_model)
        for epoch in range(1, 4):
            mio.save_model(model, registered_model, epoch=epoch, keep_last=1)

        epoch_files = sorted(ckpt.glob("model_[0-9]*.pt"))
        assert [f.name for f in epoch_files] == ["model_0003.pt"]

    def test_no_epoch_kwarg_never_creates_epoch_backups(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        for _ in range(3):
            mio.save_model(model, registered_model)  # epoch=None every time
        assert list(ckpt.glob("model_[0-9]*.pt")) == []

    def test_load_model_round_trip(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        loaded = mio.load_model(registered_model)
        for p_orig, p_loaded in zip(model.parameters(), loaded.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_load_model_weights_into_existing(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        from mole_jepa import factory

        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        cfg = reg_module.get_config(registered_model)
        target, _ = factory.build(cfg)
        mio.load_model_weights(target, registered_model)
        for p_orig, p_loaded in zip(model.parameters(), target.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_load_raises_if_missing(
        self,
        registered_model: str,
        mock_pretrained: unittest.mock.MagicMock,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            mio.load_model(registered_model)

    def test_load_model_weights_raises_if_missing(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        with pytest.raises(FileNotFoundError):
            mio.load_model_weights(model, registered_model)


# ── TestTrainState ────────────────────────────────────────────────────────────


class TestTrainState:
    def test_has_train_state_false_initially(self, registered_model: str) -> None:
        assert not mio.has_train_state(registered_model)

    def test_has_train_state_true_after_save(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=3, name=registered_model)
        assert mio.has_train_state(registered_model)

    def test_round_trip_preserves_epoch(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=7, name=registered_model)
        result = mio.load_train_state(registered_model)
        assert result is not None
        _, epoch, _ = result
        assert epoch == 7

    def test_round_trip_preserves_optimizer_state(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        _, optimizer = model_and_optimizer
        original_state = optimizer.state_dict()
        mio.save_train_state(optimizer, epoch=0, name=registered_model)
        result = mio.load_train_state(registered_model)
        assert result is not None
        opt_state, _, _ = result
        assert opt_state["param_groups"] == original_state["param_groups"]

    def test_round_trip_preserves_scheduler_state(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        _, optimizer = model_and_optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: 1.0
        )
        for _ in range(5):
            scheduler.step()
        original_sched_state = scheduler.state_dict()
        mio.save_train_state(
            optimizer, epoch=0, name=registered_model, scheduler=scheduler
        )
        result = mio.load_train_state(registered_model)
        assert result is not None
        _, _, sched_state = result
        assert sched_state is not None
        assert sched_state["last_epoch"] == original_sched_state["last_epoch"]

    def test_round_trip_scheduler_none_when_not_saved(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        """When saved without a scheduler, the loaded scheduler state is None."""
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=0, name=registered_model)
        result = mio.load_train_state(registered_model)
        assert result is not None
        _, _, sched_state = result
        assert sched_state is None

    def test_load_returns_none_when_missing(self, registered_model: str) -> None:
        assert mio.load_train_state(registered_model) is None

    def test_save_is_atomic(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=0, name=registered_model)
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        assert not (ckpt / "train_state.pt.tmp").exists()

    def test_cleanup_removes_train_state(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        _, optimizer = model_and_optimizer
        mio.save_train_state(optimizer, epoch=0, name=registered_model)
        assert mio.has_train_state(registered_model)
        mio.cleanup_train_state(registered_model)
        assert not mio.has_train_state(registered_model)

    def test_cleanup_is_safe_when_missing(self, registered_model: str) -> None:
        mio.cleanup_train_state(registered_model)  # should not raise


# ── TestListModels ────────────────────────────────────────────────────────────


class TestListModels:
    def test_empty_root_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert mio.list_models(tmp_path) == []

    def test_nonexistent_root_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert mio.list_models(tmp_path / "nonexistent") == []

    def test_finds_saved_model(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
        small_config: config_module.ModelConfig,
    ) -> None:
        from mole_jepa.registry import _entry_hash

        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        ckpt_root = reg_module.get_checkpoint_dir(registered_model).parent
        results = mio.list_models(ckpt_root)
        assert len(results) == 1
        assert results[0].config == small_config
        assert results[0].config_hash == _entry_hash(registered_model, small_config)

    def test_skips_plain_files_in_root(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        ckpt_root = reg_module.get_checkpoint_dir(registered_model).parent
        (ckpt_root / "stray.txt").write_text("garbage")
        assert len(mio.list_models(ckpt_root)) == 1

    def test_skips_dir_missing_model_pt(
        self,
        small_config: config_module.ModelConfig,
        tmp_path: pathlib.Path,
    ) -> None:
        import dataclasses
        import json

        loc = mio.model_dir(tmp_path, small_config)
        loc.mkdir(parents=True)
        (loc / "config.json").write_text(json.dumps(dataclasses.asdict(small_config)))
        assert mio.list_models(tmp_path) == []

    def test_skips_dir_missing_config_json(
        self,
        model_and_optimizer: tuple[models.MoLeJEPA, torch.optim.Optimizer],
        registered_model: str,
    ) -> None:
        model, _ = model_and_optimizer
        mio.save_model(model, registered_model)
        ckpt = reg_module.get_checkpoint_dir(registered_model)
        (ckpt / "config.json").unlink()
        assert mio.list_models(ckpt.parent) == []

    def test_filter_by_field(
        self,
        mock_pretrained: unittest.mock.MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        from mole_jepa import factory

        ckpt_root = tmp_path / "ckpts"
        for embed_dim in [16, 32, 64]:
            cfg = config_module.ModelConfig(embed_dim=embed_dim, predictor_n_layers=2)
            model, _ = factory.build(cfg)
            name = f"model_{embed_dim}"
            reg_module.register(name, config=cfg, checkpoint_dir=ckpt_root)
            mio.save_model(model, name)

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
            reg_module.register(name, config=cfg, checkpoint_dir=ckpt_root)
            mio.save_model(model, name)

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
        for embed_dim in [16, 32, 64]:
            cfg = config_module.ModelConfig(embed_dim=embed_dim, predictor_n_layers=2)
            model, _ = factory.build(cfg)
            name = f"model_{embed_dim}"
            reg_module.register(name, config=cfg, checkpoint_dir=ckpt_root)
            mio.save_model(model, name)
            time.sleep(0.01)

        results = mio.list_models(ckpt_root)
        assert results[0].config.embed_dim == 64
        assert results[-1].config.embed_dim == 16
