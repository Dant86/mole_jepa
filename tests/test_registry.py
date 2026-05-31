"""Unit tests for mole_jepa.registry."""

import pathlib

import pytest

from mole_jepa import config as config_module
from mole_jepa import registry
from mole_jepa.registry import _entry_hash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**kwargs: object) -> config_module.ModelConfig:
    """Return a ModelConfig with sensible defaults overridden by kwargs."""
    return config_module.ModelConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestRegister
# ---------------------------------------------------------------------------


class TestRegister:
    def test_checkpoint_dir_derives_hash_subdir(self, tmp_path: pathlib.Path) -> None:
        """checkpoint_dir is root / sha256(name|config): name and config both matter."""
        cfg = _cfg()
        entry = registry.register(
            "my_model",
            config=cfg,
            checkpoint_dir=tmp_path / "ckpts",
            registry_dir=tmp_path,
        )
        assert entry.checkpoint_dir == tmp_path / "ckpts" / _entry_hash("my_model", cfg)

    def test_different_configs_give_different_paths(
        self, tmp_path: pathlib.Path
    ) -> None:
        cfg_a = _cfg(embed_dim=64)
        cfg_b = _cfg(embed_dim=128)
        entry_a = registry.register(
            "a", config=cfg_a, checkpoint_dir=tmp_path, registry_dir=tmp_path
        )
        entry_b = registry.register(
            "b", config=cfg_b, checkpoint_dir=tmp_path, registry_dir=tmp_path
        )
        assert entry_a.checkpoint_dir != entry_b.checkpoint_dir

    def test_registry_json_created(self, tmp_path: pathlib.Path) -> None:
        registry.register(
            "a",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        assert (tmp_path / "registry.json").exists()

    def test_entry_survives_round_trip(self, tmp_path: pathlib.Path) -> None:
        cfg = _cfg(embed_dim=64)
        registry.register(
            "roundtrip",
            config=cfg,
            checkpoint_dir=tmp_path / "ckpts",
            description="hello",
            registry_dir=tmp_path,
        )
        entry = registry.get_entry("roundtrip", tmp_path)
        assert entry.name == "roundtrip"
        assert entry.config == cfg
        assert entry.description == "hello"
        expected = tmp_path / "ckpts" / _entry_hash("roundtrip", cfg)
        assert entry.checkpoint_dir == expected

    def test_duplicate_raises_without_overwrite(self, tmp_path: pathlib.Path) -> None:
        registry.register(
            "dup",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                "dup",
                config=_cfg(),
                checkpoint_dir=tmp_path,
                registry_dir=tmp_path,
            )

    def test_overwrite_replaces_entry(self, tmp_path: pathlib.Path) -> None:
        cfg_old = _cfg()
        cfg_new = _cfg(embed_dim=64)
        registry.register(
            "ow",
            config=cfg_old,
            checkpoint_dir=tmp_path / "ckpts",
            registry_dir=tmp_path,
        )
        registry.register(
            "ow",
            config=cfg_new,
            checkpoint_dir=tmp_path / "ckpts",
            registry_dir=tmp_path,
            overwrite=True,
        )
        entry = registry.get_entry("ow", tmp_path)
        assert entry.checkpoint_dir == tmp_path / "ckpts" / _entry_hash("ow", cfg_new)
        assert entry.config.embed_dim == 64

    def test_raises_when_checkpoint_dir_missing_and_no_env(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No checkpoint_dir and $CHECKPOINT_DIR unset → RuntimeError."""
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        # Also ensure .env loading doesn't find a file
        import mole_jepa.env as env_mod

        env_mod._load_dotenv.cache_clear()
        with pytest.raises(RuntimeError, match="CHECKPOINT_DIR"):
            registry.register(
                "bad",
                config=_cfg(),
                # checkpoint_dir omitted → falls back to env
                registry_dir=tmp_path,
            )

    def test_creates_registry_dir_if_absent(self, tmp_path: pathlib.Path) -> None:
        reg_dir = tmp_path / "deep" / "nested"
        registry.register(
            "nested",
            config=_cfg(),
            checkpoint_dir=tmp_path / "ckpts",
            registry_dir=reg_dir,
        )
        assert reg_dir.is_dir()

    def test_created_at_is_iso8601(self, tmp_path: pathlib.Path) -> None:
        import datetime

        registry.register(
            "ts_check",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        entry = registry.get_entry("ts_check", tmp_path)
        # Verify it parses as a valid ISO 8601 datetime.
        parsed = datetime.datetime.fromisoformat(entry.created_at)
        assert parsed.tzinfo is not None  # must be timezone-aware


# ---------------------------------------------------------------------------
# TestGetEntry / GetConfig / GetCheckpointDir
# ---------------------------------------------------------------------------


class TestGetEntry:
    def test_missing_raises_key_error(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="ghost"):
            registry.get_entry("ghost", tmp_path)

    def test_key_error_lists_available_models(self, tmp_path: pathlib.Path) -> None:
        registry.register(
            "alpha",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        with pytest.raises(KeyError, match="alpha"):
            registry.get_entry("beta", tmp_path)

    def test_get_config_returns_model_config(self, tmp_path: pathlib.Path) -> None:
        cfg = _cfg(embed_dim=32)
        registry.register(
            "cfg_check",
            config=cfg,
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        assert registry.get_config("cfg_check", tmp_path) == cfg

    def test_get_checkpoint_dir_returns_derived_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        cfg = _cfg()
        registry.register(
            "ckpt_check",
            config=cfg,
            checkpoint_dir=tmp_path / "base",
            registry_dir=tmp_path,
        )
        assert registry.get_checkpoint_dir("ckpt_check", tmp_path) == (
            tmp_path / "base" / _entry_hash("ckpt_check", cfg)
        )


# ---------------------------------------------------------------------------
# TestEnvFallback
# ---------------------------------------------------------------------------


class TestEnvFallback:
    """registry_dir=None falls back to $REGISTRY_PATH via env.registry_dir()."""

    def test_register_and_get_via_env(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mole_jepa.env as env_mod

        monkeypatch.setenv("REGISTRY_PATH", str(tmp_path))
        env_mod._load_dotenv.cache_clear()
        cfg = _cfg()
        registry.register(
            "env_model", config=cfg, checkpoint_dir=tmp_path, registry_dir=None
        )
        entry = registry.get_entry("env_model", registry_dir=None)
        assert entry.name == "env_model"
        assert entry.config == cfg


# ---------------------------------------------------------------------------
# TestListEntries
# ---------------------------------------------------------------------------


class TestListEntries:
    def test_empty_registry_returns_empty_list(self, tmp_path: pathlib.Path) -> None:
        assert registry.list_entries(tmp_path) == []

    def test_nonexistent_registry_dir_returns_empty_list(
        self, tmp_path: pathlib.Path
    ) -> None:
        # list_entries must not crash on a directory that doesn't exist yet.
        assert registry.list_entries(tmp_path / "no_such_dir") == []

    def test_returns_all_registered_entries(self, tmp_path: pathlib.Path) -> None:
        for name in ("alpha", "beta", "gamma"):
            registry.register(
                name,
                config=_cfg(),
                checkpoint_dir=tmp_path,
                registry_dir=tmp_path,
            )
        entries = registry.list_entries(tmp_path)
        assert {e.name for e in entries} == {"alpha", "beta", "gamma"}

    def test_returns_entries_sorted_by_name(self, tmp_path: pathlib.Path) -> None:
        for name in ("zeta", "alpha", "mu"):
            registry.register(
                name,
                config=_cfg(),
                checkpoint_dir=tmp_path,
                registry_dir=tmp_path,
            )
        entries = registry.list_entries(tmp_path)
        assert [e.name for e in entries] == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# TestSchemaVersion
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_returns_current_version(self, tmp_path: pathlib.Path) -> None:
        registry.register(
            "sv",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        assert registry.schema_version(tmp_path) == registry.CURRENT_SCHEMA_VERSION

    def test_returns_current_version_when_file_absent(
        self, tmp_path: pathlib.Path
    ) -> None:
        # _load() returns a skeleton with CURRENT_SCHEMA_VERSION when the file
        # doesn't yet exist, so schema_version() should match it.
        assert registry.schema_version(tmp_path) == registry.CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# TestDeregister
# ---------------------------------------------------------------------------


class TestDeregister:
    def test_removes_entry(self, tmp_path: pathlib.Path) -> None:
        registry.register(
            "to_remove",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        registry.deregister("to_remove", tmp_path)
        assert registry.list_entries(tmp_path) == []

    def test_missing_raises_key_error(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="ghost"):
            registry.deregister("ghost", tmp_path)

    def test_other_entries_survive(self, tmp_path: pathlib.Path) -> None:
        for name in ("keep", "drop"):
            registry.register(
                name,
                config=_cfg(),
                checkpoint_dir=tmp_path,
                registry_dir=tmp_path,
            )
        registry.deregister("drop", tmp_path)
        entries = registry.list_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0].name == "keep"


# ---------------------------------------------------------------------------
# TestAtomicWrite / TestApplyMigration
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_no_tmp_file_left_after_register(self, tmp_path: pathlib.Path) -> None:
        registry.register(
            "atomic",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


class TestApplyMigration:
    def test_raises_if_up_fn_is_none(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError):
            registry.apply_migration(tmp_path, None)  # type: ignore[arg-type]

    def test_up_fn_receives_and_returns_raw_dict(self, tmp_path: pathlib.Path) -> None:
        registry.register(
            "m",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        received: list[dict] = []

        def up(data: dict) -> dict:
            received.append(data)
            data["schema_version"] = 99
            return data

        registry.apply_migration(tmp_path, up)
        assert len(received) == 1
        assert "models" in received[0]
        assert registry.schema_version(tmp_path) == 99

    def test_up_fn_runs_inside_lock(self, tmp_path: pathlib.Path) -> None:
        """Mutations made by up_fn must be persisted after apply_migration returns."""
        registry.register(
            "pre",
            config=_cfg(),
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )

        def up(data: dict) -> dict:
            # Simulate a migration that adds a new field to every entry.
            for entry in data["models"].values():
                entry["migrated"] = True
            data["schema_version"] = 2
            return data

        registry.apply_migration(tmp_path, up)

        # Read back the raw JSON and verify the mutation was saved.
        import json

        raw = json.loads((tmp_path / "registry.json").read_text())
        assert raw["schema_version"] == 2
        assert all(e.get("migrated") for e in raw["models"].values())


# ---------------------------------------------------------------------------
# TestConcurrentAccess
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestEntryHash  — name-aware checkpoint directory
# ---------------------------------------------------------------------------


class TestEntryHash:
    """The checkpoint path must encode both the model name and the config."""

    def test_different_names_same_config_give_different_paths(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Key regression: two scaling-curve models with identical configs must
        land in separate directories so they don't share checkpoints."""
        cfg = _cfg(embed_dim=768)
        e1 = registry.register(
            "scale_500k", config=cfg, checkpoint_dir=tmp_path, registry_dir=tmp_path
        )
        e2 = registry.register(
            "scale_1m", config=cfg, checkpoint_dir=tmp_path, registry_dir=tmp_path
        )
        assert e1.checkpoint_dir != e2.checkpoint_dir

    def test_same_name_same_config_is_deterministic(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Registering the same (name, config) pair always produces the same path."""
        cfg = _cfg()
        e1 = registry.register(
            "det", config=cfg, checkpoint_dir=tmp_path, registry_dir=tmp_path
        )
        # Re-register with overwrite to get a fresh entry for the same pair.
        e2 = registry.register(
            "det",
            config=cfg,
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
            overwrite=True,
        )
        assert e1.checkpoint_dir == e2.checkpoint_dir

    def test_same_name_different_config_gives_different_paths(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Overwriting a name with a different config must produce a new path."""
        cfg_a = _cfg(embed_dim=64)
        cfg_b = _cfg(embed_dim=128)
        e1 = registry.register(
            "overwrite_me",
            config=cfg_a,
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
        )
        e2 = registry.register(
            "overwrite_me",
            config=cfg_b,
            checkpoint_dir=tmp_path,
            registry_dir=tmp_path,
            overwrite=True,
        )
        assert e1.checkpoint_dir != e2.checkpoint_dir

    def test_hash_is_64_hex_chars(self, tmp_path: pathlib.Path) -> None:
        cfg = _cfg()
        entry = registry.register(
            "hash_len", config=cfg, checkpoint_dir=tmp_path, registry_dir=tmp_path
        )
        name = entry.checkpoint_dir.name
        assert len(name) == 64
        assert all(c in "0123456789abcdef" for c in name)

    def test_entry_hash_matches_stored_path(self, tmp_path: pathlib.Path) -> None:
        """_entry_hash must agree with the path stored in the registry."""
        cfg = _cfg(embed_dim=32)
        entry = registry.register(
            "h_check", config=cfg, checkpoint_dir=tmp_path, registry_dir=tmp_path
        )
        assert entry.checkpoint_dir == tmp_path / _entry_hash("h_check", cfg)


class TestConcurrentAccess:
    def test_concurrent_writes_do_not_corrupt_registry(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Interleaved register() calls must not lose entries."""
        import threading

        errors: list[Exception] = []

        def _register(name: str) -> None:
            try:
                registry.register(
                    name,
                    config=_cfg(),
                    checkpoint_dir=tmp_path,
                    registry_dir=tmp_path,
                )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_register, args=(f"model_{i}",)) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        entries = registry.list_entries(tmp_path)
        assert len(entries) == 8
