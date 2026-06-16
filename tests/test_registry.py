"""Unit tests for mole_jepa.registry (Supabase backend)."""

import datetime
import pathlib

import pytest

from mole_jepa import config as config_module
from mole_jepa import registry
from mole_jepa.registry import _entry_hash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**kwargs: object) -> config_module.ModelConfig:
    return config_module.ModelConfig(**kwargs)  # type: ignore[arg-type]


def _register(name: str, **kwargs: object) -> registry.RegistryEntry:
    cfg = kwargs.pop("config", _cfg())  # type: ignore[assignment]
    ckpt = kwargs.pop("checkpoint_dir", "/tmp/ckpts")
    return registry.register(name, cfg, checkpoint_dir=ckpt, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestRegister
# ---------------------------------------------------------------------------


class TestRegister:
    def test_checkpoint_dir_derives_hash_subdir(self) -> None:
        cfg = _cfg()
        entry = _register("my_model", config=cfg, checkpoint_dir="/tmp/ckpts")
        assert entry.checkpoint_dir == pathlib.Path("/tmp/ckpts") / _entry_hash(
            "my_model", cfg
        )

    def test_different_configs_give_different_paths(self) -> None:
        cfg_a = _cfg(embed_dim=64)
        cfg_b = _cfg(embed_dim=128)
        e_a = _register("a", config=cfg_a)
        e_b = _register("b", config=cfg_b)
        assert e_a.checkpoint_dir != e_b.checkpoint_dir

    def test_entry_survives_round_trip(self) -> None:
        cfg = _cfg(embed_dim=64)
        _register(
            "roundtrip", config=cfg, checkpoint_dir="/tmp/ckpts", description="hello"
        )
        entry = registry.get_entry("roundtrip")
        assert entry.name == "roundtrip"
        assert entry.config == cfg
        assert entry.description == "hello"
        assert entry.checkpoint_dir == pathlib.Path("/tmp/ckpts") / _entry_hash(
            "roundtrip", cfg
        )

    def test_duplicate_raises_without_overwrite(self) -> None:
        _register("dup")
        with pytest.raises(ValueError, match="already registered"):
            _register("dup")

    def test_overwrite_replaces_entry(self) -> None:
        cfg_old = _cfg()
        cfg_new = _cfg(embed_dim=64)
        _register("ow", config=cfg_old)
        registry.register("ow", cfg_new, checkpoint_dir="/tmp/ckpts", overwrite=True)
        entry = registry.get_entry("ow")
        assert entry.config.embed_dim == 64
        assert entry.checkpoint_dir == pathlib.Path("/tmp/ckpts") / _entry_hash(
            "ow", cfg_new
        )

    def test_raises_when_checkpoint_dir_missing_and_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        import mole_jepa.env as env_mod

        env_mod._load_dotenv.cache_clear()
        # Block both .env candidates (cwd and editable-install project root) —
        # otherwise this picks up the real project .env, which has
        # CHECKPOINT_DIR set, defeating the "no env" case under test.
        monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)
        with pytest.raises(RuntimeError, match="CHECKPOINT_DIR"):
            registry.register("bad", _cfg())

    def test_created_at_is_iso8601(self) -> None:
        _register("ts_check")
        entry = registry.get_entry("ts_check")
        parsed = datetime.datetime.fromisoformat(entry.created_at)
        assert parsed.tzinfo is not None


# ---------------------------------------------------------------------------
# TestGetEntry / GetConfig / GetCheckpointDir
# ---------------------------------------------------------------------------


class TestGetEntry:
    def test_missing_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="ghost"):
            registry.get_entry("ghost")

    def test_key_error_lists_available_models(self) -> None:
        _register("alpha")
        with pytest.raises(KeyError, match="alpha"):
            registry.get_entry("beta")

    def test_get_config_returns_model_config(self) -> None:
        cfg = _cfg(embed_dim=32)
        _register("cfg_check", config=cfg)
        assert registry.get_config("cfg_check") == cfg

    def test_get_checkpoint_dir_returns_derived_path(self) -> None:
        cfg = _cfg()
        _register("ckpt_check", config=cfg, checkpoint_dir="/tmp/base")
        assert registry.get_checkpoint_dir("ckpt_check") == (
            pathlib.Path("/tmp/base") / _entry_hash("ckpt_check", cfg)
        )


# ---------------------------------------------------------------------------
# TestListEntries
# ---------------------------------------------------------------------------


class TestListEntries:
    def test_empty_registry_returns_empty_list(self) -> None:
        assert registry.list_entries() == []

    def test_returns_all_registered_entries(self) -> None:
        for name in ("alpha", "beta", "gamma"):
            _register(name)
        entries = registry.list_entries()
        assert {e.name for e in entries} == {"alpha", "beta", "gamma"}

    def test_returns_entries_sorted_by_name(self) -> None:
        for name in ("zeta", "alpha", "mu"):
            _register(name)
        entries = registry.list_entries()
        assert [e.name for e in entries] == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# TestDeregister
# ---------------------------------------------------------------------------


class TestDeregister:
    def test_removes_entry(self) -> None:
        _register("to_remove")
        registry.deregister("to_remove")
        assert registry.list_entries() == []

    def test_missing_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="ghost"):
            registry.deregister("ghost")

    def test_other_entries_survive(self) -> None:
        for name in ("keep", "drop"):
            _register(name)
        registry.deregister("drop")
        entries = registry.list_entries()
        assert len(entries) == 1
        assert entries[0].name == "keep"


# ---------------------------------------------------------------------------
# TestEntryHash
# ---------------------------------------------------------------------------


class TestEntryHash:
    def test_different_names_same_config_give_different_paths(self) -> None:
        """Scaling-curve models with identical configs must not share dirs."""
        cfg = _cfg(embed_dim=768)
        e1 = _register("scale_500k", config=cfg)
        e2 = _register("scale_1m", config=cfg)
        assert e1.checkpoint_dir != e2.checkpoint_dir

    def test_same_name_same_config_is_deterministic(self) -> None:
        cfg = _cfg()
        e1 = _register("det", config=cfg)
        e2 = registry.register("det", cfg, checkpoint_dir="/tmp/ckpts", overwrite=True)
        assert e1.checkpoint_dir == e2.checkpoint_dir

    def test_same_name_different_config_gives_different_paths(self) -> None:
        cfg_a = _cfg(embed_dim=64)
        cfg_b = _cfg(embed_dim=128)
        e1 = _register("overwrite_me", config=cfg_a)
        e2 = registry.register(
            "overwrite_me", cfg_b, checkpoint_dir="/tmp/ckpts", overwrite=True
        )
        assert e1.checkpoint_dir != e2.checkpoint_dir

    def test_hash_is_64_hex_chars(self) -> None:
        cfg = _cfg()
        entry = _register("hash_len", config=cfg)
        name = entry.checkpoint_dir.name
        assert len(name) == 64
        assert all(c in "0123456789abcdef" for c in name)

    def test_entry_hash_matches_stored_path(self) -> None:
        cfg = _cfg(embed_dim=32)
        entry = _register("h_check", config=cfg, checkpoint_dir="/tmp/ckpts")
        assert entry.checkpoint_dir == pathlib.Path("/tmp/ckpts") / _entry_hash(
            "h_check", cfg
        )


# ---------------------------------------------------------------------------
# TestRowToEntry
# ---------------------------------------------------------------------------


class TestRowToEntry:
    def test_config_stored_as_json_string_is_parsed(
        self, fake_supabase: object
    ) -> None:
        """Supabase's jsonb columns sometimes come back as a JSON string
        rather than an already-parsed dict, depending on client version."""
        import json

        cfg = _cfg(embed_dim=64)
        _register("string_config", config=cfg)
        # Mutate the fake store directly to simulate a string-encoded config.
        row = fake_supabase._store["string_config"]  # type: ignore[attr-defined]
        row["config"] = json.dumps(row["config"])

        entry = registry.get_entry("string_config")
        assert entry.config.embed_dim == 64


# ---------------------------------------------------------------------------
# TestSbClient
# ---------------------------------------------------------------------------


class TestSbClient:
    def test_returns_cached_client_without_reinit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = object()
        monkeypatch.setattr(registry, "_sb_client", sentinel)
        assert registry._sb() is sentinel

    def test_raises_import_error_when_supabase_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registry, "_sb_client", None)
        monkeypatch.setattr("mole_jepa.env._load_dotenv", lambda: None)
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "supabase":
                raise ImportError("no supabase")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="supabase"):
            registry._sb()

    def test_raises_runtime_error_when_env_vars_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registry, "_sb_client", None)
        monkeypatch.setattr("mole_jepa.env._load_dotenv", lambda: None)
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
        with pytest.raises(RuntimeError, match="SUPABASE_URL"):
            registry._sb()

    def test_creates_and_caches_client_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registry, "_sb_client", None)
        monkeypatch.setattr("mole_jepa.env._load_dotenv", lambda: None)
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

        created = object()
        fake_supabase_module = type(
            "module", (), {"create_client": lambda url, key: created}
        )
        monkeypatch.setitem(__import__("sys").modules, "supabase", fake_supabase_module)

        client = registry._sb()
        assert client is created
        assert registry._sb_client is created
