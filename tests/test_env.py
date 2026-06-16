"""Tests for mole_jepa.env — env-var accessors and .env file parsing."""

from __future__ import annotations

import pathlib

import pytest

import mole_jepa.env as env_mod
from mole_jepa import env

# ── helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the lru_cache on _load_dotenv before each test."""
    env_mod._load_dotenv.cache_clear()


# ── TestCheckpointDir ─────────────────────────────────────────────────────────


class TestCheckpointDir:
    def test_returns_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHECKPOINT_DIR", "/some/checkpoints")
        assert env.checkpoint_dir() == "/some/checkpoints"

    def test_raises_when_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        # _load_dotenv falls back to the real project .env (editable-install
        # path) which has CHECKPOINT_DIR set — block both candidates so this
        # test exercises the genuinely-unset case.
        monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)
        with pytest.raises(RuntimeError, match="CHECKPOINT_DIR"):
            env.checkpoint_dir()


# ── TestParseEnvFile ──────────────────────────────────────────────────────────


class TestParseEnvFile:
    def _write_env(self, path: pathlib.Path, content: str) -> pathlib.Path:
        env_file = path / ".env"
        env_file.write_text(content)
        return env_file

    def test_sets_simple_key_value(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MY_VAR", raising=False)
        f = self._write_env(tmp_path, "MY_VAR=hello\n")
        env_mod._parse_env_file(f)
        import os

        assert os.environ["MY_VAR"] == "hello"

    def test_ignores_comment_lines(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SHOULD_NOT_EXIST", raising=False)
        f = self._write_env(tmp_path, "# SHOULD_NOT_EXIST=bad\n")
        env_mod._parse_env_file(f)
        import os

        assert "SHOULD_NOT_EXIST" not in os.environ

    def test_ignores_blank_lines(self, tmp_path: pathlib.Path) -> None:
        f = self._write_env(tmp_path, "\n   \n\nKEY=val\n")
        # Should not raise
        env_mod._parse_env_file(f)

    def test_strips_double_quotes(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("QUOTED", raising=False)
        f = self._write_env(tmp_path, 'QUOTED="my value"\n')
        env_mod._parse_env_file(f)
        import os

        assert os.environ["QUOTED"] == "my value"

    def test_strips_single_quotes(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SQ", raising=False)
        f = self._write_env(tmp_path, "SQ='single'\n")
        env_mod._parse_env_file(f)
        import os

        assert os.environ["SQ"] == "single"

    def test_strips_inline_comments(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("IC", raising=False)
        f = self._write_env(tmp_path, "IC=value # this is a comment\n")
        env_mod._parse_env_file(f)
        import os

        assert os.environ["IC"] == "value"

    def test_does_not_override_existing_var(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXISTING", "original")
        f = self._write_env(tmp_path, "EXISTING=overridden\n")
        env_mod._parse_env_file(f)
        import os

        assert os.environ["EXISTING"] == "original"

    def test_ignores_lines_without_equals(self, tmp_path: pathlib.Path) -> None:
        f = self._write_env(tmp_path, "NOEQUALS\nKEY=val\n")
        # Should not raise
        env_mod._parse_env_file(f)


# ── TestLoadDotenv ────────────────────────────────────────────────────────────


class TestLoadDotenv:
    def test_loads_from_cwd_dot_env(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_load_dotenv() populates os.environ from .env in the cwd."""
        monkeypatch.delenv("FROM_DOTENV_FILE", raising=False)
        (tmp_path / ".env").write_text("FROM_DOTENV_FILE=found\n")
        monkeypatch.chdir(tmp_path)
        env_mod._load_dotenv()
        import os

        assert os.environ.get("FROM_DOTENV_FILE") == "found"

    def test_dotenv_does_not_override_shell_env(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SHELL_SET", "shell_value")
        (tmp_path / ".env").write_text("SHELL_SET=file_value\n")
        monkeypatch.chdir(tmp_path)
        env_mod._load_dotenv()
        import os

        assert os.environ["SHELL_SET"] == "shell_value"

    def test_no_env_file_is_safe(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_load_dotenv() must not raise when no .env file exists."""
        monkeypatch.chdir(tmp_path)
        env_mod._load_dotenv()  # should not raise

    def test_checkpoint_dir_reads_from_dot_env(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """checkpoint_dir() surfaces a value from .env when env var is absent."""
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        (tmp_path / ".env").write_text(f"CHECKPOINT_DIR={tmp_path}\n")
        monkeypatch.chdir(tmp_path)
        assert env.checkpoint_dir() == str(tmp_path)
