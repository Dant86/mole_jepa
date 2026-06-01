"""Environment variable accessors for MoLeJEPA.

Reads ``CHECKPOINT_DIR``, ``SUPABASE_URL``, and ``SUPABASE_ANON_KEY`` from
``os.environ``.  If a variable is not already present, the module attempts to
populate it from a ``.env`` file in the current working directory (or the
project root for editable installs) before raising.

The ``.env`` file is parsed **once** and does **not** override variables that
are already set — so a value exported in a SLURM script (``source .env``,
``export CHECKPOINT_DIR=…``) always takes precedence over the file.

Usage::

    from mole_jepa import env

    ckpts = env.checkpoint_dir()  # raises RuntimeError if not set
"""

from __future__ import annotations

import os
import pathlib
from functools import lru_cache

# ── .env loading ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_dotenv() -> None:
    """Parse the first .env file found and populate os.environ (no-override)."""
    candidates = [
        pathlib.Path.cwd() / ".env",
        # project root when package is installed in editable mode
        pathlib.Path(__file__).parents[2] / ".env",
    ]
    for env_file in candidates:
        if env_file.exists():
            _parse_env_file(env_file)
            return


def _parse_env_file(path: pathlib.Path) -> None:
    """Read *path* and set missing env variables from its KEY=VALUE lines."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Strip inline comments: VALUE # comment → VALUE
        value = value.split("#")[0].strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


# ── public accessors ──────────────────────────────────────────────────────────


def checkpoint_dir() -> str:
    """Return the checkpoint root directory from ``$CHECKPOINT_DIR``.

    Loads ``.env`` from the current working directory if the variable is not
    already set.

    Returns:
        The value of ``$CHECKPOINT_DIR``.

    Raises:
        RuntimeError: If ``CHECKPOINT_DIR`` is not set in the environment or
            in a ``.env`` file.
    """
    _load_dotenv()
    try:
        return os.environ["CHECKPOINT_DIR"]
    except KeyError:
        raise RuntimeError(
            "CHECKPOINT_DIR is not set. "
            "Export it in your shell, or add CHECKPOINT_DIR=<path> to your .env file."
        ) from None
