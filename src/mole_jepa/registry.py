"""Supabase-backed model registry for MoLeJEPA experiments.

All model configs and checkpoint paths are stored in a Postgres table on
`Supabase <https://supabase.com>`_ and accessed via the REST API.  This
lets you register, inspect, and delete model configs from any machine
(laptop, CI, cluster) without touching a shared filesystem.

Setup
-----
1. Create a free project at https://supabase.com.
2. Run this SQL once in the Supabase SQL Editor::

       create table public.models (
         name           text primary key,
         created_at     timestamptz not null default now(),
         description    text not null default '',
         checkpoint_dir text not null,
         config         jsonb not null
       );

3. Copy the **Project URL** and **anon public key** from *Settings → API*.
4. Set env vars (add to ``.env`` or export in your shell)::

       SUPABASE_URL=https://<project-ref>.supabase.co
       SUPABASE_ANON_KEY=<anon-public-key>

Key properties
--------------
- **Stable paths**: ``checkpoint_dir`` is written once at registration and
  never recomputed.  Adding new fields to :class:`~mole_jepa.config.ModelConfig`
  does *not* change existing entries.
- **Name-aware hash**: ``checkpoint_dir`` is derived as
  ``checkpoint_dir / sha256(name + "|" + config.serialize())``.  Two entries
  with identical configs but different names always land in separate directories.

Usage::

    from mole_jepa import registry
    from mole_jepa.config import ModelConfig

    # Register (SUPABASE_URL + SUPABASE_ANON_KEY must be set)
    registry.register(
        "vit_base_bert_jepa_frozen",
        config=ModelConfig(),
        checkpoint_dir="/scratch/vpathak/checkpoints",
        description="Baseline frozen ViT run",
    )

    # Inspect
    for entry in registry.list_entries():
        print(entry.name, entry.checkpoint_dir)

    # Look up
    cfg  = registry.get_config("vit_base_bert_jepa_frozen")
    ckpt = registry.get_checkpoint_dir("vit_base_bert_jepa_frozen")
"""

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
from typing import Any

from mole_jepa import config as config_module

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RegistryEntry:
    """One row in the Supabase registry.

    Attributes:
        name: Human-readable identifier used to look up the entry.
        created_at: ISO 8601 timestamp of initial registration.
        description: Free-text note (empty string if omitted).
        checkpoint_dir: Absolute path to the directory that holds
            ``model.pt``, ``config.json``, and ``train_state.pt``.
            Always derived as
            ``<checkpoint_root> / sha256(name + "|" + config.serialize())``.
        config: The :class:`~mole_jepa.config.ModelConfig` registered for
            this model.
    """

    name: str
    created_at: str
    description: str
    checkpoint_dir: pathlib.Path
    config: config_module.ModelConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _entry_hash(name: str, config: config_module.ModelConfig) -> str:
    """Return a stable checkpoint-subdirectory name for a ``(name, config)`` pair.

    Hashes both the model *name* and the serialised config so that two entries
    with identical configs but different names (e.g. a data-scaling ablation)
    always land in different directories.

    Args:
        name: Human-readable model name passed to :func:`register`.
        config: The :class:`~mole_jepa.config.ModelConfig` for this entry.

    Returns:
        A 64-character SHA-256 hex string.
    """
    payload = f"{name}|{config.serialize()}".encode()
    return hashlib.sha256(payload).hexdigest()


def _row_to_entry(row: dict[str, Any]) -> RegistryEntry:
    """Convert a raw Supabase row dict to a :class:`RegistryEntry`."""
    config_data = row["config"]
    if isinstance(config_data, str):
        config_data = json.loads(config_data)
    return RegistryEntry(
        name=row["name"],
        created_at=row["created_at"],
        description=row.get("description", ""),
        checkpoint_dir=pathlib.Path(row["checkpoint_dir"]),
        config=config_module.ModelConfig(**config_data),
    )


# ---------------------------------------------------------------------------
# Supabase client (lazy, module-level singleton)
# ---------------------------------------------------------------------------

_sb_client: Any = None


def _sb() -> Any:
    """Return a lazily-created Supabase client.

    Reads ``$SUPABASE_URL`` and ``$SUPABASE_ANON_KEY`` from the environment
    (with ``.env`` fallback).  The client is cached for the lifetime of the
    process.

    Returns:
        A ``supabase.Client`` instance.

    Raises:
        ImportError: If the ``supabase`` package is not installed.
        RuntimeError: If ``SUPABASE_URL`` or ``SUPABASE_ANON_KEY`` are not set.
    """
    global _sb_client
    if _sb_client is not None:
        return _sb_client

    # Load .env so the env vars are available when running locally.
    from mole_jepa import env as env_module

    env_module._load_dotenv()

    try:
        from supabase import create_client  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'supabase' package is required. Install it with:  uv add supabase"
        ) from exc

    try:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_ANON_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            f"{exc.args[0]} is not set. "
            "Add it to your .env file or export it in your shell."
        ) from exc

    _sb_client = create_client(url, key)
    return _sb_client


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def get_entry(name: str) -> RegistryEntry:
    """Return the :class:`RegistryEntry` for *name*.

    Args:
        name: Registered model name.

    Returns:
        The matching :class:`RegistryEntry`.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    result = _sb().table("models").select("*").eq("name", name).execute()
    if not result.data:
        all_names = sorted(
            r["name"] for r in _sb().table("models").select("name").execute().data
        )
        raise KeyError(
            f"Model {name!r} not found in registry. Registered models: {all_names}"
        )
    return _row_to_entry(result.data[0])


def get_config(name: str) -> config_module.ModelConfig:
    """Return the :class:`~mole_jepa.config.ModelConfig` for *name*.

    Args:
        name: Registered model name.

    Returns:
        The registered :class:`~mole_jepa.config.ModelConfig`.
    """
    return get_entry(name).config


def get_checkpoint_dir(name: str) -> pathlib.Path:
    """Return the checkpoint directory path for *name*.

    Args:
        name: Registered model name.

    Returns:
        Absolute path to the checkpoint directory.
    """
    return get_entry(name).checkpoint_dir


def list_entries() -> list[RegistryEntry]:
    """Return all entries in the registry, sorted by name.

    Returns:
        List of :class:`RegistryEntry` objects sorted by ``name``.
    """
    result = _sb().table("models").select("*").order("name").execute()
    return [_row_to_entry(row) for row in result.data]


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------


def register(
    name: str,
    config: config_module.ModelConfig,
    *,
    checkpoint_dir: str | pathlib.Path | None = None,
    description: str = "",
    overwrite: bool = False,
) -> RegistryEntry:
    """Register a model config in the Supabase registry.

    The checkpoint directory is derived as::

        checkpoint_dir / sha256(name + "|" + config.serialize())

    Including the model name in the hash guarantees that two entries with
    identical configs but different names always land in separate directories.
    The derived path is stored verbatim so it never changes even if new fields
    are added to :class:`~mole_jepa.config.ModelConfig`.

    Args:
        name: Human-readable identifier (e.g. ``"vit_base_bert_jepa_frozen"``).
        config: The :class:`~mole_jepa.config.ModelConfig` to register.
        checkpoint_dir: Root directory under which all model checkpoints live
            (e.g. ``/scratch/vpathak/checkpoints``).  The hash-named subdir is
            appended automatically.  Defaults to ``$CHECKPOINT_DIR``.
        description: Optional human note stored alongside the entry.
        overwrite: If ``True``, silently replace an entry with the same name.

    Returns:
        The newly created :class:`RegistryEntry`.

    Raises:
        ValueError: If *name* already exists and ``overwrite=False``.
        RuntimeError: If *checkpoint_dir* is ``None`` and ``$CHECKPOINT_DIR``
            is not set, or if Supabase credentials are missing.
    """
    if checkpoint_dir is None:
        from mole_jepa import env as env_module

        checkpoint_dir = env_module.checkpoint_dir()

    sb = _sb()
    existing = sb.table("models").select("name").eq("name", name).execute()
    if existing.data and not overwrite:
        raise ValueError(
            f"Model {name!r} is already registered. Pass overwrite=True to replace it."
        )

    resolved_dir = pathlib.Path(checkpoint_dir) / _entry_hash(name, config)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    row = {
        "name": name,
        "created_at": now,
        "description": description,
        "checkpoint_dir": str(resolved_dir),
        "config": dataclasses.asdict(config),
    }
    sb.table("models").upsert(row, on_conflict="name").execute()

    return RegistryEntry(
        name=name,
        created_at=now,
        description=description,
        checkpoint_dir=resolved_dir,
        config=config,
    )


def deregister(name: str) -> None:
    """Remove a model from the registry.

    This does **not** delete any checkpoint files on disk.

    Args:
        name: Registered model name to remove.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    sb = _sb()
    existing = sb.table("models").select("name").eq("name", name).execute()
    if not existing.data:
        raise KeyError(f"Model {name!r} not found in registry.")
    sb.table("models").delete().eq("name", name).execute()
