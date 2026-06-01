"""Model registry for MoLeJEPA experiments.

Two backends are supported, selected automatically at call time:

**File backend** (default)
    A JSON file at ``{registry_dir}/registry.json``.  All writes are
    protected by an exclusive ``flock`` sidecar so concurrent processes on the
    same NFS mount cannot corrupt the file.  Pass an explicit *registry_dir*
    (or set ``$REGISTRY_PATH``) to use this backend.

**Supabase backend** (remote)
    A Postgres table on `Supabase <https://supabase.com>`_ accessed via the
    REST API.  Activated automatically when *registry_dir* is ``None`` *and*
    ``$SUPABASE_URL`` is set.  Lets you register, inspect, and delete model
    configs from any machine (laptop, cluster, CI) without touching the NFS
    filesystem.

    Required env vars::

        SUPABASE_URL=https://<project-ref>.supabase.co
        SUPABASE_ANON_KEY=<anon-public-key>

    Create the table once in the Supabase SQL editor::

        create table public.models (
          name           text primary key,
          created_at     timestamptz not null default now(),
          description    text not null default '',
          checkpoint_dir text not null,
          config         jsonb not null
        );

Key properties
--------------
- **Stable paths**: ``checkpoint_dir`` is written once at registration and
  never recomputed.  Adding new fields to :class:`~mole_jepa.config.ModelConfig`
  does *not* change existing entries.
- **Name-aware hash**: ``checkpoint_dir`` is derived as
  ``checkpoint_dir / sha256(name + "|" + config.serialize())``.  Two entries
  with identical configs but different names always land in separate directories.
- **NFS-safe writes**: the file backend acquires an exclusive ``flock`` before
  every read-modify-write cycle.
- **Schema versioning** (file backend only): the file carries a
  ``schema_version`` int.  When :class:`~mole_jepa.config.ModelConfig` gains a
  new field, bump :data:`CURRENT_SCHEMA_VERSION`, write a migration in
  ``apps/registry/migrations/``, and run ``apps/registry/migrate.py`` on the
  cluster.

Usage::

    from mole_jepa import registry
    from mole_jepa.config import ModelConfig

    # ── file backend (default, reads $REGISTRY_PATH) ──────────────────────
    entry = registry.get_entry("vit_base_bert_jepa_frozen")

    # ── Supabase backend (set SUPABASE_URL + SUPABASE_ANON_KEY in env) ────
    registry.register(
        "my_experiment",
        config=ModelConfig(...),
        checkpoint_dir="/scratch/vpathak/checkpoints",
        description="Lower LR, frozen ViT",
    )
"""

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
from collections.abc import Callable
from typing import Any

from filelock import FileLock

from mole_jepa import config as config_module

CURRENT_SCHEMA_VERSION: int = 1

_REGISTRY_FILE = "registry.json"
_LOCK_FILE = "registry.lock"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RegistryEntry:
    """One row in the registry.

    Attributes:
        name: Human-readable identifier used to look up the entry.
        created_at: ISO 8601 timestamp of initial registration.
        description: Free-text note (empty string if omitted).
        checkpoint_dir: Absolute path to the directory that holds
            ``model.pt``, ``config.json``, and ``train_state.pt``.
            Always derived as ``<checkpoint_root> / _entry_hash(name, config)``.
        config: The :class:`~mole_jepa.config.ModelConfig` registered for
            this model.
    """

    name: str
    created_at: str
    description: str
    checkpoint_dir: pathlib.Path
    config: config_module.ModelConfig


# ---------------------------------------------------------------------------
# Internal helpers — shared
# ---------------------------------------------------------------------------


def _entry_hash(name: str, config: config_module.ModelConfig) -> str:
    """Derive a stable, unique checkpoint-directory name for a registry entry.

    The hash is computed over both the model *name* and the serialised
    :class:`~mole_jepa.config.ModelConfig` so that two entries with identical
    configs but different names always land in different directories.

    Args:
        name: Human-readable model name passed to :func:`register`.
        config: The :class:`~mole_jepa.config.ModelConfig` for this entry.

    Returns:
        A 64-character SHA-256 hex string uniquely identifying the
        ``(name, config)`` pair.
    """
    payload = f"{name}|{config.serialize()}".encode()
    return hashlib.sha256(payload).hexdigest()


def _entry_from_raw(name: str, raw: dict[str, Any]) -> RegistryEntry:
    return RegistryEntry(
        name=name,
        created_at=raw["created_at"],
        description=raw.get("description", ""),
        checkpoint_dir=pathlib.Path(raw["checkpoint_dir"]),
        config=config_module.ModelConfig(**raw["config"]),
    )


def _entry_to_raw(entry: RegistryEntry) -> dict[str, Any]:
    return {
        "created_at": entry.created_at,
        "description": entry.description,
        "checkpoint_dir": str(entry.checkpoint_dir),
        "config": dataclasses.asdict(entry.config),
    }


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _supabase_mode(registry_dir: str | pathlib.Path | None) -> bool:
    """Return ``True`` when the Supabase backend should be used.

    Activated when *registry_dir* is ``None`` **and** ``$SUPABASE_URL`` is set
    in the environment.  Passing an explicit *registry_dir* always forces the
    file backend, even when ``$SUPABASE_URL`` is present.

    Args:
        registry_dir: The value passed by the caller — ``None`` means "use
            the default backend".

    Returns:
        ``True`` if Supabase mode is active.
    """
    return registry_dir is None and bool(os.environ.get("SUPABASE_URL"))


# ---------------------------------------------------------------------------
# Internal I/O helpers — file backend
# ---------------------------------------------------------------------------


def _registry_path(registry_dir: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(registry_dir) / _REGISTRY_FILE


def _lock_path(registry_dir: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(registry_dir) / _LOCK_FILE


def _registry_lock(registry_dir: str | pathlib.Path) -> FileLock:
    """Return an exclusive ``FileLock`` for the registry sidecar file.

    Uses ``filelock.FileLock``, which calls ``fcntl.flock`` on POSIX — the
    standard advisory lock on modern Linux NFS4 mounts used by SLURM clusters.
    The lock directory is created if it does not already exist.

    Args:
        registry_dir: Path to the registry directory on NFS.

    Returns:
        A :class:`filelock.FileLock` context manager.
    """
    pathlib.Path(registry_dir).mkdir(parents=True, exist_ok=True)
    return FileLock(_lock_path(registry_dir))


def _load(registry_dir: str | pathlib.Path) -> dict[str, Any]:
    """Read the registry JSON, returning an empty skeleton if absent."""
    path = _registry_path(registry_dir)
    if not path.exists():
        return {"schema_version": CURRENT_SCHEMA_VERSION, "models": {}}
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _save(data: dict[str, Any], registry_dir: str | pathlib.Path) -> None:
    """Atomically write the registry JSON (write-then-rename)."""
    registry_dir = pathlib.Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    tmp = _registry_path(registry_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, _registry_path(registry_dir))


def _resolve_registry_dir(
    registry_dir: str | pathlib.Path | None,
) -> str | pathlib.Path:
    """Return *registry_dir*, falling back to ``$REGISTRY_PATH`` if ``None``."""
    if registry_dir is not None:
        return registry_dir
    from mole_jepa import env as env_module  # late import avoids circular deps

    return env_module.registry_dir()


# ---------------------------------------------------------------------------
# Internal I/O helpers — Supabase backend
# ---------------------------------------------------------------------------

_sb_client: Any = None


def _sb() -> Any:
    """Return a lazily-created, module-level Supabase client.

    Reads ``$SUPABASE_URL`` and ``$SUPABASE_ANON_KEY`` from the environment.
    The client is cached for the lifetime of the process.

    Returns:
        A ``supabase.Client`` instance.

    Raises:
        ImportError: If the ``supabase`` package is not installed.
        KeyError: If ``$SUPABASE_URL`` or ``$SUPABASE_ANON_KEY`` are not set.
    """
    global _sb_client
    if _sb_client is None:
        try:
            from supabase import create_client  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'supabase' package is required for remote registry access. "
                "Install it with:  uv add supabase"
            ) from exc
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_ANON_KEY"]
        _sb_client = create_client(url, key)
    return _sb_client


def _sb_row_to_entry(row: dict[str, Any]) -> RegistryEntry:
    """Convert a raw Supabase table row to a :class:`RegistryEntry`."""
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


def _sb_get_entry(name: str) -> RegistryEntry:
    result = _sb().table("models").select("*").eq("name", name).execute()
    if not result.data:
        raise KeyError(f"Model {name!r} not found in Supabase registry.")
    return _sb_row_to_entry(result.data[0])


def _sb_list_entries() -> list[RegistryEntry]:
    result = _sb().table("models").select("*").order("name").execute()
    return [_sb_row_to_entry(row) for row in result.data]


def _sb_register(
    name: str,
    config: config_module.ModelConfig,
    *,
    checkpoint_dir: str | pathlib.Path,
    description: str = "",
    overwrite: bool = False,
) -> RegistryEntry:
    sb = _sb()
    existing = sb.table("models").select("name").eq("name", name).execute()
    if existing.data and not overwrite:
        raise ValueError(
            f"Model {name!r} is already registered in Supabase. "
            "Pass overwrite=True to replace it."
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


def _sb_deregister(name: str) -> None:
    sb = _sb()
    existing = sb.table("models").select("name").eq("name", name).execute()
    if not existing.data:
        raise KeyError(f"Model {name!r} not found in Supabase registry.")
    sb.table("models").delete().eq("name", name).execute()


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def get_entry(
    name: str,
    registry_dir: str | pathlib.Path | None = None,
) -> RegistryEntry:
    """Return the :class:`RegistryEntry` for *name*.

    Args:
        name: Registered model name.
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.  When ``None`` and ``$SUPABASE_URL`` is set,
            the Supabase backend is used instead.

    Returns:
        The matching :class:`RegistryEntry`.

    Raises:
        KeyError: If *name* is not in the registry.
        RuntimeError: If *registry_dir* is ``None``, neither
            ``$REGISTRY_PATH`` nor ``$SUPABASE_URL`` is set.
    """
    if _supabase_mode(registry_dir):
        return _sb_get_entry(name)

    rdir = _resolve_registry_dir(registry_dir)
    data = _load(rdir)
    if name not in data["models"]:
        available = sorted(data["models"])
        raise KeyError(
            f"Model {name!r} not found in registry at {rdir!r}. "
            f"Registered models: {available}"
        )
    return _entry_from_raw(name, data["models"][name])


def get_config(
    name: str,
    registry_dir: str | pathlib.Path | None = None,
) -> config_module.ModelConfig:
    """Return the :class:`~mole_jepa.config.ModelConfig` for *name*.

    Args:
        name: Registered model name.
        registry_dir: Path to the registry directory.  Defaults to
            ``$REGISTRY_PATH`` (file) or ``$SUPABASE_URL`` (remote).

    Returns:
        The registered :class:`~mole_jepa.config.ModelConfig`.
    """
    return get_entry(name, registry_dir).config


def get_checkpoint_dir(
    name: str,
    registry_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Return the checkpoint directory path for *name*.

    Args:
        name: Registered model name.
        registry_dir: Path to the registry directory.  Defaults to
            ``$REGISTRY_PATH`` (file) or ``$SUPABASE_URL`` (remote).

    Returns:
        Absolute path to the checkpoint directory (``<root> / config_hash``).
    """
    return get_entry(name, registry_dir).checkpoint_dir


def list_entries(
    registry_dir: str | pathlib.Path | None = None,
) -> list[RegistryEntry]:
    """Return all entries in the registry, sorted by name.

    Args:
        registry_dir: Path to the registry directory.  Defaults to
            ``$REGISTRY_PATH`` (file) or ``$SUPABASE_URL`` (remote).
            Pass an explicit path to force the file backend.

    Returns:
        List of :class:`RegistryEntry` objects sorted by ``name``.
    """
    if _supabase_mode(registry_dir):
        return _sb_list_entries()

    rdir = _resolve_registry_dir(registry_dir)
    if not pathlib.Path(rdir).exists():
        return []
    data = _load(rdir)
    return sorted(
        (_entry_from_raw(name, raw) for name, raw in data["models"].items()),
        key=lambda e: e.name,
    )


def schema_version(registry_dir: str | pathlib.Path | None = None) -> int:
    """Return the current ``schema_version`` from the registry file.

    Not meaningful for the Supabase backend (migrations are handled via the
    Supabase dashboard or ``psql``); returns :data:`CURRENT_SCHEMA_VERSION`
    in that mode.

    Args:
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.

    Returns:
        Schema version integer.
    """
    if _supabase_mode(registry_dir):
        return CURRENT_SCHEMA_VERSION

    rdir = _resolve_registry_dir(registry_dir)
    return int(_load(rdir).get("schema_version", 0))


# ---------------------------------------------------------------------------
# Write API  (all mutations go through the flock / Supabase upsert)
# ---------------------------------------------------------------------------


def register(
    name: str,
    config: config_module.ModelConfig,
    *,
    checkpoint_dir: str | pathlib.Path | None = None,
    description: str = "",
    registry_dir: str | pathlib.Path | None = None,
    overwrite: bool = False,
) -> RegistryEntry:
    """Register a model config in the registry.

    The checkpoint directory is derived as::

        checkpoint_dir / sha256(name + "|" + config.serialize())

    Including the model name in the hash guarantees that two entries with
    identical configs but different names (e.g. a data-scaling ablation)
    always land in separate directories.  The derived path is stored
    verbatim in the registry entry so it never changes even if new fields
    are added to :class:`~mole_jepa.config.ModelConfig`.

    Backend selection:

    * Pass an explicit *registry_dir* → **file backend** (NFS JSON).
    * Leave *registry_dir* as ``None`` and set ``$SUPABASE_URL`` →
      **Supabase backend** (remote Postgres).
    * Leave both unset → file backend, reads ``$REGISTRY_PATH``.

    Args:
        name: Human-readable identifier (e.g. ``"vit_base_bert_jepa_frozen"``).
        config: The :class:`~mole_jepa.config.ModelConfig` to register.
        checkpoint_dir: Root directory under which all model checkpoints live
            (e.g. ``/scratch/vpathak/checkpoints``).  The hash-named subdir
            is appended automatically.  Defaults to ``$CHECKPOINT_DIR`` (via
            :func:`mole_jepa.env.checkpoint_dir`).
        description: Optional human note stored alongside the entry.
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH`` (file) or Supabase (remote).
        overwrite: If ``True``, silently replace an entry with the same name.

    Returns:
        The newly created :class:`RegistryEntry`.

    Raises:
        ValueError: If *name* already exists and ``overwrite=False``.
        RuntimeError: If *checkpoint_dir* or *registry_dir* are ``None`` and
            the corresponding environment variable is not set.
    """
    if checkpoint_dir is None:
        from mole_jepa import env as env_module

        checkpoint_dir = env_module.checkpoint_dir()

    if _supabase_mode(registry_dir):
        return _sb_register(
            name,
            config,
            checkpoint_dir=checkpoint_dir,
            description=description,
            overwrite=overwrite,
        )

    rdir = _resolve_registry_dir(registry_dir)
    resolved_dir = pathlib.Path(checkpoint_dir) / _entry_hash(name, config)

    with _registry_lock(rdir):
        data = _load(rdir)
        if name in data["models"] and not overwrite:
            raise ValueError(
                f"Model {name!r} is already registered. "
                "Pass overwrite=True to replace it."
            )
        entry = RegistryEntry(
            name=name,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            description=description,
            checkpoint_dir=resolved_dir,
            config=config,
        )
        data["models"][name] = _entry_to_raw(entry)
        _save(data, rdir)

    return entry


def deregister(
    name: str,
    registry_dir: str | pathlib.Path | None = None,
) -> None:
    """Remove a model from the registry.

    This does **not** delete any checkpoint files on disk.

    Args:
        name: Registered model name to remove.
        registry_dir: Path to the registry directory.  Defaults to
            ``$REGISTRY_PATH`` (file) or ``$SUPABASE_URL`` (remote).

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if _supabase_mode(registry_dir):
        _sb_deregister(name)
        return

    rdir = _resolve_registry_dir(registry_dir)
    with _registry_lock(rdir):
        data = _load(rdir)
        if name not in data["models"]:
            raise KeyError(f"Model {name!r} not found in registry at {rdir!r}.")
        del data["models"][name]
        _save(data, rdir)


def apply_migration(
    registry_dir: str | pathlib.Path | None = None,
    up_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Apply a schema-migration function to the raw registry data under a lock.

    The entire read-transform-write cycle is protected by an exclusive
    :func:`filelock.FileLock` so concurrent callers cannot interleave.
    This is the only public entry point that grants access to the raw
    registry dict; it exists solely for ``apps/registry/migrate.py``.

    The *up_fn* receives the full registry dict (including ``schema_version``
    and ``models``) and must return the transformed dict.  It is responsible
    for bumping ``data["schema_version"]`` as appropriate.

    Not supported in Supabase mode — use the Supabase dashboard or ``psql``
    to run SQL migrations instead.

    Args:
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.
        up_fn: Pure function ``dict → dict`` that applies one or more
            schema migrations.  Called exactly once, inside the lock.

    Raises:
        ValueError: If *up_fn* is ``None``.
        NotImplementedError: If called in Supabase mode.
    """
    if up_fn is None:
        raise ValueError("up_fn is required.")
    if _supabase_mode(registry_dir):
        raise NotImplementedError(
            "apply_migration is not supported in Supabase mode. "
            "Run SQL migrations directly via the Supabase dashboard or psql."
        )
    rdir = _resolve_registry_dir(registry_dir)
    with _registry_lock(rdir):
        data = _load(rdir)
        data = up_fn(data)
        _save(data, rdir)
