"""NFS-backed model registry for MoLeJEPA experiments.

The registry is a JSON file at ``{registry_dir}/registry.json`` that maps
human-readable model names to their :class:`~mole_jepa.config.ModelConfig`
and the checkpoint directory derived from the config hash.

Key properties
--------------
- **Stable paths**: ``checkpoint_dir`` is written once at registration and
  never recomputed.  Adding new fields to :class:`~mole_jepa.config.ModelConfig`
  does *not* change existing entries.
- **Single path convention**: every model's checkpoints live at
  ``checkpoint_dir / config.serialize()``.  There is no concept of an
  "explicit" path — the hash subdir is always derived automatically.
- **NFS-safe writes**: all mutations acquire an exclusive ``flock`` on a
  ``.lock`` sidecar before reading, modifying, and atomically replacing the
  JSON file.
- **Schema versioning**: the file carries a ``schema_version`` int.  When
  :class:`~mole_jepa.config.ModelConfig` gains a new field, bump
  :data:`CURRENT_SCHEMA_VERSION`, write a migration in
  ``apps/registry/migrations/``, and run ``apps/registry/migrate.py`` on
  the cluster.
- **Env defaults**: ``registry_dir`` and ``checkpoint_dir`` parameters
  default to ``None``, which falls back to :func:`mole_jepa.env.registry_dir`
  and :func:`mole_jepa.env.checkpoint_dir` respectively.

Checkpoint directory convention
--------------------------------
:func:`register` derives the checkpoint directory as::

    checkpoint_dir / sha256(name + "|" + config.serialize())

where ``checkpoint_dir`` is the root under which all model checkpoints live
(e.g. ``/scratch/vpathak/checkpoints``).  Including the model name in the
hash guarantees that two entries with identical configs but different names
always land in separate directories.  The derived path is stored verbatim in
the registry entry so it remains stable even if
:class:`~mole_jepa.config.ModelConfig` gains new fields later.

Usage::

    from mole_jepa import registry

    # Read (registry_dir defaults to $REGISTRY_PATH)
    entry = registry.get_entry("vit_base_bert_jepa_frozen")
    cfg   = entry.config
    ckpt  = entry.checkpoint_dir

    # Write (acquires flock)
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
    """One row in the NFS registry.

    Attributes:
        name: Human-readable identifier used to look up the entry.
        created_at: ISO 8601 timestamp of initial registration.
        description: Free-text note (empty string if omitted).
        checkpoint_dir: Absolute path to the directory that holds
            ``model.pt``, ``config.json``, and ``train_state.pt``.
            Always derived as ``<checkpoint_root> / config.serialize()``.
        config: The :class:`~mole_jepa.config.ModelConfig` registered for
            this model.
    """

    name: str
    created_at: str
    description: str
    checkpoint_dir: pathlib.Path
    config: config_module.ModelConfig


# ---------------------------------------------------------------------------
# Internal I/O helpers
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


def _resolve_registry_dir(
    registry_dir: str | pathlib.Path | None,
) -> str | pathlib.Path:
    """Return *registry_dir*, falling back to ``$REGISTRY_PATH`` if ``None``."""
    if registry_dir is not None:
        return registry_dir
    from mole_jepa import env as env_module  # late import avoids circular deps

    return env_module.registry_dir()


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
            ``$REGISTRY_PATH`` (via :func:`mole_jepa.env.registry_dir`).

    Returns:
        The matching :class:`RegistryEntry`.

    Raises:
        KeyError: If *name* is not in the registry.
        RuntimeError: If *registry_dir* is ``None`` and ``$REGISTRY_PATH``
            is not set.
    """
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
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.

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
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.

    Returns:
        Absolute path to the checkpoint directory (``<root> / config_hash``).
    """
    return get_entry(name, registry_dir).checkpoint_dir


def list_entries(
    registry_dir: str | pathlib.Path | None = None,
) -> list[RegistryEntry]:
    """Return all entries in the registry, sorted by name.

    Args:
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.

    Returns:
        List of :class:`RegistryEntry` objects sorted by ``name``.
    """
    rdir = _resolve_registry_dir(registry_dir)
    data = _load(rdir)
    return sorted(
        (_entry_from_raw(name, raw) for name, raw in data["models"].items()),
        key=lambda e: e.name,
    )


def schema_version(registry_dir: str | pathlib.Path | None = None) -> int:
    """Return the current ``schema_version`` from the registry file.

    Args:
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.

    Returns:
        Schema version integer (0 if the file does not exist yet).
    """
    rdir = _resolve_registry_dir(registry_dir)
    return int(_load(rdir).get("schema_version", 0))


# ---------------------------------------------------------------------------
# Write API  (all mutations go through the flock)
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
    """Register a model config in the NFS registry.

    The checkpoint directory is derived as::

        checkpoint_dir / sha256(name + "|" + config.serialize())

    Including the model name in the hash guarantees that two entries with
    identical configs but different names (e.g. a data-scaling ablation)
    always land in separate directories.  The derived path is stored
    verbatim in the registry entry so it never changes even if new fields
    are added to :class:`~mole_jepa.config.ModelConfig`.

    Args:
        name: Human-readable identifier (e.g. ``"vit_base_bert_jepa_frozen"``).
        config: The :class:`~mole_jepa.config.ModelConfig` to register.
        checkpoint_dir: Root directory under which all model checkpoints live
            (e.g. ``/scratch/vpathak/checkpoints``).  The hash-named subdir
            is appended automatically.  Defaults to ``$CHECKPOINT_DIR`` (via
            :func:`mole_jepa.env.checkpoint_dir`).
        description: Optional human note stored alongside the entry.
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.
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
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.

    Raises:
        KeyError: If *name* is not in the registry.
    """
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

    Args:
        registry_dir: Path to the registry directory on NFS.  Defaults to
            ``$REGISTRY_PATH``.
        up_fn: Pure function ``dict → dict`` that applies one or more
            schema migrations.  Called exactly once, inside the lock.
    """
    if up_fn is None:
        raise ValueError("up_fn is required.")
    rdir = _resolve_registry_dir(registry_dir)
    with _registry_lock(rdir):
        data = _load(rdir)
        data = up_fn(data)
        _save(data, rdir)
