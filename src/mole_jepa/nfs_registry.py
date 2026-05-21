"""NFS-backed model registry for MoLeJEPA experiments.

The registry is a JSON file at ``{registry_dir}/registry.json`` that maps
human-readable model names to their :class:`~mole_jepa.config.ModelConfig`
and explicit checkpoint directory.

Key properties
--------------
- **Stable paths**: ``checkpoint_dir`` is written once at registration and
  never recomputed.  Adding new fields to :class:`~mole_jepa.config.ModelConfig`
  does *not* change existing entries.
- **NFS-safe writes**: all mutations acquire an exclusive ``flock`` on a
  ``.lock`` sidecar before reading, modifying, and atomically replacing the
  JSON file.
- **Schema versioning**: the file carries a ``schema_version`` int.  When
  :class:`~mole_jepa.config.ModelConfig` gains a new field, bump
  :data:`CURRENT_SCHEMA_VERSION`, write a migration in
  ``apps/registry/migrations/``, and run ``apps/registry/migrate.py`` on
  the cluster.

Checkpoint directory convention
--------------------------------
By default, :func:`register` derives the checkpoint directory as::

    checkpoint_base / config.serialize()

This matches the convention used by the original hash-based system so that
:func:`import_from_registry_py` can point new entries at existing directories
without moving any data.  Pass ``checkpoint_dir`` explicitly to override.

Usage::

    from mole_jepa import nfs_registry

    # Read
    entry = nfs_registry.get_entry("vit_base_bert_jepa_frozen", registry_dir)
    cfg   = entry.config
    ckpt  = entry.checkpoint_dir

    # Write (acquires flock)
    nfs_registry.register(
        "my_experiment",
        config=ModelConfig(...),
        checkpoint_base="/scratch/vpathak/checkpoints",
        description="Lower LR, frozen ViT",
        registry_dir="/scratch/vpathak/registry",
    )
"""

import dataclasses
import datetime
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


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def get_entry(
    name: str,
    registry_dir: str | pathlib.Path,
) -> RegistryEntry:
    """Return the :class:`RegistryEntry` for *name*.

    Args:
        name: Registered model name.
        registry_dir: Path to the registry directory on NFS.

    Returns:
        The matching :class:`RegistryEntry`.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    data = _load(registry_dir)
    if name not in data["models"]:
        available = sorted(data["models"])
        raise KeyError(
            f"Model {name!r} not found in registry at {registry_dir!r}. "
            f"Registered models: {available}"
        )
    return _entry_from_raw(name, data["models"][name])


def get_config(
    name: str,
    registry_dir: str | pathlib.Path,
) -> config_module.ModelConfig:
    """Return the :class:`~mole_jepa.config.ModelConfig` for *name*.

    Args:
        name: Registered model name.
        registry_dir: Path to the registry directory on NFS.

    Returns:
        The registered :class:`~mole_jepa.config.ModelConfig`.
    """
    return get_entry(name, registry_dir).config


def get_checkpoint_dir(
    name: str,
    registry_dir: str | pathlib.Path,
) -> pathlib.Path:
    """Return the checkpoint directory path for *name*.

    Args:
        name: Registered model name.
        registry_dir: Path to the registry directory on NFS.

    Returns:
        Absolute path to the checkpoint directory.
    """
    return get_entry(name, registry_dir).checkpoint_dir


def list_entries(
    registry_dir: str | pathlib.Path,
) -> list[RegistryEntry]:
    """Return all entries in the registry, sorted by name.

    Args:
        registry_dir: Path to the registry directory on NFS.

    Returns:
        List of :class:`RegistryEntry` objects sorted by ``name``.
    """
    data = _load(registry_dir)
    return sorted(
        (_entry_from_raw(name, raw) for name, raw in data["models"].items()),
        key=lambda e: e.name,
    )


def schema_version(registry_dir: str | pathlib.Path) -> int:
    """Return the current ``schema_version`` from the registry file.

    Args:
        registry_dir: Path to the registry directory on NFS.

    Returns:
        Schema version integer (0 if the file does not exist yet).
    """
    return int(_load(registry_dir).get("schema_version", 0))


# ---------------------------------------------------------------------------
# Write API  (all mutations go through _Lock)
# ---------------------------------------------------------------------------


def register(
    name: str,
    config: config_module.ModelConfig,
    *,
    checkpoint_base: str | pathlib.Path | None = None,
    checkpoint_dir: str | pathlib.Path | None = None,
    description: str = "",
    registry_dir: str | pathlib.Path,
    overwrite: bool = False,
) -> RegistryEntry:
    """Register a model config in the NFS registry.

    Exactly one of *checkpoint_base* or *checkpoint_dir* must be provided.

    When *checkpoint_base* is given, the checkpoint directory is derived as::

        checkpoint_base / config.serialize()

    This matches the convention used throughout the codebase so that the
    path can be found even if the registry file is lost.  The derived path
    is stored verbatim in the entry and never recomputed, so it remains
    stable if :class:`~mole_jepa.config.ModelConfig` gains new fields.

    When *checkpoint_dir* is given, it is used directly (useful for pointing
    at pre-existing directories without renaming them).

    Args:
        name: Human-readable identifier (e.g. ``"vit_base_bert_jepa_frozen"``).
        config: The :class:`~mole_jepa.config.ModelConfig` to register.
        checkpoint_base: Root directory; the hash subdir is appended
            automatically.  Mutually exclusive with *checkpoint_dir*.
        checkpoint_dir: Explicit checkpoint directory.  Mutually exclusive
            with *checkpoint_base*.
        description: Optional human note stored alongside the entry.
        registry_dir: Path to the registry directory on NFS.
        overwrite: If ``True``, silently replace an entry with the same name.

    Returns:
        The newly created :class:`RegistryEntry`.

    Raises:
        ValueError: If neither or both of *checkpoint_base* / *checkpoint_dir*
            are provided, or if *name* already exists and ``overwrite=False``.
    """
    if checkpoint_dir is None and checkpoint_base is None:
        raise ValueError(
            "Provide either checkpoint_base (hash subdir is appended) "
            "or checkpoint_dir (explicit path)."
        )
    if checkpoint_dir is not None and checkpoint_base is not None:
        raise ValueError("checkpoint_base and checkpoint_dir are mutually exclusive.")

    resolved_dir: pathlib.Path
    if checkpoint_dir is not None:
        resolved_dir = pathlib.Path(checkpoint_dir)
    else:
        resolved_dir = pathlib.Path(checkpoint_base) / config.serialize()  # type: ignore[arg-type]

    with _registry_lock(registry_dir):
        data = _load(registry_dir)
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
        _save(data, registry_dir)

    return entry


def deregister(
    name: str,
    registry_dir: str | pathlib.Path,
) -> None:
    """Remove a model from the registry.

    This does **not** delete any checkpoint files on disk.

    Args:
        name: Registered model name to remove.
        registry_dir: Path to the registry directory on NFS.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    with _registry_lock(registry_dir):
        data = _load(registry_dir)
        if name not in data["models"]:
            raise KeyError(f"Model {name!r} not found in registry at {registry_dir!r}.")
        del data["models"][name]
        _save(data, registry_dir)


def apply_migration(
    registry_dir: str | pathlib.Path,
    up_fn: Callable[[dict[str, Any]], dict[str, Any]],
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
        registry_dir: Path to the registry directory on NFS.
        up_fn: Pure function ``dict → dict`` that applies one or more
            schema migrations.  Called exactly once, inside the lock.
    """
    with _registry_lock(registry_dir):
        data = _load(registry_dir)
        data = up_fn(data)
        _save(data, registry_dir)
