"""Model persistence and loading for MoLeJEPA.

All public save/load functions accept a **model name** and resolve the
checkpoint directory from the NFS registry.  Registry and checkpoint
directories default to the ``$REGISTRY_PATH`` and ``$CHECKPOINT_DIR``
environment variables (with ``.env`` fallback via :mod:`mole_jepa.env`).

Artifacts
---------
Two distinct artifacts live inside each model's checkpoint directory:

``model.pt``
    Model state dict only.  Written by :func:`save_model` and read by
    :func:`load_model` / :func:`load_model_weights`.  This is the permanent
    artifact that outlives a training run.

``config.json``
    Serialised :class:`~mole_jepa.config.ModelConfig`.  Written once by
    :func:`save_model` alongside ``model.pt``.

``train_state.pt``
    Epoch number and optimiser state dict.  Written by
    :func:`save_train_state` and read by :func:`load_train_state`.  This is
    ephemeral — it exists only while training is in progress and is removed
    by :func:`cleanup_train_state` once the run completes successfully.

Low-level utilities
-------------------
:func:`model_dir` and :func:`list_models` do not require a registry and
remain available for notebook exploration of a checkpoint root directory.
"""

import dataclasses
import json
import os
import pathlib
from typing import Any

import torch

from mole_jepa import config as config_module
from mole_jepa import factory, models

_MODEL_FILE = "model.pt"
_CONFIG_FILE = "config.json"
_TRAIN_STATE_FILE = "train_state.pt"


@dataclasses.dataclass
class ModelInfo:
    """Metadata about a saved model on disk.

    Attributes:
        path: Directory containing ``model.pt`` and ``config.json``.
        config_hash: SHA-256 hex string identifying the config (directory name).
        config: Deserialised :class:`~mole_jepa.config.ModelConfig`.
    """

    path: pathlib.Path
    config_hash: str
    config: config_module.ModelConfig


# ── internal path helpers ─────────────────────────────────────────────────────


def model_dir(
    root: str | pathlib.Path,
    config: config_module.ModelConfig,
) -> pathlib.Path:
    """Return the checkpoint subdirectory for *config* inside *root*.

    This is a low-level utility for notebook exploration.  Training code
    should use the name-based API (:func:`save_model`, :func:`load_model`,
    etc.) which resolve *root* from the NFS registry automatically.

    Args:
        root: Root directory holding all model subdirectories.
        config: The model config whose hash names the subdirectory.

    Returns:
        ``root / config.serialize()``.
    """
    return pathlib.Path(root) / config.serialize()


def _checkpoint_dir_for(
    name: str,
    registry_dir: str | pathlib.Path | None,
) -> pathlib.Path:
    """Look up *name* in the registry and return its checkpoint directory."""
    from mole_jepa import nfs_registry

    entry = nfs_registry.get_entry(name, registry_dir)
    return entry.checkpoint_dir


def _save_model_to(
    model: models.MoLeJEPA,
    config: config_module.ModelConfig,
    checkpoint_dir: pathlib.Path,
) -> None:
    """Write ``model.pt`` (and ``config.json`` on first save) atomically."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = checkpoint_dir / _CONFIG_FILE
    if not config_path.exists():
        config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2))
    tmp = checkpoint_dir / (_MODEL_FILE + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, checkpoint_dir / _MODEL_FILE)


def _load_model_weights_from(
    model: models.MoLeJEPA,
    checkpoint_dir: pathlib.Path,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load ``model.pt`` from *checkpoint_dir* into *model* in-place."""
    model_path = checkpoint_dir / _MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run training first, or check that the model name is correct."
        )
    state_dict: dict[str, Any] = torch.load(
        model_path, map_location=map_location, weights_only=True
    )
    model.load_state_dict(state_dict)


def _save_train_state_to(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: pathlib.Path,
) -> None:
    """Write ``train_state.pt`` atomically to *checkpoint_dir*."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    tmp = checkpoint_dir / (_TRAIN_STATE_FILE + ".tmp")
    torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()}, tmp)
    os.replace(tmp, checkpoint_dir / _TRAIN_STATE_FILE)


def _load_train_state_from(
    checkpoint_dir: pathlib.Path,
) -> tuple[dict[str, Any], int] | None:
    """Read ``train_state.pt`` from *checkpoint_dir*, or return ``None``."""
    path = checkpoint_dir / _TRAIN_STATE_FILE
    if not path.exists():
        return None
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return state["optimizer_state_dict"], state["epoch"]


# ── name-based public API ─────────────────────────────────────────────────────


def save_model(
    model: models.MoLeJEPA,
    name: str,
    *,
    registry_dir: str | pathlib.Path | None = None,
) -> None:
    """Save model weights (and config) to the checkpoint directory for *name*.

    Writes ``model.pt`` atomically via a temp-file rename, and ``config.json``
    on the first call (never overwritten on subsequent saves).  The checkpoint
    directory is resolved from the NFS registry entry for *name*.

    Args:
        model: The model whose weights to persist.
        name: Registered model name (looked up in the NFS registry).
        registry_dir: Registry directory.  Defaults to ``$REGISTRY_PATH``.
    """
    from mole_jepa import nfs_registry

    entry = nfs_registry.get_entry(name, registry_dir)
    _save_model_to(model, entry.config, entry.checkpoint_dir)


def load_model(
    name: str,
    *,
    registry_dir: str | pathlib.Path | None = None,
    map_location: str | torch.device = "cpu",
) -> models.MoLeJEPA:
    """Build a model from its registered config and load its saved weights.

    Constructs the model via :func:`~mole_jepa.factory.build`, then loads
    the weights from the registry entry's checkpoint directory.  Intended
    for notebook use where no model has been built yet.  In a training loop
    where the model is already constructed, use :func:`load_model_weights`
    to avoid the redundant ``from_pretrained`` call.

    Args:
        name: Registered model name.
        registry_dir: Registry directory.  Defaults to ``$REGISTRY_PATH``.
        map_location: Device to load tensors onto.  Defaults to ``"cpu"``.

    Returns:
        The loaded :class:`~mole_jepa.models.MoLeJEPA` model.

    Raises:
        FileNotFoundError: If no ``model.pt`` exists for *name*.
    """
    from mole_jepa import nfs_registry

    entry = nfs_registry.get_entry(name, registry_dir)
    model, _ = factory.build(entry.config)
    _load_model_weights_from(model, entry.checkpoint_dir, map_location=map_location)
    return model


def load_model_weights(
    model: models.MoLeJEPA,
    name: str,
    *,
    registry_dir: str | pathlib.Path | None = None,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load saved weights into an already-constructed model in-place.

    Args:
        model: An already-constructed :class:`~mole_jepa.models.MoLeJEPA`
            instance to load weights into.
        name: Registered model name.
        registry_dir: Registry directory.  Defaults to ``$REGISTRY_PATH``.
        map_location: Device to load tensors onto.

    Raises:
        FileNotFoundError: If no ``model.pt`` exists for *name*.
    """
    ckpt = _checkpoint_dir_for(name, registry_dir)
    _load_model_weights_from(model, ckpt, map_location=map_location)


def has_train_state(
    name: str,
    *,
    registry_dir: str | pathlib.Path | None = None,
) -> bool:
    """Return ``True`` if a resumable train state exists for *name*.

    Args:
        name: Registered model name.
        registry_dir: Registry directory.  Defaults to ``$REGISTRY_PATH``.

    Returns:
        ``True`` if ``train_state.pt`` is present in the checkpoint directory.
    """
    ckpt = _checkpoint_dir_for(name, registry_dir)
    return (ckpt / _TRAIN_STATE_FILE).exists()


def save_train_state(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    name: str,
    *,
    registry_dir: str | pathlib.Path | None = None,
) -> None:
    """Save optimizer state and epoch for training resumption.

    Writes ``train_state.pt`` atomically.  Remove it with
    :func:`cleanup_train_state` once training completes successfully.

    Args:
        optimizer: The optimizer whose state to persist.
        epoch: Most recently completed epoch (0-indexed).
        name: Registered model name.
        registry_dir: Registry directory.  Defaults to ``$REGISTRY_PATH``.
    """
    ckpt = _checkpoint_dir_for(name, registry_dir)
    _save_train_state_to(optimizer, epoch, ckpt)


def load_train_state(
    name: str,
    *,
    registry_dir: str | pathlib.Path | None = None,
) -> tuple[dict[str, Any], int] | None:
    """Load optimizer state and epoch from a saved train state.

    Args:
        name: Registered model name.
        registry_dir: Registry directory.  Defaults to ``$REGISTRY_PATH``.

    Returns:
        ``(optimizer_state_dict, epoch)`` if a train state file exists,
        ``None`` if no train state is present (fresh run).
    """
    ckpt = _checkpoint_dir_for(name, registry_dir)
    return _load_train_state_from(ckpt)


def cleanup_train_state(
    name: str,
    *,
    registry_dir: str | pathlib.Path | None = None,
) -> None:
    """Remove the ephemeral train state file for *name*.

    Safe to call even if no train state exists.

    Args:
        name: Registered model name.
        registry_dir: Registry directory.  Defaults to ``$REGISTRY_PATH``.
    """
    ckpt = _checkpoint_dir_for(name, registry_dir)
    (ckpt / _TRAIN_STATE_FILE).unlink(missing_ok=True)


# ── exploration utility ───────────────────────────────────────────────────────


def list_models(
    root: str | pathlib.Path,
    **filters: object,
) -> list[ModelInfo]:
    """Return metadata for every saved model found under *root*.

    Scans *root* for subdirectories that contain both ``model.pt`` and
    ``config.json``.  Does not require an NFS registry — useful for
    exploring a checkpoint root directory in a notebook.

    Results are sorted by ``model.pt`` modification time, most recent first.

    Args:
        root: Root directory holding all model subdirectories.
        **filters: Optional equality filters on
            :class:`~mole_jepa.config.ModelConfig` fields.  For example::

                list_models(root, embed_dim=256, contrastive=True)

            returns only models whose config has ``embed_dim == 256`` and
            ``contrastive == True``.

    Returns:
        List of :class:`ModelInfo` sorted by ``model.pt`` mtime descending.

    Raises:
        ValueError: If any filter key is not a valid ``ModelConfig`` field.
    """
    valid_fields = {f.name for f in dataclasses.fields(config_module.ModelConfig)}
    unknown = set(filters) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown ModelConfig field(s): {sorted(unknown)}. "
            f"Valid fields: {sorted(valid_fields)}"
        )

    results: list[ModelInfo] = []
    root_path = pathlib.Path(root)
    if not root_path.exists():
        return results

    for subdir in root_path.iterdir():
        if not subdir.is_dir():
            continue
        model_path = subdir / _MODEL_FILE
        config_path = subdir / _CONFIG_FILE
        if not model_path.exists() or not config_path.exists():
            continue

        cfg = config_module.ModelConfig(**json.loads(config_path.read_text()))

        if filters and not all(getattr(cfg, k) == v for k, v in filters.items()):
            continue

        results.append(ModelInfo(path=subdir, config_hash=subdir.name, config=cfg))

    return sorted(
        results,
        key=lambda m: (m.path / _MODEL_FILE).stat().st_mtime,
        reverse=True,
    )
