"""Model persistence, loading, and registry for MoLeJEPA.

Artifacts
---------
Two distinct artifacts live inside each model subdirectory:

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


# ── paths ─────────────────────────────────────────────────────────────────────


def model_dir(
    root: str | pathlib.Path,
    config: config_module.ModelConfig,
) -> pathlib.Path:
    """Return the subdirectory for a given config inside ``root``.

    Args:
        root: Root directory holding all model subdirectories.
        config: The model config whose hash names the subdirectory.

    Returns:
        ``root / config.serialize()``.
    """
    return pathlib.Path(root) / config.serialize()


# ── model artifact ────────────────────────────────────────────────────────────


def save_model(
    model: models.MoLeJEPA,
    config: config_module.ModelConfig,
    root: str | pathlib.Path,
) -> None:
    """Save model weights and config atomically.

    Writes ``model.pt`` (state dict only) and ``config.json`` to the
    config-specific subdirectory of ``root``.  ``model.pt`` is first written
    to a ``.tmp`` file and then renamed into place so that a mid-write crash
    cannot corrupt an existing save.

    Args:
        model: The model whose weights to persist.
        config: The config used to build the model.
        root: Root directory holding all model subdirectories.
    """
    loc = model_dir(root, config)
    os.makedirs(loc, exist_ok=True)

    config_path = loc / _CONFIG_FILE
    if not config_path.exists():
        config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2))

    tmp = loc / (_MODEL_FILE + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, loc / _MODEL_FILE)


def load_model(
    config: config_module.ModelConfig,
    root: str | pathlib.Path,
    *,
    map_location: str | torch.device = "cpu",
) -> models.MoLeJEPA:
    """Build a model and load its saved weights.

    Constructs the model via :func:`~mole_jepa.factory.build` and loads the
    saved state dict.  Intended for notebook use where no model has been
    built yet.  If you have already constructed the model (e.g. in a
    training loop), prefer :func:`load_model_weights` to avoid a redundant
    ``from_pretrained`` call.

    Args:
        config: Config identifying which model to load.
        root: Root directory holding all model subdirectories.
        map_location: Device to load tensors onto.  Defaults to ``"cpu"``
            so notebooks work without a GPU.

    Returns:
        The loaded model, ready for inference.

    Raises:
        FileNotFoundError: If no saved model exists for ``config``.
    """
    model, _ = factory.build(config)
    load_model_weights(model, config, root, map_location=map_location)
    return model


def load_model_weights(
    model: models.MoLeJEPA,
    config: config_module.ModelConfig,
    root: str | pathlib.Path,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load saved weights into an already-constructed model in-place.

    Args:
        model: An already-constructed :class:`~mole_jepa.models.MoLeJEPA`
            instance to load weights into.
        config: Config identifying which saved model to load from.
        root: Root directory holding all model subdirectories.
        map_location: Device to load tensors onto.

    Raises:
        FileNotFoundError: If no saved model exists for ``config``.
    """
    loc = model_dir(root, config)
    model_path = loc / _MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run training first, or check the config matches the saved run."
        )
    state_dict: dict[str, Any] = torch.load(
        model_path, map_location=map_location, weights_only=True
    )
    model.load_state_dict(state_dict)


# ── train state (ephemeral) ───────────────────────────────────────────────────


def save_train_state(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: config_module.ModelConfig,
    root: str | pathlib.Path,
) -> None:
    """Save optimizer state and epoch for training resumption.

    Writes ``train_state.pt`` atomically.  This file is ephemeral — remove
    it with :func:`cleanup_train_state` once training completes successfully.

    Args:
        optimizer: The optimizer whose state to persist.
        epoch: Most recently completed epoch (0-indexed).
        config: Config identifying which model this train state belongs to.
        root: Root directory holding all model subdirectories.
    """
    loc = model_dir(root, config)
    os.makedirs(loc, exist_ok=True)

    tmp = loc / (_TRAIN_STATE_FILE + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, loc / _TRAIN_STATE_FILE)


def load_train_state(
    config: config_module.ModelConfig,
    root: str | pathlib.Path,
) -> tuple[dict[str, Any], int] | None:
    """Load optimizer state and epoch from a saved train state.

    Args:
        config: Config identifying which model's train state to load.
        root: Root directory holding all model subdirectories.

    Returns:
        ``(optimizer_state_dict, epoch)`` if a train state file exists,
        ``None`` if no train state is present (fresh run).
    """
    path = model_dir(root, config) / _TRAIN_STATE_FILE
    if not path.exists():
        return None
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return state["optimizer_state_dict"], state["epoch"]


def has_train_state(
    config: config_module.ModelConfig,
    root: str | pathlib.Path,
) -> bool:
    """Return ``True`` if a train state file exists for ``config``.

    Args:
        config: Config identifying which model's train state to check.
        root: Root directory holding all model subdirectories.

    Returns:
        ``True`` if ``train_state.pt`` exists, ``False`` otherwise.
    """
    return (model_dir(root, config) / _TRAIN_STATE_FILE).exists()


def cleanup_train_state(
    config: config_module.ModelConfig,
    root: str | pathlib.Path,
) -> None:
    """Remove the ephemeral train state file for ``config``.

    Safe to call even if no train state exists.

    Args:
        config: Config identifying which model's train state to remove.
        root: Root directory holding all model subdirectories.
    """
    (model_dir(root, config) / _TRAIN_STATE_FILE).unlink(missing_ok=True)


# ── explicit-path variants ────────────────────────────────────────────────────
# These functions accept a direct ``checkpoint_dir`` (the exact directory
# containing model.pt / train_state.pt) instead of ``(root, config)``.
# They are used by the training script when the checkpoint path is resolved
# from the NFS registry rather than derived from the config hash at runtime.
# The original ``(root, config)`` functions remain available for notebook use.


def save_model_at(
    model: models.MoLeJEPA,
    checkpoint_dir: str | pathlib.Path,
    config: config_module.ModelConfig | None = None,
) -> None:
    """Save model weights (and optionally config) to an explicit directory.

    Writes ``model.pt`` atomically via a temp-file rename.  If *config* is
    provided and ``config.json`` does not yet exist, it is written alongside.

    Args:
        model: The model whose weights to persist.
        checkpoint_dir: Directory to write into (created if absent).
        config: If given, serialised to ``config.json`` on first save.
    """
    loc = pathlib.Path(checkpoint_dir)
    os.makedirs(loc, exist_ok=True)
    if config is not None:
        config_path = loc / _CONFIG_FILE
        if not config_path.exists():
            config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2))
    tmp = loc / (_MODEL_FILE + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, loc / _MODEL_FILE)


def load_model_weights_at(
    model: models.MoLeJEPA,
    checkpoint_dir: str | pathlib.Path,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load saved weights into an already-constructed model in-place.

    Args:
        model: Model instance to load weights into.
        checkpoint_dir: Directory containing ``model.pt``.
        map_location: Device to load tensors onto.

    Raises:
        FileNotFoundError: If ``model.pt`` does not exist.
    """
    model_path = pathlib.Path(checkpoint_dir) / _MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run training first, or check the checkpoint directory."
        )
    state_dict: dict[str, Any] = torch.load(
        model_path, map_location=map_location, weights_only=True
    )
    model.load_state_dict(state_dict)


def save_train_state_at(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: str | pathlib.Path,
) -> None:
    """Save optimizer state and epoch to an explicit directory (atomic).

    Args:
        optimizer: Optimizer whose state to persist.
        epoch: Most recently completed epoch (0-indexed).
        checkpoint_dir: Directory to write into (created if absent).
    """
    loc = pathlib.Path(checkpoint_dir)
    os.makedirs(loc, exist_ok=True)
    tmp = loc / (_TRAIN_STATE_FILE + ".tmp")
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        tmp,
    )
    os.replace(tmp, loc / _TRAIN_STATE_FILE)


def load_train_state_at(
    checkpoint_dir: str | pathlib.Path,
) -> tuple[dict[str, Any], int] | None:
    """Load optimizer state and epoch from an explicit directory.

    Args:
        checkpoint_dir: Directory containing ``train_state.pt``.

    Returns:
        ``(optimizer_state_dict, epoch)`` if a train state file exists,
        ``None`` otherwise.
    """
    path = pathlib.Path(checkpoint_dir) / _TRAIN_STATE_FILE
    if not path.exists():
        return None
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return state["optimizer_state_dict"], state["epoch"]


def has_train_state_at(checkpoint_dir: str | pathlib.Path) -> bool:
    """Return ``True`` if ``train_state.pt`` exists in *checkpoint_dir*.

    Args:
        checkpoint_dir: Directory to check.
    """
    return (pathlib.Path(checkpoint_dir) / _TRAIN_STATE_FILE).exists()


def cleanup_train_state_at(checkpoint_dir: str | pathlib.Path) -> None:
    """Remove ``train_state.pt`` from *checkpoint_dir* (safe if absent).

    Args:
        checkpoint_dir: Directory whose train state to remove.
    """
    (pathlib.Path(checkpoint_dir) / _TRAIN_STATE_FILE).unlink(missing_ok=True)


# ── registry ──────────────────────────────────────────────────────────────────


def list_models(
    root: str | pathlib.Path,
    **filters: object,
) -> list[ModelInfo]:
    """Return metadata for every saved model found under ``root``.

    A subdirectory is treated as a valid model entry only if it contains
    *both* ``model.pt`` and ``config.json``.  Directories missing either
    file (e.g. an in-progress or corrupt save) are silently skipped.

    Results are sorted by ``model.pt`` modification time, most recent first,
    which is usually the most useful order in a notebook.

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
