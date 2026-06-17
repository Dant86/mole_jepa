"""Model persistence and loading for MoLeJEPA.

All public save/load functions accept a **model name** and resolve the
checkpoint directory from the Supabase registry entry for that name.

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
import re
import shutil
from typing import Any

import torch

from mole_jepa import config as config_module
from mole_jepa import factory, models

_MODEL_FILE = "model.pt"
_CONFIG_FILE = "config.json"
_TRAIN_STATE_FILE = "train_state.pt"
_CHECKPOINT_FILE = "checkpoint.pt"
_EPOCH_CKPT_RE_V2 = re.compile(r"^checkpoint_(\d{4})\.pt$")


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


def _checkpoint_dir_for(name: str) -> pathlib.Path:
    """Look up *name* in the registry and return its checkpoint directory."""
    from mole_jepa import registry

    return registry.get_entry(name).checkpoint_dir


_EPOCH_CKPT_RE = re.compile(r"^model_(\d{4})\.pt$")


def _prune_epoch_checkpoints(checkpoint_dir: pathlib.Path, keep_last: int) -> None:
    """Delete the oldest epoch-stamped checkpoints, keeping *keep_last*."""
    epoch_files = sorted(
        (int(m.group(1)), f)
        for f in checkpoint_dir.iterdir()
        if (m := _EPOCH_CKPT_RE.match(f.name))
    )
    for _, path in epoch_files[:-keep_last]:
        path.unlink(missing_ok=True)


def _prune_combined_checkpoints(checkpoint_dir: pathlib.Path, keep_last: int) -> None:
    """Delete the oldest combined epoch-stamped checkpoints, keeping *keep_last*."""
    epoch_files = sorted(
        (int(m.group(1)), f)
        for f in checkpoint_dir.iterdir()
        if (m := _EPOCH_CKPT_RE_V2.match(f.name))
    )
    for _, path in epoch_files[:-keep_last]:
        path.unlink(missing_ok=True)


def _save_model_to(
    model: models.MoLeJEPA,
    config: config_module.ModelConfig,
    checkpoint_dir: pathlib.Path,
    *,
    epoch: int | None = None,
    keep_last: int = 2,
) -> None:
    """Write ``model.pt`` (and ``config.json`` on first save) atomically.

    If *epoch* is provided, the current ``model.pt`` is copied to
    ``model_{epoch:04d}.pt`` before being overwritten, giving a rolling
    backup of the last *keep_last* completed epochs.  This ensures a
    preemption mid-save never leaves the only checkpoint corrupted.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = checkpoint_dir / _CONFIG_FILE
    if not config_path.exists():
        config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2))
    current = checkpoint_dir / _MODEL_FILE
    if epoch is not None and current.exists():
        shutil.copy2(current, checkpoint_dir / f"model_{epoch:04d}.pt")
        _prune_epoch_checkpoints(checkpoint_dir, keep_last)
    tmp = checkpoint_dir / (_MODEL_FILE + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, current)


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
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Write ``train_state.pt`` atomically to *checkpoint_dir*."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    tmp = checkpoint_dir / (_TRAIN_STATE_FILE + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        },
        tmp,
    )
    os.replace(tmp, checkpoint_dir / _TRAIN_STATE_FILE)


def _load_train_state_from(
    checkpoint_dir: pathlib.Path,
) -> tuple[dict[str, Any], int, dict[str, Any] | None] | None:
    """Read ``train_state.pt`` from *checkpoint_dir*, or return ``None``."""
    path = checkpoint_dir / _TRAIN_STATE_FILE
    if not path.exists():
        return None
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return (
        state["optimizer_state_dict"],
        state["epoch"],
        state.get("scheduler_state_dict"),
    )


def _save_checkpoint_to(
    model: models.MoLeJEPA,
    config: config_module.ModelConfig,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: pathlib.Path,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    keep_last: int = 3,
    local_tmpdir: str | pathlib.Path | None = None,
) -> None:
    """Write a combined checkpoint (weights + optimizer + scheduler + epoch) atomically.

    Writes to *local_tmpdir* first (fast local NVMe) if provided, verifies
    the file loads cleanly, then copies to NFS and does a single atomic rename.
    This eliminates the race where model.pt and train_state.pt are written
    separately and SIGKILL hits between them, producing mismatched state.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = checkpoint_dir / _CONFIG_FILE
    if not config_path.exists():
        config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2))

    current = checkpoint_dir / _CHECKPOINT_FILE
    if current.exists():
        shutil.copy2(current, checkpoint_dir / f"checkpoint_{epoch:04d}.pt")
        _prune_combined_checkpoints(checkpoint_dir, keep_last)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    nfs_tmp = checkpoint_dir / (_CHECKPOINT_FILE + ".tmp")
    if local_tmpdir and os.path.isdir(local_tmpdir):
        local_tmp = pathlib.Path(local_tmpdir) / (_CHECKPOINT_FILE + ".tmp")
        torch.save(payload, local_tmp)
        try:
            torch.load(local_tmp, map_location="cpu", weights_only=False)
        except Exception as exc:
            local_tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Checkpoint verification failed: {exc}") from exc
        shutil.copy2(local_tmp, nfs_tmp)
        local_tmp.unlink(missing_ok=True)
    else:
        torch.save(payload, nfs_tmp)
        try:
            torch.load(nfs_tmp, map_location="cpu", weights_only=False)
        except Exception as exc:
            nfs_tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Checkpoint verification failed: {exc}") from exc

    os.replace(nfs_tmp, current)


def _load_checkpoint_from(
    model: models.MoLeJEPA,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: pathlib.Path,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> int | None:
    """Load a combined checkpoint into *model*, *optimizer*, and *scheduler*.

    Returns the saved epoch, or ``None`` if no combined checkpoint exists.
    """
    path = checkpoint_dir / _CHECKPOINT_FILE
    if not path.exists():
        return None
    state: dict[str, Any] = torch.load(
        path, map_location=map_location, weights_only=False
    )
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return int(state["epoch"])


# ── name-based public API ─────────────────────────────────────────────────────


def save_model(
    model: models.MoLeJEPA,
    name: str,
    *,
    epoch: int | None = None,
    keep_last: int = 2,
) -> None:
    """Save model weights (and config) to the checkpoint directory for *name*.

    Writes ``model.pt`` atomically via a temp-file rename, and ``config.json``
    on the first call (never overwritten on subsequent saves).  The checkpoint
    directory is resolved from the Supabase registry entry for *name*.

    When *epoch* is provided, the previous ``model.pt`` is copied to
    ``model_{epoch:04d}.pt`` before being overwritten, maintaining a rolling
    backup of the last *keep_last* completed epochs.

    Args:
        model: The model whose weights to persist.
        name: Registered model name.
        epoch: Completed epoch number used to name the backup checkpoint.
            Pass ``None`` (default) to skip rotation (e.g. final save).
        keep_last: Number of epoch-stamped backups to retain.  Defaults to 2.
    """
    from mole_jepa import registry

    entry = registry.get_entry(name)
    _save_model_to(
        model, entry.config, entry.checkpoint_dir, epoch=epoch, keep_last=keep_last
    )


def load_model(
    name: str,
    *,
    map_location: str | torch.device = "cpu",
    attn_implementation: str | None = None,
) -> models.MoLeJEPA:
    """Build a model from its registered config and load its saved weights.

    Constructs the model via :func:`~mole_jepa.factory.build`, then loads
    the weights from the registry entry's checkpoint directory.  Intended
    for notebook use where no model has been built yet.  In a training loop
    where the model is already constructed, use :func:`load_model_weights`
    to avoid the redundant ``from_pretrained`` call.

    Args:
        name: Registered model name.
        map_location: Device to load tensors onto.  Defaults to ``"cpu"``.
        attn_implementation: Override the attention backend from the registered
            config.  Useful for inference where ``"sdpa"`` is preferred over
            ``"flash_attention_2"`` to avoid Triton JIT requirements.

    Returns:
        The loaded :class:`~mole_jepa.models.MoLeJEPA` model.

    Raises:
        FileNotFoundError: If no ``model.pt`` exists for *name*.
    """
    import dataclasses

    from mole_jepa import registry

    entry = registry.get_entry(name)
    cfg = (
        dataclasses.replace(entry.config, attn_implementation=attn_implementation)
        if attn_implementation is not None
        else entry.config
    )
    model, _ = factory.build(cfg)
    _load_model_weights_from(model, entry.checkpoint_dir, map_location=map_location)
    return model


def load_model_weights(
    model: models.MoLeJEPA,
    name: str,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load saved weights into an already-constructed model in-place.

    Args:
        model: An already-constructed :class:`~mole_jepa.models.MoLeJEPA`
            instance to load weights into.
        name: Registered model name.
        map_location: Device to load tensors onto.

    Raises:
        FileNotFoundError: If no ``model.pt`` exists for *name*.
    """
    ckpt = _checkpoint_dir_for(name)
    _load_model_weights_from(model, ckpt, map_location=map_location)


def save_checkpoint(
    model: models.MoLeJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    name: str,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    keep_last: int = 3,
    local_tmpdir: str | pathlib.Path | None = None,
) -> None:
    """Save a combined checkpoint (weights + optimizer + scheduler + epoch) atomically.

    Uses a single file so that model weights and optimizer state are always
    in sync — the two-file race (where SIGKILL between save_model and
    save_train_state leaves mismatched state on resume) cannot occur.

    When *local_tmpdir* is provided (e.g. ``os.environ.get("TMPDIR")``), the
    checkpoint is written to fast local storage first, verified, then copied
    to the NFS checkpoint directory and renamed atomically.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer whose state to persist.
        epoch: Most recently completed (or in-progress) epoch number.
        name: Registered model name.
        scheduler: Optional LR scheduler to persist.
        keep_last: Number of epoch-stamped backups to retain.
        local_tmpdir: Optional path to fast local scratch space (e.g. $TMPDIR).
    """
    from mole_jepa import registry

    entry = registry.get_entry(name)
    _save_checkpoint_to(
        model,
        entry.config,
        optimizer,
        epoch,
        entry.checkpoint_dir,
        scheduler=scheduler,
        keep_last=keep_last,
        local_tmpdir=local_tmpdir,
    )


def load_checkpoint(
    model: models.MoLeJEPA,
    optimizer: torch.optim.Optimizer,
    name: str,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> int | None:
    """Load a combined checkpoint into *model*, *optimizer*, and *scheduler*.

    Falls back to the legacy two-file format (``model.pt`` + ``train_state.pt``)
    when no combined ``checkpoint.pt`` is present, enabling in-place upgrades of
    existing runs.

    Args:
        model: An already-constructed model to load weights into.
        optimizer: The optimizer to restore.
        name: Registered model name.
        scheduler: Optional LR scheduler to restore.
        map_location: Device to load tensors onto.

    Returns:
        The saved epoch number, or ``None`` if no resumable checkpoint exists.
    """
    ckpt = _checkpoint_dir_for(name)
    epoch = _load_checkpoint_from(
        model, optimizer, ckpt, scheduler=scheduler, map_location=map_location
    )
    if epoch is not None:
        return epoch
    # Legacy fallback: separate model.pt + train_state.pt written by old code.
    if not (ckpt / _MODEL_FILE).exists():
        return None
    _load_model_weights_from(model, ckpt, map_location=map_location)
    train_state = _load_train_state_from(ckpt)
    if train_state is None:
        return None
    opt_state, saved_epoch, sched_state = train_state
    optimizer.load_state_dict(opt_state)
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)
    return saved_epoch


def has_train_state(name: str) -> bool:
    """Return ``True`` if a resumable train state exists for *name*.

    Args:
        name: Registered model name.

    Returns:
        ``True`` if ``checkpoint.pt`` or ``train_state.pt`` is present.
    """
    ckpt = _checkpoint_dir_for(name)
    return (ckpt / _CHECKPOINT_FILE).exists() or (ckpt / _TRAIN_STATE_FILE).exists()


def save_train_state(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    name: str,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Save optimizer state, LR scheduler state, and epoch for training resumption.

    Writes ``train_state.pt`` atomically.  Remove it with
    :func:`cleanup_train_state` once training completes successfully.

    Args:
        optimizer: The optimizer whose state to persist.
        epoch: Most recently completed epoch (0-indexed).
        name: Registered model name.
        scheduler: Optional LR scheduler whose state to persist alongside the
            optimizer.  Pass ``None`` (default) when no scheduler is used.
    """
    ckpt = _checkpoint_dir_for(name)
    _save_train_state_to(optimizer, epoch, ckpt, scheduler=scheduler)


def load_train_state(
    name: str,
) -> tuple[dict[str, Any], int, dict[str, Any] | None] | None:
    """Load optimizer state, epoch, and optional scheduler state.

    Args:
        name: Registered model name.

    Returns:
        ``(optimizer_state_dict, epoch, scheduler_state_dict)`` if a train
        state file exists — ``scheduler_state_dict`` is ``None`` when the
        checkpoint was saved without a scheduler.  Returns ``None`` entirely
        if no train state is present (fresh run).
    """
    ckpt = _checkpoint_dir_for(name)
    return _load_train_state_from(ckpt)


def cleanup_train_state(name: str) -> None:
    """Remove the ephemeral train state file for *name*.

    Safe to call even if no train state exists.

    Args:
        name: Registered model name.
    """
    ckpt = _checkpoint_dir_for(name)
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
