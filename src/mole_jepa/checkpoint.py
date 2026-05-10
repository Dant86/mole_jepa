"""Checkpoint save, load, and registry for MoLeJEPA models."""

import dataclasses
import json
import os
import pathlib
from typing import Any

import torch

from mole_jepa import config as config_module
from mole_jepa import factory, models

_CHECKPOINT_FILE = "checkpoint.pt"
_FINAL_MODEL_FILE = "model.pt"


@dataclasses.dataclass
class CheckpointInfo:
    """Metadata about a checkpoint on disk.

    Attributes:
        path: Directory containing the checkpoint file(s).
        config_hash: SHA-256 hex string identifying the config (directory name).
        epoch: Most recently completed epoch stored in the checkpoint.
        config: Deserialized :class:`~mole_jepa.config.ModelConfig`, or
            ``None`` if no ``config.json`` was found alongside the checkpoint.
    """

    path: pathlib.Path
    config_hash: str
    epoch: int
    config: config_module.ModelConfig | None


def location(
    checkpoint_dir: str | pathlib.Path,
    config: config_module.ModelConfig,
) -> pathlib.Path:
    """Return the subdirectory for a given config inside ``checkpoint_dir``.

    Args:
        checkpoint_dir: Root directory holding all checkpoint subdirectories.
        config: The model config whose hash names the subdirectory.

    Returns:
        ``checkpoint_dir / config.serialize()``.
    """
    return pathlib.Path(checkpoint_dir) / config.serialize()


def save(
    model: models.MoLeJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: config_module.ModelConfig,
    checkpoint_dir: str | pathlib.Path,
) -> None:
    """Save model and optimiser state atomically.

    Writes to ``checkpoint.pt.tmp`` first, then renames it into place so that
    a ``SIGKILL`` mid-write cannot corrupt an existing checkpoint. Also writes
    ``config.json`` alongside so the checkpoint can be identified without
    knowing the config upfront.

    Args:
        model: The model whose weights to persist.
        optimizer: The optimiser whose state to persist.
        epoch: Most recently completed epoch (0-indexed).
        config: The config used to build the model. Stored as ``config.json``.
        checkpoint_dir: Root directory holding all checkpoint subdirectories.
    """
    loc = location(checkpoint_dir, config)
    os.makedirs(loc, exist_ok=True)

    # Write config.json so the registry can reconstruct the config.
    config_path = loc / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2))

    tmp = loc / (f"{_CHECKPOINT_FILE}.tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, loc / _CHECKPOINT_FILE)


def load(
    config: config_module.ModelConfig,
    checkpoint_dir: str | pathlib.Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[models.MoLeJEPA, int]:
    """Load a model from the checkpoint matching ``config``.

    Builds the model via :func:`~mole_jepa.factory.build`, loads the saved
    weights, and returns the model ready for inference or resumed training.

    Args:
        config: Config identifying which checkpoint to load.
        checkpoint_dir: Root directory holding all checkpoint subdirectories.
        map_location: Device to load tensors onto. Defaults to ``"cpu"`` so
            notebooks work without a GPU.

    Returns:
        ``(model, epoch)`` — the loaded model and the last completed epoch.

    Raises:
        FileNotFoundError: If no checkpoint exists for ``config``.
    """
    loc = location(checkpoint_dir, config)
    ckpt_path = loc / _CHECKPOINT_FILE
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            "Run training first, or check the config matches the saved run."
        )

    model, _ = factory.build(config)
    state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model, state["epoch"]


def save_final(
    model: models.MoLeJEPA,
    epoch: int,
    config: config_module.ModelConfig,
    checkpoint_dir: str | pathlib.Path,
) -> None:
    """Save the final model weights after a completed training run.

    Writes only the model state dict (no optimiser state) to ``model.pt``.

    Args:
        model: The fully trained model.
        epoch: Last completed epoch (0-indexed).
        config: Config used to build the model.
        checkpoint_dir: Root directory holding all checkpoint subdirectories.
    """
    loc = location(checkpoint_dir, config)
    os.makedirs(loc, exist_ok=True)
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        loc / _FINAL_MODEL_FILE,
    )


def list_checkpoints(
    checkpoint_dir: str | pathlib.Path,
) -> list[CheckpointInfo]:
    """Return metadata for every checkpoint found under ``checkpoint_dir``.

    Each subdirectory that contains a ``checkpoint.pt`` is treated as one
    checkpoint entry. If a ``config.json`` is present alongside it, the config
    is deserialized and included.

    Args:
        checkpoint_dir: Root directory holding all checkpoint subdirectories.

    Returns:
        List of :class:`CheckpointInfo` sorted by epoch (ascending).
    """
    results: list[CheckpointInfo] = []
    root = pathlib.Path(checkpoint_dir)
    if not root.exists():
        return results

    for subdir in sorted(root.iterdir()):
        ckpt_path = subdir / _CHECKPOINT_FILE
        if not ckpt_path.exists():
            continue

        state: dict[str, Any] = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
        epoch: int = state["epoch"]

        cfg: config_module.ModelConfig | None = None
        config_path = subdir / "config.json"
        if config_path.exists():
            cfg = config_module.ModelConfig(**json.loads(config_path.read_text()))

        results.append(
            CheckpointInfo(
                path=subdir,
                config_hash=subdir.name,
                epoch=epoch,
                config=cfg,
            )
        )

    return sorted(results, key=lambda c: c.epoch)
