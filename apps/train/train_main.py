"""Driver file for training."""

import argparse
import dataclasses
import hashlib
import os
import pathlib
import re
import signal
import sys
from typing import Any, Literal

import torch
from torch import nn

from mole_jepa import losses, models, regularizers, test_statistics


@dataclasses.dataclass
class ModelConfig:
    """Model configuration.

    The :class:`ModelConfig` class contains the minimal set of attributes
    required to represent a MoLeJEPA.
    """

    contrastive: bool = False
    sigreg_dist: Literal["gaussian", "laplace"] = "gaussian"

    def serialize(self) -> str:
        """Serialize a :class:`ModelConfig` instance to a hash string.

        Returns:
            A unique string identifier for this config.
        """
        s_repr = sum(
            [f"{k}={v}" for k, v in dataclasses.asdict(self)], start=""
        ).encode("utf-8")

        return hashlib.sha256(s_repr).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        A :class:`Namespace` object.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        acton="store_true",
        help="If specified, load the latest model checkpoint and resume training.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        help="Directory from which to read/write model checkpoints.",
    )

    parser.add_argument(
        "--use-contrastive-loss",
        help="If specified, use InfoNCE loss. Defaults to SIGReg.",
        action="store_true",
    )

    parser.add_argument(
        "--sigreg-dist",
        help="Distribution to regularize with respect to.",
        options=["gaussian", "laplace"],
    )

    parser.add_argument("--num_epochs", type=int, help="Number of train epochs.")

    return parser.parse_args()


def construct_model_config(args: argparse.Namespace) -> ModelConfig:
    """Construct a :class:`ModelConfig` class.

    Args:
        args: :class:`Namespace` object from CLI args.

    Returns:
        A :class:`ModelConfig` instance.
    """
    return ModelConfig(
        contrastive=args.use_contrastive_loss,
        sigreg_dist=args.sigreg_dist,
    )


def construct_model_and_loss(
    config: ModelConfig,
) -> tuple[models.MoLeJEPA, nn.Module]:
    """Recover a :class:`MoleJEPA` instance and loss function.

    Args:
        config: A :class:`ModelConfig` instance.

    Returns:
        A :class:`MoleJepa` instance and a loss :class:`Module`.
    """
    if config.contrastive:
        loss = losses.InfoNCELoss()
    else:
        loss = losses.JEPALoss(
            regularizers.SIGReg(
                test_statistics.epps_pulley(
                    config.sigreg_dist,
                ),
            )
        )

    model = models.MoLeJEPA(
        image_encoder=models.ImageEncoder(),
        text_encoder=models.TextEncoder(),
        predictor=models.Predictor(256, 512),
    )

    return model, loss


def model_location(checkpoint_dir: str, config: ModelConfig) -> pathlib.Path:
    """Construct the model checkpoint location.

    Args:
        checkpoint_dir: Directory in which all checkpoints are stored.
        config: A :class:`ModelConfig` instance.

    Returns:
        A :class:`Path` in which to save all checkpoint files for this model.
    """
    return pathlib.Path(checkpoint_dir) / config.serialize()


def save_model(
    model: models.MoLeJEPA,
    epoch: int,
    model_loc: pathlib.Path,
) -> None:
    """Save a :class:`MoLeJEPA` state.

    Args:
        model: A :class:`MoleJEPA` instance.
        epoch: Most recent epoch.
        model_loc: Location in which to save model state.
    """
    os.makedirs(model_loc, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        model_loc / f"epoch_{epoch}.pt",
    )


def load_model(model_loc: pathlib.Path) -> tuple[dict[str, Any], int]:
    """Load a :class:`MoLeJEPA` instance from the most recent checkpoint.

    Args:
        model_loc: Location from which to read checkpoints.

    Returns:
        The model's state dictionary and number of most recent epoch.
    """
    pattern = re.compile(r"^epoch_(\d+)\.pt$")
    epoch = max(
        int(m.group(1)) for f in os.listdir(model_loc) if (m := pattern.match(f))
    )

    data = torch.load(model_loc)

    return data["model_state_dict"], epoch


def main() -> None:
    """Main script body."""
    args = parse_args()
    config = construct_model_config(args)
    loc = model_location(args.checkpoint_dir, config)

    model, loss = construct_model_and_loss(config)
    start_epoch = 0

    if args.resume:
        state_dict, start_epoch = load_model(loc)
        model.load_state_dict(state_dict)

    for epoch in range(start_epoch, args.num_epochs):
        # Train model.

        if (epoch + 1) % 5 == 0:
            save_model(model, epoch, loc)


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, lambda *_: sys.exit(99))

    main()
