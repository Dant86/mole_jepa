"""Driver file for training."""

import argparse
import os
import pathlib
import re
import signal
import sys
from typing import Any

import torch

from mole_jepa import config as config_module
from mole_jepa import models


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        A :class:`Namespace` object.
    """
    cfg = config_module.ModelConfig()
    parser = argparse.ArgumentParser()

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the latest checkpoint and resume training.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory from which to read/write model checkpoints.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs.",
    )

    # ── shared ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=cfg.embed_dim,
        help="Shared embedding dimension for both encoders and the predictor.",
    )

    # ── encoders ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--image-encoder-model-name",
        default=cfg.image_encoder_model_name,
        help="HuggingFace model identifier for the image encoder (ViT).",
    )
    parser.add_argument(
        "--text-encoder-model-name",
        default=cfg.text_encoder_model_name,
        help="HuggingFace model identifier for the text encoder (LM).",
    )

    # ── predictor ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--predictor-hidden-dim",
        type=int,
        default=cfg.predictor_hidden_dim,
        help="Hidden width of the MLP predictor.",
    )
    parser.add_argument(
        "--predictor-n-layers",
        type=int,
        default=cfg.predictor_n_layers,
        help="Total number of linear layers in the MLP predictor.",
    )

    # ── loss ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--use-contrastive-loss",
        action="store_true",
        help="Use InfoNCE loss instead of JEPALoss with SIGReg.",
    )
    parser.add_argument(
        "--sigreg-dist",
        choices=["gaussian", "laplace"],
        default=cfg.sigreg_dist,
        help="Target distribution for the SIGReg Epps-Pulley statistic.",
    )
    parser.add_argument(
        "--sigreg-n-directions",
        type=int,
        default=cfg.sigreg_n_directions,
        help="Number of random projection directions used by SIGReg.",
    )
    parser.add_argument(
        "--jepa-reg-weight",
        type=float,
        default=cfg.jepa_reg_weight,
        help="Weighting factor λ for the SIGReg term in JEPALoss.",
    )
    parser.add_argument(
        "--info-nce-temperature",
        type=float,
        default=cfg.info_nce_temperature,
        help="Softmax temperature for InfoNCELoss.",
    )

    return parser.parse_args()


def construct_model_config(args: argparse.Namespace) -> config_module.ModelConfig:
    """Construct a :class:`~mole_jepa.config.ModelConfig` from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A :class:`~mole_jepa.config.ModelConfig` instance.
    """
    return config_module.ModelConfig(
        embed_dim=args.embed_dim,
        image_encoder_model_name=args.image_encoder_model_name,
        text_encoder_model_name=args.text_encoder_model_name,
        predictor_hidden_dim=args.predictor_hidden_dim,
        predictor_n_layers=args.predictor_n_layers,
        contrastive=args.use_contrastive_loss,
        sigreg_dist=args.sigreg_dist,
        sigreg_n_directions=args.sigreg_n_directions,
        jepa_reg_weight=args.jepa_reg_weight,
        info_nce_temperature=args.info_nce_temperature,
    )


def model_location(
    checkpoint_dir: str, config: config_module.ModelConfig
) -> pathlib.Path:
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

    model, loss = config_module.build(config)
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
