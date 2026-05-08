"""Driver file for training."""

import argparse
import datetime
import json
import os
import pathlib
import signal
import sys
from typing import Any

import datasets as hf_datasets
import torch
import torch.utils.data

from mole_jepa import config as config_module
from mole_jepa import data as data_module
from mole_jepa import losses, models

_CHECKPOINT_FILE = "checkpoint.pt"
_FINAL_MODEL_FILE = "model.pt"
_STATS_FILE = "stats.jsonl"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        A :class:`Namespace` object.
    """
    mcfg = config_module.ModelConfig()
    dcfg = config_module.DataConfig()
    parser = argparse.ArgumentParser()

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the latest checkpoint and resume training.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory from which to read/write model checkpoints.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )

    # ── optimiser ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 disables).",
    )

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--hf-dataset-name",
        default="conceptual_captions",
        help="HuggingFace dataset identifier for the training set.",
    )
    parser.add_argument(
        "--hf-dataset-split",
        default="train",
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--val-hf-dataset-name",
        default=None,
        help=(
            "HuggingFace dataset identifier for the validation set. "
            "Defaults to --hf-dataset-name when omitted."
        ),
    )
    parser.add_argument(
        "--val-hf-dataset-split",
        default="validation",
        help="Dataset split to use for validation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=dcfg.batch_size,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=dcfg.num_workers,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=dcfg.max_seq_length,
        help="Maximum token sequence length; longer captions are truncated.",
    )

    # ── shared model ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=mcfg.embed_dim,
        help="Shared embedding dimension for both encoders and the predictor.",
    )

    # ── encoders ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--image-encoder-model-name",
        default=mcfg.image_encoder_model_name,
        help="HuggingFace model identifier for the image encoder (ViT).",
    )
    parser.add_argument(
        "--text-encoder-model-name",
        default=mcfg.text_encoder_model_name,
        help="HuggingFace model identifier for the text encoder (LM).",
    )

    # ── predictor ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--predictor-hidden-dim",
        type=int,
        default=mcfg.predictor_hidden_dim,
        help="Hidden width of the MLP predictor.",
    )
    parser.add_argument(
        "--predictor-n-layers",
        type=int,
        default=mcfg.predictor_n_layers,
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
        default=mcfg.sigreg_dist,
        help="Target distribution for the SIGReg Epps-Pulley statistic.",
    )
    parser.add_argument(
        "--sigreg-n-directions",
        type=int,
        default=mcfg.sigreg_n_directions,
        help="Number of random projection directions used by SIGReg.",
    )
    parser.add_argument(
        "--jepa-reg-weight",
        type=float,
        default=mcfg.jepa_reg_weight,
        help="Weighting factor λ for the SIGReg term in JEPALoss.",
    )
    parser.add_argument(
        "--info-nce-temperature",
        type=float,
        default=mcfg.info_nce_temperature,
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


def construct_data_config(
    args: argparse.Namespace,
    model_config: config_module.ModelConfig,
) -> config_module.DataConfig:
    """Construct a :class:`~mole_jepa.config.DataConfig` from CLI arguments.

    Processor and tokenizer model names are derived from ``model_config`` so
    they always stay in sync with the encoder configuration.

    Args:
        args: Parsed CLI arguments.
        model_config: Model config whose encoder names are reused.

    Returns:
        A :class:`~mole_jepa.config.DataConfig` instance.
    """
    return config_module.DataConfig(
        image_processor_model_name=model_config.image_encoder_model_name,
        tokenizer_model_name=model_config.text_encoder_model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def build_loader(
    dataset_name: str,
    split: str,
    data_config: config_module.DataConfig,
    device: torch.device,
) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
    """Stream a HuggingFace dataset split and wrap it in a :class:`DataLoader`.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split name.
        data_config: Data configuration for transforms and batch sizing.
        device: Training device (used to enable ``pin_memory`` for CUDA).

    Returns:
        A :class:`DataLoader` yielding ``(pixel_values, input_ids,
        attention_mask)`` triples.
    """
    hf_ds = hf_datasets.load_dataset(dataset_name, split=split, streaming=True)
    dataset = data_module.CC3MDataset(hf_ds, data_config)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        worker_init_fn=data_module.CC3MDataset.worker_init_fn,
        pin_memory=(device.type == "cuda"),
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


def save_checkpoint(
    model: models.MoLeJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_loc: pathlib.Path,
) -> None:
    """Save model and optimiser state to a single checkpoint file.

    Always writes to ``checkpoint.pt`` inside ``model_loc``, overwriting any
    previous checkpoint. The completed epoch is stored inside the file so it
    can be recovered without relying on the filename.

    Args:
        model: The model whose weights to persist.
        optimizer: The optimiser whose state to persist.
        epoch: Most recently completed epoch (0-indexed).
        model_loc: Directory in which to write the checkpoint.
    """
    os.makedirs(model_loc, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_loc / _CHECKPOINT_FILE,
    )


def load_checkpoint(
    model_loc: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
    """Load the checkpoint from ``model_loc``.

    Args:
        model_loc: Directory containing ``checkpoint.pt``.

    Returns:
        A ``(model_state_dict, optimizer_state_dict, epoch)`` triple.
        ``optimizer_state_dict`` is ``None`` if the checkpoint predates
        optimizer state persistence.
    """
    checkpoint = torch.load(
        model_loc / _CHECKPOINT_FILE,
        weights_only=True,
    )
    return (
        checkpoint["model_state_dict"],
        checkpoint.get("optimizer_state_dict"),
        checkpoint["epoch"],
    )


def save_final_model(
    model: models.MoLeJEPA,
    epoch: int,
    model_loc: pathlib.Path,
) -> None:
    """Save the final model weights after a completed training run.

    Writes only the model state dict (no optimiser state) to ``model.pt``
    inside ``model_loc``. The optimiser is not needed after training ends.

    Args:
        model: The fully trained model.
        epoch: Last completed epoch (0-indexed).
        model_loc: Directory in which to write the file.
    """
    os.makedirs(model_loc, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        },
        model_loc / _FINAL_MODEL_FILE,
    )


def log_stats(
    model_loc: pathlib.Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """Append one stats record to ``stats.jsonl`` inside ``model_loc``.

    Each line is a self-contained JSON object, making the file easy to tail,
    parse incrementally, or load with ``pandas.read_json(..., lines=True)``.

    Args:
        model_loc: Directory in which to write the stats file.
        epoch: Completed epoch number (0-indexed).
        train_loss: Average training loss for this epoch.
        val_loss: Average validation loss for this epoch.
    """
    record = {
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    os.makedirs(model_loc, exist_ok=True)
    with open(model_loc / _STATS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _compute_loss(
    loss_fn: torch.nn.Module,
    output: models.MoLeJEPAOutput,
) -> torch.Tensor:
    """Dispatch to the correct loss module and return a scalar loss tensor.

    Args:
        loss_fn: Either a :class:`~mole_jepa.losses.InfoNCELoss` or
            :class:`~mole_jepa.losses.JEPALoss` module.
        output: Output from a :class:`~mole_jepa.models.MoLeJEPA` forward pass.

    Returns:
        Scalar loss tensor.
    """
    if isinstance(loss_fn, losses.InfoNCELoss):
        return loss_fn(output.z_v, output.z_t)
    return loss_fn(output).loss


def train_step(
    model: models.MoLeJEPA,
    loss_fn: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run a single forward/backward pass and return the scalar loss.

    Args:
        model: The MoLeJEPA model.
        loss_fn: Either a :class:`~mole_jepa.losses.JEPALoss` or
            :class:`~mole_jepa.losses.InfoNCELoss` module.
        batch: ``(pixel_values, input_ids, attention_mask)`` tensors.
        optimizer: The optimiser to step.
        device: Device on which tensors live.
        grad_clip: Max gradient norm (0 disables clipping).

    Returns:
        Scalar training loss for this batch.
    """
    pixel_values, input_ids, attention_mask = batch
    pixel_values = pixel_values.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    optimizer.zero_grad()
    output = model(pixel_values, input_ids, attention_mask)
    loss = _compute_loss(loss_fn, output)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item()


def train_epoch(
    model: models.MoLeJEPA,
    loss_fn: torch.nn.Module,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float,
    log_interval: int = 100,
) -> float:
    """Train for one full pass over the dataloader.

    Args:
        model: The MoLeJEPA model.
        loss_fn: Loss module.
        loader: DataLoader yielding training batches.
        optimizer: The optimiser to use.
        device: Device on which to run the forward pass.
        epoch: Current epoch number (for logging).
        grad_clip: Max gradient norm passed to :func:`train_step`.
        log_interval: Print a progress line every this many steps.

    Returns:
        Average loss over all batches in this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        step_loss = train_step(model, loss_fn, batch, optimizer, device, grad_clip)
        total_loss += step_loss
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(f"epoch {epoch:>3}  step {batch_idx + 1:>6}  loss {step_loss:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    print(f"epoch {epoch:>3}  avg_loss {avg_loss:.4f}")
    return avg_loss


def eval_epoch(
    model: models.MoLeJEPA,
    loss_fn: torch.nn.Module,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    epoch: int,
) -> float:
    """Evaluate the model over the validation dataloader.

    Runs in :func:`torch.no_grad` with the model in eval mode.

    Args:
        model: The MoLeJEPA model.
        loss_fn: Loss module.
        loader: DataLoader yielding validation batches.
        device: Device on which to run the forward pass.
        epoch: Current epoch number (for logging).

    Returns:
        Average validation loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            pixel_values, input_ids, attention_mask = batch
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            output = model(pixel_values, input_ids, attention_mask)
            loss = _compute_loss(loss_fn, output)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    avg_val_loss = total_loss / max(n_batches, 1)
    print(f"epoch {epoch:>3}  val_loss  {avg_val_loss:.4f}")
    return avg_val_loss


def main() -> None:
    """Main script body."""
    args = parse_args()
    device = _get_device()
    print(f"device: {device}")

    model_config = construct_model_config(args)
    data_config = construct_data_config(args, model_config)

    loc = model_location(args.checkpoint_dir, model_config)
    model, loss_fn = config_module.build(model_config)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    if args.resume:
        model_state, opt_state, start_epoch = load_checkpoint(loc)
        model.load_state_dict(model_state)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        start_epoch += 1  # resume from the next epoch

    # ── preemption handler ────────────────────────────────────────────────────
    # SIGUSR1 is sent by SLURM five minutes before the wall-time limit.
    # Save a checkpoint so the job can be requeued with --resume.
    current_epoch = start_epoch

    def _checkpoint_and_exit(*_: object) -> None:
        print(f"\nSIGUSR1 — saving checkpoint at epoch {current_epoch} and exiting.")
        save_checkpoint(model, optimizer, current_epoch, loc)
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _checkpoint_and_exit)

    # ── data ──────────────────────────────────────────────────────────────────
    loader = build_loader(
        args.hf_dataset_name, args.hf_dataset_split, data_config, device
    )
    val_loader = build_loader(
        args.val_hf_dataset_name or args.hf_dataset_name,
        args.val_hf_dataset_split,
        data_config,
        device,
    )

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.num_epochs):
        current_epoch = epoch

        train_loss = train_epoch(
            model,
            loss_fn,
            loader,
            optimizer,
            device,
            epoch=epoch,
            grad_clip=args.grad_clip,
        )

        if (epoch + 1) % 5 == 0:
            val_loss = eval_epoch(model, loss_fn, val_loader, device, epoch)
            log_stats(loc, epoch, train_loss, val_loss)

    # Training completed successfully.
    # Save the final model weights, then remove the partial checkpoint — it was
    # only needed for resumption and is no longer useful.
    final_epoch = args.num_epochs - 1
    save_final_model(model, final_epoch, loc)
    (loc / _CHECKPOINT_FILE).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
