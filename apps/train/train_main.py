"""Driver file for training."""

import argparse
import contextlib
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import signal
import sys
import time
from typing import Any

import datasets as hf_datasets
import torch
import torch.utils.data

from mole_jepa import config as config_module
from mole_jepa import data as data_module
from mole_jepa import factory, losses, model_io, models, nfs_registry

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
    dcfg = config_module.DataConfig()
    parser = argparse.ArgumentParser()

    # ── model / registry ──────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        required=True,
        metavar="NAME",
        help="Named model config looked up in the NFS registry.",
    )
    parser.add_argument(
        "--registry-path",
        default=os.environ.get("REGISTRY_PATH"),
        metavar="DIR",
        help=(
            "Directory containing registry.json. "
            "Defaults to $REGISTRY_PATH from the environment."
        ),
    )

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Override the checkpoint directory from the registry. "
            "If omitted, the path stored in the registry entry is used."
        ),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Stop each epoch after this many batches. "
            "Useful for a quick smoke-test on the cluster (e.g. --max-batches 1)."
        ),
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help=(
            "Run validation and log stats every this many epochs (default: 1). "
            "Checkpoints are saved every epoch regardless."
        ),
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
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help=(
            "Enable gradient checkpointing on the text encoder (and image "
            "encoder if not frozen).  Recomputes activations during the "
            "backward pass instead of storing them, reducing activation "
            "memory ~8× at a ~25%% compute overhead.  Allows larger batch "
            "sizes on memory-limited GPUs."
        ),
    )

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--hf-dataset-name",
        default="pixparse/cc3m-wds",
        help="Local shard directory or HuggingFace dataset identifier.",
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
        "--prefetch-factor",
        type=int,
        default=dcfg.prefetch_factor,
        help=(
            "Batches each worker preloads ahead of consumption. "
            "Reduce to 1 if workers are OOM-killed at large batch sizes "
            "(each step holds prefetch_factor × num_workers × batch_size "
            "images in /dev/shm)."
        ),
    )
    parser.add_argument(
        "--val-shard-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of WebDataset shards reserved for validation "
            "(last N%% of the sorted shard list).  Only used when "
            "--hf-dataset-name is a local directory.  Set to 0 to "
            "skip validation entirely."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=dcfg.max_seq_length,
        help="Maximum token sequence length; longer captions are truncated.",
    )
    parser.add_argument(
        "--image-field",
        default=dcfg.image_field,
        help="Dataset field name containing the image (default: 'jpg').",
    )
    parser.add_argument(
        "--caption-field",
        default=dcfg.caption_field,
        help="Dataset field name containing the caption (default: 'txt').",
    )

    return parser.parse_args()


def resolve_entry(
    args: argparse.Namespace,
) -> tuple[config_module.ModelConfig, pathlib.Path]:
    """Resolve the model config and checkpoint directory from the NFS registry.

    The checkpoint directory comes from the registry entry unless
    ``--checkpoint-dir`` is explicitly provided on the command line.

    Args:
        args: Parsed CLI arguments (must include ``args.config``,
            ``args.registry_path``, and optionally ``args.checkpoint_dir``).

    Returns:
        ``(model_config, checkpoint_dir)`` tuple.

    Raises:
        SystemExit: If ``args.registry_path`` is unset, or if ``args.config``
            is not found in the registry.
    """
    if not args.registry_path:
        raise SystemExit(
            "No registry path set. "
            "Pass --registry-path or set $REGISTRY_PATH in your environment."
        )
    try:
        entry = nfs_registry.get_entry(args.config, args.registry_path)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    checkpoint_dir = (
        pathlib.Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else entry.checkpoint_dir
    )
    return entry.config, checkpoint_dir


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
        prefetch_factor=args.prefetch_factor,
        image_field=args.image_field,
        caption_field=args.caption_field,
    )


def _is_local_path(name: str) -> bool:
    """Return ``True`` if *name* looks like a local filesystem path.

    A name is treated as local when it starts with ``/``, ``./``, or ``~/``,
    or when it resolves to an existing directory.  Everything else is assumed
    to be a HuggingFace dataset identifier (e.g. ``pixparse/cc3m-wds``).

    Args:
        name: The ``--hf-dataset-name`` value from the CLI.

    Returns:
        ``True`` for local paths, ``False`` for HF identifiers.
    """
    return (
        name.startswith("/")
        or name.startswith("./")
        or name.startswith("~/")
        or pathlib.Path(name).expanduser().is_dir()
    )


def build_loader(
    dataset_name: str,
    split: str,
    data_config: config_module.DataConfig,
    device: torch.device,
    *,
    tar_paths: list[str] | None = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
    """Stream a dataset and wrap it in a :class:`DataLoader`.

    Automatically selects the backend based on *dataset_name*:

    * **Local path** — if *dataset_name* is a directory (absolute, ``./``,
      or ``~/`` prefix, or an existing path), the WebDataset backend is used.
      Pass a pre-computed *tar_paths* list to specify exactly which shards to
      load (e.g. for a train/val split); if omitted, all shards under
      *dataset_name* are used.

    * **HuggingFace identifier** — otherwise the dataset is streamed via
      :func:`datasets.load_dataset` and wrapped in
      :class:`~mole_jepa.data.CC3MDataset`.

    Args:
        dataset_name: HF dataset identifier *or* local shard directory.
        split: Dataset split name (only used for the HF backend).
        data_config: Data configuration for transforms and batch sizing.
        device: Training device (enables ``pin_memory`` for CUDA).
        tar_paths: Explicit shard list for the WebDataset backend.  Ignored
            when using the HF backend.
        shuffle: Whether to shuffle shards and samples.  Pass ``False`` for
            the validation loader.

    Returns:
        A :class:`DataLoader` yielding ``(pixel_values, input_ids,
        attention_mask)`` triples.
    """
    if _is_local_path(dataset_name):
        paths = tar_paths or data_module.find_tar_shards(dataset_name)
        print(f"  WebDataset: {len(paths):,} shards from {dataset_name}")
        return data_module.build_datacomp_loader(
            paths, data_config, device, shuffle=shuffle
        )

    hf_ds = hf_datasets.load_dataset(dataset_name, split=split, streaming=True)
    dataset = data_module.CC3MDataset(hf_ds, data_config)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        prefetch_factor=(
            data_config.prefetch_factor if data_config.num_workers > 0 else None
        ),
        worker_init_fn=data_module.CC3MDataset.worker_init_fn,
        pin_memory=(device.type == "cuda"),
        # Keep workers alive between epochs — avoids re-spawning processes and
        # re-initialising transforms / tokenizers at the start of every epoch.
        persistent_workers=(data_config.num_workers > 0),
        # Python 3.14 changed the default multiprocessing start method on Linux
        # from 'fork' to 'forkserver'.  Forkserver requires worker arguments to
        # be picklable, but our dataset holds closures (the compiled torchvision
        # transform and tokenizer wrapper) that are intentionally not picklable.
        # The whole point of building transforms in __init__ is that fork-based
        # workers inherit them from the parent process without serialisation.
        multiprocessing_context="fork" if data_config.num_workers > 0 else None,
    )


def log_stats(
    model_loc: pathlib.Path,
    epoch: int,
    train_stats: dict[str, float],
    train_batches: int,
    val_stats: dict[str, float] | None = None,
    val_batches: int | None = None,
) -> None:
    """Append one stats record to ``stats.jsonl`` inside ``model_loc``.

    Each line is a self-contained JSON object, making the file easy to tail,
    parse incrementally, or load with ``pandas.read_json(..., lines=True)``.

    Train stats are written every epoch. Val stats are optional and only
    present in records for epochs where validation was run.

    For :class:`~mole_jepa.losses.JEPALoss`, the record includes
    ``train_loss``, ``train_loss_mse``, ``train_loss_reg_image``, and
    ``train_loss_reg_text`` (and their ``val_`` equivalents when present).

    Args:
        model_loc: Directory in which to write the stats file.
        epoch: Completed epoch number (0-indexed).
        train_stats: Mapping of component name to average training value
            (e.g. ``{"loss": …, "loss_mse": …, "loss_reg_image": …,
            "loss_reg_text": …}``).
        train_batches: Number of training batches processed this epoch.
        val_stats: Same structure as ``train_stats`` for the validation set.
            ``None`` when validation was not run this epoch.
        val_batches: Number of validation batches processed this epoch.
            ``None`` when validation was not run this epoch.
    """
    record: dict[str, Any] = {
        "epoch": epoch,
        "train_batches": train_batches,
        **{f"train_{k}": v for k, v in train_stats.items()},
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if val_stats is not None and val_batches is not None:
        record["val_batches"] = val_batches
        record.update({f"val_{k}": v for k, v in val_stats.items()})
    os.makedirs(model_loc, exist_ok=True)
    with open(model_loc / _STATS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _forward_batch(
    model: models.MoLeJEPA,
    loss_fn: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Move a batch to ``device``, run the model, and compute loss components.

    Args:
        model: The MoLeJEPA model.
        loss_fn: Either a :class:`~mole_jepa.losses.JEPALoss` or
            :class:`~mole_jepa.losses.InfoNCELoss` module.
        batch: ``(pixel_values, input_ids, attention_mask)`` tensors.
        device: Device to transfer tensors to before the forward pass.

    Returns:
        ``(loss_tensor, components)`` — same contract as
        :func:`_forward_loss`.
    """
    pixel_values, input_ids, attention_mask = batch
    pixel_values = pixel_values.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    output = model(pixel_values, input_ids, attention_mask)
    return _forward_loss(loss_fn, output)


def _forward_loss(
    loss_fn: torch.nn.Module,
    output: models.MoLeJEPAOutput,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dispatch to the correct loss module and return the loss with components.

    Args:
        loss_fn: Either a :class:`~mole_jepa.losses.InfoNCELoss` or
            :class:`~mole_jepa.losses.JEPALoss` module.
        output: Output from a :class:`~mole_jepa.models.MoLeJEPA` forward pass.

    Returns:
        ``(loss_tensor, components)`` where ``components`` maps component name
        to its scalar float value. For :class:`~mole_jepa.losses.JEPALoss`
        this includes ``"loss"``, ``"loss_mse"``, ``"loss_reg_image"``, and
        ``"loss_reg_text"``; for InfoNCE just ``"loss"``.
    """
    if isinstance(loss_fn, losses.InfoNCELoss):
        loss = loss_fn(output.z_v, output.z_t)
        return loss, {"loss": loss.item()}
    jepa_out = loss_fn(output)
    return jepa_out.loss, {
        "loss": jepa_out.loss.item(),
        "loss_mse": jepa_out.loss_mse.item(),
        "loss_reg_image": jepa_out.loss_reg_image.item(),
        "loss_reg_text": jepa_out.loss_reg_text.item(),
    }


def run_epoch(
    model: models.MoLeJEPA,
    loss_fn: torch.nn.Module,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    epoch: int,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
    log_interval: int = 100,
    max_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    """Run one epoch of training or evaluation.

    Args:
        model: The MoLeJEPA model.
        loss_fn: Loss module.
        loader: DataLoader yielding batches.
        device: Device on which to run the forward pass.
        epoch: Current epoch number (for logging).
        train: If ``True``, run in training mode (gradient updates enabled).
            If ``False``, run in eval mode under :func:`torch.no_grad`.
        optimizer: Required when ``train=True``; ignored otherwise.
        grad_clip: Max gradient norm (0 disables clipping). Training only.
        log_interval: Print a per-step line every this many steps. Training only.
        max_batches: Stop after this many batches. ``None`` means no limit.

    Returns:
        Tuple of ``(avg_components, n_batches)`` where ``avg_components``
        maps component name (e.g. ``"loss"``, ``"loss_mse"``) to its epoch
        average.
    """
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    batch_size: int = loader.batch_size or 1  # type: ignore[assignment]

    epoch_t0 = time.monotonic()
    interval_t0 = epoch_t0

    # BF16 autocast on CUDA: halves activation memory and uses tensor-core throughput.
    # No GradScaler needed — BF16 has the same exponent range as float32.
    amp_ctx = torch.amp.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=(device.type == "cuda"),
    )
    ctx = contextlib.nullcontext() if train else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            with amp_ctx:
                loss, components = _forward_batch(model, loss_fn, batch, device)

            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            if train and (batch_idx + 1) % log_interval == 0:
                now = time.monotonic()
                samples_per_sec = (log_interval * batch_size) / (now - interval_t0)
                interval_t0 = now
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                parts = "  ".join(f"{k} {v:.4f}" for k, v in components.items())
                print(
                    f"[{ts}]  epoch {epoch:>3}  step {batch_idx + 1:>6}"
                    f"  {parts}  samples/s {samples_per_sec:,.0f}"
                )

    elapsed = time.monotonic() - epoch_t0
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    avg_samples_per_sec = (n_batches * batch_size) / elapsed if elapsed > 0 else 0.0
    label = "train" if train else "val  "
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    avgs = {k: v / max(n_batches, 1) for k, v in totals.items()}
    parts = "  ".join(f"{k} {v:.4f}" for k, v in avgs.items())
    print(
        f"[{ts}]  epoch {epoch:>3}  {label}  {parts}"
        f"  batches {n_batches}  elapsed {elapsed_str}"  # noqa: E501
        f"  samples/s {avg_samples_per_sec:,.0f}"
    )
    return avgs, n_batches


def main() -> None:
    """Main script body."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    device = _get_device()
    print(f"device: {device}")

    model_config, checkpoint_dir = resolve_entry(args)
    data_config = construct_data_config(args, model_config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"config:         {args.config}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    for field in dataclasses.fields(model_config):
        print(f"  {field.name} = {getattr(model_config, field.name)!r}")

    # Build DataLoaders before moving the model to CUDA.  Workers are forked
    # at first iteration, so creating them here ensures they don't inherit the
    # CUDA context — each inherited CUDA context costs ~1.5 GB of address space
    # and SLURM counts it against the job's --mem budget even with fork CoW.
    train_dataset_name = args.hf_dataset_name
    val_dataset_name = args.val_hf_dataset_name or args.hf_dataset_name

    # For local WebDataset directories, split the shard list once so that
    # train and val always see disjoint shards regardless of which path is
    # passed for each.
    train_tar_paths: list[str] | None = None
    val_tar_paths: list[str] | None = None
    if _is_local_path(train_dataset_name):
        all_shards = data_module.find_tar_shards(train_dataset_name)
        train_tar_paths, val_tar_paths = data_module.split_shards(
            all_shards, args.val_shard_fraction
        )
        print(
            f"Shard split: {len(train_tar_paths):,} train  "
            f"{len(val_tar_paths):,} val  "
            f"(val_fraction={args.val_shard_fraction})"
        )

    loader = build_loader(
        train_dataset_name,
        args.hf_dataset_split,
        data_config,
        device,
        tar_paths=train_tar_paths,
        shuffle=True,
    )
    val_loader = build_loader(
        val_dataset_name,
        args.val_hf_dataset_split,
        data_config,
        device,
        tar_paths=val_tar_paths if val_tar_paths else None,
        shuffle=False,
    )

    model, loss_fn = factory.build(model_config)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    if args.gradient_checkpointing:
        # Enable gradient checkpointing on every trainable transformer encoder.
        # This recomputes activations during the backward pass instead of storing
        # all of them, cutting activation memory ~8× at ~25% compute overhead.
        model.text_encoder.lm.gradient_checkpointing_enable()
        if not model_config.freeze_image_encoder:
            model.image_encoder.vit.gradient_checkpointing_enable()
        frozen = model_config.freeze_image_encoder
        suffix = " (text encoder only — ViT is frozen)" if frozen else ""
        print(f"Gradient checkpointing enabled{suffix}")

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── resume detection ──────────────────────────────────────────────────────
    # Auto-resume if a train state exists in the registry-resolved directory.
    # The path comes from the registry entry, so it is stable across config
    # field additions (no hash recomputation at resume time).
    resuming = model_io.has_train_state_at(checkpoint_dir)

    start_epoch = 0
    if resuming:
        model_io.load_model_weights_at(model, checkpoint_dir, map_location=device)
        train_state = model_io.load_train_state_at(checkpoint_dir)
        assert train_state is not None
        opt_state_dict, last_epoch = train_state
        optimizer.load_state_dict(opt_state_dict)
        # load_state_dict restores the saved LR/weight-decay, which would
        # silently ignore any --lr / --weight-decay passed at the command line.
        # Override the param groups so CLI flags always win on resume.
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["weight_decay"] = args.weight_decay
        start_epoch = last_epoch + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting fresh run")

    # ── preemption handler ────────────────────────────────────────────────────
    # On the DSI cluster, preemption sends SIGUSR1 with a 5-minute grace period
    # (configured via --signal=B:USR1@300), then SIGKILL. Exiting with code 99
    # tells Slurm to requeue the job automatically (--requeue).
    current_epoch = start_epoch

    def _checkpoint_and_exit(*_: object) -> None:
        print(f"\nSIGUSR1 — saving checkpoint at epoch {current_epoch}.")
        model_io.save_model_at(model, checkpoint_dir, config=model_config)
        model_io.save_train_state_at(optimizer, current_epoch, checkpoint_dir)
        # Explicitly tear down DataLoader workers before exiting.  If we call
        # sys.exit() while a _MultiProcessingDataLoaderIter is alive, the
        # worker processes are orphaned and Python's multiprocessing resource
        # tracker warns about ~21 leaked semaphores at shutdown.  Nulling the
        # _iterator attribute triggers the iterator's __del__, which calls
        # _shutdown_workers() and joins the worker processes cleanly.
        for dl in (loader, val_loader):
            if getattr(dl, "_iterator", None) is not None:
                dl._iterator = None  # type: ignore[assignment]
        sys.exit(99)

    signal.signal(signal.SIGUSR1, _checkpoint_and_exit)

    # ── stats file ────────────────────────────────────────────────────────────
    # Truncate on a fresh run; on resume, keep history and continue appending.
    if not resuming:
        (checkpoint_dir / _STATS_FILE).unlink(missing_ok=True)

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.num_epochs):
        current_epoch = epoch

        train_stats, train_batches = run_epoch(
            model,
            loss_fn,
            loader,
            device,
            epoch,
            train=True,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            max_batches=args.max_batches,
        )

        val_stats: dict[str, float] | None = None
        val_batches: int | None = None
        if (epoch + 1) % args.val_interval == 0:
            val_stats, val_batches = run_epoch(
                model,
                loss_fn,
                val_loader,
                device,
                epoch,
                train=False,
                max_batches=args.max_batches,
            )

        log_stats(
            checkpoint_dir, epoch, train_stats, train_batches, val_stats, val_batches
        )
        model_io.save_model_at(model, checkpoint_dir, config=model_config)
        model_io.save_train_state_at(optimizer, epoch, checkpoint_dir)

    # Training completed successfully.
    # Save the final model weights, then remove the ephemeral train state —
    # it is only needed for resumption and is no longer useful.
    model_io.save_model_at(model, checkpoint_dir, config=model_config)
    model_io.cleanup_train_state_at(checkpoint_dir)


if __name__ == "__main__":
    main()
