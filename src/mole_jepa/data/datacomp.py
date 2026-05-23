"""DataComp WebDataset loader for locally downloaded tar shards.

Reads the WebDataset tar files produced by ``prepare_datacomp_main.py``
(img2dataset with ``--output_format webdataset``).  Each sample in a shard
contains three files keyed by the same stem:

* ``{key}.jpg``  — JPEG-encoded image (resized to 256 px shorter edge)
* ``{key}.txt``  — plain-text caption (``re_caption_condition_diverse_topk``)
* ``{key}.json`` — sidecar metadata (uid, img2dataset stats)

The loader is used when the dataset path is a local directory rather than a
HuggingFace identifier.  Shard splitting between DataLoader workers is handled
internally by WebDataset — no ``worker_init_fn`` is required.
"""

from __future__ import annotations

import collections.abc
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.utils.data
import webdataset as wds

from mole_jepa import config as config_module
from mole_jepa.data import transforms


def find_tar_shards(shards_root: str | Path) -> list[str]:
    """Return sorted list of all ``.tar`` shard paths under *shards_root*.

    Searches recursively so the chunked layout produced by
    ``prepare_datacomp_main.py`` (``shards/chunk_NNNN/*.tar``) is handled
    automatically.

    Args:
        shards_root: Root directory to search.

    Returns:
        Sorted list of absolute path strings.

    Raises:
        FileNotFoundError: If no ``.tar`` files are found.
    """
    paths = sorted(str(p) for p in Path(shards_root).rglob("*.tar"))
    if not paths:
        raise FileNotFoundError(f"No .tar shards found under {shards_root}")
    return paths


def build_datacomp_loader(
    tar_paths: list[str],
    data_config: config_module.DataConfig,
    device: torch.device,
    *,
    shuffle: bool = True,
    shuffle_buffer: int = 1_000,
) -> torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Wrap a list of WebDataset tar shards in a :class:`DataLoader`.

    Transforms are built once in the calling process.  Workers are forked so
    they inherit the compiled torchvision pipeline and tokenizer without
    re-constructing them from HuggingFace on each worker startup.

    To cap the number of samples per epoch, pass ``max_batches`` to
    :func:`~mole_jepa.train.train_main.run_epoch` rather than slicing here.
    WebDataset shard splitting distributes shards across DataLoader workers,
    so a per-dataset slice would apply independently to each worker's stream
    rather than to the global sample count.

    Args:
        tar_paths: List of ``.tar`` shard paths to read.  Pass a subset for
            train/val splitting (see :func:`split_shards`).
        data_config: Data configuration for transforms and batch sizing.
        device: Training device (enables ``pin_memory`` for CUDA).
        shuffle: If ``True``, shuffle the shard order each epoch and apply a
            sample-level shuffle buffer.  Set to ``False`` for validation.
        shuffle_buffer: Number of decoded samples to hold in the
            sample-shuffle buffer.  Higher values improve randomisation at
            the cost of RAM (~200 KB per PIL image slot).  Only used when
            ``shuffle=True``.

    Returns:
        A :class:`DataLoader` yielding
        ``(pixel_values, input_ids, attention_mask)`` triples as batch
        tensors of shape ``(B, C, H, W)``, ``(B, L)``, ``(B, L)``.
    """
    image_transform = transforms.build_image_transform(
        data_config.image_processor_model_name,
        train=shuffle,  # shuffle=True for train loader, False for val
    )
    tokenize = transforms.build_tokenizer(
        data_config.tokenizer_model_name, data_config.max_seq_length
    )
    image_field = data_config.image_field  # "jpg"
    caption_field = data_config.caption_field  # "txt"

    def _preprocess(
        sample: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from PIL import Image  # local import — already a dep, avoids top-level cost

        raw = sample[image_field]
        if isinstance(raw, bytes):
            import io

            raw = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            raw = raw.convert("RGB")  # type: ignore[union-attr]
        pixel_values = image_transform(raw)

        caption = sample[caption_field]
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8", errors="replace")
        input_ids, attention_mask = tokenize(str(caption))

        return pixel_values, input_ids, attention_mask

    pipeline: collections.abc.Iterable = wds.WebDataset(
        tar_paths,
        shardshuffle=shuffle,
        nodesplitter=wds.split_by_node,
        # Skip corrupt or truncated tars rather than crashing the job.
        # warn_and_continue logs a warning and moves to the next shard.
        handler=wds.warn_and_continue,
    ).decode("pil", handler=wds.warn_and_continue)

    if shuffle:
        pipeline = pipeline.shuffle(shuffle_buffer)  # type: ignore[union-attr]

    # Pass handler here too so a bad JPEG inside an otherwise-valid tar is
    # skipped rather than aborting the epoch.
    dataset = pipeline.map(_preprocess, handler=wds.warn_and_continue)  # type: ignore[union-attr]

    return torch.utils.data.DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        prefetch_factor=(
            data_config.prefetch_factor if data_config.num_workers > 0 else None
        ),
        pin_memory=(device.type == "cuda"),
        # Do NOT use persistent_workers with WebDataset: the iterator is
        # exhausted at epoch end and needs to be rebuilt each epoch.
        persistent_workers=False,
        # Transforms are closures — not picklable with spawn.  Use fork so
        # workers inherit the pre-built pipeline from the parent process.
        multiprocessing_context="fork" if data_config.num_workers > 0 else None,
    )


def count_samples(tar_paths: list[str], *, n_workers: int = 32) -> int:
    """Count the total number of samples across a list of WebDataset tar shards.

    Each img2dataset shard contains exactly one ``.jpg`` file per sample
    (alongside ``.txt`` and ``.json`` sidecars), so the sample count equals
    the number of ``.jpg`` tar members.  Tar headers are read in parallel
    using threads (I/O-bound), so even 1 000+ shards finish in under a
    minute on typical NFS.  Corrupt or unreadable shards are silently
    skipped and contribute 0 to the total.

    Args:
        tar_paths: List of ``.tar`` shard paths (e.g. from
            :func:`find_tar_shards`).
        n_workers: Number of parallel threads for header reads.

    Returns:
        Total number of samples across all shards.
    """

    def _count_one(path: str) -> int:
        try:
            with tarfile.open(path, "r:*") as tf:
                return sum(1 for m in tf.getmembers() if m.name.endswith(".jpg"))
        except Exception:
            return 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return sum(pool.map(_count_one, tar_paths))


def split_shards(
    tar_paths: list[str],
    val_fraction: float,
) -> tuple[list[str], list[str]]:
    """Split a shard list into training and validation sets.

    Uses the *last* ``round(len * val_fraction)`` shards for validation so
    that a stable prefix is always used for training (deterministic splits
    across reruns).

    Args:
        tar_paths: Full sorted shard list from :func:`find_tar_shards`.
        val_fraction: Fraction of shards to reserve for validation
            (e.g. ``0.05`` for 5 %).  Set to ``0`` to use all shards for
            training and return an empty val list.

    Returns:
        ``(train_paths, val_paths)`` tuple.
    """
    if val_fraction <= 0:
        return tar_paths, []
    n_val = max(1, round(len(tar_paths) * val_fraction))
    return tar_paths[:-n_val], tar_paths[-n_val:]
