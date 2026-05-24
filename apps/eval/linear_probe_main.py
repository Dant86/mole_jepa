"""Linear probe evaluation for MoLeJEPA image encoders.

Encodes all images from a classification dataset with a frozen model, then
trains a single linear layer on the resulting embeddings.  No fine-tuning of
the encoder — weights are frozen throughout.

Supported datasets (all fetched from HuggingFace):

* ``tanganke/stl10``    — 10 classes, 5 k train / 8 k test (default)
* ``uoft-cs/cifar100``  — 100 fine / 20 coarse classes, 50 k train / 10 k test

Usage::

    uv run python apps/eval/linear_probe_main.py \
        --config vit_small_miniml_jepa_frozen \
        --dataset tanganke/stl10

    # Evaluate multiple models side-by-side on CIFAR-100
    uv run python apps/eval/linear_probe_main.py \
        --config vit_small_miniml_jepa_frozen \
                 vit_small_miniml_jepa_unfrozen \
                 vit_small_miniml_infonce_frozen \
        --dataset uoft-cs/cifar100

Results are printed to stdout and appended to ``linear_probe_results.jsonl``
in each model's checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import datasets as hf_datasets
import torch
import torch.nn as nn
import torch.utils.data

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mole_jepa import config as config_module  # noqa: E402
from mole_jepa import model_io, models, nfs_registry  # noqa: E402
from mole_jepa.data import transforms as data_transforms  # noqa: E402

_RESULTS_FILE = "linear_probe_results.jsonl"

# ── dataset configs ───────────────────────────────────────────────────────────

_DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "tanganke/stl10": {
        "train_split": "train",
        "test_split": "test",
        "image_field": "image",
        "label_fields": {"label": 10},
    },
    "uoft-cs/cifar100": {
        "train_split": "train",
        "test_split": "test",
        "image_field": "img",
        "label_fields": {"fine_label": 100, "coarse_label": 20},
    },
}


# ── device ────────────────────────────────────────────────────────────────────


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── args ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Linear probe evaluation for MoLeJEPA image encoders."
    )

    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        metavar="NAME",
        help="One or more registered model names to evaluate.",
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
    parser.add_argument(
        "--dataset",
        default="tanganke/stl10",
        choices=list(_DATASET_CONFIGS),
        metavar="DATASET",
        help=(
            "HuggingFace dataset to use. "
            f"Choices: {list(_DATASET_CONFIGS)}. "
            "Default: tanganke/stl10."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for image encoding (default: 256).",
    )

    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=100,
        help="Epochs to train the linear head (default: 100).",
    )
    parser.add_argument(
        "--probe-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the linear head (default: 1e-3).",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=512,
        help="Batch size for linear head training (default: 512).",
    )

    return parser.parse_args()


# ── encoding ──────────────────────────────────────────────────────────────────


@torch.inference_mode()
def encode_dataset(
    model: models.MoLeJEPA,
    hf_ds: hf_datasets.Dataset,  # type: ignore[name-defined]
    image_field: str,
    image_transform: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Encode all images in *hf_ds* with the frozen image encoder.

    Args:
        model: MoLeJEPA model whose image encoder to use.
        hf_ds: HuggingFace dataset split.
        image_field: Name of the PIL image column.
        image_transform: Callable that converts a PIL image to a tensor.
        batch_size: Encoding batch size.
        device: Inference device.

    Returns:
        A tuple ``(embeddings, labels)`` where ``embeddings`` is a float
        tensor of shape ``(N, embed_dim)`` on CPU and ``labels`` is a dict
        mapping each label column name to a long tensor of shape ``(N,)``.
    """
    model.eval()

    label_cols = [
        c
        for c in hf_ds.column_names
        if c != image_field  # type: ignore[union-attr]
    ]

    def _collate(
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        imgs = torch.stack([image_transform(ex[image_field]) for ex in batch])
        lbls = {
            k: torch.tensor([ex[k] for ex in batch], dtype=torch.long)
            for k in label_cols
        }
        return imgs, lbls

    # num_workers=0: the HF dataset is already in RAM and the bottleneck is
    # GPU throughput, not data loading.  Workers would require pickling the
    # _collate closure (which captures image_transform), which fails under
    # Python 3.14's forkserver start method on Linux.
    loader = torch.utils.data.DataLoader(
        hf_ds,  # type: ignore[arg-type]
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_collate,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    all_embs: list[torch.Tensor] = []
    all_labels: dict[str, list[torch.Tensor]] = {k: [] for k in label_cols}

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        emb = model.image_encoder(imgs)
        all_embs.append(emb.cpu())
        for k, v in lbls.items():
            all_labels[k].append(v)

    embeddings = torch.cat(all_embs, dim=0)
    labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items()}
    return embeddings, labels


# ── linear probe ──────────────────────────────────────────────────────────────


def train_linear_probe(
    train_embs: torch.Tensor,
    train_labels: torch.Tensor,
    n_classes: int,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> nn.Linear:
    """Train a single linear layer on frozen embeddings.

    Args:
        train_embs: Training embeddings of shape ``(N, embed_dim)``.
        train_labels: Integer class labels of shape ``(N,)``.
        n_classes: Number of output classes.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        device: Training device.

    Returns:
        The trained :class:`torch.nn.Linear` module (on CPU).
    """
    embed_dim = train_embs.shape[1]
    head = nn.Linear(embed_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(train_embs, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for emb_batch, lbl_batch in loader:
            emb_batch = emb_batch.to(device)
            lbl_batch = lbl_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(head(emb_batch), lbl_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            avg = total_loss / len(loader)
            print(f"    epoch {epoch + 1:3d}/{epochs}  loss={avg:.4f}")

    return head.cpu()


@torch.inference_mode()
def evaluate_probe(
    head: nn.Linear,
    test_embs: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute top-1 accuracy of *head* on *test_embs*.

    Args:
        head: Trained linear head.
        test_embs: Test embeddings of shape ``(N, embed_dim)``.
        test_labels: Ground-truth integer labels of shape ``(N,)``.
        device: Evaluation device.

    Returns:
        Top-1 accuracy as a float in ``[0, 1]``.
    """
    head = head.to(device)
    head.eval()
    logits = head(test_embs.to(device))
    preds = logits.argmax(dim=-1).cpu()
    return (preds == test_labels).float().mean().item()


# ── result helpers ────────────────────────────────────────────────────────────


def _fmt_row(
    name: str,
    dataset: str,
    scores: dict[str, float],
) -> str:
    parts = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
    return f"  {name:<40s}  {dataset}  {parts}"


def _save_result(
    checkpoint_dir: pathlib.Path,
    name: str,
    dataset: str,
    scores: dict[str, float],
) -> None:
    """Append a JSON result line to *checkpoint_dir/linear_probe_results.jsonl*."""
    record = {"model": name, "dataset": dataset, **scores}
    results_path = checkpoint_dir / _RESULTS_FILE
    with results_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"  Results appended to {results_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run linear probe evaluation for one or more registered models."""
    args = parse_args()
    device = _get_device()
    print(f"Device: {device}")

    ds_cfg = _DATASET_CONFIGS[args.dataset]

    print(f"\nLoading dataset '{args.dataset}' …")
    t0 = time.perf_counter()
    train_hf = hf_datasets.load_dataset(  # type: ignore[call-overload]
        args.dataset, split=ds_cfg["train_split"]
    )
    test_hf = hf_datasets.load_dataset(  # type: ignore[call-overload]
        args.dataset, split=ds_cfg["test_split"]
    )
    print(
        f"  train={len(train_hf):,}  test={len(test_hf):,} "  # type: ignore[arg-type]
        f"in {time.perf_counter() - t0:.1f}s"
    )

    all_results: list[tuple[str, dict[str, float]]] = []

    for name in args.config:
        print(f"\n{'=' * 70}")
        print(f"Model: {name}")
        print(f"{'=' * 70}")

        # ── load model ────────────────────────────────────────────────────────
        print("Loading model …")
        t0 = time.perf_counter()
        model = model_io.load_model(
            name,
            registry_dir=args.registry_path,
            map_location=device,
        )
        model.to(device)
        model.eval()
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

        # Build the image transform matching what was used during training.
        entry = nfs_registry.get_entry(name, args.registry_path)
        data_cfg = config_module.DataConfig(
            image_processor_model_name=entry.config.image_encoder_model_name,
        )
        image_transform = data_transforms.build_image_transform(
            data_cfg.image_processor_model_name, train=False
        )

        # ── encode ────────────────────────────────────────────────────────────
        print("Encoding train split …")
        t0 = time.perf_counter()
        train_embs, train_labels = encode_dataset(
            model,
            train_hf,  # type: ignore[arg-type]
            ds_cfg["image_field"],
            image_transform,
            args.batch_size,
            device,
        )
        print(f"  {train_embs.shape} in {time.perf_counter() - t0:.1f}s")

        print("Encoding test split …")
        t0 = time.perf_counter()
        test_embs, test_labels = encode_dataset(
            model,
            test_hf,  # type: ignore[arg-type]
            ds_cfg["image_field"],
            image_transform,
            args.batch_size,
            device,
        )
        print(f"  {test_embs.shape} in {time.perf_counter() - t0:.1f}s")

        # ── linear probe per label field ──────────────────────────────────────
        scores: dict[str, float] = {}
        for label_field, n_classes in ds_cfg["label_fields"].items():
            print(
                f"\nTraining linear probe: {label_field} "
                f"({n_classes} classes, {args.probe_epochs} epochs) …"
            )
            t0 = time.perf_counter()
            head = train_linear_probe(
                train_embs,
                train_labels[label_field],
                n_classes,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                batch_size=args.probe_batch_size,
                device=device,
            )
            acc = evaluate_probe(head, test_embs, test_labels[label_field], device)
            scores[label_field] = acc
            print(
                f"  {label_field}: top-1={acc:.3f} in {time.perf_counter() - t0:.1f}s"
            )

        all_results.append((name, scores))
        print(_fmt_row(name, args.dataset, scores))
        _save_result(entry.checkpoint_dir, name, args.dataset, scores)

    # ── summary ───────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        for name, scores in all_results:
            print(_fmt_row(name, args.dataset, scores))

    print("\nDone.")


if __name__ == "__main__":
    main()
