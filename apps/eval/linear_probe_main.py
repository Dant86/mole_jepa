r"""Linear probe evaluation for MoLeJEPA image encoders.

Encodes all images from a classification dataset with a frozen encoder, then
trains a single linear layer on the resulting embeddings.  No fine-tuning of
the encoder — weights are frozen throughout.

Three encoders are compared for each registered model:

``full``
    The trained image encoder: ViT backbone + trained projection head.
    This is what the model actually uses at inference time.

``backbone``
    ViT backbone only (no projection), using weights from the trained model.
    Output is the CLS token in the ViT's native hidden dimension.  Tells you
    whether the projection head adds or removes discriminative information.

``baseline``
    Vanilla ViT loaded fresh from HuggingFace with no training on our data.
    This is the quality ceiling for the ViT's native representations — if the
    trained models score lower, our training degraded the ViT.

Supported datasets (all fetched from HuggingFace):

* ``tanganke/stl10``    — 10 classes, 5 k train / 8 k test (default)
* ``uoft-cs/cifar100``  — 100 fine / 20 coarse classes, 50 k train / 10 k test

Usage::

    uv run python apps/eval/linear_probe_main.py \\
        --config vit_small_miniml_jepa_frozen \\
        --dataset tanganke/stl10

    # Evaluate multiple models side-by-side on CIFAR-100
    uv run python apps/eval/linear_probe_main.py \\
        --config vit_small_miniml_jepa_frozen \\
                 vit_small_miniml_jepa_unfrozen \\
                 vit_small_miniml_infonce_frozen \\
        --dataset uoft-cs/cifar100

Results are printed to stdout and appended to ``linear_probe_results.jsonl``
in each model's checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
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

import transformers  # noqa: E402

from mole_jepa import config as config_module  # noqa: E402
from mole_jepa import model_io, models, registry  # noqa: E402
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


def _make_loader(
    hf_ds: hf_datasets.Dataset,  # type: ignore[name-defined]
    image_field: str,
    image_transform: Any,
    batch_size: int,
    device: torch.device,
) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
    label_cols = [c for c in hf_ds.column_names if c != image_field]  # type: ignore[union-attr]

    def _collate(
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        imgs = torch.stack([image_transform(ex[image_field]) for ex in batch])
        lbls = {
            k: torch.tensor([ex[k] for ex in batch], dtype=torch.long)
            for k in label_cols
        }
        return imgs, lbls

    # num_workers=0: HF dataset is already in RAM; bottleneck is GPU throughput.
    # Workers require pickling _collate (which captures image_transform), which
    # fails under Python 3.14's forkserver start method on Linux.
    return torch.utils.data.DataLoader(
        hf_ds,  # type: ignore[arg-type]
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_collate,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )


@torch.inference_mode()
def encode_full(
    model: models.MoLeJEPA,
    hf_ds: hf_datasets.Dataset,  # type: ignore[name-defined]
    image_field: str,
    image_transform: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Encode images through the full image encoder (ViT backbone + projection).

    Args:
        model: MoLeJEPA model whose image encoder to use.
        hf_ds: HuggingFace dataset split.
        image_field: Name of the PIL image column.
        image_transform: Callable that converts a PIL image to a tensor.
        batch_size: Encoding batch size.
        device: Inference device.

    Returns:
        ``(embeddings, labels)`` where ``embeddings`` is shape
        ``(N, embed_dim)`` on CPU and ``labels`` maps each label column to a
        long tensor of shape ``(N,)``.
    """
    model.eval()
    loader = _make_loader(hf_ds, image_field, image_transform, batch_size, device)
    all_embs: list[torch.Tensor] = []
    all_labels: dict[str, list[torch.Tensor]] = {}

    for imgs, lbls in loader:
        if not all_labels:
            all_labels = {k: [] for k in lbls}
        emb = model.image_encoder(imgs.to(device))
        all_embs.append(emb.cpu())
        for k, v in lbls.items():
            all_labels[k].append(v)

    return torch.cat(all_embs, dim=0), {k: torch.cat(v) for k, v in all_labels.items()}


@torch.inference_mode()
def encode_backbone(
    model: models.MoLeJEPA,
    hf_ds: hf_datasets.Dataset,  # type: ignore[name-defined]
    image_field: str,
    image_transform: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Encode images through the ViT backbone only (no projection head).

    Uses the trained ViT weights but skips the projection, so the output is
    the CLS token in the ViT's native hidden dimension.  Isolates whether the
    projection head is adding or removing discriminative information.

    Args:
        model: MoLeJEPA model whose ViT backbone to use.
        hf_ds: HuggingFace dataset split.
        image_field: Name of the PIL image column.
        image_transform: Callable that converts a PIL image to a tensor.
        batch_size: Encoding batch size.
        device: Inference device.

    Returns:
        ``(embeddings, labels)`` where ``embeddings`` is shape
        ``(N, vit_hidden_dim)`` on CPU.
    """
    vit = model.image_encoder.vit
    vit.eval()
    loader = _make_loader(hf_ds, image_field, image_transform, batch_size, device)
    all_embs: list[torch.Tensor] = []
    all_labels: dict[str, list[torch.Tensor]] = {}

    for imgs, lbls in loader:
        if not all_labels:
            all_labels = {k: [] for k in lbls}
        cls = vit(pixel_values=imgs.to(device)).last_hidden_state[:, 0]
        all_embs.append(cls.cpu())
        for k, v in lbls.items():
            all_labels[k].append(v)

    return torch.cat(all_embs, dim=0), {k: torch.cat(v) for k, v in all_labels.items()}


@torch.inference_mode()
def encode_baseline(
    model_name: str,
    hf_ds: hf_datasets.Dataset,  # type: ignore[name-defined]
    image_field: str,
    image_transform: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Encode images with a vanilla ViT loaded fresh from HuggingFace.

    No training on our data.  This is the quality ceiling for the ViT's native
    representations — scores lower than this indicate our training degraded the
    ViT's discriminative features.

    Args:
        model_name: HuggingFace model identifier (e.g.
            ``"WinKawaks/vit-small-patch16-224"``).
        hf_ds: HuggingFace dataset split.
        image_field: Name of the PIL image column.
        image_transform: Callable that converts a PIL image to a tensor.
        batch_size: Encoding batch size.
        device: Inference device.

    Returns:
        ``(embeddings, labels)`` where ``embeddings`` is shape
        ``(N, vit_hidden_dim)`` on CPU.
    """
    vit = transformers.AutoModel.from_pretrained(model_name).to(device)
    vit.eval()
    loader = _make_loader(hf_ds, image_field, image_transform, batch_size, device)
    all_embs: list[torch.Tensor] = []
    all_labels: dict[str, list[torch.Tensor]] = {}

    for imgs, lbls in loader:
        if not all_labels:
            all_labels = {k: [] for k in lbls}
        cls = vit(pixel_values=imgs.to(device)).last_hidden_state[:, 0]
        all_embs.append(cls.cpu())
        for k, v in lbls.items():
            all_labels[k].append(v)

    return torch.cat(all_embs, dim=0), {k: torch.cat(v) for k, v in all_labels.items()}


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
    encoder: str,
    dataset: str,
    scores: dict[str, float],
) -> str:
    parts = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
    return f"  {name:<40s}  {encoder:<10s}  {dataset}  {parts}"


def _save_result(
    checkpoint_dir: pathlib.Path,
    name: str,
    encoder: str,
    dataset: str,
    scores: dict[str, float],
) -> None:
    """Append a JSON result line to *checkpoint_dir/linear_probe_results.jsonl*."""
    record = {"model": name, "encoder": encoder, "dataset": dataset, **scores}
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

    # All results accumulated for the summary table.
    all_results: list[tuple[str, str, dict[str, float]]] = []

    # Baseline cache: load fresh ViT once per unique model name.
    baseline_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for name in args.config:
        print(f"\n{'=' * 70}")
        print(f"Model: {name}")
        print(f"{'=' * 70}")

        print("Loading model …")
        t0 = time.perf_counter()
        model = model_io.load_model(
            name,
            map_location=device,
        )
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

        entry = registry.get_entry(name)
        image_model_name = entry.config.image_encoder_model_name
        data_cfg = config_module.DataConfig(
            image_processor_model_name=image_model_name,
        )
        image_transform = data_transforms.build_image_transform(
            data_cfg.image_processor_model_name, train=False
        )

        # ── full encoder (backbone + projection) ─────────────────────────────
        print("Encoding with full encoder (backbone + projection) …")
        t0 = time.perf_counter()
        train_embs_full, train_labels = encode_full(
            model,
            train_hf,
            ds_cfg["image_field"],  # type: ignore[arg-type]
            image_transform,
            args.batch_size,
            device,
        )
        test_embs_full, test_labels = encode_full(
            model,
            test_hf,
            ds_cfg["image_field"],  # type: ignore[arg-type]
            image_transform,
            args.batch_size,
            device,
        )
        print(f"  {train_embs_full.shape}  in {time.perf_counter() - t0:.1f}s")

        scores_full: dict[str, float] = {}
        for label_field, n_classes in ds_cfg["label_fields"].items():
            print(f"\n  Probing full  [{label_field}, {n_classes} classes] …")
            head = train_linear_probe(
                train_embs_full,
                train_labels[label_field],
                n_classes,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                batch_size=args.probe_batch_size,
                device=device,
            )
            scores_full[label_field] = evaluate_probe(
                head, test_embs_full, test_labels[label_field], device
            )
        print(_fmt_row(name, "full", args.dataset, scores_full))
        all_results.append((name, "full", scores_full))
        _save_result(entry.checkpoint_dir, name, "full", args.dataset, scores_full)

        # ── backbone only (no projection) ────────────────────────────────────
        print("\nEncoding with backbone only (no projection) …")
        t0 = time.perf_counter()
        train_embs_bb, _ = encode_backbone(
            model,
            train_hf,
            ds_cfg["image_field"],  # type: ignore[arg-type]
            image_transform,
            args.batch_size,
            device,
        )
        test_embs_bb, _ = encode_backbone(
            model,
            test_hf,
            ds_cfg["image_field"],  # type: ignore[arg-type]
            image_transform,
            args.batch_size,
            device,
        )
        print(f"  {train_embs_bb.shape}  in {time.perf_counter() - t0:.1f}s")

        scores_bb: dict[str, float] = {}
        for label_field, n_classes in ds_cfg["label_fields"].items():
            print(f"\n  Probing backbone  [{label_field}, {n_classes} classes] …")
            head = train_linear_probe(
                train_embs_bb,
                train_labels[label_field],
                n_classes,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                batch_size=args.probe_batch_size,
                device=device,
            )
            scores_bb[label_field] = evaluate_probe(
                head, test_embs_bb, test_labels[label_field], device
            )
        print(_fmt_row(name, "backbone", args.dataset, scores_bb))
        all_results.append((name, "backbone", scores_bb))
        _save_result(entry.checkpoint_dir, name, "backbone", args.dataset, scores_bb)

        # ── vanilla baseline (once per unique image model name) ───────────────
        if image_model_name not in baseline_cache:
            print(f"\nEncoding with vanilla {image_model_name} (baseline) …")
            t0 = time.perf_counter()
            train_embs_base, _ = encode_baseline(
                image_model_name,
                train_hf,
                ds_cfg["image_field"],  # type: ignore[arg-type]
                image_transform,
                args.batch_size,
                device,
            )
            test_embs_base, _ = encode_baseline(
                image_model_name,
                test_hf,
                ds_cfg["image_field"],  # type: ignore[arg-type]
                image_transform,
                args.batch_size,
                device,
            )
            baseline_cache[image_model_name] = (train_embs_base, test_embs_base)
            print(f"  {train_embs_base.shape}  in {time.perf_counter() - t0:.1f}s")
        else:
            print(f"\nBaseline ({image_model_name}): cached")
            train_embs_base, test_embs_base = baseline_cache[image_model_name]

        scores_base: dict[str, float] = {}
        for label_field, n_classes in ds_cfg["label_fields"].items():
            print(f"\n  Probing baseline  [{label_field}, {n_classes} classes] …")
            head = train_linear_probe(
                train_embs_base,
                train_labels[label_field],
                n_classes,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                batch_size=args.probe_batch_size,
                device=device,
            )
            scores_base[label_field] = evaluate_probe(
                head, test_embs_base, test_labels[label_field], device
            )
        print(_fmt_row("", "baseline", args.dataset, scores_base))
        all_results.append((name, "baseline", scores_base))
        _save_result(entry.checkpoint_dir, name, "baseline", args.dataset, scores_base)

    # ── summary ───────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        prev_model = ""
        for model_name, encoder, scores in all_results:
            if model_name != prev_model:
                print()
                prev_model = model_name
            label = model_name if encoder == "full" else ""
            print(_fmt_row(label, encoder, args.dataset, scores))

    print("\nDone.")


if __name__ == "__main__":
    main()
