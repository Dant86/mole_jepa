r"""Text classification linear probe for MoLeJEPA text encoders.

Encodes all sentences from a text classification dataset with a frozen
encoder, then trains a single linear layer on the resulting embeddings.
No fine-tuning of the encoder — weights are frozen throughout.

Three encoders are compared for each registered model:

``full``
    The trained text encoder: MiniLM backbone + trained projection head.

``backbone``
    MiniLM backbone only (no projection), using weights from the trained
    model.  Output is in the backbone's native hidden dimension.  Tells you
    whether the projection head adds or removes discriminative information.

``baseline``
    Vanilla MiniLM loaded fresh from HuggingFace with no training on our
    data.  This is the quality ceiling for pure text representations — scores
    lower than this indicate our training degraded MiniLM.

Supported datasets (all fetched from HuggingFace):

* ``fancyzhx/ag_news`` — 4 classes, 120 k train / 7.6 k test (default)

Usage::

    uv run python apps/eval/text_probe_main.py \\
        --config vit_small_miniml_jepa_unfrozen_lam05_v2 \\
                 vit_small_miniml_jepa_unfrozen_lam05_laplace \\
                 vit_small_miniml_infonce_frozen_v3

Results are printed to stdout and appended to ``text_probe_results.jsonl``
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
import torch.nn.functional as F
import torch.utils.data

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import transformers  # noqa: E402

from mole_jepa import model_io, models, registry  # noqa: E402
from mole_jepa.data import transforms as data_transforms  # noqa: E402

_RESULTS_FILE = "text_probe_results.jsonl"

_DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "fancyzhx/ag_news": {
        "train_split": "train",
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "n_classes": 4,
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
        description="Text classification linear probe for MoLeJEPA text encoders."
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
        default="fancyzhx/ag_news",
        choices=list(_DATASET_CONFIGS),
        metavar="DATASET",
        help=(
            "HuggingFace dataset to use. "
            f"Choices: {list(_DATASET_CONFIGS)}. "
            "Default: fancyzhx/ag_news."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for text encoding (default: 256).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help=(
            "Max token sequence length (default: 64). "
            "Should match the value used during training."
        ),
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


def _mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings weighted by the attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.inference_mode()
def encode_full(
    model: models.MoLeJEPA,
    sentences: list[str],
    tokenize: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode sentences through the full text encoder (backbone + projection).

    Args:
        model: MoLeJEPA model whose text encoder to use.
        sentences: List of strings to encode.
        tokenize: Tokenizer callable returning ``(input_ids, attention_mask)``.
        batch_size: Encoding batch size.
        device: Inference device.

    Returns:
        L2-normalised embeddings of shape ``(N, embed_dim)`` on CPU.
    """
    model.eval()
    parts: list[torch.Tensor] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        ids, masks = zip(*[tokenize(s) for s in batch])
        input_ids = torch.stack(list(ids)).to(device)
        attention_mask = torch.stack(list(masks)).to(device)
        emb = model.text_encoder(input_ids, attention_mask)
        parts.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(parts, dim=0)


@torch.inference_mode()
def encode_backbone(
    model: models.MoLeJEPA,
    sentences: list[str],
    tokenize: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode sentences through the backbone only (no projection head).

    Uses the trained MiniLM weights but skips the projection.  Output is in
    the backbone's native hidden dimension.

    Args:
        model: MoLeJEPA model whose text encoder backbone to use.
        sentences: List of strings to encode.
        tokenize: Tokenizer callable returning ``(input_ids, attention_mask)``.
        batch_size: Encoding batch size.
        device: Inference device.

    Returns:
        L2-normalised embeddings of shape ``(N, backbone_dim)`` on CPU.
    """
    lm = model.text_encoder.lm
    lm.eval()
    parts: list[torch.Tensor] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        ids, masks = zip(*[tokenize(s) for s in batch])
        input_ids = torch.stack(list(ids)).to(device)
        attention_mask = torch.stack(list(masks)).to(device)
        hidden = lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        emb = _mean_pool(hidden, attention_mask)
        parts.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(parts, dim=0)


@torch.inference_mode()
def encode_baseline(
    model_name: str,
    sentences: list[str],
    max_seq_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode sentences with vanilla MiniLM loaded fresh from HuggingFace.

    No training on our data.  Quality ceiling for pure text representations.

    Args:
        model_name: HuggingFace model identifier.
        sentences: List of strings to encode.
        max_seq_length: Tokenizer truncation length.
        batch_size: Encoding batch size.
        device: Inference device.

    Returns:
        L2-normalised embeddings of shape ``(N, backbone_dim)`` on CPU.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    lm = transformers.AutoModel.from_pretrained(model_name).to(device)
    lm.eval()
    parts: list[torch.Tensor] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        hidden = lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        emb = _mean_pool(hidden, attention_mask)
        parts.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(parts, dim=0)


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

    Returns:
        Top-1 accuracy as a float in ``[0, 1]``.
    """
    head = head.to(device)
    head.eval()
    preds = head(test_embs.to(device)).argmax(dim=-1).cpu()
    return (preds == test_labels).float().mean().item()


# ── result helpers ────────────────────────────────────────────────────────────


def _fmt_row(name: str, encoder: str, dataset: str, acc: float) -> str:
    return f"  {name:<44s}  {encoder:<10s}  {dataset}  acc={acc:.3f}"


def _save_result(
    checkpoint_dir: pathlib.Path,
    name: str,
    encoder: str,
    dataset: str,
    acc: float,
) -> None:
    record = {"model": name, "encoder": encoder, "dataset": dataset, "acc": acc}
    results_path = checkpoint_dir / _RESULTS_FILE
    with results_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"  Results appended to {results_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run text classification linear probe for one or more registered models."""
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
    train_sentences: list[str] = train_hf[ds_cfg["text_field"]]  # type: ignore[index]
    test_sentences: list[str] = test_hf[ds_cfg["text_field"]]  # type: ignore[index]
    train_labels = torch.tensor(  # type: ignore[call-overload]
        train_hf[ds_cfg["label_field"]],
        dtype=torch.long,  # type: ignore[index]
    )
    test_labels = torch.tensor(  # type: ignore[call-overload]
        test_hf[ds_cfg["label_field"]],
        dtype=torch.long,  # type: ignore[index]
    )
    print(
        f"  train={len(train_sentences):,}  test={len(test_sentences):,} "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    all_results: list[tuple[str, str, float]] = []
    baseline_cache: dict[str, torch.Tensor] = {}

    for name in args.config:
        print(f"\n{'=' * 70}")
        print(f"Model: {name}")
        print(f"{'=' * 70}")

        print("Loading model …")
        t0 = time.perf_counter()
        model = model_io.load_model(name, map_location=device)
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

        entry = registry.get_entry(name)
        text_model_name = entry.config.text_encoder_model_name
        tokenize = data_transforms.build_tokenizer(text_model_name, args.max_seq_length)
        n_classes = ds_cfg["n_classes"]

        # ── full encoder (backbone + projection) ─────────────────────────────
        print("Encoding with full encoder …")
        t0 = time.perf_counter()
        train_embs_full = encode_full(
            model, train_sentences, tokenize, args.batch_size, device
        )
        test_embs_full = encode_full(
            model, test_sentences, tokenize, args.batch_size, device
        )
        print(f"  {train_embs_full.shape}  in {time.perf_counter() - t0:.1f}s")
        print(f"\n  Probing full  [{n_classes} classes] …")
        head = train_linear_probe(
            train_embs_full,
            train_labels,
            n_classes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            device=device,
        )
        acc_full = evaluate_probe(head, test_embs_full, test_labels, device)
        print(_fmt_row(name, "full", args.dataset, acc_full))
        all_results.append((name, "full", acc_full))
        _save_result(entry.checkpoint_dir, name, "full", args.dataset, acc_full)

        # ── backbone only (no projection) ────────────────────────────────────
        print("\nEncoding with backbone only …")
        t0 = time.perf_counter()
        train_embs_bb = encode_backbone(
            model, train_sentences, tokenize, args.batch_size, device
        )
        test_embs_bb = encode_backbone(
            model, test_sentences, tokenize, args.batch_size, device
        )
        print(f"  {train_embs_bb.shape}  in {time.perf_counter() - t0:.1f}s")
        print(f"\n  Probing backbone  [{n_classes} classes] …")
        head = train_linear_probe(
            train_embs_bb,
            train_labels,
            n_classes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            device=device,
        )
        acc_bb = evaluate_probe(head, test_embs_bb, test_labels, device)
        print(_fmt_row(name, "backbone", args.dataset, acc_bb))
        all_results.append((name, "backbone", acc_bb))
        _save_result(entry.checkpoint_dir, name, "backbone", args.dataset, acc_bb)

        # ── vanilla baseline (once per unique text model name) ────────────────
        if text_model_name not in baseline_cache:
            print(f"\nEncoding with vanilla {text_model_name} (baseline) …")
            t0 = time.perf_counter()
            train_embs_base = encode_baseline(
                text_model_name,
                train_sentences,
                args.max_seq_length,
                args.batch_size,
                device,
            )
            test_embs_base = encode_baseline(
                text_model_name,
                test_sentences,
                args.max_seq_length,
                args.batch_size,
                device,
            )
            baseline_cache[text_model_name] = test_embs_base
            # Probe baseline train embeddings once, reuse for all models.
            print(f"  {train_embs_base.shape}  in {time.perf_counter() - t0:.1f}s")
            print(f"\n  Probing baseline  [{n_classes} classes] …")
            head = train_linear_probe(
                train_embs_base,
                train_labels,
                n_classes,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                batch_size=args.probe_batch_size,
                device=device,
            )
            acc_base = evaluate_probe(head, test_embs_base, test_labels, device)
            # Store trained head result so we don't re-probe for subsequent models.
            baseline_cache[f"{text_model_name}_acc"] = torch.tensor(acc_base)  # type: ignore[assignment]
        else:
            print(f"\nBaseline ({text_model_name}): cached")
            acc_base = float(baseline_cache[f"{text_model_name}_acc"].item())  # type: ignore[union-attr]

        print(_fmt_row("", "baseline", args.dataset, acc_base))
        all_results.append((name, "baseline", acc_base))
        _save_result(entry.checkpoint_dir, name, "baseline", args.dataset, acc_base)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    prev_model = ""
    for model_name, encoder, acc in all_results:
        if model_name != prev_model:
            print()
            prev_model = model_name
        label = model_name if encoder == "full" else ""
        print(_fmt_row(label, encoder, args.dataset, acc))

    print("\nDone.")


if __name__ == "__main__":
    main()
