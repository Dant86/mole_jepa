"""Semantic Textual Similarity (STS-B) evaluation for MoLeJEPA text encoders.

Measures how well cosine similarity between text embeddings correlates with
human-annotated sentence similarity scores (Spearman and Pearson rho).

Three encoders are compared for each registered model:

``full``
    The trained text encoder: MiniLM backbone + trained projection head.
    This is what the model actually uses at inference time.

``backbone``
    MiniLM backbone only (no projection), using weights from the trained
    model.  Tells you whether the projection head adds or removes information.

``baseline``
    Vanilla MiniLM loaded fresh from HuggingFace with no training on our
    data.  This is the ceiling for text-only quality — if the trained models
    score lower, our training degraded MiniLM's semantic representations.

Dataset: ``mteb/stsbenchmark-sts`` test split (~1.4 k sentence pairs, scores
0–5).

Usage::

    uv run python apps/eval/sts_main.py \
        --config vit_small_miniml_jepa_frozen \
                 vit_small_miniml_jepa_unfrozen \
                 vit_small_miniml_infonce_frozen
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
import torch.nn.functional as F

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import transformers  # noqa: E402

from mole_jepa import model_io, models, registry  # noqa: E402
from mole_jepa.data import transforms as data_transforms  # noqa: E402

_RESULTS_FILE = "sts_results.jsonl"
_STS_DATASET = "mteb/stsbenchmark-sts"
_STS_SPLIT = "test"
_STS_SCORE_MAX = 5.0  # raw scores are 0–5; normalise to 0–1 for display


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
        description="STS-B evaluation for MoLeJEPA text encoders."
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

    Uses the trained MiniLM weights but skips the projection, so the output
    is in the backbone's native 384-d space.  Isolates whether the projection
    head is adding or removing information relative to the full encoder.

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
    """Encode sentences with vanilla MiniLM (no training on our data).

    Loads a fresh copy of *model_name* from HuggingFace, applies mean
    pooling, and L2-normalises.  This is the quality ceiling for pure text
    semantics — scores lower than this indicate training degraded MiniLM.

    Args:
        model_name: HuggingFace model identifier (e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``).
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


# ── metrics ───────────────────────────────────────────────────────────────────


def spearman_pearson(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    """Compute Spearman and Pearson correlation between two 1-D tensors.

    Args:
        pred: Predicted similarity scores of shape ``(N,)``.
        target: Ground-truth similarity scores of shape ``(N,)``.

    Returns:
        ``(spearman_rho, pearson_r)`` as floats.
    """
    from scipy.stats import pearsonr, spearmanr

    p = pred.numpy()
    t = target.numpy()
    spearman = float(spearmanr(p, t).correlation)  # type: ignore[attr-defined]
    pearson = float(pearsonr(p, t).statistic)  # type: ignore[attr-defined]
    return spearman, pearson


def cosine_similarities(
    embs1: torch.Tensor,
    embs2: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise cosine similarity between two aligned embedding matrices.

    Both inputs are assumed to be already L2-normalised.

    Args:
        embs1: Embeddings of shape ``(N, d)``.
        embs2: Embeddings of shape ``(N, d)``.

    Returns:
        Similarity tensor of shape ``(N,)``.
    """
    return (embs1 * embs2).sum(dim=-1)


# ── result helpers ────────────────────────────────────────────────────────────


def _fmt_row(
    name: str,
    encoder: str,
    spearman: float,
    pearson: float,
) -> str:
    return (
        f"  {name:<40s}  {encoder:<10s}  spearman={spearman:.4f}  pearson={pearson:.4f}"
    )


def _save_result(
    checkpoint_dir: pathlib.Path,
    name: str,
    encoder: str,
    spearman: float,
    pearson: float,
) -> None:
    """Append a JSON result line to *checkpoint_dir/sts_results.jsonl*."""
    record = {
        "model": name,
        "encoder": encoder,
        "dataset": _STS_DATASET,
        "split": _STS_SPLIT,
        "spearman": spearman,
        "pearson": pearson,
    }
    results_path = checkpoint_dir / _RESULTS_FILE
    with results_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run STS-B evaluation for one or more registered models."""
    args = parse_args()
    device = _get_device()
    print(f"Device: {device}")

    print(f"\nLoading {_STS_DATASET} ({_STS_SPLIT} split) …")
    t0 = time.perf_counter()
    sts = hf_datasets.load_dataset(_STS_DATASET, split=_STS_SPLIT)  # type: ignore[call-overload]
    sentences1: list[str] = sts["sentence1"]  # type: ignore[index]
    sentences2: list[str] = sts["sentence2"]  # type: ignore[index]
    gt_scores = torch.tensor(  # type: ignore[call-overload]
        [s / _STS_SCORE_MAX for s in sts["score"]],
        dtype=torch.float32,  # type: ignore[index]
    )
    print(f"  {len(sentences1):,} pairs in {time.perf_counter() - t0:.1f}s")

    # Track all results for the summary table.
    all_results: list[tuple[str, str, float, float]] = []

    # ── baseline: vanilla MiniLM, loaded once per unique model name ──────────
    # We only need one baseline per text encoder model name, not one per
    # registered config.  Cache by model name to avoid redundant HF downloads.
    baseline_cache: dict[str, tuple[float, float]] = {}

    for name in args.config:
        print(f"\n{'=' * 70}")
        print(f"Model: {name}")
        print(f"{'=' * 70}")

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

        entry = registry.get_entry(name, args.registry_path)
        text_model_name = entry.config.text_encoder_model_name
        tokenize = data_transforms.build_tokenizer(text_model_name, args.max_seq_length)

        # ── full encoder (backbone + projection) ─────────────────────────────
        print("Encoding with full encoder (backbone + projection) …")
        t0 = time.perf_counter()
        embs1_full = encode_full(model, sentences1, tokenize, args.batch_size, device)
        embs2_full = encode_full(model, sentences2, tokenize, args.batch_size, device)
        sims_full = cosine_similarities(embs1_full, embs2_full)
        sp_full, pe_full = spearman_pearson(sims_full, gt_scores)
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.1f}s  →  {_fmt_row('', 'full', sp_full, pe_full).strip()}")
        all_results.append((name, "full", sp_full, pe_full))
        _save_result(entry.checkpoint_dir, name, "full", sp_full, pe_full)

        # ── backbone only (no projection) ────────────────────────────────────
        print("Encoding with backbone only (no projection) …")
        t0 = time.perf_counter()
        embs1_bb = encode_backbone(model, sentences1, tokenize, args.batch_size, device)
        embs2_bb = encode_backbone(model, sentences2, tokenize, args.batch_size, device)
        sims_bb = cosine_similarities(embs1_bb, embs2_bb)
        sp_bb, pe_bb = spearman_pearson(sims_bb, gt_scores)
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.1f}s  →  {_fmt_row('', 'backbone', sp_bb, pe_bb).strip()}")
        all_results.append((name, "backbone", sp_bb, pe_bb))
        _save_result(entry.checkpoint_dir, name, "backbone", sp_bb, pe_bb)

        # ── vanilla baseline (once per unique text model name) ────────────────
        if text_model_name not in baseline_cache:
            print(f"Encoding with vanilla {text_model_name} (baseline) …")
            t0 = time.perf_counter()
            embs1_base = encode_baseline(
                text_model_name,
                sentences1,
                args.max_seq_length,
                args.batch_size,
                device,
            )
            embs2_base = encode_baseline(
                text_model_name,
                sentences2,
                args.max_seq_length,
                args.batch_size,
                device,
            )
            sims_base = cosine_similarities(embs1_base, embs2_base)
            sp_base, pe_base = spearman_pearson(sims_base, gt_scores)
            baseline_cache[text_model_name] = (sp_base, pe_base)
            print(
                f"  {time.perf_counter() - t0:.1f}s  →  "
                f"{_fmt_row('', 'baseline', sp_base, pe_base).strip()}"
            )
        else:
            sp_base, pe_base = baseline_cache[text_model_name]
            print(f"Baseline ({text_model_name}): cached")

        all_results.append((name, "baseline", sp_base, pe_base))

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    prev_model = ""
    for model_name, encoder, spearman, pearson in all_results:
        if model_name != prev_model:
            print()
            prev_model = model_name
        label = model_name if encoder == "full" else ""
        print(_fmt_row(label, encoder, spearman, pearson))

    print("\nDone.")


if __name__ == "__main__":
    main()
