"""Image-text retrieval evaluation on COCO.

Loads a trained MoLeJEPA model by registry name, encodes all COCO validation
images and captions, and reports R@1/5/10 in both retrieval directions.

Usage::

    uv run python apps/eval/retrieval_main.py \
        --config vit_small_miniml_jepa_frozen \
        --coco-split validation

    # Evaluate multiple models in one pass
    uv run python apps/eval/retrieval_main.py \
        --config vit_small_miniml_jepa_frozen \
                 vit_small_miniml_jepa_unfrozen \
                 vit_small_miniml_infonce_frozen

Results are printed to stdout in a human-readable table and written to a
``retrieval_results.jsonl`` file in the model's checkpoint directory (one
JSON line per evaluated model).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

import datasets as hf_datasets
import torch

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mole_jepa import config as config_module  # noqa: E402
from mole_jepa import model_io, models, nfs_registry  # noqa: E402
from mole_jepa.data.coco import COCODataset  # noqa: E402
from mole_jepa.evaluation.retrieval import (  # noqa: E402
    RetrievalResult,
    retrieval_metrics,
)

_RESULTS_FILE = "retrieval_results.jsonl"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MoLeJEPA model(s) on COCO retrieval."
    )

    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        metavar="NAME",
        help=(
            "One or more registered model names to evaluate. "
            "Results are printed side-by-side."
        ),
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

    # ── evaluation dataset ────────────────────────────────────────────────────
    parser.add_argument(
        "--coco-dataset",
        default="clip-benchmark/wds_flickr30k",
        metavar="DATASET",
        help=(
            "HuggingFace dataset identifier (must be Parquet-native, not a "
            "loading script). Defaults to 'clip-benchmark/wds_flickr30k' "
            "(31k images, 5 captions each, no auth required). "
            "For COCO proper use 'yerevann/coco-karpathy' with "
            "--coco-split test --caption-field sentences."
        ),
    )
    parser.add_argument(
        "--coco-split",
        default="test",
        metavar="SPLIT",
        help="Dataset split to evaluate on (default: 'test').",
    )
    parser.add_argument(
        "--image-field",
        default="jpg",
        metavar="FIELD",
        help=(
            "Name of the image column in the dataset "
            "(default: 'jpg' for clip-benchmark/wds_flickr30k)."
        ),
    )
    parser.add_argument(
        "--caption-field",
        default="txt",
        metavar="FIELD",
        help=(
            "Name of the captions column in the dataset "
            "(default: 'txt' for clip-benchmark/wds_flickr30k)."
        ),
    )
    parser.add_argument(
        "--captions-per-image",
        type=int,
        default=5,
        help="Number of captions per image in the evaluation set (default: 5).",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help=(
            "Treat each dataset row as a single (image, caption) pair and "
            "group consecutive --captions-per-image rows per image. "
            "Required for clip-benchmark/wds_flickr30k and other WebDataset "
            "repos where the caption field is a plain string."
        ),
    )

    # ── data / inference ──────────────────────────────────────────────────────
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help=(
            "Max token sequence length for caption encoding (default: 64). "
            "Should match the value used during training."
        ),
    )

    return parser.parse_args()


# ── encoding helpers ──────────────────────────────────────────────────────────


@torch.inference_mode()
def encode_images(
    model: models.MoLeJEPA,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    use_predictor: bool = False,
) -> torch.Tensor:
    """Encode all images from *loader* and return the stacked embeddings.

    For JEPA models, pass ``use_predictor=True`` to run image embeddings
    through the predictor before retrieval.  The predictor maps ``z_v`` into
    the text embedding space (``ẑ_t``), which is what should be compared
    against text embeddings at retrieval time.  For contrastive models both
    encoders are directly aligned, so the predictor is not needed.

    Args:
        model: The MoLeJEPA model whose image encoder to use.
        loader: DataLoader yielding ``(pixel_values,)`` tuples.
        device: Target device for inference.
        use_predictor: If ``True``, pass ``z_v`` through ``model.predictor``
            before returning.

    Returns:
        Float tensor of shape ``(N, embed_dim)`` on CPU.
    """
    model.eval()
    parts: list[torch.Tensor] = []
    for (pixel_values,) in loader:
        pixel_values = pixel_values.to(device)
        emb = model.image_encoder(pixel_values)
        if use_predictor:
            emb = model.predictor(emb)
        parts.append(emb.cpu())
    return torch.cat(parts, dim=0)


@torch.inference_mode()
def encode_captions(
    model: models.MoLeJEPA,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> torch.Tensor:
    """Encode all captions from *loader* and return the stacked embeddings.

    Args:
        model: The MoLeJEPA model whose text encoder to use.
        loader: DataLoader yielding ``(input_ids, attention_mask)`` tuples.
        device: Target device for inference.

    Returns:
        Float tensor of shape ``(N_caps, embed_dim)`` on CPU.
    """
    model.eval()
    parts: list[torch.Tensor] = []
    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        emb = model.text_encoder(input_ids, attention_mask)
        parts.append(emb.cpu())
    return torch.cat(parts, dim=0)


# ── result formatting ─────────────────────────────────────────────────────────


def _fmt_row(name: str, r: RetrievalResult) -> str:
    return (
        f"  {name:<40s}  "
        f"i2t  R@1={r.i2t_r1:.3f}  R@5={r.i2t_r5:.3f}  R@10={r.i2t_r10:.3f}  "
        f"  t2i  R@1={r.t2i_r1:.3f}  R@5={r.t2i_r5:.3f}  R@10={r.t2i_r10:.3f}"
    )


def _save_result(
    checkpoint_dir: pathlib.Path,
    name: str,
    result: RetrievalResult,
    dataset: str,
    split: str,
) -> None:
    """Append a JSON result line to *checkpoint_dir/retrieval_results.jsonl*."""
    record = {
        "model": name,
        "dataset": dataset,
        "split": split,
        **dataclasses.asdict(result),
    }
    results_path = checkpoint_dir / _RESULTS_FILE
    with results_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"  Results appended to {results_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run retrieval evaluation for one or more registered models."""
    args = parse_args()
    device = _get_device()
    print(f"Device: {device}")

    # Load the evaluation dataset once — shared across all models.
    print(f"\nLoading dataset '{args.coco_dataset}' split='{args.coco_split}' …")
    t0 = time.perf_counter()
    hf_ds = hf_datasets.load_dataset(  # type: ignore[call-overload]
        args.coco_dataset,
        split=args.coco_split,
    )
    print(f"  Loaded {len(hf_ds):,} examples in {time.perf_counter() - t0:.1f}s")

    all_results: list[tuple[str, RetrievalResult]] = []

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

        # ── derive data config from the model config ──────────────────────────
        # The image processor and tokenizer share the same HF identifier as the
        # respective encoder backbone, so we can derive DataConfig from
        # ModelConfig without storing it separately in the registry.
        entry = nfs_registry.get_entry(name, args.registry_path)
        data_cfg = config_module.DataConfig(
            image_processor_model_name=entry.config.image_encoder_model_name,
            tokenizer_model_name=entry.config.text_encoder_model_name,
            max_seq_length=args.max_seq_length,
        )

        # ── build COCO dataset ────────────────────────────────────────────────
        print("Pre-processing evaluation data …")
        t0 = time.perf_counter()
        coco = COCODataset(
            hf_dataset=hf_ds,
            config=data_cfg,
            captions_per_image=args.captions_per_image,
            caption_field=args.caption_field,
            image_field=args.image_field,
            flat=args.flat,
        )
        print(
            f"  {coco.n_images:,} images × {coco.captions_per_image} captions "
            f"in {time.perf_counter() - t0:.1f}s"
        )

        img_loader = coco.image_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        cap_loader = coco.caption_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # ── encode ────────────────────────────────────────────────────────────
        # For JEPA models the predictor maps z_v into the text embedding space;
        # retrieval compares ẑ_t = predictor(z_v) against z_t.
        # For contrastive models both encoders are directly aligned.
        use_predictor = not entry.config.contrastive
        print(
            f"Encoding images "
            f"({'z_v → predictor → ẑ_t' if use_predictor else 'z_v direct'}) …"
        )
        t0 = time.perf_counter()
        image_embs = encode_images(
            model, img_loader, device, use_predictor=use_predictor
        )
        print(f"  {image_embs.shape} in {time.perf_counter() - t0:.1f}s")

        print("Encoding captions …")
        t0 = time.perf_counter()
        text_embs = encode_captions(model, cap_loader, device)
        print(f"  {text_embs.shape} in {time.perf_counter() - t0:.1f}s")

        # ── metrics ───────────────────────────────────────────────────────────
        result = retrieval_metrics(
            image_embs,
            text_embs,
            captions_per_image=coco.captions_per_image,
        )
        all_results.append((name, result))
        print(_fmt_row(name, result))
        _save_result(
            entry.checkpoint_dir,
            name,
            result,
            args.coco_dataset,
            args.coco_split,
        )

    # ── summary table ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        for name, result in all_results:
            print(_fmt_row(name, result))

    print("\nDone.")


if __name__ == "__main__":
    main()
