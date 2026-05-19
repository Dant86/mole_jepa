r"""Filter Recap-DataComp-1B metadata and download images as WebDataset shards.

Two-phase pipeline:

1. **Filter** — stream ``mlfoundations/recap-datacomp-1b`` from HuggingFace,
   apply quality filters (CLIP score + re-caption length), and write a
   filtered Parquet file to ``{output_dir}/filtered.parquet``.  Scanning
   stops once ``--target-samples`` accepted rows are collected.

2. **Download** — invoke ``img2dataset`` on the filtered Parquet to fetch,
   resize (shorter edge → 256 px), and pack images into WebDataset tar
   shards under ``{output_dir}/shards/``.

Resuming:
    Re-running skips Phase 1 when ``filtered.parquet`` already exists
    (use ``--force-filter`` to redo it).  ``img2dataset`` has its own
    resume logic and skips already-written shards automatically.

Storage estimate:
    40 M images × ~25 KB (256 px JPEG, q=85) ≈ 1 TB.

Usage::

    uv run --group data python apps/prepare_datacomp/prepare_datacomp_main.py \
        --output-dir /scratch/vpathak/datacomp \
        --target-samples 40_000_000

Column names (run with --list-columns to inspect the first row)::

    --url-col       url             (image URL)
    --caption-col   re_caption      (LLaVA synthetic caption)
    --clip-col      clip_l14_score  (CLIP ViT-L/14 similarity)
    --uid-col       uid             (unique sample identifier)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASET_NAME = "mlfoundations/recap-datacomp-1b"
_SHARD_SIZE = 5_000  # images per WebDataset tar shard
_RESIZE_PX = 256  # shorter edge; image processor crops to 224
_BATCH_SIZE = 100_000  # rows collected per pyarrow write batch
_DEFAULT_PROCESSES = 16  # img2dataset --processes_count
_DEFAULT_THREADS = 64  # img2dataset --thread_count (per process)

# Default column names in mlfoundations/recap-datacomp-1b.
_DEFAULT_URL_COL = "url"
_DEFAULT_CAPTION_COL = "re_caption"
_DEFAULT_CLIP_COL = "clip_l14_score"
_DEFAULT_UID_COL = "uid"


# ---------------------------------------------------------------------------
# Phase 1: metadata filter
# ---------------------------------------------------------------------------


def _filter(
    output_dir: Path,
    *,
    target_samples: int,
    clip_threshold: float,
    min_caption_words: int,
    max_caption_words: int,
    url_col: str,
    caption_col: str,
    clip_col: str,
    uid_col: str,
    shuffle_buffer: int,
    hf_token: str | None,
) -> Path:
    """Stream Recap-DataComp-1B, filter, and write a Parquet of accepted rows.

    Args:
        output_dir: Directory in which to write ``filtered.parquet``.
        target_samples: Stop accepting rows after this many pass all filters.
        clip_threshold: Minimum CLIP ViT-L/14 score.
        min_caption_words: Reject captions shorter than this many words.
        max_caption_words: Reject captions longer than this many words.
        url_col: Dataset column name for the image URL.
        caption_col: Dataset column name for the text caption.
        clip_col: Dataset column name for the CLIP score.
        uid_col: Dataset column name for the unique identifier.
        shuffle_buffer: Shuffle-buffer size for streaming (0 = no shuffle).
        hf_token: HuggingFace token for authenticated access.

    Returns:
        Path to the written ``filtered.parquet`` file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    out_path = output_dir / "filtered.parquet"
    print(f"Phase 1 — filtering {_DATASET_NAME}")
    print(
        f"  clip >= {clip_threshold}  |  "
        f"caption words: [{min_caption_words}, {max_caption_words}]  |  "
        f"target: {target_samples:,}"
    )

    ds = load_dataset(
        _DATASET_NAME,
        split="train",
        streaming=True,
        token=hf_token,
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=42, buffer_size=shuffle_buffer)

    schema = pa.schema(
        [
            pa.field("url", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("uid", pa.string()),
        ]
    )

    accepted = 0
    scanned = 0
    batch_urls: list[str] = []
    batch_captions: list[str] = []
    batch_uids: list[str] = []

    writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")

    def _flush() -> None:
        writer.write_table(
            pa.table(
                {
                    "url": batch_urls[:],
                    "caption": batch_captions[:],
                    "uid": batch_uids[:],
                },
                schema=schema,
            )
        )
        batch_urls.clear()
        batch_captions.clear()
        batch_uids.clear()

    try:
        for row in ds:
            scanned += 1

            # ── field extraction ──────────────────────────────────────────
            url = row.get(url_col) or ""
            caption = row.get(caption_col) or ""
            uid = str(row.get(uid_col) or "")
            clip = row.get(clip_col)

            # ── quality filters ───────────────────────────────────────────
            if not url:
                continue
            if not caption:
                continue
            if clip is None or float(clip) < clip_threshold:
                continue
            words = len(caption.split())
            if words < min_caption_words or words > max_caption_words:
                continue

            # ── accept ────────────────────────────────────────────────────
            batch_urls.append(url)
            batch_captions.append(caption)
            batch_uids.append(uid)
            accepted += 1

            if accepted % _BATCH_SIZE == 0:
                _flush()
                pct = 100 * accepted / target_samples
                print(
                    f"  scanned {scanned:>12,}  accepted {accepted:>10,}"
                    f"  ({pct:.1f}% of target)"
                )

            if accepted >= target_samples:
                break
    finally:
        if batch_urls:
            _flush()
        writer.close()

    print(
        f"Phase 1 done — scanned {scanned:,}, accepted {accepted:,}. "
        f"Written to {out_path}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Phase 2: image download via img2dataset
# ---------------------------------------------------------------------------


def _download(
    parquet_path: Path,
    output_dir: Path,
    *,
    processes: int,
    threads: int,
) -> None:
    """Invoke img2dataset to download and pack images from the filtered Parquet.

    Images are resized so the shorter edge is ``_RESIZE_PX`` pixels, then
    packed into WebDataset tar shards under ``output_dir/shards/``.

    The ``url`` and ``caption`` columns written by :func:`_filter` map
    directly to img2dataset's ``url_col`` / ``caption_col``.  The ``uid``
    column is preserved as a sidecar JSON field in each shard sample.

    Args:
        parquet_path: Path to the filtered Parquet produced by :func:`_filter`.
        output_dir: Root directory; shards are written to a ``shards/``
            subdirectory.
        processes: Number of download processes (``img2dataset
            --processes_count``).
        threads: Threads per process (``img2dataset --thread_count``).
    """
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "img2dataset",
        "--url_list",
        str(parquet_path),
        "--input_format",
        "parquet",
        "--url_col",
        "url",
        "--caption_col",
        "caption",
        "--output_folder",
        str(shards_dir),
        "--output_format",
        "webdataset",
        "--image_size",
        str(_RESIZE_PX),
        "--resize_mode",
        "keep_ratio",
        "--resize_only_if_bigger",
        "True",
        "--number_sample_per_shard",
        str(_SHARD_SIZE),
        "--processes_count",
        str(processes),
        "--thread_count",
        str(threads),
        "--save_additional_columns",
        '["uid"]',
        "--distributor",
        "multiprocessing",
        "--enable_wandb",
        "False",
    ]

    print("\nPhase 2 — downloading images")
    print("  " + " \\\n    ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"\nimg2dataset exited with code {result.returncode}. "
            "Check the output above for details. Partial shards in "
            f"{shards_dir} are safe to leave in place — re-running will "
            "resume from where it left off.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    print(f"\nPhase 2 done — shards written to {shards_dir}")


# ---------------------------------------------------------------------------
# Column inspection helper
# ---------------------------------------------------------------------------


def _list_columns(hf_token: str | None) -> None:
    """Print the column names and a sample row from the dataset, then exit."""
    from datasets import load_dataset

    print(f"Loading one row from {_DATASET_NAME} to inspect columns…")
    ds = load_dataset(
        _DATASET_NAME,
        split="train",
        streaming=True,
        token=hf_token,
    )
    row = next(iter(ds))
    print("\nColumns and sample values:")
    for k, v in row.items():
        snippet = str(v)
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        print(f"  {k!r:30s} {snippet!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse arguments and run the pipeline."""
    import os

    parser = argparse.ArgumentParser(
        description=(
            "Filter Recap-DataComp-1B and download images as WebDataset shards."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for filtered.parquet and shards/.",
    )

    # ── filter params ────────────────────────────────────────────────────────
    parser.add_argument(
        "--target-samples",
        type=int,
        default=40_000_000,
        help="Stop after accepting this many samples in Phase 1.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.28,
        help="Minimum CLIP ViT-L/14 image–text similarity score.",
    )
    parser.add_argument(
        "--min-caption-words",
        type=int,
        default=5,
        help="Reject captions shorter than this many words.",
    )
    parser.add_argument(
        "--max-caption-words",
        type=int,
        default=200,
        help="Reject captions longer than this many words.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=100_000,
        help=(
            "Streaming shuffle-buffer size for Phase 1. "
            "0 disables shuffling (samples in dataset order)."
        ),
    )
    parser.add_argument(
        "--force-filter",
        action="store_true",
        help="Redo Phase 1 even if filtered.parquet already exists.",
    )

    # ── column names ─────────────────────────────────────────────────────────
    parser.add_argument("--url-col", default=_DEFAULT_URL_COL)
    parser.add_argument("--caption-col", default=_DEFAULT_CAPTION_COL)
    parser.add_argument("--clip-col", default=_DEFAULT_CLIP_COL)
    parser.add_argument("--uid-col", default=_DEFAULT_UID_COL)

    # ── download params ───────────────────────────────────────────────────────
    default_procs = int(os.environ.get("SLURM_CPUS_PER_TASK", _DEFAULT_PROCESSES))
    parser.add_argument(
        "--processes",
        type=int,
        default=default_procs,
        help="img2dataset --processes_count (download parallelism).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=_DEFAULT_THREADS,
        help="img2dataset --thread_count (threads per process).",
    )

    # ── phase control ────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Run Phase 1 only (no img2dataset call).",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Run Phase 2 only (filtered.parquet must already exist).",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print dataset column names from the first row and exit.",
    )

    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    if args.list_columns:
        _list_columns(hf_token)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "filtered.parquet"

    # ── Phase 1: filter ───────────────────────────────────────────────────────
    if args.skip_filter:
        if not parquet_path.exists():
            print(
                f"ERROR: --skip-filter set but {parquet_path} does not exist.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Skipping Phase 1 — using existing {parquet_path}")
    elif parquet_path.exists() and not args.force_filter:
        print(
            f"Skipping Phase 1 — {parquet_path} already exists "
            "(pass --force-filter to redo)."
        )
    else:
        parquet_path = _filter(
            output_dir,
            target_samples=args.target_samples,
            clip_threshold=args.clip_threshold,
            min_caption_words=args.min_caption_words,
            max_caption_words=args.max_caption_words,
            url_col=args.url_col,
            caption_col=args.caption_col,
            clip_col=args.clip_col,
            uid_col=args.uid_col,
            shuffle_buffer=args.shuffle_buffer,
            hf_token=hf_token,
        )

    # ── Phase 2: download ─────────────────────────────────────────────────────
    if args.skip_download:
        print("Skipping Phase 2 (--skip-download).")
        return

    _download(
        parquet_path,
        output_dir,
        processes=args.processes,
        threads=args.threads,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    shards_dir = output_dir / "shards"
    n_shards = len(list(shards_dir.glob("*.tar")))
    size_gb = sum(f.stat().st_size for f in shards_dir.glob("*.tar")) / 1e9
    print(f"\n{'─' * 60}")
    print(f"  Shards: {n_shards:,}  ·  On disk: {size_gb:.1f} GB")
    print("\nPass to the train script:")
    print(f"  --hf-dataset-name      {shards_dir}")
    print("  --val-hf-dataset-name  <your-val-dir>")


if __name__ == "__main__":
    main()
