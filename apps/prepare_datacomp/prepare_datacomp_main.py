r"""Filter Recap-DataComp-1B metadata and download images as WebDataset shards.

Two-phase pipeline:

1. **Filter** — list all parquet shards in ``mlfoundations/recap-datacomp-1b``
   via ``HfFileSystem``, read only the required columns (no full-file
   download), apply quality filters (CLIP score + re-caption length), and
   write a filtered Parquet to ``{output_dir}/filtered.parquet``.
   *N* shards are processed in parallel with a ``ThreadPoolExecutor``,
   which is I/O-bound (network reads) and releases the GIL for pandas work.

2. **Download** — invoke ``img2dataset`` on the filtered Parquet to fetch,
   resize (shorter edge → 256 px), and pack images into WebDataset tar
   shards under ``{output_dir}/shards/``.

Performance targets (single H200 node, 64 CPUs):
    Phase 1  ~30–90 min   (depends on HF network speed and shard count)
    Phase 2  ~8–18 h      (40 M × 25 KB, 4 096 concurrent connections)
    Total    fits overnight for a ~40 M image corpus

Resuming:
    Re-running skips Phase 1 when ``filtered.parquet`` already exists
    (use ``--force-filter`` to redo it).  ``img2dataset`` has its own
    resume logic and skips already-written shards automatically.

Storage estimate:
    40 M images × ~25 KB (256 px JPEG, q=85) ≈ 1 TB.

URL mortality:
    DataComp URLs can go dead after dataset creation.  The default
    ``--target-samples`` of 60 M over-samples so that ~40 M images are
    recovered even with 30 % URL mortality.  Tune with ``--target-samples``
    if you have a sense of the current dead-URL rate.

Usage::

    uv run --group data python apps/prepare_datacomp/prepare_datacomp_main.py \
        --output-dir /scratch/vpathak/datacomp \
        --target-samples 60_000_000

Column names (run with --list-columns to inspect the first row)::

    --url-col       url             (image URL)
    --caption-col   re_caption      (LLaVA synthetic caption)
    --clip-col      clip_l14_score  (CLIP ViT-L/14 similarity)
    --uid-col       uid             (unique sample identifier)
"""

import argparse
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASET_NAME = "mlfoundations/recap-datacomp-1b"
_SHARD_SIZE = 10_000  # images per WebDataset tar shard
_RESIZE_PX = 256  # shorter edge; image processor crops to 224
_DEFAULT_FILTER_WORKERS = 16  # parallel parquet-shard readers in Phase 1
_DEFAULT_IMG2DATASET_PROCESSES = 64  # img2dataset --processes_count
_DEFAULT_IMG2DATASET_THREADS = 64  # img2dataset --thread_count per process
_DEFAULT_STORAGE_LIMIT_GB = 980.0  # terminate Phase 2 before filling the disk
_STORAGE_POLL_INTERVAL_S = 30  # how often the monitor thread checks disk usage

# Default column names in mlfoundations/recap-datacomp-1b.
_DEFAULT_URL_COL = "url"
_DEFAULT_CAPTION_COL = "re_caption"
_DEFAULT_CLIP_COL = "clip_l14_score"
_DEFAULT_UID_COL = "uid"


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------


def _list_shards(token: str | None) -> list[str]:
    """Return shuffled list of parquet shard paths in the dataset repo.

    Uses ``HfFileSystem`` to list files directly without downloading them.
    Paths are shuffled so parallel workers sample diverse parts of the
    dataset.

    Args:
        token: HuggingFace auth token (can be ``None`` for public repos).

    Returns:
        Shuffled list of repo-relative parquet paths, e.g.
        ``["data/train-00000-of-01024.parquet", ...]``.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    # HfFileSystem.ls stubs type the return as list[str | dict] regardless of
    # the detail= flag; cast to the concrete type we requested.
    all_files: list[str] = cast(
        list[str], fs.ls(f"datasets/{_DATASET_NAME}", detail=False)
    )
    # Recurse into subdirectories (common layout: data/*.parquet)
    parquet_paths: list[str] = []
    for entry in all_files:
        if entry.endswith(".parquet"):
            parquet_paths.append(entry)
        else:
            try:
                sub: list[str] = cast(list[str], fs.ls(entry, detail=False))
                parquet_paths.extend(p for p in sub if p.endswith(".parquet"))
            except Exception:  # noqa: BLE001
                pass

    if not parquet_paths:
        raise RuntimeError(
            f"No parquet files found under datasets/{_DATASET_NAME}. "
            "Run with --list-columns to inspect the repo structure."
        )

    random.seed(42)
    random.shuffle(parquet_paths)
    return parquet_paths


def _filter_shard(
    args: tuple[
        str,  # hf_path  — full HfFileSystem path to the parquet shard
        str | None,  # token
        float,  # clip_threshold
        int,  # min_caption_words
        int,  # max_caption_words
        str,  # url_col
        str,  # caption_col
        str,  # clip_col
        str,  # uid_col
    ],
) -> tuple[str, Any, Any]:
    """Read one parquet shard, apply quality filters, return accepted rows.

    This function runs inside a ``ThreadPoolExecutor`` worker thread.
    ``HfFileSystem`` I/O releases the GIL, so multiple workers make
    genuine progress in parallel.

    Args:
        args: Packed tuple of all parameters (required for
            ``ThreadPoolExecutor.submit``).

    Returns:
        ``(hf_path, filtered_df, n_scanned)`` on success, or
        ``(hf_path, None, None)`` if the shard could not be read.
        The second element is a ``pandas.DataFrame`` or ``None``;
        typed as ``Any`` to avoid a top-level pandas import.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    (
        hf_path,
        token,
        clip_threshold,
        min_words,
        max_words,
        url_col,
        caption_col,
        clip_col,
        uid_col,
    ) = args

    try:
        fs = HfFileSystem(token=token)
        table = pq.read_table(
            hf_path,
            columns=[url_col, caption_col, clip_col, uid_col],
            filesystem=fs,
        )
        df = table.to_pandas()
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read {hf_path}: {exc}", file=sys.stderr)
        return hf_path, None, None

    n_scanned = len(df)

    # ── quality filters ───────────────────────────────────────────────────
    mask = df[url_col].notna() & (df[url_col] != "")
    mask &= df[caption_col].notna() & (df[caption_col] != "")
    mask &= df[clip_col].notna() & (df[clip_col].astype(float) >= clip_threshold)
    df = df[mask]

    word_counts = df[caption_col].str.split().str.len()
    df = df[(word_counts >= min_words) & (word_counts <= max_words)]

    # Normalise column names for img2dataset / downstream use.
    df = df[[url_col, caption_col, uid_col]].rename(
        columns={caption_col: "caption", uid_col: "uid", url_col: "url"}
    )

    return hf_path, df, n_scanned


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
    num_workers: int,
    hf_token: str | None,
) -> Path:
    """Stream Recap-DataComp-1B shards in parallel, filter, write Parquet.

    Shards are listed from the HuggingFace repo, shuffled for diversity,
    then dispatched to a ``ThreadPoolExecutor``.  Each worker reads only
    the required columns directly from HF (no full-shard download to disk),
    applies pandas-based quality filters, and returns the accepted rows.
    Results are written to ``{output_dir}/filtered.parquet`` in batches as
    workers complete.  Scanning stops once ``target_samples`` rows have been
    accepted.

    Args:
        output_dir: Directory in which to write ``filtered.parquet``.
        target_samples: Stop accepting rows after this many pass all filters.
        clip_threshold: Minimum CLIP ViT-L/14 score.
        min_caption_words: Reject captions shorter than this many words.
        max_caption_words: Reject captions longer than this many words.
        url_col: Dataset column containing the image URL.
        caption_col: Dataset column containing the text caption.
        clip_col: Dataset column containing the CLIP similarity score.
        uid_col: Dataset column containing the unique sample identifier.
        num_workers: Parallel shard-reader threads.
        hf_token: HuggingFace auth token.

    Returns:
        Path to the written ``filtered.parquet`` file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = output_dir / "filtered.parquet"
    print(f"Phase 1 — filtering {_DATASET_NAME}  (workers={num_workers})")
    print(
        f"  clip >= {clip_threshold}  |  "
        f"caption words: [{min_caption_words}, {max_caption_words}]  |  "
        f"target: {target_samples:,}"
    )

    shards = _list_shards(hf_token)
    print(f"  found {len(shards):,} parquet shards")

    schema = pa.schema(
        [
            pa.field("url", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("uid", pa.string()),
        ]
    )

    accepted = 0
    scanned = 0
    shards_done = 0

    writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")

    worker_args = [
        (
            shard,
            hf_token,
            clip_threshold,
            min_caption_words,
            max_caption_words,
            url_col,
            caption_col,
            clip_col,
            uid_col,
        )
        for shard in shards
    ]

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_filter_shard, a): a[0] for a in worker_args}
            for future in as_completed(futures):
                hf_path, df, n_scanned = future.result()
                shards_done += 1

                if df is None or n_scanned is None:
                    continue

                scanned += int(n_scanned)

                if accepted >= target_samples:
                    continue  # drain remaining futures (don't write)

                # Trim the shard if it would push us over the target.
                remaining = target_samples - accepted
                if len(df) > remaining:
                    df = df.iloc[:remaining]

                writer.write_table(pa.Table.from_pandas(df, schema=schema))
                accepted += len(df)

                pct = 100 * accepted / target_samples
                print(
                    f"  shard {shards_done:>5}/{len(shards):,}  "
                    f"scanned {scanned:>12,}  "
                    f"accepted {accepted:>10,}  ({pct:.1f}%)"
                )

                if accepted >= target_samples:
                    # Cancel pending futures — we have enough rows.
                    for f in futures:
                        f.cancel()
                    print("  target reached — cancelling remaining shards")
    finally:
        writer.close()

    print(
        f"\nPhase 1 done — "
        f"shards processed: {shards_done:,}  "
        f"scanned: {scanned:,}  "
        f"accepted: {accepted:,}"
    )
    print(f"Written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Phase 2: image download via img2dataset
# ---------------------------------------------------------------------------


def _monitor_storage(
    shards_dir: Path,
    limit_bytes: int,
    process: subprocess.Popen[bytes],
    stop_event: threading.Event,
) -> None:
    """Poll shard directory size and terminate *process* if limit is exceeded.

    Runs as a daemon thread alongside the img2dataset subprocess.  Only
    completed ``.tar`` files are counted (img2dataset writes to a temp path
    then atomically renames), so the check reflects committed, readable data.

    Args:
        shards_dir: Directory where img2dataset writes tar shards.
        limit_bytes: Terminate *process* once this many bytes are on disk.
        process: The running img2dataset ``Popen`` handle.
        stop_event: Set by the main thread when img2dataset exits normally,
            so the monitor doesn't race to read a dead process.
    """
    while not stop_event.is_set():
        # Sum only completed shards; ignore in-progress temp files.
        used = sum(f.stat().st_size for f in shards_dir.glob("*.tar") if f.is_file())
        used_gb = used / 1e9
        if used >= limit_bytes:
            print(
                f"\n[storage monitor] {used_gb:.1f} GB >= limit "
                f"{limit_bytes / 1e9:.1f} GB — sending SIGTERM to img2dataset.",
                flush=True,
            )
            try:
                process.terminate()
            except OSError:
                pass  # process already gone
            return
        time.sleep(_STORAGE_POLL_INTERVAL_S)


def _download(
    parquet_path: Path,
    output_dir: Path,
    *,
    processes: int,
    threads: int,
    storage_limit_gb: float,
) -> None:
    """Invoke img2dataset to download and pack images from the filtered Parquet.

    Images are resized so the shorter edge is ``_RESIZE_PX`` pixels (the
    ViT image processor crops to 224 anyway), then packed into WebDataset
    tar shards under ``output_dir/shards/``.

    Parallelism:
        ``processes × threads`` concurrent HTTP connections.  With
        ``processes=64`` and ``threads=64`` that is 4 096 concurrent
        downloads — enough to saturate a 10 Gbit cluster uplink.

    Timeout / retries:
        ``--timeout 5`` fails dead URLs quickly (most fail in < 100 ms).
        ``--retries 2`` handles transient network hiccups without wasting
        time on truly dead URLs.

    Storage guard:
        A daemon thread polls the shard directory every
        ``_STORAGE_POLL_INTERVAL_S`` seconds.  When the total size of
        completed ``.tar`` files reaches ``storage_limit_gb`` GB, the
        thread sends ``SIGTERM`` to img2dataset and exits.  This exit is
        treated as a clean stop (not an error), so the script reports
        success and prints the final shard count.

    The ``url`` and ``caption`` columns produced by :func:`_filter` map
    directly to img2dataset's ``url_col`` / ``caption_col``.  The ``uid``
    column is preserved as a sidecar JSON field in every shard sample.

    Args:
        parquet_path: Path to the filtered Parquet produced by :func:`_filter`.
        output_dir: Root directory; shards land in ``shards/``.
        processes: ``img2dataset --processes_count``.
        threads: ``img2dataset --thread_count`` (per process).
        storage_limit_gb: Terminate img2dataset once this many GB of
            completed shards are written.
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
        "--timeout",
        "5",
        "--retries",
        "2",
        "--save_additional_columns",
        '["uid"]',
        "--distributor",
        "multiprocessing",
        "--enable_wandb",
        "False",
    ]

    print("\nPhase 2 — downloading images")
    print(
        f"  processes={processes}  threads={threads}  "
        f"({processes * threads:,} concurrent connections)"
    )
    print(f"  storage limit: {storage_limit_gb:.0f} GB")
    print("  " + " \\\n    ".join(cmd))

    limit_bytes = int(storage_limit_gb * 1e9)
    stop_event = threading.Event()

    proc = subprocess.Popen(cmd)  # noqa: S603
    monitor = threading.Thread(
        target=_monitor_storage,
        args=(shards_dir, limit_bytes, proc, stop_event),
        daemon=True,
    )
    monitor.start()

    returncode = proc.wait()
    stop_event.set()  # tell the monitor thread to exit its loop
    monitor.join(timeout=5)

    # SIGTERM (returncode -15) means the storage monitor fired — that's
    # a clean, intentional stop, not an error.
    terminated_by_monitor = returncode in (-15, 143)

    if terminated_by_monitor:
        used_gb = (
            sum(f.stat().st_size for f in shards_dir.glob("*.tar") if f.is_file()) / 1e9
        )
        print(
            f"\nPhase 2 stopped by storage guard at {used_gb:.1f} GB "
            f"(limit {storage_limit_gb:.0f} GB)."
        )
    elif returncode != 0:
        print(
            f"\nimg2dataset exited with code {returncode}. "
            f"Partial shards in {shards_dir} are safe to leave — "
            "re-running resumes from the last completed shard.",
            file=sys.stderr,
        )
        sys.exit(returncode)
    else:
        print(f"\nPhase 2 done — shards written to {shards_dir}")


# ---------------------------------------------------------------------------
# Column inspection helper
# ---------------------------------------------------------------------------


def _list_columns(hf_token: str | None) -> None:
    """Print repo structure and column names from the first shard, then exit."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=hf_token)
    print(f"Listing top-level entries in datasets/{_DATASET_NAME}:")
    top: list[dict[str, Any]] = cast(
        list[dict[str, Any]],
        fs.ls(f"datasets/{_DATASET_NAME}", detail=True),
    )
    for entry in top[:20]:
        print(f"  {entry.get('type', '?'):6s}  {entry.get('name', '?')}")

    # Find first parquet and sample one row.
    parquet_files: list[str] = [
        str(e["name"]) for e in top if str(e.get("name", "")).endswith(".parquet")
    ]
    if not parquet_files:
        for entry in top:
            try:
                sub: list[str] = cast(
                    list[str], fs.ls(str(entry["name"]), detail=False)
                )
                parquet_files = [p for p in sub if p.endswith(".parquet")]
                if parquet_files:
                    break
            except Exception:  # noqa: BLE001
                pass

    if not parquet_files:
        print("\nNo parquet files found — check repo structure above.")
        return

    import pyarrow.parquet as pq

    sample_path = parquet_files[0]
    print(f"\nSampling first row from {sample_path}:")
    schema = pq.read_schema(sample_path, filesystem=fs)
    print("\nColumn names:")
    for name in schema.names:
        print(f"  {name!r}")


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
        default=60_000_000,
        help=(
            "Collect this many rows in Phase 1. Set higher than your image "
            "target to account for URL mortality (default 60 M → ~40 M "
            "live images assuming ~30%% dead URLs)."
        ),
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
        "--num-filter-workers",
        type=int,
        default=_DEFAULT_FILTER_WORKERS,
        help=(
            "Parallel shard-reader threads for Phase 1. "
            "Each thread reads one parquet shard from HF over the network."
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
    default_procs = int(
        os.environ.get("SLURM_CPUS_PER_TASK", _DEFAULT_IMG2DATASET_PROCESSES)
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=default_procs,
        help=(
            "img2dataset --processes_count. Defaults to SLURM_CPUS_PER_TASK when set."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=_DEFAULT_IMG2DATASET_THREADS,
        help="img2dataset --thread_count (threads per process).",
    )
    parser.add_argument(
        "--storage-limit-gb",
        type=float,
        default=_DEFAULT_STORAGE_LIMIT_GB,
        help=(
            "Terminate img2dataset once this many GB of completed shards "
            "are written to disk. Prevents filling the scratch partition."
        ),
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
        help="Print dataset repo structure and column names, then exit.",
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
            num_workers=args.num_filter_workers,
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
        storage_limit_gb=args.storage_limit_gb,
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
