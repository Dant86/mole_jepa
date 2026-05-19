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
    --caption-col   re_caption      (LLaVA-1.5 recaption)
    --clip-col      re_clip_score   (CLIP score for the recaption)
    --uid-col       key             (unique sample identifier)
"""

import argparse
import os
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASET_NAME = "UCSC-VLAA/Recap-DataComp-1B"
_SHARD_SIZE = 10_000  # images per WebDataset tar shard
_RESIZE_PX = 256  # shorter edge; image processor crops to 224
_DEFAULT_FILTER_WORKERS = 16  # parallel parquet-shard readers in Phase 1
_DEFAULT_IMG2DATASET_PROCESSES = 64  # img2dataset --processes_count
_DEFAULT_IMG2DATASET_THREADS = 64  # img2dataset --thread_count per process
_DEFAULT_STORAGE_LIMIT_GB = 980.0  # terminate Phase 2 before filling the disk
_STORAGE_POLL_INTERVAL_S = 30  # how often the monitor thread checks disk usage
# The "default" config has known conversion errors; condition_diverse_topk is clean.
_DEFAULT_SUBSET = "condition_diverse_topk"

# Default column names in UCSC-VLAA/Recap-DataComp-1B.
_DEFAULT_URL_COL = "url"
_DEFAULT_CAPTION_COL = "re_caption"
_DEFAULT_CLIP_COL = "re_clip_score"
_DEFAULT_UID_COL = "key"


# ---------------------------------------------------------------------------
# SLURM preemption support
# ---------------------------------------------------------------------------

# Set by the SIGUSR1 handler; checked in _filter and _download to trigger a
# clean exit with code 99 so SLURM requeues the job.
_preempt_requested: threading.Event = threading.Event()

# Populated by _download while img2dataset is running so the signal handler
# can forward SIGTERM to it immediately.
_img2dataset_proc: subprocess.Popen[bytes] | None = None


def _install_preempt_handler() -> None:
    """Install a SIGUSR1 handler for SLURM preemption.

    When the batch script receives ``--signal=B:USR1@300`` from SLURM it
    forwards SIGUSR1 to this process.  The handler sets ``_preempt_requested``
    and, if img2dataset is running, sends it SIGTERM so it exits cleanly
    before Slurm hard-kills the job.  The main thread detects the flag after
    img2dataset exits and calls ``sys.exit(99)`` to trigger a requeue.
    """

    def _handler(signum: int, frame: object) -> None:  # noqa: ARG001
        print(
            "\n[preempt] SIGUSR1 received — stopping cleanly for Slurm requeue.",
            flush=True,
        )
        _preempt_requested.set()
        proc = _img2dataset_proc
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass  # process already gone

    signal.signal(signal.SIGUSR1, _handler)


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------


def _list_shards(token: str | None, subset: str) -> list[str]:
    """Return shuffled list of ``hf://`` parquet URLs in the dataset repo.

    Uses ``list_repo_files`` (direct Hub API) for discovery, which takes the
    repo ID directly and avoids any filesystem path ambiguity.  Each returned
    path is a full ``hf://datasets/{repo}/{file}`` URL ready for
    ``pyarrow.parquet.read_table``.

    Args:
        token: HuggingFace auth token (can be ``None`` for public repos).
        subset: Only return parquet files whose repo-relative path contains
            this string.  Used to target a single dataset config and avoid
            broken or irrelevant shards from other configs.

    Returns:
        Shuffled list of ``hf://`` URLs filtered to *subset*.
    """
    from huggingface_hub import list_repo_files

    all_files = list(
        list_repo_files(
            _DATASET_NAME,
            repo_type="dataset",
            token=token,
        )
    )
    parquet_paths = [
        f"hf://datasets/{_DATASET_NAME}/{f}"
        for f in all_files
        if f.endswith(".parquet")
    ]

    # Filter to the requested subset so broken configs are not touched.
    parquet_paths = [p for p in parquet_paths if subset in p]

    if not parquet_paths:
        raise RuntimeError(
            f"No parquet files found for subset {subset!r} in {_DATASET_NAME}. "
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
) -> tuple[str, Any, Any, dict[str, int]]:
    """Read one parquet shard, apply quality filters, return accepted rows.

    This function runs inside a ``ThreadPoolExecutor`` worker thread.
    ``HfFileSystem`` I/O releases the GIL, so multiple workers make
    genuine progress in parallel.

    Args:
        args: Packed tuple of all parameters (required for
            ``ThreadPoolExecutor.submit``).

    Returns:
        ``(hf_path, filtered_df, n_scanned, stats)`` on success, or
        ``(hf_path, None, None, {})`` if the shard could not be read.
        *stats* keys: ``after_url``, ``after_caption``, ``after_clip``,
        ``after_words`` — row counts surviving each successive filter.
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
        # hf_path is a full hf://datasets/... URL; pyarrow resolves it via
        # fsspec + huggingface_hub.  Pass an explicit HfFileSystem so the
        # auth token is forwarded to every request.
        fs = HfFileSystem(token=token)
        table = pq.read_table(
            hf_path,
            columns=[url_col, caption_col, clip_col, uid_col],
            filesystem=fs,
        )
        df = table.to_pandas()
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read {hf_path}: {exc}", file=sys.stderr)
        return hf_path, None, None, {}

    n_scanned = len(df)

    # ── quality filters (each step tracked for diagnostics) ───────────────
    df = df[df[url_col].notna() & (df[url_col] != "")]
    after_url = len(df)

    df = df[df[caption_col].notna() & (df[caption_col] != "")]
    after_caption = len(df)

    if clip_threshold > 0:
        df = df[df[clip_col].notna() & (df[clip_col].astype(float) >= clip_threshold)]
    after_clip = len(df)

    word_counts = df[caption_col].str.split().str.len()
    word_mask = word_counts >= min_words
    if max_words > 0:
        word_mask &= word_counts <= max_words
    df = df[word_mask]
    after_words = len(df)

    # Normalise column names for img2dataset / downstream use.
    df = df[[url_col, caption_col, uid_col]].rename(
        columns={caption_col: "caption", uid_col: "uid", url_col: "url"}
    )

    stats = {
        "after_url": after_url,
        "after_caption": after_caption,
        "after_clip": after_clip,
        "after_words": after_words,
    }
    return hf_path, df, n_scanned, stats


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
    subset: str,
) -> Path | None:
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
        subset: Parquet path substring used to select the dataset config
            (e.g. ``"condition_diverse_topk"``).

    Returns:
        Path to the written ``filtered.parquet`` file, or ``None`` if the
        job was preempted mid-filter (caller should exit 99).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = output_dir / "filtered.parquet"
    # Write to a temp file and atomic-rename on success.  This way a
    # mid-Phase-1 preemption or crash never leaves a partial file that would
    # be mistaken for a complete one on the next (requeued) run.
    tmp_path = out_path.with_suffix(".parquet.tmp")
    tmp_path.unlink(missing_ok=True)

    print(f"Phase 1 — filtering {_DATASET_NAME}/{subset}  (workers={num_workers})")
    print(
        f"  clip >= {clip_threshold}  |  "
        f"caption words: [{min_caption_words}, {max_caption_words}]  |  "
        f"target: {target_samples:,}"
    )

    shards = _list_shards(hf_token, subset)
    print(f"  found {len(shards):,} parquet shards in {subset!r}")

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
    # Running per-filter totals for diagnostics.
    tot_after_url = 0
    tot_after_caption = 0
    tot_after_clip = 0
    tot_after_words = 0

    writer = pq.ParquetWriter(str(tmp_path), schema, compression="snappy")

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

    preempted = False
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_filter_shard, a): a[0] for a in worker_args}
            for future in as_completed(futures):
                # Check preemption on every completed shard.
                if _preempt_requested.is_set():
                    print(
                        "\n[preempt] Stopping Phase 1 early — will rerun on next job.",
                        flush=True,
                    )
                    for f in futures:
                        f.cancel()
                    preempted = True
                    break

                hf_path, df, n_scanned, stats = future.result()
                shards_done += 1

                if df is None or n_scanned is None:
                    continue

                scanned += int(n_scanned)
                tot_after_url += stats.get("after_url", 0)
                tot_after_caption += stats.get("after_caption", 0)
                tot_after_clip += stats.get("after_clip", 0)
                tot_after_words += stats.get("after_words", 0)

                if accepted >= target_samples:
                    continue  # drain remaining futures (don't write)

                # Trim the shard if it would push us over the target.
                remaining = target_samples - accepted
                if len(df) > remaining:
                    df = df.iloc[:remaining]

                writer.write_table(pa.Table.from_pandas(df, schema=schema))
                accepted += len(df)

                pct = 100 * accepted / target_samples
                # Print a diagnostic line every 10 shards so filter
                # rejection rates are visible without flooding the log.
                if shards_done % 10 == 0 or accepted >= target_samples:

                    def _pct(a: int, b: int) -> str:
                        return f"{100 * a / b:.0f}%" if b else "n/a"

                    print(
                        f"  shard {shards_done:>5}/{len(shards):,}  "
                        f"scanned {scanned:>10,}  "
                        f"accepted {accepted:>8,}  ({pct:.1f}%)  "
                        f"[pass: url {_pct(tot_after_url, scanned)}  "
                        f"cap {_pct(tot_after_caption, scanned)}  "
                        f"clip {_pct(tot_after_clip, scanned)}  "
                        f"words {_pct(tot_after_words, scanned)}]"
                    )

                if accepted >= target_samples:
                    # Cancel pending futures — we have enough rows.
                    for f in futures:
                        f.cancel()
                    print("  target reached — cancelling remaining shards")
    finally:
        writer.close()
        if preempted:
            # Delete the incomplete temp file so the next run reruns Phase 1.
            tmp_path.unlink(missing_ok=True)

    if preempted:
        return None  # signal caller to exit 99

    # Atomic rename: only visible as the final path once fully written.
    os.replace(tmp_path, out_path)

    def _pct(a: int, b: int) -> str:
        return f"{100 * a / b:.1f}%" if b else "n/a"

    print(
        f"\nPhase 1 done — "
        f"shards processed: {shards_done:,}  "
        f"scanned: {scanned:,}  "
        f"accepted: {accepted:,}"
    )
    print("Filter funnel (cumulative pass rate vs scanned):")
    print(f"  has url       {_pct(tot_after_url, scanned):>7}  ({tot_after_url:,})")
    print(
        f"  has caption   {_pct(tot_after_caption, scanned):>7}"
        f"  ({tot_after_caption:,})"
    )
    print(
        f"  clip>={clip_threshold:.2f}    "
        f"{_pct(tot_after_clip, scanned):>7}  ({tot_after_clip:,})"
    )
    print(f"  word count    {_pct(tot_after_words, scanned):>7}  ({tot_after_words:,})")
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
) -> bool:
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

    Returns:
        ``True`` if the job was preempted (caller should exit 99),
        ``False`` for a normal or storage-guard stop.
    """
    global _img2dataset_proc  # noqa: PLW0603

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
    _img2dataset_proc = proc  # expose to USR1 handler
    monitor = threading.Thread(
        target=_monitor_storage,
        args=(shards_dir, limit_bytes, proc, stop_event),
        daemon=True,
    )
    monitor.start()

    returncode = proc.wait()
    _img2dataset_proc = None
    stop_event.set()  # tell the monitor thread to exit its loop
    monitor.join(timeout=5)

    # Preemption: USR1 handler sent SIGTERM to img2dataset before we exit.
    # Report cleanly and tell the caller to exit 99 for requeue.
    if _preempt_requested.is_set():
        used_gb = (
            sum(f.stat().st_size for f in shards_dir.glob("*.tar") if f.is_file()) / 1e9
        )
        print(
            f"\n[preempt] Phase 2 stopped for requeue at {used_gb:.1f} GB. "
            "img2dataset will resume from the last completed shard on next run.",
            flush=True,
        )
        return True  # caller exits 99

    # SIGTERM (returncode -15 / 143) means the storage monitor fired — that's
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

    return False


# ---------------------------------------------------------------------------
# Column inspection helper
# ---------------------------------------------------------------------------


def _list_columns(hf_token: str | None) -> None:
    """Print repo structure and column names from the first shard, then exit."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, list_repo_files

    print(f"Files in {_DATASET_NAME}:")
    all_files = list(
        list_repo_files(_DATASET_NAME, repo_type="dataset", token=hf_token)
    )
    for f in all_files[:20]:
        print(f"  {f}")
    if len(all_files) > 20:
        print(f"  ... ({len(all_files)} total)")

    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    if not parquet_files:
        print("\nNo parquet files found — check repo structure above.")
        return

    sample_url = f"hf://datasets/{_DATASET_NAME}/{parquet_files[0]}"
    print(f"\nSampling schema from {sample_url}:")
    fs = HfFileSystem(token=hf_token)
    schema = pq.read_schema(sample_url, filesystem=fs)
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

    # ── dataset subset ───────────────────────────────────────────────────────
    parser.add_argument(
        "--subset",
        default=_DEFAULT_SUBSET,
        help=(
            "Only read parquet files whose path contains this string. "
            "Use to target a specific dataset config and skip broken ones. "
            f"Default: {_DEFAULT_SUBSET!r}."
        ),
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
        default=0.2,
        help=(
            "Minimum re_clip_score (image vs LLaVA recaption). "
            "LLaVA recaptions are long and verbose, so CLIP scores them "
            "lower than short alt-text — 0.2 is a reasonable starting point. "
            "Set to 0.0 to disable. Check the 'clip' column in the filter "
            "funnel to tune."
        ),
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
        default=1000,
        help=(
            "Reject captions longer than this many words. "
            "LLaVA recaptions are verbose paragraphs (often 200–500 words) "
            "so the default is generous. Set to 0 to disable."
        ),
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

    # Install SIGUSR1 handler before doing any work so preemption is always
    # caught, even during Phase 1.
    _install_preempt_handler()

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
        result = _filter(
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
            subset=args.subset,
        )
        if result is None:
            # Preempted mid-Phase 1 — exit 99 for requeue.
            sys.exit(99)
        parquet_path = result

    # ── Phase 2: download ─────────────────────────────────────────────────────
    if args.skip_download:
        print("Skipping Phase 2 (--skip-download).")
        return

    preempted = _download(
        parquet_path,
        output_dir,
        processes=args.processes,
        threads=args.threads,
        storage_limit_gb=args.storage_limit_gb,
    )

    if preempted:
        sys.exit(99)

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
