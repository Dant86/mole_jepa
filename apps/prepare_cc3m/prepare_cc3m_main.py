r"""Download CC3M images and pack into WebDataset tar shards.

Images are resized to 256×256 before saving (~12 KB each vs ~150 KB originals).
The model's image processor crops to 224 anyway, so storing full-resolution
images wastes space. Dead URLs are skipped; expect ~60 % success rate,
yielding ~2 M images in ~25 GB.

Output shards use the same ``jpg`` / ``txt`` layout as ``pixparse/cc3m-wds``,
so the existing :class:`~mole_jepa.data.CC3MDataset` and train script work
unchanged — just pass the output directory as ``--hf-dataset-name``.

Usage::

    uv run python apps/prepare_cc3m/prepare_cc3m_main.py \
        --output-dir /scratch/vpathak/hf_cache/cc3m-local
"""

import argparse
import concurrent.futures
import io
import os
import tarfile
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset
from PIL import Image

_SHARD_SIZE = 5_000  # images per tar shard
_RESIZE_PX = 256  # resize shorter edge; image processor crops to 224
_TIMEOUT_S = 3  # per-image HTTP timeout — most dead URLs fail in <100 ms
_JPEG_QUALITY = 85
_DEFAULT_WORKERS = 64  # HTTP is I/O-bound; more threads than cores is fine
_STORAGE_BUDGET_GB = 45.0  # stop writing shards before the disk fills up


def _fetch(args: tuple[int, str, str]) -> tuple[int, str | None, bytes | None]:
    """Fetch one image URL and return resized JPEG bytes.

    Args:
        args: ``(index, url, caption)`` triple.

    Returns:
        ``(index, caption, jpeg_bytes)`` on success, or
        ``(index, None, None)`` if the URL is dead or the image is corrupt.
    """
    import urllib.request

    idx, url, caption = args
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        scale = _RESIZE_PX / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        return idx, caption, buf.getvalue()
    except Exception:  # noqa: BLE001
        return idx, None, None


def _write_shard(
    output_dir: Path,
    shard_idx: int,
    items: list[tuple[int, str, bytes]],
) -> Path:
    """Write one WebDataset tar shard.

    Args:
        output_dir: Directory in which to write the shard.
        shard_idx: Zero-based shard number, used to name the file.
        items: List of ``(index, caption, jpeg_bytes)`` tuples to pack.

    Returns:
        Path to the written shard file.
    """
    path = output_dir / f"cc3m-train-{shard_idx:05d}.tar"
    with tarfile.open(path, "w") as tf:
        for idx, caption, jpg_bytes in items:
            key = f"{idx:08d}"
            for ext, content in (("jpg", jpg_bytes), ("txt", caption.encode())):
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))
    return path


def _progress_path(output_dir: Path, split: str) -> Path:
    return output_dir / f".progress_{split}.json"


def _load_progress(output_dir: Path, split: str) -> tuple[int, int, int, int]:
    """Read saved progress for a split.

    Returns:
        ``(urls_consumed, ok, skip, shard_idx)`` — all zero if no progress file.
    """
    import json

    p = _progress_path(output_dir, split)
    if not p.exists():
        return 0, 0, 0, 0
    data = json.loads(p.read_text())
    return data["urls_consumed"], data["ok"], data["skip"], data["shard_idx"]


def _save_progress(
    output_dir: Path, split: str, urls_consumed: int, ok: int, skip: int, shard_idx: int
) -> None:
    import json

    _progress_path(output_dir, split).write_text(
        json.dumps(
            {
                "urls_consumed": urls_consumed,
                "ok": ok,
                "skip": skip,
                "shard_idx": shard_idx,
            }
        )
    )


def _process_split(split: str, output_dir: Path, max_workers: int) -> None:
    """Download all reachable images for one dataset split.

    Resumes automatically from saved progress if the job was previously
    interrupted — already-written shards are left untouched.

    Args:
        split: HuggingFace split name, e.g. ``"train"`` or ``"validation"``.
        output_dir: Directory in which to write tar shards.
        max_workers: Number of parallel download threads.
    """
    print(f"\n── {split} ──")
    ds = load_dataset(
        "google-research-datasets/conceptual_captions",
        split=split,
    )

    urls_consumed, ok, skip, shard_idx = _load_progress(output_dir, split)
    if urls_consumed:
        print(
            f"  resuming from URL {urls_consumed:,} "
            f"(shard {shard_idx:05d}, ok={ok:,} skip={skip:,})"
        )

    pending: list[tuple[int, str, bytes]] = []

    args_iter = (
        (
            i,
            cast(dict[str, Any], row)["image_url"],
            cast(dict[str, Any], row)["caption"],
        )  # noqa: E501
        for i, row in enumerate(ds)
        if i >= urls_consumed
    )

    budget_bytes = _STORAGE_BUDGET_GB * 1e9
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, caption, jpg in pool.map(_fetch, args_iter, chunksize=256):
            urls_consumed += 1
            if jpg is None or caption is None:
                skip += 1
                continue
            ok += 1
            pending.append((idx, caption, jpg))
            if len(pending) == _SHARD_SIZE:
                used = sum(f.stat().st_size for f in output_dir.glob("*.tar"))
                if used >= budget_bytes:
                    print(
                        f"  storage budget ({_STORAGE_BUDGET_GB:.0f} GB) reached "
                        f"after {ok:,} images — stopping."
                    )
                    break
                path = _write_shard(output_dir, shard_idx, pending)
                print(f"  shard {shard_idx:05d} → {path}  (ok={ok} skip={skip})")
                pending = []
                shard_idx += 1
                _save_progress(output_dir, split, urls_consumed, ok, skip, shard_idx)

    if pending:
        path = _write_shard(output_dir, shard_idx, pending)
        print(f"  shard {shard_idx:05d} → {path}  (ok={ok} skip={skip})")
        shard_idx += 1

    _progress_path(output_dir, split).unlink(missing_ok=True)
    print(f"{split}: {ok:,} images downloaded, {skip:,} skipped")


def main() -> None:
    """Entry point: parse arguments and prepare all splits."""
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", _DEFAULT_WORKERS))

    parser = argparse.ArgumentParser(
        description="Download CC3M and pack into WebDataset shards.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which to write tar shards.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Dataset splits to download (default: train validation).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_workers,
        help=f"Parallel download threads (default: {default_workers}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        _process_split(split, output_dir, args.max_workers)

    size_gb = sum(f.stat().st_size for f in output_dir.glob("*.tar")) / 1e9
    print(f"\nTotal on disk: {size_gb:.1f} GB")
    print("\nPass to train script:")
    print(f"  --hf-dataset-name {output_dir}")


if __name__ == "__main__":
    main()
