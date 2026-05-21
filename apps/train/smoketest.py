r"""Data-pipeline throughput sweep for MoLeJEPA training.

Sweeps every combination of batch_size × num_workers × prefetch_factor and
reports samples/sec for each.  Each configuration runs in an **isolated
subprocess** — if the kernel OOM-kills a configuration its exit code is
non-zero and it is marked FAIL, leaving the remaining configurations
unaffected.

Run directly (e.g. inside an interactive SLURM session):

    uv run python apps/train/smoketest.py \\
        --hf-dataset-name /path/to/datacomp/shards \\
        --image-field jpg --caption-field txt

Or submit via sbatch:

    sbatch scripts/smoketest.sh

The output is a table of results — pick the highest-throughput OK row and
copy its settings into scripts/train.sh.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

# ── sentinel that switches the script into single-config worker mode ──────────
_WORKER_FLAG = "--bench-worker"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--hf-dataset-name",
        required=True,
        metavar="DIR",
        help="Local shard directory (the same path used in train.sh).",
    )
    parser.add_argument("--image-field", default="jpg")
    parser.add_argument("--caption-field", default="txt")

    # ── sweep grid ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        metavar="N",
        help="Batch sizes to sweep (default: 512 1024 2048 4096).",
    )
    parser.add_argument(
        "--worker-counts",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        metavar="N",
        help="num_workers values to sweep (default: 4 8 12).",
    )
    parser.add_argument(
        "--prefetch-factors",
        type=int,
        nargs="+",
        default=[2, 4],
        metavar="N",
        help="prefetch_factor values to sweep (default: 2 4).",
    )

    # ── timing ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Batches to discard before starting the timer (default: 5).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=30,
        help="Batches to time per configuration (default: 30).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=360,
        help="Per-configuration subprocess timeout in seconds (default: 360).",
    )

    # ── internal: worker mode args (hidden from help) ─────────────────────────
    parser.add_argument(
        _WORKER_FLAG, dest="bench_worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_batch-size", type=int, dest="w_batch_size", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_num-workers", type=int, dest="w_num_workers", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_prefetch", type=int, dest="w_prefetch", help=argparse.SUPPRESS
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Worker: runs one configuration, prints JSON result to stdout
# ─────────────────────────────────────────────────────────────────────────────


def _run_worker(args: argparse.Namespace) -> None:
    import torch

    from mole_jepa import config as config_module
    from mole_jepa import data as data_module

    batch_size = args.w_batch_size
    num_workers = args.w_num_workers
    prefetch = args.w_prefetch

    data_cfg = config_module.DataConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        image_field=args.image_field,
        caption_field=args.caption_field,
    )

    shards = data_module.find_tar_shards(args.hf_dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = data_module.build_datacomp_loader(shards, data_cfg, device, shuffle=False)

    total_batches = args.warmup_batches + args.num_batches
    samples_timed = 0
    t0: float | None = None

    for i, (pv, _ids, _mask) in enumerate(loader):
        if i >= total_batches:
            break
        if i == args.warmup_batches:
            t0 = time.monotonic()
        if t0 is not None:
            # Move to device — tests pin_memory / H2D transfer
            pv.to(device, non_blocking=True)
            samples_timed += pv.shape[0]

    elapsed = (time.monotonic() - t0) if t0 is not None else 0.0
    sps = samples_timed / elapsed if elapsed > 0 else 0.0

    print(json.dumps({"samples_per_sec": sps, "batches_timed": args.num_batches}))


# ─────────────────────────────────────────────────────────────────────────────
# Driver: sweep configs, spawn subprocesses, print table
# ─────────────────────────────────────────────────────────────────────────────


def _run_driver(args: argparse.Namespace) -> None:
    configs = list(
        itertools.product(args.batch_sizes, args.worker_counts, args.prefetch_factors)
    )

    col = dict(batch=8, workers=7, prefetch=8, sps=14, status=8)
    hdr = (
        f"{'batch':>{col['batch']}}  "
        f"{'workers':>{col['workers']}}  "
        f"{'prefetch':>{col['prefetch']}}  "
        f"{'samples/s':>{col['sps']}}  "
        f"status"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    results: list[tuple[int, int, int, float | None, str]] = []

    for batch_size, num_workers, prefetch in configs:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--hf-dataset-name",
            args.hf_dataset_name,
            "--image-field",
            args.image_field,
            "--caption-field",
            args.caption_field,
            "--warmup-batches",
            str(args.warmup_batches),
            "--num-batches",
            str(args.num_batches),
            _WORKER_FLAG,
            "--_batch-size",
            str(batch_size),
            "--_num-workers",
            str(num_workers),
            "--_prefetch",
            str(prefetch),
        ]

        status = "?"
        sps: float | None = None

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
        else:
            if proc.returncode != 0:
                status = "OOM/FAIL"
            else:
                try:
                    # Take the last JSON line (subprocess may print warnings before it)
                    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
                    last_line = lines[-1]
                    data = json.loads(last_line)
                    sps = data["samples_per_sec"]
                    status = "OK"
                except Exception:
                    status = "PARSE_ERR"

        sps_str = f"{sps:,.0f}" if sps is not None else ""
        print(
            f"{batch_size:>{col['batch']}}  "
            f"{num_workers:>{col['workers']}}  "
            f"{prefetch:>{col['prefetch']}}  "
            f"{sps_str:>{col['sps']}}  "
            f"{status}"
        )
        results.append((batch_size, num_workers, prefetch, sps, status))

    print(sep)

    ok = [(b, w, p, s) for b, w, p, s, st in results if st == "OK" and s is not None]
    if ok:
        best = max(ok, key=lambda x: x[3])  # highest samples/s
        print(
            f"\nBest OK configuration:  "
            f"--batch-size {best[0]}  "
            f"--num-workers {best[1]}  "
            f"--prefetch-factor {best[2]}  "
            f"→ {best[3]:,.0f} samples/s"
        )
        print("Copy those flags into scripts/train.sh.")
    else:
        print("\nNo configurations completed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: dispatch to driver or worker based on CLI flags."""
    args = _parse_args()
    if args.bench_worker:
        _run_worker(args)
    else:
        _run_driver(args)


if __name__ == "__main__":
    main()
