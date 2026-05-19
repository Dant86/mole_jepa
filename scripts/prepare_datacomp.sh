#!/usr/bin/env bash
# Filter Recap-DataComp-1B metadata and download images to WebDataset shards.
#
# Two-phase pipeline:
#   Phase 1  Read parquet shards in parallel via HfFileSystem, apply quality
#            filters, write ${DATACOMP_LOCAL_DIR}/filtered.parquet.
#            Typical runtime: 30–90 min (network I/O bound).
#   Phase 2  Call img2dataset on the filtered parquet to fetch, resize, and
#            pack ~40 M images into WebDataset tar shards under
#            ${DATACOMP_LOCAL_DIR}/shards/.
#            Typical runtime: 8–18 h (4 096 concurrent connections on 64 CPUs).
#
# Phase 1 is skipped automatically if filtered.parquet already exists.
# Phase 2 resumes from the last completed shard if the job is requeued.
#
# Usage:
#   sbatch scripts/prepare_datacomp.sh
#
# Run Phase 1 only (inspect the parquet before committing to a full download):
#   sbatch scripts/prepare_datacomp.sh --skip-download
#
# Re-run Phase 2 on an existing parquet:
#   sbatch scripts/prepare_datacomp.sh --skip-filter
#
# Any extra arguments are forwarded verbatim to prepare_datacomp_main.py.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=prepare_datacomp
#SBATCH --partition=general
#SBATCH --qos=general
# No GPU needed — both phases are CPU + network.
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
# Maximum allowed wall time. Phase 2 downloads are resumable — the job
# requeues automatically via exit code 99 until the corpus is complete.
#SBATCH --time=12:00:00
# Deliver SIGUSR1 to the batch script 300 s before wall-time, giving the
# Python process time to terminate img2dataset cleanly before we requeue.
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
# Log paths are set after sourcing .env.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
# Prevent HuggingFace tokenizers from spawning threads before fork.
export TOKENIZERS_PARALLELISM=false
# Pin NumPy/SciPy/OpenBLAS to 1 thread per process.  img2dataset spawns up to
# --processes_count worker processes, each of which imports NumPy.  Without
# this, OpenBLAS tries to create 64 threads per worker, exhausting the per-user
# thread limit (RLIMIT_NPROC) and causing "can't start new thread" failures.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Suppress albumentations version-check HTTP requests — img2dataset spawns
# 16 worker processes and each one fires this on import, adding noise to logs.
export NO_ALBUMENTATIONS_UPDATE=1
# Raise the open-file limit — img2dataset opens many sockets and tar files.
ulimit -n 65536 2>/dev/null || true

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Resolve paths relative to the submission directory, not the spool copy.
cd "$SLURM_SUBMIT_DIR"

# Load all paths and secrets from .env.
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}. Copy .env.sample and fill in your values." >&2
    exit 1
fi
source .env

# Redirect logs now that LOG_DIR is known.
mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/prepare_datacomp_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/prepare_datacomp_${SLURM_JOB_ID}.err"

mkdir -p "${DATACOMP_LOCAL_DIR}"
uv sync --group data

# ── run ───────────────────────────────────────────────────────────────────────
# Run Python in the background (setsid makes it a new session/process-group
# leader) so we can trap SIGUSR1 here in bash and forward it to Python before
# Slurm kills the job.  Python catches USR1, terminates img2dataset cleanly,
# and exits 99 to trigger a requeue.  img2dataset resumes automatically from
# the last completed shard on the next run.
#
# SLURM_CPUS_PER_TASK is read by prepare_datacomp_main.py to set the default
# --processes count for img2dataset (Phase 2 parallelism).
#
# --target-samples 60 M over-samples to account for ~30% URL mortality,
# yielding ~40 M successfully downloaded images.
# --num-filter-workers: parallel HfFileSystem threads for Phase 1.
PY_PID=""

on_preempt() {
    echo "[$(date)] USR1 received: stopping pipeline before preemption."
    if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
        kill -s USR1 -- "-${PY_PID}" 2>/dev/null || kill -USR1 "${PY_PID}" 2>/dev/null || true
    fi
    # Wait for Python to finish saving state before we exit.
    wait "${PY_PID}" || true
    echo "[$(date)] Exiting with code 99 for Slurm requeue."
    exit 99
}

on_term() {
    echo "[$(date)] TERM received: terminating without requeue."
    if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
        kill -s TERM -- "-${PY_PID}" 2>/dev/null || kill -TERM "${PY_PID}" 2>/dev/null || true
    fi
    wait "${PY_PID}" || true
    exit 143
}

trap on_preempt USR1
trap on_term TERM

setsid uv run --group data python apps/prepare_datacomp/prepare_datacomp_main.py \
    --output-dir           "${DATACOMP_LOCAL_DIR}" \
    --target-samples       60000000 \
    --num-filter-workers   "${SLURM_CPUS_PER_TASK}" \
    "$@" &
PY_PID=$!

# Wait for Python; propagate its exit code (99 = explicit requeue request).
return_code=0
wait "${PY_PID}" || return_code=$?

if [[ "${return_code}" -eq 99 ]]; then
    echo "[$(date)] Pipeline requested requeue via exit 99."
    exit 99
fi
exit "${return_code}"
