#!/usr/bin/env bash
# Filter Recap-DataComp-1B metadata and download images to WebDataset shards.
#
# Two-phase pipeline:
#   Phase 1  Stream HuggingFace metadata, apply quality filters, write
#            ${DATACOMP_LOCAL_DIR}/filtered.parquet  (~3–6 h, CPU-bound).
#   Phase 2  Call img2dataset on the filtered parquet to fetch, resize, and
#            pack ~40 M images into tar shards under
#            ${DATACOMP_LOCAL_DIR}/shards/  (~12–24 h, network-bound).
#
# Phase 1 is skipped automatically if filtered.parquet already exists.
# Phase 2 resumes from the last completed shard if the job is requeued.
#
# Usage:
#   sbatch scripts/prepare_datacomp.sh
#
# To run Phase 1 only (useful for inspecting the parquet before committing
# to a full download):
#   sbatch scripts/prepare_datacomp.sh --skip-download
#
# To (re-)run Phase 2 on an existing parquet:
#   sbatch scripts/prepare_datacomp.sh --skip-filter
#
# Any extra arguments are forwarded to prepare_datacomp_main.py.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=prepare_datacomp
#SBATCH --partition=general
#SBATCH --qos=general
# No GPU needed — download is purely CPU + network.
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
# Allow up to 48 h; Phase 2 typically finishes in 12–24 h on fast storage.
#SBATCH --time=48:00:00
# Log paths are set after sourcing .env.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
# Prevent HuggingFace tokenizers from spawning threads before fork.
export TOKENIZERS_PARALLELISM=false

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
# SLURM_CPUS_PER_TASK is read inside prepare_datacomp_main.py to set the
# default --processes count for img2dataset.
uv run --group data python apps/prepare_datacomp/prepare_datacomp_main.py \
    --output-dir      "${DATACOMP_LOCAL_DIR}" \
    --target-samples  40000000 \
    --clip-threshold  0.28 \
    "$@"
