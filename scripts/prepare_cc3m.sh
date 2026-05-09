#!/usr/bin/env bash
# Download CC3M images and pack into WebDataset shards on scratch.
# Images are pre-resized to 256×256 so they fit within the storage budget.
# Dead URLs are skipped automatically. Resumes from prior progress if rerun.
#
# Usage:
#   sbatch scripts/prepare_cc3m.sh

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=prepare_cc3m
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we redirect manually below.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$SLURM_SUBMIT_DIR"

# Load all paths and secrets from .env.
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}. Copy .env.sample and fill in your values." >&2
    exit 1
fi
source .env

# Redirect logs now that LOG_DIR is known.
mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/prepare_cc3m_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/prepare_cc3m_${SLURM_JOB_ID}.err"

mkdir -p "${CC3M_LOCAL_DIR}"
uv sync

uv run python apps/prepare_cc3m/prepare_cc3m_main.py \
    --output-dir  "${CC3M_LOCAL_DIR}" \
    --splits      train validation \
    --max-workers "${SLURM_CPUS_PER_TASK}"
