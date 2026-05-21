#!/usr/bin/env bash
# One-time import: copy configs from registry.py into the NFS registry.
#
# For each named config in src/mole_jepa/registry.py, the script computes
# the checkpoint directory path (checkpoint_base / config_hash) and writes
# an entry into the NFS registry JSON.  No data is moved or renamed.
#
# Usage:
#   sbatch scripts/import_registry.sh
#
# Dry run (shows what would be imported, writes nothing):
#   sbatch scripts/import_registry.sh --dry-run
#
# Skip configs whose checkpoint directory doesn't exist on disk:
#   sbatch scripts/import_registry.sh --skip-missing
#
# Any extra arguments are forwarded verbatim to import_from_registry_py.py.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=import_registry
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=0:05:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}. Copy .env.sample and fill in your values." >&2
    exit 1
fi
source .env

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/import_registry_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/import_registry_${SLURM_JOB_ID}.err"

# ── run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting registry import (job ${SLURM_JOB_ID})"
echo "[$(date)] Checkpoint base: ${CHECKPOINT_DIR}"
echo "[$(date)] Registry path  : ${REGISTRY_PATH}"

uv sync
uv run python apps/registry/import_from_registry_py.py \
    --checkpoint-base "${CHECKPOINT_DIR}" \
    --registry-path   "${REGISTRY_PATH}" \
    "$@"

echo "[$(date)] Done."
