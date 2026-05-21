#!/usr/bin/env bash
# Apply pending schema migrations to the NFS registry.
#
# This job is pure I/O (one JSON read + write).  It requests the minimum
# SLURM resources — a single CPU, minimal memory, short wall time — and
# does not need a GPU or multiple cores.
#
# Usage:
#   sbatch scripts/migrate_registry.sh
#
# Dry run (shows which migrations would run, writes nothing):
#   sbatch scripts/migrate_registry.sh --dry-run
#
# Any extra arguments are forwarded verbatim to migrate.py.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=migrate_registry
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=0:05:00
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we start with /dev/null and exec-redirect once LOG_DIR is known.
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
exec > "${LOG_DIR}/migrate_registry_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/migrate_registry_${SLURM_JOB_ID}.err"

# ── run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting registry migration (job ${SLURM_JOB_ID})"
echo "[$(date)] Registry path: ${REGISTRY_PATH}"

uv sync
uv run python apps/registry/migrate.py \
    --registry-path "${REGISTRY_PATH}" \
    "$@"

echo "[$(date)] Done."
