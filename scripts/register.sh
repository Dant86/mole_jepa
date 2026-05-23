#!/usr/bin/env bash
# Register a single model in the NFS registry from the cluster.
#
# All arguments are forwarded verbatim to apps/registry/register.py, so the
# full CLI is available:
#
#   # Register from a JSON config
#   sbatch scripts/register.sh \
#       --name my_model \
#       --config-json '{"embed_dim": 128, "contrastive": false}' \
#       --description "My experiment"
#
#   # Clone config from an existing entry
#   sbatch scripts/register.sh \
#       --name my_model_v2 \
#       --from-entry my_model \
#       --description "Lower LR variant"
#
#   # List all registered models
#   sbatch scripts/register.sh --list
#
#   # Remove an entry (does NOT delete checkpoint files)
#   sbatch scripts/register.sh --deregister my_model

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=mole_jepa_register
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
exec > "${LOG_DIR}/register_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/register_${SLURM_JOB_ID}.err"

# ── run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Registering model (job ${SLURM_JOB_ID})"
echo "[$(date)] Registry path  : ${REGISTRY_PATH}"
echo "[$(date)] Checkpoint dir : ${CHECKPOINT_DIR}"

uv sync
uv run python apps/registry/register.py \
    --registry-path "${REGISTRY_PATH}" \
    "$@"

echo "[$(date)] Done."
