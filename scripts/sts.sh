#!/usr/bin/env bash
# Submit a MoLeJEPA STS-B text encoder evaluation job to the DSI cluster.
#
# Usage:
#   sbatch scripts/sts.sh \
#       --config vit_small_miniml_jepa_frozen \
#                vit_small_miniml_jepa_unfrozen \
#                vit_small_miniml_infonce_frozen

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=mole_jepa_sts
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}"

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}." >&2
    exit 1
fi
source .env

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/sts_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/sts_${SLURM_JOB_ID}.err"

export HF_TOKEN="${HF_TOKEN:-}"

uv sync

# ── evaluate ──────────────────────────────────────────────────────────────────
uv run python apps/eval/sts_main.py \
    --registry-path "${REGISTRY_PATH}" \
    "$@"
