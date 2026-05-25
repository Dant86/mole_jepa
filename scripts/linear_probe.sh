#!/usr/bin/env bash
# Submit a MoLeJEPA linear probe evaluation job to the DSI cluster.
#
# Usage:
#   sbatch scripts/linear_probe.sh \
#       --config vit_small_miniml_jepa_frozen \
#                vit_small_miniml_jepa_unfrozen \
#                vit_small_miniml_infonce_frozen \
#       --dataset tanganke/stl10

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=mole_jepa_probe
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
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
exec > "${LOG_DIR}/probe_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/probe_${SLURM_JOB_ID}.err"

export HF_TOKEN="${HF_TOKEN:-}"

uv sync

# ── evaluate ──────────────────────────────────────────────────────────────────
uv run python apps/eval/linear_probe_main.py \
    --registry-path "${REGISTRY_PATH}" \
    "$@"
