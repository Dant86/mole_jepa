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
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we redirect manually below.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

# Load all paths and secrets from .env.
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}. Copy .env.sample and fill in your values." >&2
    exit 1
fi
source .env

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$SLURM_SUBMIT_DIR"

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/probe_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/probe_${SLURM_JOB_ID}.err"

mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}"

export HF_TOKEN="${HF_TOKEN:-}"

uv sync
uv pip install "flash-attn<2.8" --no-build-isolation --no-cache-dir

# ── evaluate ──────────────────────────────────────────────────────────────────
uv run python apps/eval/linear_probe_main.py \
    "$@"
