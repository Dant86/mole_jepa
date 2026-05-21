#!/usr/bin/env bash
# Sweep DataLoader settings to find the highest batch/workers/prefetch that
# fits in system RAM on the target node.
#
# Usage:
#   sbatch scripts/smoketest.sh
#
# Results are printed to ${LOG_DIR}/smoketest_<JOBID>.out.
# Copy the "Best OK" line's flags into scripts/train.sh.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=mole_jepa_smoketest
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
exec > "${LOG_DIR}/smoketest_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/smoketest_${SLURM_JOB_ID}.err"

uv sync

# ── sweep ─────────────────────────────────────────────────────────────────────
# The optimal settings depend on model size: a tiny model leaves more room for
# larger batches, while a big model shrinks the VRAM headroom.  Pass --config
# to load the real model and run genuine forward+backward passes.
#
# Required: pass --config at sbatch time, e.g.:
#   sbatch scripts/smoketest.sh --config vit_base_bert_jepa_frozen
#
# Each configuration runs in an isolated subprocess. If the kernel OOM-kills
# a subprocess its exit code is non-zero and it is marked FAIL/OOM.
# The sweep goes from smallest to largest; pick the best OK row for train.sh.

if [[ "$*" != *"--config"* ]]; then
    echo "ERROR: --config NAME is required. Example:" >&2
    echo "  sbatch scripts/smoketest.sh --config vit_base_bert_jepa_frozen" >&2
    exit 1
fi

uv run python apps/train/smoketest.py \
    --registry-path     "${REGISTRY_PATH}" \
    --hf-dataset-name   "${DATACOMP_LOCAL_DIR}/shards" \
    --image-field       jpg \
    --caption-field     txt \
    --batch-sizes       512 1024 2048 4096 \
    --worker-counts     4 8 12 \
    --prefetch-factors  2 4 \
    --warmup-batches    5 \
    --num-batches       30 \
    --timeout           360 \
    "$@"
