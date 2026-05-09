#!/usr/bin/env bash
# Submit a MoLeJEPA training job to the DSI cluster.
#
# Usage:
#   sbatch scripts/train.sh
#
# Override any #SBATCH default at submission time, e.g.:
#   sbatch --qos=protected --time=48:00:00 scripts/train.sh

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=mole_jepa_train
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we redirect manually below.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# SLURM copies the script to a spool directory; use SLURM_SUBMIT_DIR so that
# relative paths resolve correctly.
cd "$SLURM_SUBMIT_DIR"

# Load all paths and secrets from .env.
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}. Copy .env.sample and fill in your values." >&2
    exit 1
fi
source .env

# Redirect logs now that LOG_DIR is known.
mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/train_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/train_${SLURM_JOB_ID}.err"

mkdir -p "${CHECKPOINT_DIR}"
uv sync

# ── resume detection ──────────────────────────────────────────────────────────
RESUME_FLAG=""
if compgen -G "${CHECKPOINT_DIR}/*/checkpoint.pt" > /dev/null 2>&1; then
    RESUME_FLAG="--resume"
fi

# ── train ─────────────────────────────────────────────────────────────────────
uv run python apps/train/train_main.py \
    --checkpoint-dir       "${CHECKPOINT_DIR}" \
    --num-epochs           100 \
    --lr                   1e-4 \
    --weight-decay         1e-4 \
    --grad-clip            1.0 \
    --batch-size           256 \
    --num-workers          8 \
    --max-seq-length       64 \
    --hf-dataset-name      "${CC3M_LOCAL_DIR}" \
    --hf-dataset-split     train \
    --val-hf-dataset-name  "${CC3M_LOCAL_DIR}" \
    --val-hf-dataset-split train \
    --image-field          jpg \
    --caption-field        txt \
    --embed-dim            256 \
    --predictor-hidden-dim 512 \
    --predictor-n-layers   2 \
    --sigreg-n-directions  128 \
    --jepa-reg-weight      1.0 \
    $RESUME_FLAG
