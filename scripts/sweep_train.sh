#!/usr/bin/env bash
# Submit a MoLeJEPA sweep training job (A100, short runs with per-epoch retrieval eval).
#
# Usage:
#   sbatch scripts/sweep_train.sh --config sweep_sphere_lam1

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=mole_jepa_sweep
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}." >&2
    exit 1
fi
source .env

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export SUPABASE_URL="${SUPABASE_URL:-}"
export SUPABASE_ANON_KEY="${SUPABASE_ANON_KEY:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$SLURM_SUBMIT_DIR"

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/sweep_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/sweep_${SLURM_JOB_ID}.err"

mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${CHECKPOINT_DIR}"
uv sync
# No flash-attn here: every sweep config uses attn_implementation="sdpa".
# Installing it anyway is actively harmful — ModernBERT eagerly imports
# flash_attn at module load time if it's merely *installed* (regardless of
# the chosen backend), and a broken/partial build (likely on generic gpu:1
# allocations across heterogeneous nodes) crashes model construction even
# though sdpa never needed it.

# ── train ─────────────────────────────────────────────────────────────────────
PY_PID=""

on_preempt() {
    echo "[$(date)] USR1 received: checkpointing before preemption."
    if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
        kill -s USR1 -- "-${PY_PID}" 2>/dev/null || kill -USR1 "${PY_PID}" 2>/dev/null || true
    fi
    wait "${PY_PID}" || true
    echo "[$(date)] Exiting with code 99 for Slurm requeue."
    exit 99
}

on_term() {
    echo "[$(date)] TERM received: terminating without requeue."
    if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
        kill -s TERM -- "-${PY_PID}" 2>/dev/null || kill -TERM "${PY_PID}" 2>/dev/null || true
    fi
    wait "${PY_PID}" || true
    exit 143
}

trap on_preempt USR1
trap on_term TERM

setsid uv run --no-sync python apps/train/train_main.py \
    --num-epochs           8 \
    --gradient-checkpointing \
    --lr                   1e-4 \
    --weight-decay         1e-4 \
    --grad-clip            1.0 \
    --batch-size           512 \
    --num-workers          8 \
    --prefetch-factor      4 \
    --max-seq-length       64 \
    --hf-dataset-name      "${DATACOMP_LOCAL_DIR}/shards" \
    --val-shard-fraction   0.05 \
    --image-field          jpg \
    --caption-field        txt \
    --max-train-samples    2000000 \
    --text-encoder-lr-multiplier  1.0 \
    --image-encoder-lr-multiplier 1.0 \
    --eval-retrieval \
    "$@" &
PY_PID=$!

return_code=0
wait "${PY_PID}" || return_code=$?

if [[ "${return_code}" -eq 99 ]]; then
    echo "[$(date)] Training requested requeue via exit 99."
    exit 99
fi
exit "${return_code}"
