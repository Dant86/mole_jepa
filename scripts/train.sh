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
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
# Prevents the HuggingFace tokenizers Rust library from spawning threads
# before DataLoader forks workers, which can cause deadlocks or silent crashes.
export TOKENIZERS_PARALLELISM=false
# Reduces CUDA allocator fragmentation at large batch sizes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# ── train ─────────────────────────────────────────────────────────────────────
# Run Python in the background so we can trap SIGUSR1 here in bash and forward
# it to the Python process. Without this, --signal=B:USR1@300 delivers SIGUSR1
# to the batch script (this shell), not to the Python child.
#
# setsid launches Python as its own session/process-group leader so that
# `kill -- -$PY_PID` can signal the whole group if needed (e.g. if train_main
# ever spawns workers). The PID and PGID are identical for a setsid child.
#
# Pass --config NAME via "$@" at sbatch time, e.g.:
#   sbatch scripts/train.sh --config vit_base_bert_jepa_frozen
#
# The checkpoint directory is resolved from the NFS registry entry for NAME —
# no --checkpoint-dir flag is required.  Override with --checkpoint-dir if
# you need to redirect a run to a different path.
PY_PID=""

on_preempt() {
    echo "[$(date)] USR1 received: checkpointing before preemption."
    if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
        kill -s USR1 -- "-${PY_PID}" 2>/dev/null || kill -USR1 "${PY_PID}" 2>/dev/null || true
    fi
    # Wait for Python to finish saving the checkpoint before we exit.
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

setsid uv run python apps/train/train_main.py \
    --registry-path        "${REGISTRY_PATH}" \
    --num-epochs           100 \
    --lr                   3e-4 \
    --weight-decay         1e-4 \
    --grad-clip            1.0 \
    --batch-size           4096 \
    --num-workers          12 \
    --prefetch-factor      2 \
    --max-seq-length       64 \
    --hf-dataset-name      "${DATACOMP_LOCAL_DIR}/shards" \
    --val-shard-fraction   0.05 \
    --image-field          jpg \
    --caption-field        txt \
    "$@" &
PY_PID=$!

# Wait for Python; propagate its exit code (99 = explicit requeue request).
return_code=0
wait "${PY_PID}" || return_code=$?

if [[ "${return_code}" -eq 99 ]]; then
    echo "[$(date)] Training requested requeue via exit 99."
    exit 99
fi
exit "${return_code}"
