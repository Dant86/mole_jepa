# MoLeJEPA

[![codecov](https://codecov.io/github/Dant86/mole_jepa/graph/badge.svg?token=EFA7KKVN22)](https://codecov.io/github/Dant86/mole_jepa)

<div align="center">
  <img src="assets/mole.svg" width="320" alt="pixel art mole">
</div>

A multimodal JEPA that predicts text embeddings from image embeddings,
regularized with SIGReg to prevent representational collapse.

## Architecture

Image encoder `f` (ViT) and text encoder `h` (LM) both project to a shared
embedding space `ℝᵈ`. A lightweight MLP predictor `g` is trained to predict
`h(y)` from `f(x)` for paired image-text inputs `(x, y)`.

The model and loss are intentionally decoupled. `MoLeJEPA.forward` returns the
raw embeddings `(f(x), g(f(x)), h(y))`; a separate loss module is composed on
top. The default training objective combines MSE prediction loss with SIGReg
regularization on both encoder outputs to prevent representational collapse:

```
L = MSE(g(f(x)), h(y))  +  λ · (SIGReg(f(x)) + SIGReg(h(y)))
```

An InfoNCE contrastive loss is also provided as an alternative or complement.

## Project Structure

```
src/mole_jepa/
├── models/
│   ├── encoders.py       # ImageEncoder (ViT), TextEncoder (LM)
│   ├── predictor.py      # MLP predictor g
│   └── mole_jepa.py      # Top-level model and MoLeJEPAOutput
├── losses/
│   ├── jepa_loss.py      # JEPALoss (MSE + SIGReg)
│   └── info_nce.py       # InfoNCELoss
├── regularizers/
│   └── sig_reg.py        # SIGReg
└── test_statistics/
    └── epps_pulley.py    # EppsPulley base + Gaussian / Laplace variants
```

## Development

### Setup

Install [uv](https://docs.astral.sh/uv/), then install all dependencies
including dev tools:

```bash
uv sync --group dev
uv run pre-commit install
```

### Running Tests

```bash
uv run pytest
```

Coverage is measured automatically on every run and written to `coverage.xml`.
A terminal summary is printed at the end of the test output.

### Code Style

This project enforces consistent style via **ruff** (lint + format) and
**pyright** (static type checking). Pre-commit hooks run both tools
automatically on every commit. The same checks run in CI on every push and
pull request.

#### Imports

Imports follow Google style — stdlib first, then third-party, then local —
each group separated by a blank line. `ruff` (isort rules) enforces the
ordering automatically.

Import **modules**, not names from modules. The only exception is the
`typing` module, whose members may be imported directly — but only when
`typing` is the canonical home. Names that have migrated to `collections.abc`
(e.g. `Iterator`, `Generator`, `Callable`) must be imported from there as a
module, not from `typing`.

```python
# stdlib
import collections.abc
import dataclasses
from typing import Literal

# third-party
import torch
import transformers

# local
from mole_jepa import models
from mole_jepa import test_statistics
```

#### Docstrings

All public functions, classes, and methods require Google-style docstrings.
Module and package-level docstrings are optional.

```python
def encode(text: str, max_len: int = 512) -> torch.Tensor:
    """Encode a text string into a token tensor.

    Args:
        text: The input string.
        max_len: Maximum sequence length; longer sequences are truncated.

    Returns:
        A 1-D integer tensor of token IDs.

    Raises:
        ValueError: If `text` is empty.
    """
```

#### Running checks manually

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # format check (drop --check to auto-fix)
uv run pyright                # type check
```

## Training on the DSI Cluster

### 1. Environment setup

All filesystem paths and secrets live in a gitignored `.env` file. Copy the
sample and fill in your values before running any script:

```bash
cp .env.sample .env
# edit .env — set HF_HOME, HF_TOKEN, CC3M_LOCAL_DIR, CHECKPOINT_DIR, LOG_DIR
```

### 2. Prepare the dataset (one-time)

Downloads CC3M images, resizes them to 256×256, and packs them into
WebDataset tar shards on scratch. Skips dead URLs automatically and resumes
from prior progress if the job is preempted.

```bash
sbatch scripts/prepare_cc3m.sh
```

Expected runtime: 8–12 hours with 64 workers. Monitor progress in
`$LOG_DIR/prepare_cc3m_<job_id>.out` — a line is printed per completed shard
(5,000 images each). The script stops automatically when `$CC3M_LOCAL_DIR`
reaches the storage budget configured in `prepare_cc3m_main.py`.

### 3. Train

```bash
sbatch scripts/train.sh
```

The script auto-detects an existing checkpoint in `$CHECKPOINT_DIR` and passes
`--resume` if one is found. On preemption, SLURM sends `SIGUSR1` five minutes
before the wall-time limit; the train script catches it, saves a checkpoint,
and exits so the job can be requeued.

Logs are written to `$LOG_DIR/train_<job_id>.out` and `.err`.
Training stats (loss, batch counts) are appended to `stats.jsonl` inside the
checkpoint directory every 5 epochs.
