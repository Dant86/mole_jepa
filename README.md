# MoLeJEPA

[![codecov](https://codecov.io/github/Dant86/mole_jepa/graph/badge.svg?token=EFA7KKVN22)](https://codecov.io/github/Dant86/mole_jepa)

<div align="center">
  <img src="assets/digby.png" width="180" alt="Digby the burrowing mole">
</div>

A multimodal JEPA that predicts text embeddings from image embeddings,
regularized with SIGReg to prevent representational collapse.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Loading Models & Checkpoints](#loading-models--checkpoints)
3. [Training on the DSI Cluster](#training-on-the-dsi-cluster)
4. [Contributing](#contributing)

---

## Architecture

<div align="center">
  <img src="assets/arch.png" width="720" alt="MoLeJEPA architecture diagram">
</div>

Image encoder $f$ (ViT) and text encoder $h$ (LM) both project to a shared
embedding space $\mathbb{R}^d$. A lightweight MLP predictor $g$ is trained to
predict $h(y)$ from $f(x)$ for paired image-text inputs $(x, y)$.

The model and loss are intentionally decoupled. `MoLeJEPA.forward` returns the
raw embeddings $(f(x),\, g(f(x)),\, h(y))$; a separate loss module is composed
on top. The default training objective follows LeJEPA (eq. 4):

$$L = \lambda \cdot \bigl(\mathrm{SIGReg}(f(x)) + \mathrm{SIGReg}(h(y))\bigr) + (1 - \lambda) \cdot \mathrm{MSE}\bigl(g(f(x)),\, h(y)\bigr)$$

The default $\lambda = 0.05$ (per the LeJEPA paper) keeps MSE as the dominant
signal while a small regularization term prevents representational collapse.
An InfoNCE contrastive loss is also provided as an alternative.

### Project Structure

```
src/mole_jepa/
├── models/
│   ├── encoders.py       # ImageEncoder (ViT), TextEncoder (LM)
│   ├── predictor.py      # MLP predictor g
│   └── mole_jepa.py      # Top-level model and MoLeJEPAOutput
├── losses/
│   ├── jepa_loss.py      # JEPALoss (λ·MSE + (1-λ)·SIGReg)
│   └── info_nce.py       # InfoNCELoss
├── regularizers/
│   └── sig_reg.py        # SIGReg
├── test_statistics/
│   └── epps_pulley.py    # EppsPulley base + Gaussian / Laplace variants
├── config.py             # Plain-data ModelConfig and DataConfig
└── factory.py            # build(ModelConfig) → (MoLeJEPA, loss)
```

---

## Loading Models & Checkpoints

Checkpoints are stored under a root directory (set via `CHECKPOINT_DIR` in
`.env`). Each run lives in its own subdirectory named by a SHA-256 hash of its
`ModelConfig`, so different hyperparameter combinations never collide.

### List available checkpoints

```python
from mole_jepa import model_io

models = model_io.list_models("/path/to/checkpoints")
for m in models:
    print(m.config_hash[:8], m.config)
```

Filter by config fields to narrow results:

```python
contrastive_models = model_io.list_models(
    "/path/to/checkpoints",
    contrastive=True,
    embed_dim=256,
)
```

### Load a model for inference

```python
from mole_jepa import config as cfg_module, model_io

config = cfg_module.ModelConfig(
    embed_dim=256,
    contrastive=False,
    # ... other fields matching the saved run
)

model = model_io.load_model(config, "/path/to/checkpoints")
model.eval()
```

`load_model` builds the architecture via `factory.build` and loads the saved
state dict onto CPU by default. Pass `map_location="cuda"` to load directly
onto GPU:

```python
model = model_io.load_model(config, "/path/to/checkpoints", map_location="cuda")
```

### Load weights into an existing model

If you have already constructed the model (e.g. inside a training loop), use
`load_model_weights` to avoid a redundant `from_pretrained` call:

```python
from mole_jepa import factory, model_io

model, loss_fn = factory.build(config)
model_io.load_model_weights(model, config, "/path/to/checkpoints")
```

### Run inference

```python
from mole_jepa.data import transforms

# Preprocess a single image + caption
pixel_values, input_ids, attention_mask = transforms.preprocess(
    image,       # PIL.Image
    caption,     # str
    data_config, # mole_jepa.config.DataConfig
)

with torch.no_grad():
    output = model(pixel_values, input_ids, attention_mask)

# output.image_embeddings  — f(x), shape (1, embed_dim)
# output.predicted_text_embeddings — g(f(x)), shape (1, embed_dim)
# output.text_embeddings   — h(y), shape (1, embed_dim)
```

---

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

The script auto-resumes if a checkpoint for the exact current hyperparameter
configuration exists in `$CHECKPOINT_DIR` (matched by a SHA-256 hash of the
config). Changing any hyperparameter starts a fresh run automatically — no
flags needed. On preemption, SLURM sends `SIGUSR1` five minutes before the
wall-time limit; the train script catches it, saves a checkpoint, and exits so
the job can be requeued.

Logs are written to `$LOG_DIR/train_<job_id>.out` and `.err`.
Training stats (loss, batch counts) are appended to `stats.jsonl` inside the
checkpoint directory every epoch.

---

## Contributing

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
